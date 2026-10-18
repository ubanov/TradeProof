"""PrePMSnapshot: an immutable, append-only record of exactly the state the
Portfolio Manager node receives, frozen the instant before it runs.

Separate from (and does not replace) ``recommendation_record.py``'s
RecommendationRecord -- that is an audit record of what was FINALLY
recommended; this is the frozen INPUT the Portfolio Manager decided from,
so a later experiment (a different prompt, a different model) can replay
only the Portfolio Manager against the exact same evidence, without
re-running Analysts, the research debate, the Trader, or Risk.

No parallel state model is introduced: ``state`` here is the real
``AgentState`` dict the graph already threads through every node, captured
as-is (minus ``messages``, LangGraph's own non-JSON-safe tool-calling
history, which the Portfolio Manager never reads -- see
``agents/managers/portfolio_manager.py`` and
``agents/risk_mgmt/integrity_check.py``). ``run_provenance`` is the same
structure ``recommendation_record.py``'s RunRef/ModelsRef are themselves
built from (see ``tradingagents/run_provenance.py``), embedded directly
rather than re-typed -- including the resolved Decision Memory mode
(``run_provenance["execution"]["decision_memory_enabled"/"decision_memory_root"]``).

Written automatically, unconditionally, for every run that goes through
``cli/run.py::execute_analysis`` (see fork README: "PrePMSnapshot") -- no
opt-in flag. A bare ``propagate()``/backtest caller that never reaches
``execute_analysis`` does not get one, the same scoping RecommendationRecord
already has.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from tradingagents.dataflows.symbols import safe_ticker_component
from tradingagents.recommendation_record import compute_run_fingerprint, resolve_ledger_root

SCHEMA_VERSION = "1.0"

# The one AgentState key never frozen into a snapshot: LangGraph's own
# message/tool-calling history (LangChain BaseMessage objects, not
# JSON-safe) -- the Portfolio Manager and its deterministic checks never
# read it (see module docstring).
_EXCLUDED_STATE_KEYS = ("messages",)


class IncompletePrePMSnapshotError(ValueError):
    """Raised when ``state`` is missing a field a snapshot cannot exist
    without (the ticker or the analysis date). The caller logs and skips
    the write rather than breaking the actual Portfolio Manager call over
    a secondary artifact (same posture as RecommendationRecord's own
    ``IncompleteRecommendationError``)."""


class PrePMSnapshotIntegrityError(ValueError):
    """Raised by ``load_pre_pm_snapshot`` when a loaded file's schema or
    content_hash does not check out -- never auto-repaired or silently
    accepted."""


class PrePMRunRef(BaseModel):
    ticker: str
    analysis_date: str
    # Best-effort: ``config["results_dir"]`` means different things for the
    # interactive vs. headless flows (see fork README: "RecommendationRecord"),
    # so this is a pointer for manual cross-reference, never authoritative.
    results_dir: str | None = None


class PrePMSnapshot(BaseModel):
    schema_version: str = SCHEMA_VERSION
    snapshot_id: str
    created_at: str
    content_hash: str
    run_fingerprint: str
    run: PrePMRunRef
    run_provenance: dict[str, Any]
    state: dict[str, Any]


# ---------------------------------------------------------------------------
# Canonical JSON / hashing -- same convention as
# ``recommendation_record.canonical_json`` / ``evidence_archive.schema``,
# duplicated locally rather than imported across modules for this one-line
# function (established project convention, see ``dataflows/price_basis.py``).
# ---------------------------------------------------------------------------


def _canonical_json(data: Any) -> bytes:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def compute_pre_pm_content_hash(payload_without_hash: dict[str, Any]) -> str:
    """``sha256(canonical_json(payload without content_hash))``."""
    return hashlib.sha256(_canonical_json(payload_without_hash)).hexdigest()


def verify_pre_pm_snapshot_integrity(path: Path) -> bool:
    """Recompute ``content_hash`` from the file exactly as stored and compare."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    stored_hash = payload.get("content_hash")
    without_hash = {key: value for key, value in payload.items() if key != "content_hash"}
    return stored_hash == compute_pre_pm_content_hash(without_hash)


# ---------------------------------------------------------------------------
# Building the snapshot from the real AgentState dict
# ---------------------------------------------------------------------------


def build_pre_pm_snapshot(
    *, state: dict[str, Any], run_provenance: dict[str, Any], results_dir: str | None = None,
) -> PrePMSnapshot:
    """Freeze ``state`` (the Portfolio Manager node's real input) and
    ``run_provenance`` (this run's models/seed/Decision-Memory-mode/git
    provenance) into one immutable, hashable snapshot.

    ``state`` is captured as every key except ``messages`` -- never a
    hand-picked subset re-describing it, so nothing the Portfolio Manager
    (or its deterministic Risk-integrity check, which reads the same
    ``state``) actually consults can be silently missing.
    """
    ticker = str(state.get("company_of_interest") or "").strip()
    analysis_date = str(state.get("trade_date") or "").strip()
    if not ticker or not analysis_date:
        raise IncompletePrePMSnapshotError(
            "state is missing company_of_interest/trade_date; no snapshot to build"
        )

    frozen_state = {k: v for k, v in state.items() if k not in _EXCLUDED_STATE_KEYS}
    run_fingerprint = compute_run_fingerprint(
        ticker=ticker, analysis_date=analysis_date, run_provenance=run_provenance
    )

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "snapshot_id": str(uuid.uuid4()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_fingerprint": run_fingerprint,
        "run": {"ticker": ticker, "analysis_date": analysis_date, "results_dir": results_dir},
        "run_provenance": run_provenance,
        "state": frozen_state,
    }
    payload["content_hash"] = compute_pre_pm_content_hash(payload)
    return PrePMSnapshot.model_validate(payload)


# ---------------------------------------------------------------------------
# Storage: one immutable JSON file per snapshot, write-once, under the same
# ledger tree RecommendationRecord uses (a sibling of recommendations/).
# ---------------------------------------------------------------------------


def resolve_pre_pm_root(config: dict[str, Any] | None) -> Path:
    """``<ledger_root>/pre_pm`` -- a sibling of
    ``<ledger_root>/recommendations``, never mixed with ``logs/``, `debug/``
    or Decision Memory (see fork README: "PrePMSnapshot")."""
    return resolve_ledger_root(config) / "pre_pm"


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    data = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def write_pre_pm_snapshot(snapshot: PrePMSnapshot, ledger_root: Path) -> Path:
    """Write ``<ledger_root>/pre_pm/<ticker>/<snapshot_id>.json``.

    Never overwrites an existing file at that path: identical content is a
    silent no-op (idempotent replay of this same call), any difference
    raises -- a path collision essentially never happens (fresh UUID4 per
    snapshot), but the check costs nothing (same guarantee as
    ``recommendation_record.write_recommendation_record``).
    """
    payload = snapshot.model_dump(mode="json")
    ticker_dir = Path(ledger_root) / "pre_pm" / safe_ticker_component(snapshot.run.ticker)
    final_path = ticker_dir / f"{snapshot.snapshot_id}.json"
    if final_path.exists():
        existing = json.loads(final_path.read_text(encoding="utf-8"))
        if existing == payload:
            return final_path
        raise FileExistsError(f"PrePMSnapshot already exists and differs: {final_path}")
    _atomic_write_json(final_path, payload)
    return final_path


def load_pre_pm_snapshot(path: Path) -> PrePMSnapshot:
    """Load and validate a snapshot file: schema-validates via Pydantic,
    then verifies ``content_hash`` -- raises ``PrePMSnapshotIntegrityError``
    on any mismatch rather than returning a possibly-tampered snapshot."""
    if not verify_pre_pm_snapshot_integrity(path):
        raise PrePMSnapshotIntegrityError(f"PrePMSnapshot content_hash mismatch: {path}")
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return PrePMSnapshot.model_validate(payload)


# ---------------------------------------------------------------------------
# Writing hook: wraps a Portfolio Manager node function so the snapshot is
# written BEFORE it runs, from the exact same ``state`` object the node
# itself then receives unmodified -- see module docstring and
# ``graph/setup.py``. Never used during replay (``pm_replay.py`` calls
# ``create_portfolio_manager(...)`` directly, with no wrapper), so a replay
# can never create a stray new snapshot or touch an existing one.
# ---------------------------------------------------------------------------


def with_pre_pm_snapshot(node_fn, *, run_provenance: dict[str, Any], ledger_root: Path, results_dir: str | None = None):
    def wrapped(state: dict[str, Any]) -> dict:
        snapshot_ref = None
        try:
            snapshot = build_pre_pm_snapshot(
                state=state, run_provenance=run_provenance, results_dir=results_dir
            )
            written_path = write_pre_pm_snapshot(snapshot, ledger_root)
            # Merged into this node's own return below so it flows through
            # the graph exactly like portfolio_decision_structured already
            # does, reaching result.json / RecommendationRecord (fork
            # README: "PrePMSnapshot") without any separate plumbing.
            snapshot_ref = {
                "snapshot_id": snapshot.snapshot_id,
                "content_hash": snapshot.content_hash,
                "path": str(written_path),
            }
        except Exception as exc:  # never break the real analysis over this
            ticker = state.get("company_of_interest", "?")
            print(f"[WARN] PrePMSnapshot not written for {ticker}: {exc}", flush=True)
        result = node_fn(state)
        if snapshot_ref is not None:
            result = {**result, "pre_pm_snapshot_ref": snapshot_ref}
        return result

    return wrapped
