"""PM replay: re-run ONLY the Portfolio Manager against a frozen
PrePMSnapshot (see ``pre_pm_snapshot.py`` and fork README: "PrePMSnapshot"),
producing a PMReplayRecord -- never the original RecommendationRecord, never
a RecommendationOutcome.

Structurally, not just by convention, a replay cannot run Analysts, the
research debate, Research Manager, Trader, or Risk: this module never
constructs a ``TradingAgentsGraph`` or compiles the LangGraph workflow at
all. It calls ``create_portfolio_manager(...)`` -- a bare function taking a
state dict and returning a dict -- directly, exactly as the real graph node
does, against a *deep copy* of the snapshot's frozen ``state`` (so a replay
can never mutate the snapshot it was loaded from). The Portfolio Manager
node never calls a market-data tool and never consults a live
``TradingMemoryLog``; its only "past context" is whatever string the
snapshot already froze into ``state["past_context"]`` -- so a replay is
provably independent of Decision Memory and of any price/market lookup,
regardless of what the original run's Decision Memory mode was.

Generation (replaying a decision) stays deliberately separate from
evaluation: this module never looks at a later price, never computes a
return, and never decides which replay is "better" -- that remains the
evaluator/backtester's job (see fork README: "PrePMSnapshot").
"""

from __future__ import annotations

import copy
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from tradingagents.agents.managers.portfolio_manager import create_portfolio_manager
from tradingagents.llm_clients import build_llm_kwargs, create_llm_client
from tradingagents.pre_pm_snapshot import PrePMSnapshot, load_pre_pm_snapshot
from tradingagents.recommendation_record import ModelRoleRef, resolve_ledger_root

SCHEMA_VERSION = "1.0"


class PMReplayRunRef(BaseModel):
    ticker: str
    analysis_date: str


class PMReplayRecord(BaseModel):
    schema_version: str = SCHEMA_VERSION
    replay_id: str
    created_at: str
    # Which frozen input this replay used, and its hash at load time -- so a
    # replay record can always be checked against the snapshot it claims to
    # be from, independent of this file's own location on disk.
    source_pre_pm_snapshot_path: str
    source_snapshot_id: str
    source_snapshot_hash: str
    run: PMReplayRunRef
    generation: ModelRoleRef
    validator: ModelRoleRef
    validation_enabled: bool
    rating: str | None = None
    disposition: str | None = None
    final_trade_decision: str
    portfolio_decision_structured: dict[str, Any] | None = None
    # The Portfolio Manager's own validation-log entry for THIS replay call
    # only -- never the upstream entries the frozen state already carries
    # from the original run (those belong to that run, not this replay).
    validation: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Building the record from a real node call's result
# ---------------------------------------------------------------------------


def build_pm_replay_record(
    *,
    snapshot: PrePMSnapshot,
    source_snapshot_path: Path,
    generation: ModelRoleRef,
    validator: ModelRoleRef,
    validation_enabled: bool,
    node_result: dict[str, Any],
) -> PMReplayRecord:
    final_trade_decision = node_result.get("final_trade_decision", "")
    portfolio_decision_structured = node_result.get("portfolio_decision_structured")
    validation_log = node_result.get("validation_log") or []
    pm_validation = validation_log[-1] if validation_log else None
    risk_debate_state = node_result.get("risk_debate_state") or {}

    return PMReplayRecord(
        replay_id=str(uuid.uuid4()),
        created_at=datetime.now(timezone.utc).isoformat(),
        source_pre_pm_snapshot_path=str(source_snapshot_path),
        source_snapshot_id=snapshot.snapshot_id,
        source_snapshot_hash=snapshot.content_hash,
        run=PMReplayRunRef(ticker=snapshot.run.ticker, analysis_date=snapshot.run.analysis_date),
        generation=generation,
        validator=validator,
        validation_enabled=validation_enabled,
        rating=(portfolio_decision_structured or {}).get("rating"),
        disposition=(
            risk_debate_state.get("portfolio_disposition")
            or (portfolio_decision_structured or {}).get("disposition")
            or None
        ),
        final_trade_decision=final_trade_decision,
        portfolio_decision_structured=portfolio_decision_structured,
        validation=pm_validation,
    )


# ---------------------------------------------------------------------------
# Storage: many replays per snapshot, never overwriting one another.
# ---------------------------------------------------------------------------


def resolve_pm_replay_root(config: dict[str, Any] | None) -> Path:
    """``<ledger_root>/pm_replays`` -- a sibling of ``recommendations/`` and
    ``pre_pm/``."""
    return resolve_ledger_root(config) / "pm_replays"


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    data = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def write_pm_replay_record(record: PMReplayRecord, ledger_root: Path) -> Path:
    """Write ``<ledger_root>/pm_replays/<snapshot_id>/<replay_id>.json``.

    Grouped by ``source_snapshot_id`` so "what replays exist for this
    snapshot" (fork README: "PrePMSnapshot") is answered by a directory
    listing, with no separate index to keep in sync. Never overwrites: a
    path collision would need a UUID4 clash, but the check costs nothing
    (same guarantee as the rest of the ledger).
    """
    replay_dir = Path(ledger_root) / "pm_replays" / record.source_snapshot_id
    final_path = replay_dir / f"{record.replay_id}.json"
    payload = record.model_dump(mode="json")
    if final_path.exists():
        existing = json.loads(final_path.read_text(encoding="utf-8"))
        if existing == payload:
            return final_path
        raise FileExistsError(f"PMReplayRecord already exists and differs: {final_path}")
    _atomic_write_json(final_path, payload)
    return final_path


def list_pm_replays(ledger_root: Path, snapshot_id: str) -> list[Path]:
    """Every replay recorded for ``snapshot_id``, oldest first (filenames
    are UUID4s, not timestamps, so callers sort by ``created_at`` inside
    each file if ordering matters)."""
    replay_dir = Path(ledger_root) / "pm_replays" / snapshot_id
    if not replay_dir.exists():
        return []
    return sorted(replay_dir.glob("*.json"))


# ---------------------------------------------------------------------------
# Orchestration: load snapshot -> resolve models -> run PM only -> record.
# ---------------------------------------------------------------------------


def run_pm_replay(
    *,
    snapshot_path: Path,
    generation_model: str | None = None,
    generation_reasoning_effort: str | None = None,
    validator_model: str | None = None,
    validator_reasoning_effort: str | None = None,
    validation_enabled: bool | None = None,
    llm_provider: str | None = None,
    ledger_root: Path | None = None,
) -> tuple[PMReplayRecord, Path]:
    """Load ``snapshot_path``, run ONLY the Portfolio Manager against its
    frozen state, and persist a new, independent PMReplayRecord.

    Every ``*_model``/``*_reasoning_effort``/``validation_enabled``/
    ``llm_provider`` argument is optional: omitted, it falls back to
    whatever the snapshot's own ``run_provenance["models"]`` recorded for
    the original run (see fork README: "PrePMSnapshot") -- the same
    `--generation-llm`/`--generation-reasoning-effort`/`--validator-llm`/
    `--validator-reasoning-effort`/`--validation`/`--no-validation` options
    ``run``/`batch` already expose, reused rather than duplicated.
    """
    snapshot = load_pre_pm_snapshot(snapshot_path)

    prov_models = snapshot.run_provenance.get("models") or {}
    gen_block = prov_models.get("generation") or {}
    val_block = prov_models.get("validator") or {}

    resolved_provider = llm_provider or gen_block.get("provider") or val_block.get("provider") or "ollama"
    resolved_generation_model = generation_model or gen_block.get("name")
    resolved_generation_effort = generation_reasoning_effort or gen_block.get("reasoning_effort")
    resolved_validation_enabled = (
        validation_enabled if validation_enabled is not None
        else bool(val_block.get("validation_enabled"))
    )
    resolved_validator_model = validator_model or val_block.get("name")
    resolved_validator_effort = validator_reasoning_effort or val_block.get("reasoning_effort")

    if not resolved_generation_model:
        raise ValueError(
            "No generation model resolved: the snapshot's run_provenance has none, "
            "and no --generation-llm was given."
        )
    if resolved_validation_enabled and not resolved_validator_model:
        raise ValueError(
            "Validation is enabled but no validator model resolved: the snapshot's "
            "run_provenance has none, and no --validator-llm was given."
        )

    # Same two-call construction TradingAgentsGraph.__init__ uses for these
    # roles, duplicated here rather than refactored into a shared helper --
    # this module deliberately never constructs a TradingAgentsGraph (see
    # module docstring: a replay must not be able to run the full graph).
    role_config = {
        "generation_llm": resolved_generation_model,
        "generation_reasoning_effort": resolved_generation_effort,
        "validator_llm": resolved_validator_model,
        "validator_reasoning_effort": resolved_validator_effort,
        "llm_provider": resolved_provider,
    }
    generation_client = create_llm_client(
        provider=resolved_provider, model=resolved_generation_model,
        **build_llm_kwargs(role_config, role="generation"),
    )
    generation_llm = generation_client.get_llm()

    validator_llm = None
    if resolved_validation_enabled:
        validator_client = create_llm_client(
            provider=resolved_provider, model=resolved_validator_model,
            **build_llm_kwargs(role_config, role="validator"),
        )
        validator_llm = validator_client.get_llm()

    node_fn = create_portfolio_manager(generation_llm, validator_llm=validator_llm)
    # Deep copy: node_fn returns a new dict and does not mutate its input in
    # place, but this guarantees it structurally -- a replay can never touch
    # the snapshot it was loaded from (fork README: "PrePMSnapshot").
    node_result = node_fn(copy.deepcopy(snapshot.state))

    record = build_pm_replay_record(
        snapshot=snapshot,
        source_snapshot_path=Path(snapshot_path),
        generation=ModelRoleRef(
            name=resolved_generation_model, reasoning_effort=resolved_generation_effort,
            provider=resolved_provider,
        ),
        validator=(
            ModelRoleRef(
                name=resolved_validator_model, reasoning_effort=resolved_validator_effort,
                provider=resolved_provider,
            )
            if resolved_validation_enabled else ModelRoleRef()
        ),
        validation_enabled=resolved_validation_enabled,
        node_result=node_result,
    )
    resolved_ledger_root = Path(ledger_root) if ledger_root is not None else resolve_ledger_root(None)
    written_path = write_pm_replay_record(record, resolved_ledger_root)
    return record, written_path
