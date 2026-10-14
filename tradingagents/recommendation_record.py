"""RecommendationRecord v1: an immutable, append-only ledger entry for one
final TradeProof recommendation (see fork README: "RecommendationRecord").

Separate from (and does not replace) ``decision_log.py``'s mutable markdown
memory -- that log is agent working memory fed back into future prompts;
this is an audit record of exactly what was recommended, under what
code/models/prompts, with what evidence. A record is never an outcome: it
says what TradeProof recommended, not whether the market agreed.
RecommendationOutcome (did the market move as predicted) is a separate,
future piece of work -- not implemented here.

Structured state (``result["structured"]``, ``result["run_provenance"]``) is
the source of truth throughout; nothing here is reconstructed from regex'd
markdown. The only structurally-REQUIRED input is a parsed
``portfolio_decision`` (rating + disposition) -- that is the recommendation
itself, so its absence means there is nothing to record (see
``Incomplete RecommendationError``). Every other field degrades to null/empty
when its underlying structured object never parsed, rather than failing the
write or inventing a value.

Write order (ties correctness to what's actually on disk, never computed
ahead of it): pipeline finishes -> result.json and complete_report.md are
written -> both are hashed -> the record is built -> record_hash is computed
over the record -> the record is written. See ``record_recommendation``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

import tradingagents.default_config as default_config_module
from tradingagents.dataflows.symbols import safe_ticker_component

SCHEMA_VERSION = "1.0"


class IncompleteRecommendationError(ValueError):
    """Raised when a successful run still lacks the one field a
    RecommendationRecord cannot exist without: a parsed ``portfolio_decision``
    (rating + disposition). The caller logs and skips the ledger write rather
    than letting this fail the run itself (same "never break the run over a
    secondary artifact" posture as the partial-report save in cli/run.py)."""


# ---------------------------------------------------------------------------
# Horizon: preserve the narrative string, parse best-effort alongside it --
# never collapse "5-10 trading sessions" into a single invented integer.
# ---------------------------------------------------------------------------


class HorizonUnit(str, Enum):
    TRADING_SESSIONS = "TRADING_SESSIONS"
    TRADING_DAYS = "TRADING_DAYS"
    CALENDAR_DAYS = "CALENDAR_DAYS"
    WEEKS = "WEEKS"
    MONTHS = "MONTHS"
    UNKNOWN = "UNKNOWN"


class HorizonParseStatus(str, Enum):
    PARSED_RANGE = "PARSED_RANGE"
    PARSED_SINGLE = "PARSED_SINGLE"
    UNPARSEABLE = "UNPARSEABLE"
    MISSING = "MISSING"


class Horizon(BaseModel):
    raw: str | None = None
    unit: HorizonUnit = HorizonUnit.UNKNOWN
    min: float | None = None
    max: float | None = None
    parse_status: HorizonParseStatus = HorizonParseStatus.MISSING


def _normalize_for_matching(text: str) -> str:
    """Lowercase, accent-stripped copy used ONLY for pattern matching --
    ``raw`` itself is never altered, so a Spanish "5-10 sesiones" still
    displays exactly as the recommendation produced it."""
    decomposed = unicodedata.normalize("NFKD", text)
    without_accents = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return without_accents.lower()


# English + Spanish (the two languages this project's output_language
# actually produces today -- see fork README: "RecommendationRecord").
# Deterministic keyword matching, never LLM translation: adding a language
# means adding one more alternation here, not a new dependency.
_UNIT_PATTERNS: tuple[tuple[re.Pattern, HorizonUnit], ...] = (
    (re.compile(r"\btrading\s+sessions?\b|\bsesiones?\b"), HorizonUnit.TRADING_SESSIONS),
    (
        re.compile(
            r"\btrading\s+days?\b|\bdias?\s+(?:de\s+negociaci(?:o|ó)n|h(?:a|á)biles?)\b"
        ),
        HorizonUnit.TRADING_DAYS,
    ),
    (
        re.compile(r"\bcalendar\s+days?\b|\bdias?\s+(?:naturales?|calendario)\b"),
        HorizonUnit.CALENDAR_DAYS,
    ),
    (re.compile(r"\bweeks?\b|\bsemanas?\b"), HorizonUnit.WEEKS),
    (re.compile(r"\bmonths?\b|\bmeses?\b"), HorizonUnit.MONTHS),
)
_RANGE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:-|–|—|to)\s*(\d+(?:\.\d+)?)")
_SINGLE_RE = re.compile(r"(\d+(?:\.\d+)?)")


def parse_horizon(raw: str | None) -> Horizon:
    """Best-effort structured parse of a free-text horizon string, in
    English or Spanish (whichever this run's ``output_language`` produced).

    ``raw`` is always preserved verbatim regardless of ``parse_status`` --
    matching itself runs against a lowercased, accent-stripped COPY (see
    ``_normalize_for_matching``), never against ``raw`` in place. A range or
    single value is only recognized alongside a known unit keyword (never a
    bare number, which carries no meaning without a unit) -- an unrecognized
    unit or no number at all is ``UNPARSEABLE``, never a guess.
    """
    if not raw or not raw.strip():
        return Horizon(raw=raw, unit=HorizonUnit.UNKNOWN, parse_status=HorizonParseStatus.MISSING)

    normalized = _normalize_for_matching(raw)

    unit = HorizonUnit.UNKNOWN
    for pattern, candidate in _UNIT_PATTERNS:
        if pattern.search(normalized):
            unit = candidate
            break

    if unit is HorizonUnit.UNKNOWN:
        return Horizon(raw=raw, unit=unit, parse_status=HorizonParseStatus.UNPARSEABLE)

    range_match = _RANGE_RE.search(normalized)
    if range_match:
        low, high = float(range_match.group(1)), float(range_match.group(2))
        return Horizon(
            raw=raw, unit=unit, min=min(low, high), max=max(low, high),
            parse_status=HorizonParseStatus.PARSED_RANGE,
        )

    single_match = _SINGLE_RE.search(normalized)
    if single_match:
        value = float(single_match.group(1))
        return Horizon(raw=raw, unit=unit, min=value, max=value, parse_status=HorizonParseStatus.PARSED_SINGLE)

    return Horizon(raw=raw, unit=unit, parse_status=HorizonParseStatus.UNPARSEABLE)


def resolve_horizon(
    *, portfolio_decision: dict[str, Any], debate: dict[str, Any], config: dict[str, Any]
) -> Horizon:
    """The record's ``horizon``, preferring a structured, non-localized
    source for ``unit``/``min``/``max`` over parsing the Portfolio Manager's
    own (possibly translated) free text.

    ``investment_debate_state["research_horizon"]`` (falling back to
    ``config["research_horizon"]``) is the shared horizon the whole debate
    was anchored to -- a plain config value the LLM never touches, so it is
    always in English and always exactly "N-M trading sessions"-shaped,
    regardless of ``output_language``. When it's available, it is parsed for
    ``unit``/``min``/``max`` instead of the PM's own ``time_horizon`` text --
    this is what actually fixed a real run_provenance validation bug, where
    ``output_language=Spanish`` made the Portfolio Manager write "5-10
    sesiones" and the (then English-only) parser read that as UNPARSEABLE.

    ``raw`` is still always exactly what the recommendation itself said
    (the PM's own ``time_horizon`` when present, else this same structured
    source) -- never silently replaced. In the common case the PM's horizon
    is a restatement of the shared debate horizon, so ``raw`` and the parsed
    fields agree; if the PM deliberately states a DIFFERENT holding period,
    ``raw`` still shows that exact text even though ``unit``/``min``/``max``
    here reflect the shared debate horizon instead -- a known, accepted v1
    tradeoff (never parsing fragile, highly-variable translated free text)
    over a deterministic, always-correct source for the numeric fields.
    """
    pm_horizon = portfolio_decision.get("time_horizon")
    structured_source = debate.get("research_horizon") or config.get("research_horizon")
    raw = pm_horizon or structured_source

    if structured_source:
        parsed = parse_horizon(structured_source)
        return parsed.model_copy(update={"raw": raw})
    return parse_horizon(raw)


# ---------------------------------------------------------------------------
# Record schema
# ---------------------------------------------------------------------------


class ReproducibilityStatus(str, Enum):
    CLEAN_COMMIT = "CLEAN_COMMIT"
    DIRTY_WORKTREE = "DIRTY_WORKTREE"
    UNKNOWN = "UNKNOWN"


def _reproducibility_status(git: dict[str, Any]) -> ReproducibilityStatus:
    """Derived from run_provenance's own git block -- never blocks the write
    (a dirty worktree is recorded, not rejected); future metrics can exclude
    DIRTY_WORKTREE runs by default without this ledger refusing to hold them."""
    if git.get("commit") is None or git.get("dirty") is None:
        return ReproducibilityStatus.UNKNOWN
    return ReproducibilityStatus.DIRTY_WORKTREE if git["dirty"] else ReproducibilityStatus.CLEAN_COMMIT


class GitRef(BaseModel):
    commit: str | None = None
    branch: str | None = None
    dirty: bool | None = None


class SeedRef(BaseModel):
    configured: int | None = None
    base_used: int | None = None


class RunRef(BaseModel):
    ticker: str
    analysis_date: str
    tradeproof_version: str | None = None
    git: GitRef
    seed: SeedRef
    language: str | None = None
    depth: int | str | None = None
    selected_analysts: list[str] = Field(default_factory=list)
    reproducibility_status: ReproducibilityStatus = ReproducibilityStatus.UNKNOWN


class ModelRoleRef(BaseModel):
    name: str | None = None
    reasoning_effort: str | None = None
    provider: str | None = None


class ModelsRef(BaseModel):
    generation: ModelRoleRef
    review: ModelRoleRef
    validator: ModelRoleRef
    validation_enabled: bool = False


class DecisionRef(BaseModel):
    research_manager_recommendation: str | None = None
    trader_action: str | None = None
    portfolio_rating: str | None = None
    portfolio_disposition: str | None = None


class TradePlanRef(BaseModel):
    """Only fields the Trader's own structured proposal actually carries are
    ever non-null. ``take_profit``/``risk_reward``/``atr_stop_distance``/
    ``atr_target_distance`` stay null in v1: TraderProposal has no
    take_profit field today, so none of these are computable without
    inventing one -- the closest existing equivalents are the Bull/Bear
    research-level trade metrics under ``research.bull_trade_metrics`` /
    ``research.bear_trade_metrics`` (computed from each side's OWN
    take_profit/stop_loss claims during debate, not the Trader's)."""

    action: str | None = None
    entry_price: float | None = None
    entry_price_low: float | None = None
    entry_price_high: float | None = None
    reference_entry: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    risk_reward: float | None = None
    atr_stop_distance: float | None = None
    atr_target_distance: float | None = None


class ResearchRef(BaseModel):
    bull_conviction: int | None = None
    bear_conviction: int | None = None
    bull_trade_metrics: dict[str, Any] | None = None
    bear_trade_metrics: dict[str, Any] | None = None
    verifier_status: str | None = None
    verifier_passes: int = 0
    repair_rounds: int = 0
    repaired_agents: list[str] = Field(default_factory=list)
    manager_integrity_status: str | None = None
    risk_integrity_status: str | None = None
    validation_summary: dict[str, Any] | None = None


class EvidenceRef(BaseModel):
    source_coverage: dict[str, Any] | None = None
    bull_data_gaps: list[str] = Field(default_factory=list)
    bear_data_gaps: list[str] = Field(default_factory=list)


class ArtifactsRef(BaseModel):
    """``result_path``/``report_path`` are absolute, normalized paths to the
    run's own artifacts -- never relative to this record's own location.
    The record lives under ``ledger_root`` (see ``write_recommendation_record``),
    a separate tree from ``run_output_dir`` where these artifacts actually
    are, so a relative-to-self convention ("go up two directories") would be
    pure ambiguity here. Explicit absolute paths, plus the sha256 hashes
    below, are the v1 answer; cross-machine portability is explicitly
    deferred (see fork README: "RecommendationRecord")."""

    result_path: str
    result_sha256: str
    report_path: str | None = None
    report_sha256: str | None = None
    prompt_hashes: dict[str, str] = Field(default_factory=dict)
    trace_manifest_path: str | None = None


class RecommendationRecord(BaseModel):
    schema_version: str = SCHEMA_VERSION
    recommendation_id: str
    created_at: str
    record_hash: str
    run_fingerprint: str
    run: RunRef
    models: ModelsRef
    decision: DecisionRef
    trade_plan: TradePlanRef
    horizon: Horizon
    research: ResearchRef
    evidence: EvidenceRef
    artifacts: ArtifactsRef
    portfolio_context: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Canonical JSON / hashing
# ---------------------------------------------------------------------------


def canonical_json(data: Any) -> bytes:
    """Stable byte encoding for hashing: sorted keys, fixed separators, UTF-8."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return _sha256_hex(Path(path).read_bytes())


def compute_record_hash(payload_without_hash: dict[str, Any]) -> str:
    """``sha256(canonical_json(record without record_hash))`` -- see
    ``verify_recommendation_record`` for the matching recomputation."""
    return _sha256_hex(canonical_json(payload_without_hash))


def verify_recommendation_record(path: Path) -> bool:
    """Recompute ``record_hash`` from the file exactly as stored (no
    timestamps or any other field regenerated) and compare."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    stored_hash = payload.get("record_hash")
    without_hash = {key: value for key, value in payload.items() if key != "record_hash"}
    return stored_hash == compute_record_hash(without_hash)


def compute_run_fingerprint(
    *, ticker: str, analysis_date: str, run_provenance: dict[str, Any]
) -> str:
    """Reproducible fingerprint of the EFFECTIVE configuration -- distinct
    from ``recommendation_id`` (a fresh identity per record, never
    reproducible). Two runs of the same ticker/date with the same models,
    prompts, and base seed get the same fingerprint even though each gets
    its own recommendation_id; a changed model, prompt, or seed changes it
    (see tests)."""
    fingerprint_input = {
        "ticker": ticker,
        "analysis_date": analysis_date,
        "git_commit": (run_provenance.get("git") or {}).get("commit"),
        "models": run_provenance.get("models"),
        "prompt_hashes": (run_provenance.get("prompts") or {}).get("hashes"),
        "base_seed": (run_provenance.get("seed") or {}).get("base_used"),
        "selected_analysts": (run_provenance.get("execution") or {}).get("selected_analysts"),
        "language": (run_provenance.get("execution") or {}).get("language"),
        "depth": (run_provenance.get("execution") or {}).get("depth"),
    }
    return _sha256_hex(canonical_json(fingerprint_input))


# ---------------------------------------------------------------------------
# Building the record from structured state
# ---------------------------------------------------------------------------


def _latest_conviction(history: list | None, thesis: dict | None) -> int | None:
    if history:
        return history[-1]
    if thesis:
        return (thesis.get("conviction") or {}).get("total")
    return None


def build_recommendation_record(
    *,
    execution: Any,
    result: dict[str, Any],
    config: dict[str, Any],
    result_path: str,
    result_sha256: str,
    report_path: str | None,
    report_sha256: str | None,
) -> RecommendationRecord:
    """Build one record from this run's already-finished structured state.

    ``result`` is exactly the dict ``result.json`` holds for this run
    (``result["structured"]``/``result["run_provenance"]`` are the source of
    truth); ``execution.final_state`` is read only for the shared research
    horizon fallback, which is not duplicated into ``result``.

    Raises ``IncompleteRecommendationError`` if ``portfolio_decision`` never
    parsed -- the one field without which there is no recommendation to
    record.
    """
    structured = result.get("structured") or {}
    portfolio_decision = structured.get("portfolio_decision")
    if not portfolio_decision:
        raise IncompleteRecommendationError(
            "portfolio_decision did not parse structurally; no recommendation to record"
        )
    run_provenance = result.get("run_provenance")
    if not run_provenance:
        raise IncompleteRecommendationError("run_provenance missing; cannot fingerprint this run")

    research_plan = structured.get("research_plan") or {}
    trader_proposal = structured.get("trader_proposal") or {}
    final_state = getattr(execution, "final_state", {}) or {}
    debate = final_state.get("investment_debate_state") or {}

    ticker = execution.ticker
    analysis_date = execution.analysis_date

    git = run_provenance.get("git") or {}
    run_ref = RunRef(
        ticker=ticker,
        analysis_date=analysis_date,
        tradeproof_version=run_provenance.get("tradeproof_version"),
        git=GitRef(**git),
        seed=SeedRef(**(run_provenance.get("seed") or {})),
        language=(run_provenance.get("execution") or {}).get("language"),
        depth=(run_provenance.get("execution") or {}).get("depth"),
        selected_analysts=(run_provenance.get("execution") or {}).get("selected_analysts") or [],
        reproducibility_status=_reproducibility_status(git),
    )

    prov_models = run_provenance.get("models") or {}
    validator_block = prov_models.get("validator") or {}
    models_ref = ModelsRef(
        generation=ModelRoleRef(**(prov_models.get("generation") or {})),
        review=ModelRoleRef(**(prov_models.get("review") or {})),
        validator=ModelRoleRef(
            name=validator_block.get("name"),
            reasoning_effort=validator_block.get("reasoning_effort"),
            provider=validator_block.get("provider"),
        ),
        validation_enabled=bool(validator_block.get("validation_enabled")),
    )

    decision_ref = DecisionRef(
        research_manager_recommendation=research_plan.get("recommendation"),
        trader_action=trader_proposal.get("action"),
        portfolio_rating=portfolio_decision.get("rating"),
        portfolio_disposition=portfolio_decision.get("disposition"),
    )

    trade_plan_ref = TradePlanRef(
        action=trader_proposal.get("action"),
        entry_price=trader_proposal.get("entry_price"),
        entry_price_low=trader_proposal.get("entry_price_low"),
        entry_price_high=trader_proposal.get("entry_price_high"),
        reference_entry=trader_proposal.get("reference_entry"),
        stop_loss=trader_proposal.get("stop_loss"),
    )

    horizon = resolve_horizon(portfolio_decision=portfolio_decision, debate=debate, config=config)

    bull_thesis = structured.get("bull_thesis") or {}
    bear_thesis = structured.get("bear_thesis") or {}
    research_ref = ResearchRef(
        bull_conviction=_latest_conviction(structured.get("bull_conviction_history"), bull_thesis),
        bear_conviction=_latest_conviction(structured.get("bear_conviction_history"), bear_thesis),
        bull_trade_metrics=structured.get("bull_trade_metrics") or None,
        bear_trade_metrics=structured.get("bear_trade_metrics") or None,
        verifier_status=debate.get("verification_status"),
        verifier_passes=debate.get("verifier_pass_count", 0),
        repair_rounds=debate.get("repair_rounds", 0),
        repaired_agents=debate.get("repaired_agents") or [],
        manager_integrity_status=structured.get("manager_integrity_status"),
        risk_integrity_status=structured.get("risk_integrity_status"),
        validation_summary=result.get("validation"),
    )

    evidence_ref = EvidenceRef(
        source_coverage=result.get("source_coverage"),
        bull_data_gaps=bull_thesis.get("data_gaps") or [],
        bear_data_gaps=bear_thesis.get("data_gaps") or [],
    )

    artifacts_ref = ArtifactsRef(
        result_path=result_path,
        result_sha256=result_sha256,
        report_path=report_path,
        report_sha256=report_sha256,
        prompt_hashes=(run_provenance.get("prompts") or {}).get("hashes") or {},
        trace_manifest_path=(run_provenance.get("trace") or {}).get("manifest_path"),
    )

    run_fingerprint = compute_run_fingerprint(
        ticker=ticker, analysis_date=analysis_date, run_provenance=run_provenance
    )

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "recommendation_id": str(uuid.uuid4()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_fingerprint": run_fingerprint,
        "run": run_ref.model_dump(mode="json"),
        "models": models_ref.model_dump(mode="json"),
        "decision": decision_ref.model_dump(mode="json"),
        "trade_plan": trade_plan_ref.model_dump(mode="json"),
        "horizon": horizon.model_dump(mode="json"),
        "research": research_ref.model_dump(mode="json"),
        "evidence": evidence_ref.model_dump(mode="json"),
        "artifacts": artifacts_ref.model_dump(mode="json"),
        "portfolio_context": None,
    }
    payload["record_hash"] = compute_record_hash(payload)
    return RecommendationRecord.model_validate(payload)


# ---------------------------------------------------------------------------
# Storage: one immutable JSON file per recommendation, write-once.
# ---------------------------------------------------------------------------


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """``.tmp`` + ``fsync`` + ``os.replace`` -- the file is never opened for
    modification again after this returns (see module docstring)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    data = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def resolve_ledger_root(config: dict[str, Any] | None) -> Path:
    """The one global ledger location every frontend writes to, independent
    of any run's own ``output_dir``/``--output-dir`` (see module docstring
    and fork README: "RecommendationRecord"). Falls back to
    ``DEFAULT_CONFIG["ledger_root"]`` (itself ``TRADINGAGENTS_LEDGER_ROOT``
    or ``~/.tradingagents/ledger``) for any config dict that was built
    without going through ``DEFAULT_CONFIG.copy()`` -- e.g. a bare dict a
    test constructs directly.

    Looks up ``default_config_module.DEFAULT_CONFIG`` dynamically at call
    time rather than importing the name directly -- a direct
    ``from tradingagents.default_config import DEFAULT_CONFIG`` would
    capture whatever dict object existed at THIS module's own import time,
    which goes stale the moment anything (e.g. a test exercising
    TRADINGAGENTS_* env overrides) reloads ``default_config`` and rebinds
    its ``DEFAULT_CONFIG`` attribute to a new dict.
    """
    config = config or {}
    return Path(config.get("ledger_root") or default_config_module.DEFAULT_CONFIG["ledger_root"])


def write_recommendation_record(record: RecommendationRecord, ledger_root: Path) -> Path:
    """Write ``<ledger_root>/recommendations/<ticker>/<recommendation_id>.json``.

    ``ledger_root`` is the global ledger location (see ``resolve_ledger_root``),
    never the run's own ``output_dir`` -- the two are deliberately separate
    trees in v1.

    Never overwrites an existing file at that path: identical content is a
    silent no-op (idempotent replay), any difference raises. In practice a
    path collision essentially never happens (fresh UUID4 per record), but
    the check costs nothing and matches the "write-once" guarantee the
    moment any future caller reuses an id.
    """
    payload = record.model_dump(mode="json")
    ticker_dir = Path(ledger_root) / "recommendations" / safe_ticker_component(record.run.ticker)
    final_path = ticker_dir / f"{record.recommendation_id}.json"
    if final_path.exists():
        existing = json.loads(final_path.read_text(encoding="utf-8"))
        if existing == payload:
            return final_path
        raise FileExistsError(f"RecommendationRecord already exists and differs: {final_path}")
    _atomic_write_json(final_path, payload)
    return final_path


def record_recommendation(
    *,
    execution: Any,
    result: dict[str, Any],
    config: dict[str, Any],
    result_json_path: Path,
    report_path: Path | None,
) -> Path | None:
    """The one entry point every normal frontend calls on the success path,
    strictly after ``result.json`` and the report are both already written
    to disk (see module docstring: write order).

    Artifact references are absolute, normalized paths to those two files --
    never relative to the record's own location, which now lives under a
    separate ``ledger_root`` tree (see ``resolve_ledger_root``), not inside
    the run's own ``output_dir``.

    Returns the written path, or ``None`` when nothing was persisted. Two
    distinct, intentionally non-fatal cases both log a ``[WARN]`` and return
    ``None`` rather than raising into the caller -- a successful run's own
    artifacts (result.json, the report) must never disappear or look
    incomplete just because the ledger had a problem:

    - ``IncompleteRecommendationError``: the run's own ``portfolio_decision``
      never parsed structurally, so there is nothing valid to record;
    - ``OSError`` (permission denied, disk full, read-only filesystem, or a
      genuine UUID4 collision with different content) while writing to
      ``ledger_root``: the record could not be persisted, but nothing about
      the run itself is rolled back or hidden.
    """
    result_sha256 = sha256_file(result_json_path)
    result_path_abs = str(Path(result_json_path).resolve())
    report_sha256 = None
    report_path_abs = None
    if report_path is not None and Path(report_path).exists():
        report_sha256 = sha256_file(report_path)
        report_path_abs = str(Path(report_path).resolve())

    try:
        record = build_recommendation_record(
            execution=execution, result=result, config=config,
            result_path=result_path_abs, result_sha256=result_sha256,
            report_path=report_path_abs, report_sha256=report_sha256,
        )
    except IncompleteRecommendationError as exc:
        print(f"[WARN] RecommendationRecord not written for {execution.ticker}: {exc}", flush=True)
        return None

    ledger_root = resolve_ledger_root(config)
    try:
        return write_recommendation_record(record, ledger_root)
    except OSError as exc:
        print(
            f"[WARN] RecommendationRecord could not be persisted to the ledger "
            f"({ledger_root}) for {execution.ticker}: {exc}. The run's own "
            f"result.json/report are unaffected.",
            flush=True,
        )
        return None
