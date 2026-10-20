"""run2 orchestrator -- binary First Barrier experiment:

    Market Data
        -> Deterministic Market Features
        -> Market Regime Analyzer
        -> Binary First Barrier Predictor
        -> Deterministic Validation (schema + fail-closed degrade)
        -> LONG / SHORT

Answers exactly one question: from the reference price on ``analysis_date``,
within the next ``horizon_sessions`` trading sessions, which barrier
(+``barrier_pct``% or -``barrier_pct``%) will be reached first? The
predictor MUST choose LONG or SHORT -- NO_POSITION is deliberately not a
valid prediction. A prior version of run2 allowed NO_POSITION as a third
predictor output; in practice the LLM used it as a dominant "mixed signals"
escape hatch, which defeated the point of measuring raw predictive power
(fork spec: "run2" binary simplification -- "el LLM utiliza NO_POSITION
como salida dominante cuando las senales son mixtas"). There is no
confidence threshold gating the decision either -- a structurally valid
LONG/SHORT signal from the Predictor IS the final decision, regardless of
its (diagnostic-only) confidence.

The only way to end up without a LONG/SHORT decision is ``UNRESOLVED``,
which represents EXCLUSIVELY a technical/structural pipeline failure (a
stage produced invalid structured output even after its one repair
attempt) -- never market uncertainty, and never fabricated from an invalid
or missing signal (fail-closed -- see ``REASON_*`` below).

Structurally separate from `tradingagents run`: this module never imports
``tradingagents.graph`` (``TradingAgentsGraph``/``GraphSetup``), never
constructs the Bull/Bear debate, Research Manager, Trader, Risk, or
Portfolio Manager, and never touches Decision Memory, news, fundamentals, or
social dataflows. Its only shared imports are the generic building blocks
documented in each module's own docstring (LLM client construction,
structured-output helpers, seed/provenance, market-data dataflows).

Also never imports ``tradingagents.run2.evaluator``: whether a barrier was
actually reached first is knowable only from data AFTER ``analysis_date``,
and this module must never have access to it (see ``evaluator.py``'s own
module docstring and ``tests/test_run2_cli.py``'s import-boundary test).

The only values borrowed from ``tradingagents.default_config`` are the
already-configured ``llm_provider``/``backend_url``/model defaults (pure
reads, never mutated) -- so a user who already set up `.env` for `run` does
not have to repeat every ``--generation-llm`` flag just to try run2.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.llm_clients import build_llm_kwargs, create_llm_client
from tradingagents.llm_trace import SeedContext, set_active_seed_context
from tradingagents.run2.baselines import momentum_baseline, moving_average_baseline
from tradingagents.run2.config import resolve_run2_config
from tradingagents.run2.market_data import (
    compute_market_features,
    fetch_market_context,
    fetch_price_reference,
)
from tradingagents.run2.prompts import prompt_fingerprints
from tradingagents.run2.schemas import RUN2_SCHEMA_VERSION, FinalDecision
from tradingagents.run2.stages import run_barrier_signal, run_regime_analysis
from tradingagents.run2.storage import (
    automatic_run2_dir,
    default_run2_root,
    run2_runs_base,
    write_debug_call,
    write_report,
    write_result,
)
from tradingagents.run_provenance import build_run_provenance
from tradingagents.run_seed import (
    SEED_CAPABLE_PROVIDERS,
    resolve_base_seed,
    validate_configured_seed,
)

REASON_REGIME_STAGE_FAILED = "regime_stage_failed"
REASON_SIGNAL_STAGE_FAILED = "signal_stage_failed"
REASON_VALID_SIGNAL = "valid_binary_first_barrier_signal"


def _validate_analysis_date(value: str) -> str:
    """Same ``YYYY-MM-DD``, not-in-the-future rule as
    ``tradingagents.graph.trading_graph._validate_trade_date``, duplicated
    locally (not imported) so run2 never imports from ``tradingagents.graph``
    (see module docstring)."""
    text = str(value)
    try:
        canonical = datetime.strptime(text, "%Y-%m-%d").strftime("%Y-%m-%d") == text
    except ValueError:
        canonical = False
    if not canonical:
        raise ValueError(f"analysis_date must be in YYYY-MM-DD format, got {value!r}")
    if text > datetime.now(timezone.utc).strftime("%Y-%m-%d"):
        raise ValueError(f"analysis_date cannot be in the future: {text}")
    return text


def _build_report_markdown(result: dict[str, Any]) -> str:
    final = result["final_decision"]
    lines = [
        f"# run2 (First Barrier) report: {result['ticker']} as of {result['analysis_date']}",
        "",
        f"- Horizon: {result['horizon_sessions']} trading sessions; barrier: +/-{result['barrier_pct']}%",
        f"- **Final decision: {final['action']}** (reason: {final['reason']})",
        "",
    ]
    if result.get("regime_analysis"):
        r = result["regime_analysis"]
        lines += [
            "## Market regime",
            f"- Primary trend: {r['primary_trend']}; secondary trend: {r['secondary_trend']}",
            f"- Momentum: {r['momentum']}; volatility regime: {r['volatility_regime']}",
            f"- Trend strength: {r['trend_strength']}; regime confidence: {r['regime_confidence']}",
            f"- Key levels: {r['key_levels']}",
            "",
        ]
    if result.get("barrier_signal"):
        s = result["barrier_signal"]
        lines += [
            "## First Barrier signal",
            f"- Direction: {s['direction']}; confidence (diagnostic only): {s['confidence']}",
            f"- Thesis: {s['thesis']}",
            "",
        ]
    baselines = result.get("baselines") or {}
    lines += [
        "## Baselines (experimental control -- never seen by the LLM stages)",
        f"- Momentum: {baselines.get('momentum')}",
        f"- Moving average: {baselines.get('moving_average')}",
        "",
        "## Provenance",
        f"- Models: {result['models']}",
        f"- Seed: {result['seed']}",
        f"- Git: {result['provenance'].get('git')}",
        "",
        "## Validation",
        f"{result['validation']}",
    ]
    return "\n".join(lines)


def run_run2_analysis(
    *,
    ticker: str,
    analysis_date: str,
    language: str | None = None,
    generation_llm: str | None = None,
    generation_reasoning_effort: str | None = None,
    validator_llm: str | None = None,
    validator_reasoning_effort: str | None = None,
    llm_provider: str | None = None,
    seed: int = 0,
    trace_llm: bool = False,
    horizon_sessions: int | None = None,
    barrier_pct: float | None = None,
    root: Path | None = None,
) -> tuple[dict[str, Any], Path]:
    """Run one full run2 (First Barrier) analysis for ``ticker``/
    ``analysis_date`` and persist it under run2's own storage root (see
    ``storage.py``). Returns ``(result, result_json_path)``.

    Every model/reasoning-effort argument left as ``None`` falls back to
    this checkout's already-configured ``DEFAULT_CONFIG`` value (the same
    model routing `run`/`batch` use) -- a pure read, see module docstring.
    ``validator_llm`` is optional even then: when it resolves to ``None``,
    each stage's one repair attempt (see ``validation.run_structured_stage``)
    falls back to that stage's own generation model. There is no
    ``review_llm`` -- the old Challenger stage it existed for is gone; both
    remaining stages use the generation model.
    """
    analysis_date = _validate_analysis_date(analysis_date)
    resolved_language = language or DEFAULT_CONFIG.get("output_language", "English")

    configured_seed = validate_configured_seed(seed)
    base_seed = resolve_base_seed(configured_seed)

    provider = llm_provider or DEFAULT_CONFIG.get("llm_provider")
    backend_url = DEFAULT_CONFIG.get("backend_url")
    resolved_generation_model = generation_llm or DEFAULT_CONFIG.get("generation_llm")
    resolved_generation_effort = generation_reasoning_effort or DEFAULT_CONFIG.get("generation_reasoning_effort")
    resolved_validator_model = validator_llm or DEFAULT_CONFIG.get("validator_llm")
    resolved_validator_effort = validator_reasoning_effort or DEFAULT_CONFIG.get("validator_reasoning_effort")

    if not resolved_generation_model:
        raise ValueError("No generation model resolved: pass --generation-llm or set TRADINGAGENTS_GENERATION_LLM.")

    # Same two-call-per-role construction TradingAgentsGraph.__init__ and
    # pm_replay.run_pm_replay use, duplicated rather than shared -- run2
    # must never import or construct TradingAgentsGraph (see module
    # docstring). decision_memory_* and max_risk_discuss_rounds are left
    # explicitly False/None (never defaulted to True) so this run's own
    # provenance never implies a Decision Memory or risk-debate concept run2
    # does not have.
    role_config = {
        "generation_llm": resolved_generation_model,
        "generation_reasoning_effort": resolved_generation_effort,
        "validator_llm": resolved_validator_model,
        "validator_reasoning_effort": resolved_validator_effort,
        "llm_provider": provider,
        "validation_enabled": bool(resolved_validator_model),
        "decision_memory_enabled": False,
        "decision_memory_root": None,
        "output_language": resolved_language,
        "max_risk_discuss_rounds": None,
    }

    generation_client = create_llm_client(
        provider=provider, model=resolved_generation_model, base_url=backend_url,
        **build_llm_kwargs(role_config, role="generation"),
    )
    generation_llm_obj = generation_client.get_llm()

    validator_llm_obj = None
    if resolved_validator_model:
        validator_client = create_llm_client(
            provider=provider, model=resolved_validator_model, base_url=backend_url,
            **build_llm_kwargs(role_config, role="validator"),
        )
        validator_llm_obj = validator_client.get_llm()

    run2_cfg = resolve_run2_config({
        "horizon_sessions": horizon_sessions,
        "barrier_pct": barrier_pct,
    })

    root_path = Path(root) if root is not None else default_run2_root()
    runs_base = run2_runs_base(root_path, ticker, analysis_date)
    run_dir = automatic_run2_dir(runs_base)

    start_monotonic = time.monotonic()
    started_at = datetime.now(timezone.utc)

    set_active_seed_context(
        SeedContext(base_seed=base_seed, ticker=ticker, provider_supports_seed=provider in SEED_CAPABLE_PROVIDERS)
    )
    # stage_index only numbers debug/trace files (1/2 for regime/signal,
    # regardless of internal repairs). total_llm_calls is the real count of
    # .invoke() calls made across every stage -- see
    # validation.run_structured_stage's "llm_call_count".
    stage_index = 0
    total_llm_calls = 0
    regime = signal = None
    regime_meta = signal_meta = None
    try:
        market_context = fetch_market_context(ticker, analysis_date)
        price_reference = fetch_price_reference(ticker, analysis_date)
        market_features = compute_market_features(ticker, analysis_date)
        market_features_json = market_features.model_dump_json(indent=2)
        baselines = {
            "momentum": momentum_baseline(market_features).model_dump(mode="json"),
            "moving_average": moving_average_baseline(market_features).model_dump(mode="json"),
        }

        stage_index += 1
        regime, regime_meta = run_regime_analysis(
            generation_llm_obj, ticker=ticker, analysis_date=analysis_date,
            horizon_sessions=run2_cfg["horizon_sessions"], market_features_json=market_features_json,
            market_context=market_context, language=resolved_language, repair_llm=validator_llm_obj,
        )
        total_llm_calls += regime_meta.get("llm_call_count", 0)
        if trace_llm:
            write_debug_call(run_dir, stage_index, "regime", regime_meta.get("prompt", ""), regime_meta.get("response_text", ""))

        if regime is None:
            final = FinalDecision(action="UNRESOLVED", reason=REASON_REGIME_STAGE_FAILED)
        else:
            stage_index += 1
            signal, signal_meta = run_barrier_signal(
                generation_llm_obj, ticker=ticker, analysis_date=analysis_date,
                horizon_sessions=run2_cfg["horizon_sessions"], barrier_pct=run2_cfg["barrier_pct"],
                reference_price=price_reference.get("close"), reference_date=price_reference.get("reference_date"),
                market_features_json=market_features_json, market_context=market_context,
                regime=regime, language=resolved_language, repair_llm=validator_llm_obj,
            )
            total_llm_calls += signal_meta.get("llm_call_count", 0)
            if trace_llm:
                write_debug_call(run_dir, stage_index, "barrier_signal", signal_meta.get("prompt", ""), signal_meta.get("response_text", ""))

            if signal is None:
                final = FinalDecision(action="UNRESOLVED", reason=REASON_SIGNAL_STAGE_FAILED)
            else:
                # No confidence threshold: a structurally valid LONG/SHORT
                # signal IS the final decision, however low its
                # (diagnostic-only) confidence -- the schema itself (see
                # schemas.BarrierSignal) already forbids anything but LONG/SHORT.
                final = FinalDecision(action=signal.direction, reason=REASON_VALID_SIGNAL)
    finally:
        set_active_seed_context(None)

    finished_at = datetime.now(timezone.utc)
    any_repair = any((meta or {}).get("repaired") for meta in (regime_meta, signal_meta))
    # prompt_hashes is passed explicitly so build_run_provenance never falls
    # back to its default hash_prompt_files() -- that hashes the main
    # pipeline's entire tradingagents/prompts/ tree (news, fundamentals,
    # Bull/Bear, Research Manager, Trader, Risk, Portfolio Manager), none of
    # which run2 reads.
    provenance = build_run_provenance(
        selected_analysts=[], config=role_config, configured_seed=configured_seed,
        base_seed=base_seed, trace_llm=trace_llm, prompt_hashes=prompt_fingerprints(repair_used=any_repair),
    )
    # Fields build_run_provenance's generic shape has no slot for, but this
    # fork spec explicitly wants inside provenance for a self-contained
    # audit trail (no need to cross-reference the rest of result.json).
    provenance["run2"] = {
        "schema_version": RUN2_SCHEMA_VERSION,
        "ticker": ticker,
        "analysis_date": analysis_date,
        "horizon_sessions": run2_cfg["horizon_sessions"],
        "barrier_pct": run2_cfg["barrier_pct"],
        "config": run2_cfg,
    }

    def _meta_without_text(meta: dict[str, Any] | None) -> dict[str, Any] | None:
        if meta is None:
            return None
        return {k: v for k, v in meta.items() if k not in ("prompt", "response_text")}

    result: dict[str, Any] = {
        "schema_version": RUN2_SCHEMA_VERSION,
        "run_id": run_dir.name,
        "created_at": started_at.isoformat(),
        "ticker": ticker,
        "analysis_date": analysis_date,
        "horizon_sessions": run2_cfg["horizon_sessions"],
        "barrier_pct": run2_cfg["barrier_pct"],
        "seed": {"configured": configured_seed, "base_used": base_seed},
        "price_reference": price_reference,
        "market_snapshot": market_context,
        "market_features": market_features.model_dump(mode="json"),
        "regime_analysis": regime.model_dump(mode="json") if regime is not None else None,
        "barrier_signal": signal.model_dump(mode="json") if signal is not None else None,
        "final_decision": final.model_dump(mode="json"),
        "baselines": baselines,
        "models": {
            "generation": {
                "name": resolved_generation_model, "reasoning_effort": resolved_generation_effort,
                "provider": provider,
            },
            "validator": {
                "name": resolved_validator_model, "reasoning_effort": resolved_validator_effort,
                "provider": provider, "validation_enabled": bool(resolved_validator_model),
            },
        },
        "reasoning_effort": {
            "generation": resolved_generation_effort,
            "validator": resolved_validator_effort,
        },
        "provenance": provenance,
        "validation": {
            "regime": _meta_without_text(regime_meta),
            "barrier_signal": _meta_without_text(signal_meta),
        },
        "timing": {
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "total_seconds": round(time.monotonic() - start_monotonic, 3),
        },
        "token_usage": {
            "llm_calls": total_llm_calls,
            "note": "per-call token counts are not tracked in this experimental pipeline",
        },
        "config": run2_cfg,
    }

    result_path = write_result(run_dir, result)
    write_report(run_dir, _build_report_markdown(result))
    return result, result_path
