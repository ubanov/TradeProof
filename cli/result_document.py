"""Neutral layer for shaping and persisting one run's result document.

Shared by every normal frontend that produces a final recommendation --
interactive (``cli/run.py::run_analysis``), non-interactive single
(``cli/headless.py::run_single_analysis``), and batch (which calls the
single-run path per ticker) -- so none of them duplicates this logic or
diverges in what a successful/failed run's ``result.json`` looks like.

Lives here, not in ``cli/headless.py``, specifically so ``cli/run.py`` can
import it too without ``cli/run.py -> cli/headless.py -> cli/run.py``
becoming circular (``cli/headless.py`` already imports several names FROM
``cli/run.py``). This module imports from neither.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from tradingagents.agents.rating import RATINGS_5_TIER, extract_choice, is_review
from tradingagents.agents.validator import summarize_validation_log
from tradingagents.recommendation_record import record_recommendation, resolve_ledger_root

_ACTIONS = ("Buy", "Hold", "Sell")
# Report state key -> analyst key, in the fixed execution order.
_REPORT_STAGES = {
    "market_report": "market",
    "sentiment_report": "social",
    "news_report": "news",
    "fundamentals_report": "fundamentals",
}


def _field(text: str, label: str) -> str | None:
    match = re.search(
        rf"^\s*\**{re.escape(label)}\**\s*:\s*(.+?)\s*$",
        text or "",
        re.IGNORECASE | re.MULTILINE,
    )
    if not match:
        return None
    return match.group(1).strip().strip("*") or None


def _number_field(text: str, label: str) -> float | None:
    value = _field(text, label)
    if value is None or value.lower() == "not provided":
        return None
    cleaned = value.replace(",", "").removeprefix("$").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def _research_execution(state: dict[str, Any]) -> dict[str, Any]:
    debate = state.get("investment_debate_state", {})
    return {
        "bull_initial_calls": int(bool(debate.get("bull_initial"))),
        "bear_initial_calls": int(bool(debate.get("bear_initial"))),
        "bull_review_calls": debate.get("bull_rebuttal_count", 0),
        "bear_review_calls": debate.get("bear_rebuttal_count", 0),
        "verifier_passes": debate.get("verifier_pass_count", 0),
        "verification_status": debate.get("verification_status") or None,
        "repair_rounds": debate.get("repair_rounds", 0),
        "repaired_agents": debate.get("repaired_agents", []),
    }


def _structured_section(state: dict[str, Any]) -> dict[str, Any]:
    """The original Pydantic objects (JSON-safe via model_dump), not the
    regex-recovered legacy fields above -- structured object is the source
    of truth, rendered markdown is presentation only (see fork README:
    "Preserve structured objects through the pipeline"). Every value here
    is None/absent exactly when that stage never produced a surviving
    structured parse (never fabricated from partial/free text), so a
    partial/failed run degrades to nulls rather than breaking.
    """
    debate = state.get("investment_debate_state", {}) or {}
    risk = state.get("risk_debate_state", {}) or {}
    return {
        "research_plan": state.get("research_plan_structured"),
        "trader_proposal": state.get("trader_proposal_structured"),
        "portfolio_decision": state.get("portfolio_decision_structured"),
        "bull_trade_metrics": debate.get("bull_trade_metrics"),
        "bear_trade_metrics": debate.get("bear_trade_metrics"),
        "bull_thesis": debate.get("bull_thesis"),
        "bear_thesis": debate.get("bear_thesis"),
        "bull_conviction_history": debate.get("bull_conviction_history"),
        "bear_conviction_history": debate.get("bear_conviction_history"),
        "manager_integrity_status": debate.get("manager_integrity_status") or None,
        "manager_integrity_findings": debate.get("manager_integrity_findings"),
        "risk_integrity_status": risk.get("risk_integrity_status") or None,
        "risk_integrity_findings": risk.get("risk_integrity_findings"),
    }


def result_document(execution, report_root: Path) -> dict[str, Any]:
    state = execution.final_state
    final_decision = state.get("final_trade_decision", "")
    trader_plan = state.get("trader_investment_plan", "")
    research_plan = state.get("investment_plan", "")
    rating = None if is_review(execution.rating) else execution.rating
    report_path = (
        execution.report_path.relative_to(report_root).as_posix()
        if execution.report_path is not None
        else None
    )
    return {
        "ticker": execution.ticker,
        "analysis_date": execution.analysis_date,
        "status": "ok",
        "rating": rating,
        "rating_parse_status": "parsed" if rating is not None else "unparsed",
        "action": extract_choice(
            trader_plan, _ACTIONS, label_words=("action", "acción", "accion")
        ),
        "entry_price": _number_field(trader_plan, "Entry Price"),
        "stop_loss": _number_field(trader_plan, "Stop Loss"),
        "summary": _field(final_decision, "Executive Summary"),
        "research_recommendation": extract_choice(
            research_plan, RATINGS_5_TIER,
            label_words=("recommendation", "rating", "recomendación", "recomendacion"),
        ),
        "portfolio_manager_rating": rating,
        "portfolio_disposition": state.get("risk_debate_state", {}).get("portfolio_disposition") or None,
        "raw_final_decision": final_decision or None,
        "report_path": report_path,
        "run_duration_seconds": round(execution.duration_seconds, 3),
        "stats": execution.stats,
        "fallback_warnings": execution.fallback_warnings,
        "research_execution": _research_execution(state),
        "validation": summarize_validation_log(state.get("validation_log", [])),
        "base_seed": execution.base_seed,
        "source_coverage": state.get("source_coverage"),
        "structured": _structured_section(state),
        "run_provenance": state.get("run_provenance"),
        "price_basis": state.get("price_basis"),
        "pre_pm_snapshot": state.get("pre_pm_snapshot_ref"),
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _partial_state_summary(final_state: dict[str, Any]) -> dict[str, Any]:
    """Compact "how far did we get" snapshot for a failed run's result.json.

    Deliberately a summary, not a raw state dump (see fork README: "Avoid
    dumping huge JSON payloads"); the saved partial_report.md and the
    per-section .md files already hold the full prose for anything that
    completed.
    """
    debate = final_state.get("investment_debate_state") or {}
    risk = final_state.get("risk_debate_state") or {}
    return {
        "analyst_reports_completed": sorted(
            analyst_key
            for report_key, analyst_key in _REPORT_STAGES.items()
            if final_state.get(report_key)
        ),
        "bull_initial_completed": bool(debate.get("bull_initial")),
        "bear_initial_completed": bool(debate.get("bear_initial")),
        "debate_round": debate.get("debate_round", 0),
        "verifier_pass_count": debate.get("verifier_pass_count", 0),
        "verification_status": debate.get("verification_status") or None,
        "repair_rounds": debate.get("repair_rounds", 0),
        "repaired_agents": debate.get("repaired_agents", []),
        "bull_repair_structured_status": debate.get("bull_repair_structured_status", "NOT_RUN"),
        "bear_repair_structured_status": debate.get("bear_repair_structured_status", "NOT_RUN"),
        "research_manager_completed": bool(debate.get("judge_decision")),
        "manager_integrity_status": debate.get("manager_integrity_status") or None,
        "manager_integrity_findings": debate.get("manager_integrity_findings", []),
        "trader_completed": bool(final_state.get("trader_investment_plan")),
        "risk_discussion_turns": risk.get("count", 0),
        "aggressive_risk_level": risk.get("aggressive_risk_level") or None,
        "conservative_risk_level": risk.get("conservative_risk_level") or None,
        "neutral_risk_level": risk.get("neutral_risk_level") or None,
        "portfolio_manager_completed": bool(risk.get("judge_decision")),
    }


def _error_result(
    ticker: str,
    analysis_date: str,
    error: Exception,
    duration: float,
    *,
    stats: dict[str, Any] | None = None,
    failure_stage: str | None = None,
    final_state: dict[str, Any] | None = None,
    report_path: str | None = None,
) -> dict:
    """A failed ticker's result -- as rich as what's actually available.

    ``stats``/``failure_stage``/``final_state``/``report_path`` are only
    populated when the failure came from a graph that had already started
    executing (``PartialExecutionError``); a ticker that never got that far
    (an invalid ticker string, a bad config) has none of this to preserve,
    so these default to the same bare shape the error path always had.
    """
    return {
        "ticker": ticker,
        "analysis_date": analysis_date,
        "status": "error",
        "rating": None,
        "rating_parse_status": "not_run",
        "action": None,
        "entry_price": None,
        "stop_loss": None,
        "summary": None,
        "research_recommendation": None,
        "portfolio_manager_rating": None,
        "raw_final_decision": None,
        "report_path": report_path,
        "run_duration_seconds": round(duration, 3),
        "stats": stats,
        "error": str(error),
        "failure_stage": failure_stage,
        "partial": _partial_state_summary(final_state) if final_state else None,
        "source_coverage": (final_state or {}).get("source_coverage"),
        "structured": _structured_section(final_state or {}),
        "run_provenance": (final_state or {}).get("run_provenance"),
        "price_basis": (final_state or {}).get("price_basis"),
    }


def persist_successful_run(
    *, execution, config: dict, output_dir: Path, report_root: Path | None = None,
    progress: Callable[[str], None] | None = None, progress_prefix: str | None = None,
) -> dict[str, Any]:
    """Canonical success-path persistence, shared by every normal frontend.

    ``complete_report.md`` is already on disk by the time this runs --
    written inside ``execute_analysis`` itself (see ``cli/run.py``), which
    every caller already passes the same ``output_dir`` into -- so it is
    never rewritten or duplicated here. This writes the canonical
    ``result.json`` under ``output_dir`` (the run's own artifacts, unchanged
    by the ledger split below) and then freezes a RecommendationRecord from
    exactly those bytes into the SEPARATE, global ``ledger_root`` (see
    ``tradingagents.recommendation_record.resolve_ledger_root`` --
    independent of ``output_dir``/``--output-dir`` on purpose, so the ledger
    is never fragmented across every run's own output directory).
    ``record_recommendation`` is a no-op, logged and swallowed, when the
    run's ``portfolio_decision`` never parsed structurally, or when the
    ledger itself could not be written (e.g. a permissions problem) -- never
    when the run's own artifacts are the issue.

    ``report_root`` lets batch mode keep ``result.json["report_path"]``
    relative to the BATCH root rather than the per-ticker ``output_dir``;
    every other caller (interactive, single) leaves it as ``output_dir``.

    ``progress``, when given, is told where the ledger lives -- the exact
    ``resolve_ledger_root(config)`` value every other ledger consumer already
    computes from (no second, divergent resolution), and the exact path
    ``record_recommendation`` itself returned (never reconstructed), or an
    explicit "not written" line when it returned ``None``. ``progress_prefix``
    matches the same convention this module's other progress lines already
    use (ticker, or "[i/n] ticker" in batch).
    """
    report_root = report_root if report_root is not None else output_dir
    result = result_document(execution, report_root)
    result_json_path = Path(output_dir) / "result.json"
    _write_json(result_json_path, result)
    ledger_root = resolve_ledger_root(config)
    record_path = record_recommendation(
        execution=execution, result=result, config=config,
        result_json_path=result_json_path, report_path=execution.report_path,
    )
    if progress is not None:
        prefix = f"{progress_prefix} - " if progress_prefix else ""
        progress(f"{prefix}Ledger root: {ledger_root}")
        progress(
            f"{prefix}Recommendation record: {record_path}"
            if record_path is not None
            else f"{prefix}Recommendation record: not written"
        )
    return result
