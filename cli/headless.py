"""Non-interactive single and sequential batch execution helpers."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import typer

from cli.display import ANALYST_ORDER
from cli.models import AnalystType
from cli.prompts import detect_asset_type, filter_analysts_for_asset_type, normalize_ticker_symbol
from cli.run import (
    PartialExecutionError,
    _build_run_config,
    execute_analysis,
    format_duration,
)
from cli.stats_handler import StatsCallbackHandler
from tradingagents.agents.rating import RATINGS_5_TIER, extract_choice, is_review
from tradingagents.agents.validator import summarize_validation_log
from tradingagents.dataflows.symbols import safe_ticker_component
from tradingagents.default_config import (
    DEFAULT_CONFIG,
    resolve_effective_roles,
    resolve_role_config,
    resolve_validation_config,
)
from tradingagents.graph.conditional_logic import RISK_ANALYSIS_TOTAL_TURNS
from tradingagents.graph.trading_graph import _validate_trade_date

DEPTH_SELECTIONS = {"shallow": 1, "medium": 3, "deep": 5}
_ACTIONS = ("Buy", "Hold", "Sell")
# Analyst display labels, keyed by analyst key in ANALYST_ORDER.
_ANALYST_LABELS = {
    "market": "market analyst",
    "social": "sentiment analyst",
    "news": "news analyst",
    "fundamentals": "fundamentals analyst",
}
# Report state key -> analyst key, in the fixed execution order.
_REPORT_STAGES = {
    "market_report": "market",
    "sentiment_report": "social",
    "news_report": "news",
    "fundamentals_report": "fundamentals",
}


def parse_depth(value: str) -> tuple[str, int]:
    name = value.strip().lower()
    if name not in DEPTH_SELECTIONS:
        choices = ", ".join(DEPTH_SELECTIONS)
        raise ValueError(f"unknown depth {value!r}; choose one of: {choices}")
    return name, DEPTH_SELECTIONS[name]


def parse_analysts(value: str) -> list[str]:
    requested = [item.strip().lower() for item in value.split(",") if item.strip()]
    invalid = sorted(set(requested) - set(ANALYST_ORDER))
    if invalid:
        raise ValueError(
            f"unknown analyst(s): {', '.join(invalid)}; "
            f"choose from: {', '.join(ANALYST_ORDER)}"
        )
    if not requested:
        raise ValueError("at least one analyst must be selected")
    return [name for name in ANALYST_ORDER if name in requested]


def parse_tickers(value: str | None, tickers_file: Path | None = None) -> list[str]:
    raw = [item.strip() for item in (value or "").split(",") if item.strip()]
    if tickers_file is not None:
        raw.extend(
            line.strip()
            for line in tickers_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    if not raw:
        raise ValueError("provide at least one ticker with --tickers or --tickers-file")

    tickers = []
    seen = set()
    for item in raw:
        ticker = normalize_ticker_symbol(item)
        safe_ticker_component(ticker)
        if ticker not in seen:
            tickers.append(ticker)
            seen.add(ticker)
    return tickers


def build_headless_config(
    depth_value: int,
    language: str,
    results_dir: Path,
    *,
    cli_deep_think_llm: str | None = None,
    cli_quick_think_llm: str | None = None,
    cli_generation_llm: str | None = None,
    cli_review_llm: str | None = None,
    cli_quick_reasoning_effort: str | None = None,
    cli_deep_reasoning_effort: str | None = None,
    cli_generation_reasoning_effort: str | None = None,
    cli_review_reasoning_effort: str | None = None,
    cli_validator_llm: str | None = None,
    cli_validator_reasoning_effort: str | None = None,
    cli_validation_enabled: bool | None = None,
) -> dict:
    """Assemble a non-interactive run's config (see fork README: "Model
    roles: Generation vs Review" / "Legacy naming").

    Model/role CLI flags: an explicit new-name flag (``cli_generation_llm`` /
    ``cli_review_llm``) wins over its legacy equivalent (``cli_quick_think_llm``
    -> generation, ``cli_deep_think_llm`` -> review); the merged pair is then
    resolved against DEFAULT_CONFIG with the same inherit semantics
    ``resolve_effective_models`` has always had (one flag alone runs the
    whole analysis on that model). Reasoning effort follows the identical
    new-over-legacy precedence, with no inherit logic (each role is
    independent).
    """
    if not language.strip():
        raise ValueError("language must not be empty")

    cli_generation_merged = cli_generation_llm or cli_quick_think_llm
    cli_review_merged = cli_review_llm or cli_deep_think_llm
    effective_review, effective_generation = resolve_effective_roles(
        DEFAULT_CONFIG, cli_review=cli_review_merged, cli_generation=cli_generation_merged
    )

    generation_reasoning_effort = (
        cli_generation_reasoning_effort or cli_quick_reasoning_effort
        or DEFAULT_CONFIG.get("generation_reasoning_effort")
    )
    review_reasoning_effort = (
        cli_review_reasoning_effort or cli_deep_reasoning_effort
        or DEFAULT_CONFIG.get("review_reasoning_effort")
    )
    validator_llm = cli_validator_llm or DEFAULT_CONFIG.get("validator_llm")
    validator_reasoning_effort = (
        cli_validator_reasoning_effort or DEFAULT_CONFIG.get("validator_reasoning_effort")
    )
    validation_enabled = (
        cli_validation_enabled if cli_validation_enabled is not None
        else DEFAULT_CONFIG.get("validation_enabled", False)
    )

    selections = {
        "research_depth": depth_value,
        # Legacy key names: _build_run_config (shared with the interactive
        # flow) only reads these; resolve_role_config below is what maps the
        # already-resolved values here onto the canonical generation_llm/
        # review_llm keys everything downstream reads.
        "quick_think_llm": effective_generation,
        "deep_think_llm": effective_review,
        "quick_reasoning_effort": generation_reasoning_effort,
        "deep_reasoning_effort": review_reasoning_effort,
        "backend_url": DEFAULT_CONFIG["backend_url"],
        "llm_provider": DEFAULT_CONFIG["llm_provider"],
        "google_thinking_level": DEFAULT_CONFIG.get("google_thinking_level"),
        "openai_reasoning_effort": DEFAULT_CONFIG.get("openai_reasoning_effort"),
        "anthropic_effort": DEFAULT_CONFIG.get("anthropic_effort"),
        "validator_llm": validator_llm,
        "validator_reasoning_effort": validator_reasoning_effort,
        "validation_enabled": validation_enabled,
        "output_language": language.strip(),
    }
    config = _build_run_config(selections, checkpoint=None)
    config["results_dir"] = str(results_dir)
    config = resolve_role_config(config)
    # Fails loudly here too (before the progress line below even prints),
    # not only in execute_analysis -- see fork README: "Generation /
    # Validation / Review".
    return resolve_validation_config(config)


def _analysts_for_ticker(analysts: list[str], ticker: str) -> tuple[list[str], str]:
    asset_type = detect_asset_type(ticker)
    selected = filter_analysts_for_asset_type(
        [AnalystType(name) for name in analysts], asset_type
    )
    if not selected:
        raise ValueError(f"no selected analysts support {ticker}")
    return [analyst.value for analyst in selected], asset_type.value


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


def _progress_reporter(
    progress: Callable[[str], None],
    prefix: str,
    config: dict,
    stats_handler: StatsCallbackHandler,
    started: float,
    selected_analysts: tuple[str, ...] | list[str] = tuple(ANALYST_ORDER),
) -> Callable[[dict], None]:
    """Build the chunk handler that announces every pipeline stage transition.

    Each phase is announced twice when its turn comes: a "requesting X" line
    when the phase starts (announced as the previous stage completes, since
    that is when the graph hands the work over) and an "X completed" line
    when its report lands in a streamed chunk.
    """
    seen_reports = set()
    pending = [key for key in ANALYST_ORDER if key in set(selected_analysts)]
    last_review_round = 0
    last_verifier_pass = 0
    last_repair_round = 0
    last_risk_count = 0
    manager_done = False
    trader_done = False
    portfolio_done = False
    last_activity_calls = 0

    def emit(stage: str) -> None:
        stats = stats_handler.get_stats()
        elapsed = format_duration(time.monotonic() - started)
        token_text = (
            f", tokens {stats['tokens_in']} in/{stats['tokens_out']} out"
            if stats["tokens_in"] or stats["tokens_out"]
            else ""
        )
        progress(
            f"{prefix} - {stage} [{elapsed}, LLM {stats['llm_calls']}, "
            f"tools {stats['tool_calls']}{token_text}]"
        )

    def on_chunk(chunk: dict) -> None:
        nonlocal last_review_round, last_verifier_pass, last_repair_round
        nonlocal last_risk_count, manager_done, trader_done, portfolio_done
        nonlocal last_activity_calls
        emitted = False

        analyst_completed = False
        for report_key, analyst_key in _REPORT_STAGES.items():
            if chunk.get(report_key) and analyst_key not in seen_reports:
                seen_reports.add(analyst_key)
                emit(f"{_ANALYST_LABELS[analyst_key]} completed")
                analyst_completed = True
                emitted = True
        if analyst_completed:
            pending[:] = [key for key in pending if key not in seen_reports]
            if pending:
                emit(f"requesting {_ANALYST_LABELS[pending[0]]} analysis")
            else:
                emit("requesting Bull/Bear research debate")
            emitted = True

        debate = chunk.get("investment_debate_state", {})
        if debate.get("bull_initial") and "bull_initial" not in seen_reports:
            seen_reports.add("bull_initial")
            emit("Bull initial thesis completed")
            emitted = True
        if debate.get("bear_initial") and "bear_initial" not in seen_reports:
            seen_reports.add("bear_initial")
            emit("Bear initial thesis completed")
            emitted = True
        review_round = debate.get("debate_round", 0)
        if review_round > last_review_round:
            last_review_round = review_round
            emit(
                f"research review {review_round}/{config['max_debate_rounds']} completed"
            )
            emitted = True
        verifier_pass = debate.get("verifier_pass_count", 0)
        if verifier_pass > last_verifier_pass:
            last_verifier_pass = verifier_pass
            emit(
                f"research verifier pass {verifier_pass}: "
                f"{debate.get('verification_status', 'unknown')}"
            )
            emitted = True
        repair_round = debate.get("repair_rounds", 0)
        if repair_round > last_repair_round:
            last_repair_round = repair_round
            repaired = ", ".join(debate.get("repaired_agents", [])) or "research team"
            emit(f"research repair {repair_round} completed ({repaired})")
            emitted = True
        if debate.get("judge_decision") and not manager_done:
            manager_done = True
            emit("research manager completed")
            status = debate.get("manager_integrity_status")
            findings = debate.get("manager_integrity_findings") or []
            if status == "WARN":
                emit(f"manager integrity check: WARN ({len(findings)} finding(s))")
            elif status:
                emit(f"manager integrity check: {status}")
            emit("requesting trader analysis")
            emitted = True

        if chunk.get("trader_investment_plan") and not trader_done:
            trader_done = True
            emit("trader completed")
            emit("requesting risk management discussion")
            emitted = True

        risk = chunk.get("risk_debate_state", {})
        risk_count = risk.get("count", 0)
        if risk_count > last_risk_count:
            last_risk_count = risk_count
            speaker = risk.get("latest_speaker") or "risk agent"
            phase = "independent assessment" if risk_count <= 3 else "cross-review"
            emit(f"risk {phase} {risk_count}/{RISK_ANALYSIS_TOTAL_TURNS}: {speaker}")
            if risk_count >= RISK_ANALYSIS_TOTAL_TURNS and not portfolio_done:
                emit("requesting portfolio manager decision")
            emitted = True
        if risk.get("judge_decision") and not portfolio_done:
            portfolio_done = True
            status = risk.get("risk_integrity_status")
            findings = risk.get("risk_integrity_findings") or []
            if status == "WARN":
                emit(f"risk integrity check: WARN ({len(findings)} finding(s))")
            elif status:
                emit(f"risk integrity check: {status}")
            emit("portfolio manager completed")
            emitted = True

        stats = stats_handler.get_stats()
        activity_calls = stats["llm_calls"] + stats["tool_calls"]
        if not emitted and activity_calls - last_activity_calls >= 4:
            emit("analysis in progress")
            emitted = True
        if emitted:
            last_activity_calls = activity_calls

    return on_chunk


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
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def run_single_analysis(
    *,
    ticker: str,
    analysis_date: str,
    analysts: list[str],
    depth_value: int,
    language: str,
    output_dir: Path,
    report_root: Path | None = None,
    progress: Callable[[str], None] | None = None,
    progress_prefix: str | None = None,
    deep_think_llm: str | None = None,
    quick_think_llm: str | None = None,
    generation_llm: str | None = None,
    review_llm: str | None = None,
    quick_reasoning_effort: str | None = None,
    deep_reasoning_effort: str | None = None,
    generation_reasoning_effort: str | None = None,
    review_reasoning_effort: str | None = None,
    validator_llm: str | None = None,
    validator_reasoning_effort: str | None = None,
    validation_enabled: bool | None = None,
    trace_llm: bool = False,
    base_seed: int | None = None,
    configured_seed: int | None = None,
) -> dict[str, Any]:
    ticker = normalize_ticker_symbol(ticker)
    safe_ticker_component(ticker)
    analysis_date = _validate_trade_date(analysis_date)
    selected_analysts, asset_type = _analysts_for_ticker(analysts, ticker)
    output_dir = output_dir.resolve()
    root = (report_root or output_dir).resolve()
    config = build_headless_config(
        depth_value, language, root,
        cli_deep_think_llm=deep_think_llm, cli_quick_think_llm=quick_think_llm,
        cli_generation_llm=generation_llm, cli_review_llm=review_llm,
        cli_quick_reasoning_effort=quick_reasoning_effort,
        cli_deep_reasoning_effort=deep_reasoning_effort,
        cli_generation_reasoning_effort=generation_reasoning_effort,
        cli_review_reasoning_effort=review_reasoning_effort,
        cli_validator_llm=validator_llm,
        cli_validator_reasoning_effort=validator_reasoning_effort,
        cli_validation_enabled=validation_enabled,
    )
    if progress is not None:
        progress(
            f"{progress_prefix or ticker} - Generation model: {config['generation_llm']}; "
            f"Review model: {config['review_llm']}"
        )
    stats_handler = StatsCallbackHandler()
    started = time.monotonic()
    on_chunk = (
        _progress_reporter(
            progress,
            progress_prefix or ticker,
            config,
            stats_handler,
            started,
            selected_analysts=selected_analysts,
        )
        if progress is not None
        else None
    )
    if progress is not None:
        # The first analyst's turn starts the moment the graph runs, so the
        # request line is emitted here; later requests ride on the chunk
        # handler, announced as the previous stage completes.
        progress(
            f"{progress_prefix or ticker} - requesting "
            f"{_ANALYST_LABELS[selected_analysts[0]]} analysis"
        )
    try:
        execution = execute_analysis(
            ticker=ticker,
            analysis_date=analysis_date,
            selected_analysts=selected_analysts,
            asset_type=asset_type,
            config=config,
            output_dir=output_dir,
            stats_handler=stats_handler,
            on_chunk=on_chunk,
            trace_llm=trace_llm,
            base_seed=base_seed,
            configured_seed=configured_seed,
        )
    except PartialExecutionError as exc:
        report_path = (
            exc.report_path.relative_to(root).as_posix()
            if exc.report_path is not None
            else None
        )
        result = _error_result(
            ticker, analysis_date, exc.cause, exc.duration_seconds,
            stats=exc.stats, failure_stage=exc.failure_stage,
            final_state=exc.final_state, report_path=report_path,
        )
        _write_json(output_dir / "result.json", result)
        return result

    result = result_document(execution, root)
    _write_json(output_dir / "result.json", result)
    return result


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
    }


def timestamped_echo(printer: Callable[..., None] = typer.echo) -> Callable[..., None]:
    """Wrap a printer so every printed line is prefixed with the wall-clock time.

    Headless runs print plain lines, not the interactive live view, so nothing
    used to say *when* a stage happened. The returned callable stamps every
    message with "YYYY-MM-DD HH:MM:SS " before passing it to ``printer``.
    """

    def echo(message: str, **kwargs: Any) -> None:
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        printer(f"{stamp} {message}", **kwargs)

    return echo


def write_batch_summary(
    output_dir: Path,
    *,
    analysis_date: str,
    depth: str,
    analysts: list[str],
    language: str,
    results: list[dict],
    base_seed: int | None = None,
) -> tuple[Path, Path]:
    successful = sum(result["status"] == "ok" for result in results)
    metadata = {
        "analysis_date": analysis_date,
        "depth": depth,
        "analysts": analysts,
        "language": language,
        "tickers": [result["ticker"] for result in results],
        "successful": successful,
        "failed": len(results) - successful,
        "base_seed": base_seed,
        "results": results,
    }
    json_path = output_dir / "summary.json"
    _write_json(json_path, metadata)

    lines = ["# TradingAgents Batch Summary", ""]
    if base_seed is not None:
        lines.append(f"Run seed: {base_seed}")
        lines.append("")
    lines += [
        "| Ticker | Status | Rating | Action | Entry | Stop | Report |",
        "|---|---|---|---|---:|---:|---|",
    ]
    for result in results:
        ok = result["status"] == "ok"
        if result.get("report_path"):
            # A failed ticker can still have a report_path: a partial report
            # saved from whatever the graph completed before it failed (see
            # PartialExecutionError). Link to it the same way a successful
            # ticker's report is linked, so a partial result is never a dead
            # end -- but it is still visibly ERROR in the Status column, and
            # the report itself opens with an unmissable incomplete-run banner.
            label = "report" if ok else "partial report"
            link = f"[{label}]({result['report_path']})"
        else:
            try:
                # The ticker itself may be the reason this result failed (an
                # unsafe/invalid string), in which case no result.json was
                # ever written for it — don't crash the whole summary over it.
                link = f"[error]({safe_ticker_component(result['ticker'])}/result.json)"
            except ValueError:
                link = "error (no report written)"
        lines.append(
            "| {ticker} | {status} | {rating} | {action} | {entry} | {stop} | {link} |".format(
                ticker=result["ticker"],
                status="OK" if ok else "ERROR",
                rating=result.get("rating") or "-",
                action=result.get("action") or "-",
                entry=result.get("entry_price") if result.get("entry_price") is not None else "-",
                stop=result.get("stop_loss") if result.get("stop_loss") is not None else "-",
                link=link,
            )
        )
    failures = [result for result in results if result["status"] == "error"]
    if failures:
        lines.extend(["", "## Failures", ""])
        for result in failures:
            stage = result.get("failure_stage")
            stage_text = f" (at {stage})" if stage else ""
            lines.append(f"- **{result['ticker']}**{stage_text}: {result['error']}")
    markdown_path = output_dir / "summary.md"
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return markdown_path, json_path


def run_batch_analysis(
    *,
    tickers: list[str],
    analysis_date: str,
    analysts: list[str],
    depth_name: str,
    depth_value: int,
    language: str,
    output_dir: Path,
    progress: Callable[[str], None],
    deep_think_llm: str | None = None,
    quick_think_llm: str | None = None,
    generation_llm: str | None = None,
    review_llm: str | None = None,
    quick_reasoning_effort: str | None = None,
    deep_reasoning_effort: str | None = None,
    generation_reasoning_effort: str | None = None,
    review_reasoning_effort: str | None = None,
    validator_llm: str | None = None,
    validator_reasoning_effort: str | None = None,
    validation_enabled: bool | None = None,
    trace_llm: bool = False,
    base_seed: int | None = None,
    configured_seed: int | None = None,
) -> tuple[list[dict], Path, Path]:
    """Run every ticker sequentially with the SAME ``base_seed`` (see
    ``tradingagents/run_seed.py``) -- resolved once by the caller, before this
    function runs, never regenerated per ticker."""
    analysis_date = _validate_trade_date(analysis_date)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    # Resolve + validate config once, up front, outside the per-ticker try/
    # except below -- a validation_enabled-without-validator_llm
    # misconfiguration (see fork README: "Generation / Validation / Review")
    # must fail the whole batch immediately with one clear error, not repeat
    # identically for every ticker in the loop.
    build_headless_config(
        depth_value, language, output_dir,
        cli_deep_think_llm=deep_think_llm, cli_quick_think_llm=quick_think_llm,
        cli_generation_llm=generation_llm, cli_review_llm=review_llm,
        cli_quick_reasoning_effort=quick_reasoning_effort,
        cli_deep_reasoning_effort=deep_reasoning_effort,
        cli_generation_reasoning_effort=generation_reasoning_effort,
        cli_review_reasoning_effort=review_reasoning_effort,
        cli_validator_llm=validator_llm,
        cli_validator_reasoning_effort=validator_reasoning_effort,
        cli_validation_enabled=validation_enabled,
    )
    results = []
    for index, ticker in enumerate(tickers, start=1):
        started = time.monotonic()
        try:
            ticker_dir = output_dir / safe_ticker_component(ticker)
        except ValueError as exc:
            # An invalid ticker string must not abort the rest of the batch —
            # record it as a failure and move on, same as any other per-ticker
            # error below. There's no safe directory to write a result.json to.
            result = _error_result(ticker, analysis_date, exc, time.monotonic() - started)
            progress(f"[{index}/{len(tickers)}] {ticker} - FAILED: {exc}")
            results.append(result)
            continue

        progress(f"[{index}/{len(tickers)}] {ticker} - starting")
        try:
            result = run_single_analysis(
                ticker=ticker,
                analysis_date=analysis_date,
                analysts=analysts,
                depth_value=depth_value,
                language=language,
                output_dir=ticker_dir,
                report_root=output_dir,
                progress=progress,
                progress_prefix=f"[{index}/{len(tickers)}] {ticker}",
                deep_think_llm=deep_think_llm,
                quick_think_llm=quick_think_llm,
                generation_llm=generation_llm,
                review_llm=review_llm,
                quick_reasoning_effort=quick_reasoning_effort,
                deep_reasoning_effort=deep_reasoning_effort,
                generation_reasoning_effort=generation_reasoning_effort,
                review_reasoning_effort=review_reasoning_effort,
                validator_llm=validator_llm,
                validator_reasoning_effort=validator_reasoning_effort,
                validation_enabled=validation_enabled,
                trace_llm=trace_llm,
                base_seed=base_seed,
                configured_seed=configured_seed,
            )
        except Exception as exc:
            # A setup-time failure that never reached graph execution (e.g. a
            # bad config) -- a graph-execution failure is already turned into
            # an "error" result dict by run_single_analysis, below, without
            # raising, so it keeps its partial artifacts intact.
            result = _error_result(
                ticker, analysis_date, exc, time.monotonic() - started
            )
            _write_json(ticker_dir / "result.json", result)
            progress(f"[{index}/{len(tickers)}] {ticker} - FAILED: {exc}")
            results.append(result)
            continue

        if result["status"] == "ok":
            stats = result["stats"]
            research = result["research_execution"]
            fallback_warnings = result.get("fallback_warnings", 0)
            fallback_suffix = (
                f" | completed with {fallback_warnings} fallback warning(s)"
                if fallback_warnings
                else ""
            )
            validation = result.get("validation") or {}
            validation_suffix = (
                f" | validation: {validation['fail']} fail, "
                f"{validation['unresolved']} unresolved"
                if validation.get("checked") and (validation["fail"] or validation["unresolved"])
                else ""
            )
            progress(
                f"[{index}/{len(tickers)}] {ticker} - completed in "
                f"{format_duration(result['run_duration_seconds'])} | "
                f"LLM {stats['llm_calls']}, tools {stats['tool_calls']}, "
                f"tokens {stats['tokens_in']} in/{stats['tokens_out']} out | "
                f"verifier {research['verification_status']} "
                f"({research['verifier_passes']} pass(es), "
                f"{research['repair_rounds']} repair(s)){fallback_suffix}{validation_suffix}"
            )
        else:
            stage = result.get("failure_stage") or "unknown"
            duration_text = format_duration(result["run_duration_seconds"])
            progress(
                f"[{index}/{len(tickers)}] {ticker} - FAILED at {stage} "
                f"after {duration_text}: {result['error']}"
            )
            if result.get("report_path"):
                progress(
                    f"[{index}/{len(tickers)}] {ticker} - Partial artifacts "
                    f"saved: {result['report_path']}"
                )
            stats = result.get("stats") or {}
            if stats:
                progress(
                    f"[{index}/{len(tickers)}] {ticker} - "
                    f"LLM {stats.get('llm_calls', 0)}, tools {stats.get('tool_calls', 0)}, "
                    f"tokens {stats.get('tokens_in', 0)} in/{stats.get('tokens_out', 0)} out"
                )
        results.append(result)

    markdown_path, json_path = write_batch_summary(
        output_dir,
        analysis_date=analysis_date,
        depth=depth_name,
        analysts=analysts,
        language=language,
        results=results,
        base_seed=base_seed,
    )
    return results, markdown_path, json_path
