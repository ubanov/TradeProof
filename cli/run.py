"""Running one analysis from the CLI: build the graph, stream it into the live view, save the report."""

import datetime
import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any

import typer
from rich.live import Live

from cli.display import (
    ANALYST_ORDER,
    AnalystWallTimeTracker,
    classify_message_type,
    console,
    create_layout,
    display_complete_report,
    message_buffer,
    update_analyst_statuses,
    update_display,
    update_research_team_status,
)
from cli.selections import get_user_selections
from cli.stats_handler import StatsCallbackHandler
from tradingagents.agents.fallback_log import fallback_count, reset_fallback_count
from tradingagents.agents.rating import is_review
from tradingagents.dataflows.source_coverage import (
    SourceCoverageCollector,
    get_source_coverage_snapshot,
    set_active_source_coverage_collector,
)
from tradingagents.dataflows.symbols import safe_ticker_component
from tradingagents.default_config import (
    DEFAULT_CONFIG,
    resolve_role_config,
    resolve_validation_config,
)
from tradingagents.graph.analyst_execution import (
    build_analyst_execution_plan,
)
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.llm_trace import (
    LLMTraceCallbackHandler,
    SeedContext,
    _git_info,
    _package_version,
    set_active_seed_context,
    set_active_tracer,
    snapshot_prompts,
    write_run_manifest,
)
from tradingagents.reporting import write_report_tree
from tradingagents.run_provenance import build_run_provenance
from tradingagents.run_seed import SEED_CAPABLE_PROVIDERS

logger = logging.getLogger(__name__)

RESEARCH_REVIEW_ROUNDS_BY_DEPTH = {1: 1, 3: 2, 5: 3}


def format_duration(seconds: float) -> str:
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

# Pipeline stages a failure can be attributed to, in execution order. Shared
# by the failure-stage inference below and any caller that wants to validate
# or display a stage name, so the vocabulary has exactly one source.
PIPELINE_STAGES: tuple[str, ...] = (
    "market_analyst",
    "sentiment_analyst",
    "news_analyst",
    "fundamentals_analyst",
    "research_initial",
    "research_review",
    "verifier",
    "repair",
    "research_manager",
    "trader",
    "risk",
    "portfolio_manager",
    "report_generation",
)

_ANALYST_STAGE_KEYS: tuple[tuple[str, str], ...] = (
    ("market_report", "market_analyst"),
    ("sentiment_report", "sentiment_analyst"),
    ("news_report", "news_analyst"),
    ("fundamentals_report", "fundamentals_analyst"),
)


def infer_failure_stage(final_state: dict[str, Any] | None) -> str:
    """Best-effort guess at the furthest pipeline stage a failed run reached,
    from whatever partial state the stream yielded before the exception.

    Checked in reverse pipeline order so the LATEST reached stage wins: e.g. a
    run whose risk discussion finished all six turns (three independent
    assessments, then the single cross-review round) but never got a
    Portfolio Manager decision failed AT Portfolio Manager, not somewhere
    earlier just because repair/verifier fields are also present.

    Not an exact science -- a node that partially mutated shared keys before
    raising could be misattributed -- but every real agent node in this graph
    only ever returns (and thus only ever contributes to a streamed chunk)
    once it has fully completed, so in practice the furthest-populated marker
    reliably identifies the stage that was in progress when the exception hit.
    """
    final_state = final_state or {}
    debate = final_state.get("investment_debate_state") or {}
    risk = final_state.get("risk_debate_state") or {}

    if final_state.get("final_trade_decision") or risk.get("judge_decision"):
        return "complete"
    if risk.get("neutral_review"):  # all six Risk turns done, no PM decision yet
        return "portfolio_manager"
    if risk.get("count", 0) > 0 or final_state.get("trader_investment_plan"):
        return "risk"
    if debate.get("judge_decision") or debate.get("manager_integrity_status"):
        return "trader"
    verification_history = debate.get("verification_history") or []
    if verification_history:
        last_pass = verification_history[-1]
        if last_pass.get("status") == "FAIL" and not debate.get("repair_triggered"):
            return "repair"
        return "research_manager"
    if debate.get("debate_round", 0) > 0 or (debate.get("bull_initial") and debate.get("bear_initial")):
        return "verifier"
    if debate.get("bull_initial") or debate.get("bear_initial"):
        return "research_review"
    for report_key, stage in _ANALYST_STAGE_KEYS:
        if not final_state.get(report_key):
            return stage
    return "research_initial"


class PartialExecutionError(RuntimeError):
    """Raised when the graph fails mid-run.

    Carries whatever state/stats were captured from the stream before the
    failure, so a caller can save a partial report and a rich error result
    instead of losing completed work (a real run lost ~2h43m of finished
    analysis, verification, repair, research management, trading, and a full
    9-turn risk discussion to an uncaught exception immediately before
    Portfolio Manager).
    """

    def __init__(
        self,
        cause: BaseException,
        *,
        final_state: dict[str, Any],
        stats: dict[str, Any],
        duration_seconds: float,
        failure_stage: str,
        report_path: Path | None,
    ):
        super().__init__(str(cause))
        self.cause = cause
        self.final_state = final_state
        self.stats = stats
        self.duration_seconds = duration_seconds
        self.failure_stage = failure_stage
        self.report_path = report_path


@dataclass(frozen=True)
class AnalysisExecution:
    ticker: str
    analysis_date: str
    final_state: dict[str, Any]
    rating: str
    report_path: Path | None
    duration_seconds: float
    stats: dict[str, Any]
    graph: TradingAgentsGraph
    fallback_warnings: int = 0
    base_seed: int | None = None


def _run_directory(config: dict, ticker: str, trade_date: str) -> Path:
    """Where this run writes, with the ticker validated as a path component.

    Every other path that interpolates a ticker checks it first; a value of
    ".." here would place the run outside the results directory.
    """
    return Path(config["results_dir"]) / safe_ticker_component(ticker) / trade_date


def _announce_checkpoint_state(graph, ticker: str, trade_date: str) -> None:
    """Say whether this run resumed a saved one, where the user can see it.

    The graph logs this, but nothing in the CLI configures logging and the live
    view owns the screen, so a resume was invisible.
    """
    if getattr(graph, "_resuming", False):
        message_buffer.add_message(
            "System", f"Resuming the saved run for {ticker} on {trade_date}"
        )
    else:
        message_buffer.add_message("System", f"Starting fresh for {ticker} on {trade_date}")


def _build_run_config(selections: dict, checkpoint: bool | None) -> dict:
    """Assemble the run config from interactive selections, honoring env precedence.

    Round counts and checkpoint follow "explicit env/flag wins": an env-applied
    value on DEFAULT_CONFIG is preserved unless the user overrode it on the CLI.
    """
    config = DEFAULT_CONFIG.copy()
    # The visible depth choice remains 1/3/5 for compatibility. Research uses
    # 1/2/3 cross-review rounds while risk keeps its existing 1/3/5 mapping.
    depth = selections["research_depth"]
    selected_rounds = {
        "max_debate_rounds": RESEARCH_REVIEW_ROUNDS_BY_DEPTH.get(depth, depth),
        "max_risk_discuss_rounds": depth,
    }
    # Explicit per-team environment overrides remain literal round counts.
    for env_var, key in (
        ("TRADINGAGENTS_MAX_DEBATE_ROUNDS", "max_debate_rounds"),
        ("TRADINGAGENTS_MAX_RISK_ROUNDS", "max_risk_discuss_rounds"),
    ):
        if os.environ.get(env_var):
            # The depth prompt still appeared (it is skipped only when both are
            # set), so say which half of the answer the environment overrode.
            console.print(
                f"[green]✓ {key} from environment:[/green] {config[key]} "
                f"(set by {env_var}, so the research depth you chose does not apply to it)"
            )
        else:
            config[key] = selected_rounds[key]
    config["quick_think_llm"] = selections["quick_think_llm"]
    config["deep_think_llm"] = selections["deep_think_llm"]
    # Reset the canonical keys copied from DEFAULT_CONFIG above (already
    # non-None, since DEFAULT_CONFIG is itself always pre-resolved) back to
    # None, so resolve_role_config -- called downstream, e.g. in
    # execute_analysis -- re-derives generation_llm/review_llm from the
    # legacy values just set above instead of keeping DEFAULT_CONFIG's own
    # stale resolution and silently ignoring this run's selection (see fork
    # README: "Legacy naming"). A selections dict that carries an explicit
    # canonical override (the headless config builder's new --generation-llm/
    # --review-llm flags) already folded it into these same legacy keys
    # before calling here, so nothing is lost.
    config["generation_llm"] = None
    config["review_llm"] = None
    config["backend_url"] = selections["backend_url"]
    config["llm_provider"] = selections["llm_provider"].lower()
    # Provider-specific thinking configuration
    config["google_thinking_level"] = selections.get("google_thinking_level")
    config["openai_reasoning_effort"] = selections.get("openai_reasoning_effort")
    config["anthropic_effort"] = selections.get("anthropic_effort")
    # Quick vs Deep reasoning effort (see fork README): per-role, independent
    # of which model each is set to. The interactive flow does not prompt for
    # these, unlike the three knobs above (selections has no such key) --
    # fall back to DEFAULT_CONFIG's own (possibly env-sourced) value instead
    # of unconditionally overwriting it with a bare None, so an interactive
    # run doesn't silently drop a TRADINGAGENTS_*_REASONING_EFFORT env var.
    # The headless config builder always sets these explicitly either way.
    config["quick_reasoning_effort"] = selections.get("quick_reasoning_effort") or DEFAULT_CONFIG.get("quick_reasoning_effort")
    config["deep_reasoning_effort"] = selections.get("deep_reasoning_effort") or DEFAULT_CONFIG.get("deep_reasoning_effort")
    config["generation_reasoning_effort"] = None
    config["review_reasoning_effort"] = None
    # Validator role (see fork README: "Generation / Validation / Review") --
    # no legacy name to resolve, so these are set directly (same
    # fall-back-to-DEFAULT_CONFIG reasoning as above), not reset for a
    # downstream resolver.
    config["validator_llm"] = selections.get("validator_llm") or DEFAULT_CONFIG.get("validator_llm")
    config["validator_reasoning_effort"] = (
        selections.get("validator_reasoning_effort") or DEFAULT_CONFIG.get("validator_reasoning_effort")
    )
    validation_enabled = selections.get("validation_enabled")
    config["validation_enabled"] = (
        validation_enabled if validation_enabled is not None else DEFAULT_CONFIG.get("validation_enabled", False)
    )
    config["output_language"] = selections.get("output_language", "English")
    # --checkpoint/--no-checkpoint overrides only when explicitly given; omitting
    # the flag preserves TRADINGAGENTS_CHECKPOINT_ENABLED / the default (#976).
    if checkpoint is not None:
        config["checkpoint_enabled"] = checkpoint
    return config


def _apply_seed_context(*, base_seed: int | None, ticker: str, config: dict) -> bool:
    """Set up this run's best-effort deterministic-replay seed context (see
    ``tradingagents/run_seed.py``), or explicitly clear any context left
    active by a previous ticker in the same batch process.

    Always called (even when ``base_seed`` is None): contextvars persist
    across sequential calls in the same thread, so a batch run that seeds
    ticker 1 but not ticker 2 must not leak ticker 1's context into ticker 2
    (same reasoning as ``_start_llm_trace``'s tracer reset).

    Returns whether this run's provider is seed-capable, for the manifest.
    """
    if base_seed is None:
        set_active_seed_context(None)
        return False
    provider_supports_seed = config.get("llm_provider", "").lower() in SEED_CAPABLE_PROVIDERS
    set_active_seed_context(
        SeedContext(base_seed=base_seed, ticker=ticker, provider_supports_seed=provider_supports_seed)
    )
    return provider_supports_seed


def _start_llm_trace(
    *, trace_llm: bool, output_dir: Path | None, ticker: str, analysis_date: str,
    selected_analysts: list[str], config: dict,
    configured_seed: int | None = None, base_seed: int | None = None,
    seed_provider_supported: bool = False,
    git_info: dict | None = None, package_version: str | None = None,
    prompt_hashes: dict[str, str] | None = None,
) -> LLMTraceCallbackHandler | None:
    """Set up opt-in LLM trace mode for this run, or explicitly clear any
    tracer left active by a previous ticker in the same batch process.

    Always called (even when trace_llm is False): contextvars persist across
    sequential calls in the same thread, so a batch run that traces ticker 1
    but not ticker 2 must not leak ticker 1's tracer into ticker 2.

    ``git_info``/``package_version``/``prompt_hashes`` are the SAME values
    already computed once for this run's unconditional provenance snapshot
    (see ``build_run_provenance`` in ``execute_analysis`` below) -- passed
    through so git is never invoked twice per run and the manifest can
    never diverge from provenance on the fields they share.
    """
    if not trace_llm or output_dir is None:
        set_active_tracer(None)
        return None

    debug_dir = Path(output_dir) / "debug"
    tracer = LLMTraceCallbackHandler(debug_dir)
    set_active_tracer(tracer)
    print(f"[DEBUG] LLM trace enabled: {debug_dir}", flush=True)

    def _on_warn(path: Path, exc: Exception) -> None:
        print(f"[WARN] LLM trace: failed to write {path} ({exc}); continuing.", flush=True)

    # snapshot_prompts still does its own read pass (it writes each file to
    # debug_dir/prompts/, which needs the content regardless), but its
    # hashing is hash_prompt_files() underneath -- the exact function
    # provenance's own prompt_hashes already came from, never a second
    # hashing implementation.
    written_prompt_hashes = snapshot_prompts(debug_dir, _on_warn)
    write_run_manifest(
        debug_dir,
        ticker=ticker,
        analysis_date=analysis_date,
        depth=config.get("max_risk_discuss_rounds"),
        analysts=selected_analysts,
        language=config.get("output_language", "English"),
        provider=config.get("llm_provider", ""),
        effective_deep_model=config.get("deep_think_llm", ""),
        effective_quick_model=config.get("quick_think_llm", ""),
        config=config,
        prompt_hashes=prompt_hashes if prompt_hashes is not None else written_prompt_hashes,
        on_warn=_on_warn,
        configured_seed=configured_seed,
        base_seed=base_seed,
        seed_provider_supported=seed_provider_supported,
        git_info=git_info,
        package_version=package_version,
    )
    return tracer


def execute_analysis(
    *,
    ticker: str,
    analysis_date: str,
    selected_analysts: list[str],
    asset_type: str,
    config: dict,
    portfolio=None,
    output_dir: Path | None = None,
    stats_handler: StatsCallbackHandler | None = None,
    on_chunk: Callable[[dict], None] | None = None,
    on_checkpoint: Callable[[TradingAgentsGraph], None] | None = None,
    trace_llm: bool = False,
    base_seed: int | None = None,
    configured_seed: int | None = None,
) -> AnalysisExecution:
    """Run one analysis through the shared interactive/headless execution path.

    ``base_seed`` is the already-resolved (see ``run_seed.resolve_base_seed``)
    best-effort deterministic-replay seed for this whole run -- None means no
    seeding at all (the default, and the only behavior for any caller that
    does not pass it, e.g. the interactive flow). ``configured_seed`` is only
    for the trace manifest, so a reader can see what the user actually asked
    for (0 = auto) alongside the base seed actually used.
    """
    # The single compatibility layer (see fork README: "Legacy naming"):
    # fills in generation_llm/review_llm/*_reasoning_effort from legacy
    # quick/deep config whenever a caller only set those, so every entry
    # point -- CLI, interactive, or a bare dict a test constructs directly --
    # resolves the same way before anything below reads the canonical keys.
    config = resolve_role_config(config)
    # Fails loudly here (before any client/graph construction) if
    # validation_enabled is set without a validator_llm -- see fork README:
    # "Generation / Validation / Review". A CLI caller also checks this
    # earlier, for a friendlier message before even parsing tickers/dates;
    # this is the safety net for any other caller (interactive, programmatic).
    config = resolve_validation_config(config)

    # Per-ticker, not per-process: a batch run must not accumulate one running
    # total across every ticker it processes.
    reset_fallback_count()
    build_analyst_execution_plan(selected_analysts)
    stats_handler = stats_handler or StatsCallbackHandler()
    # Always on (unlike the seed/trace contexts, which are opt-in) -- a batch
    # run must not accumulate one ticker's source-coverage entries into the
    # next, same reasoning as _apply_seed_context/_start_llm_trace.
    set_active_source_coverage_collector(SourceCoverageCollector())
    seed_provider_supported = _apply_seed_context(base_seed=base_seed, ticker=ticker, config=config)

    # Computed exactly once per run, regardless of --trace-llm, so git is
    # never invoked twice and the trace manifest (when active) reuses these
    # same values instead of recomputing them independently (see
    # run_provenance.py and _start_llm_trace's own git_info/package_version
    # params below).
    run_git_info = _git_info()
    run_package_version = _package_version()
    trace_manifest_path = (
        str(Path(output_dir) / "debug" / "run_manifest.json")
        if trace_llm and output_dir is not None
        else None
    )
    run_provenance = build_run_provenance(
        selected_analysts=selected_analysts, config=config,
        configured_seed=configured_seed, base_seed=base_seed,
        trace_llm=trace_llm, trace_manifest_path=trace_manifest_path,
        git_info=run_git_info, package_version=run_package_version,
    )

    tracer = _start_llm_trace(
        trace_llm=trace_llm, output_dir=output_dir, ticker=ticker,
        analysis_date=analysis_date, selected_analysts=selected_analysts, config=config,
        configured_seed=configured_seed, base_seed=base_seed,
        seed_provider_supported=seed_provider_supported,
        git_info=run_git_info, package_version=run_package_version,
        prompt_hashes=run_provenance["prompts"]["hashes"],
    )
    callbacks = [stats_handler] + ([tracer] if tracer is not None else [])
    graph = TradingAgentsGraph(
        selected_analysts,
        config=config,
        debug=True,
        callbacks=callbacks,
    )
    started = time.monotonic()
    init_agent_state = graph.create_run_state(
        ticker, analysis_date, asset_type, portfolio
    )
    args = graph.propagator.get_graph_args(callbacks=[stats_handler])
    checkpoint_tid = graph.begin_checkpoint(
        ticker, analysis_date, asset_type, portfolio
    )
    if checkpoint_tid is not None:
        args.setdefault("config", {}).setdefault("configurable", {})[
            "thread_id"
        ] = checkpoint_tid
        if on_checkpoint is not None:
            on_checkpoint(graph)

    trace = []
    stream_error: Exception | None = None
    try:
        for chunk in graph.graph.stream(graph.checkpoint_input(init_agent_state), **args):
            trace.append(chunk)
            if on_chunk is not None:
                on_chunk(chunk)

        final_state = {}
        for chunk in trace:
            final_state.update(chunk)

        graph.record_decision(ticker, analysis_date, final_state)
        graph.clear_checkpoint_on_success(
            ticker, analysis_date, asset_type, portfolio
        )
    except Exception as exc:
        # Whatever the graph yielded before failing is still real, completed
        # work (hours of it, in the run that motivated this): merge it the
        # same way the success path does, rather than losing it.
        stream_error = exc
        final_state = {}
        for chunk in trace:
            final_state.update(chunk)
    finally:
        graph.end_checkpoint()

    # Captured once, right after the graph finishes (success or failure),
    # into final_state itself -- so write_report_tree/result_document read
    # it from the exact same place every other analyst's report already
    # lives in, with no new return field needed on AnalysisExecution.
    final_state["source_coverage"] = get_source_coverage_snapshot()
    set_active_source_coverage_collector(None)
    # Unconditional (built above regardless of trace_llm) -- describes the
    # EXECUTION, not just a successful recommendation, so it is attached to
    # final_state on both the success and the partial/failed path below.
    final_state["run_provenance"] = run_provenance

    duration_seconds = time.monotonic() - started
    stats = stats_handler.get_stats()
    if tracer is not None:
        tracer.write_summary(
            stats=stats, elapsed_seconds=duration_seconds,
            generation_model=config.get("generation_llm", ""),
            review_model=config.get("review_llm", ""),
        )

    if stream_error is not None:
        failure_stage = infer_failure_stage(final_state)
        if failure_stage == "complete":
            # The graph itself finished; something after it (saving the
            # report, recording the decision) is what actually failed.
            failure_stage = "report_generation"
        partial_report_path = None
        if output_dir is not None:
            try:
                partial_report_path = write_report_tree(
                    final_state,
                    ticker,
                    output_dir,
                    error_banner=(
                        f"Execution failed at stage '{failure_stage}' after "
                        f"{format_duration(duration_seconds)}: {stream_error}"
                    ),
                    output_filename="partial_report.md",
                )
            except Exception:
                logger.warning(
                    "Could not write a partial report for %s after failure at %s",
                    ticker, failure_stage, exc_info=True,
                )
        raise PartialExecutionError(
            stream_error,
            final_state=final_state,
            stats=stats,
            duration_seconds=duration_seconds,
            failure_stage=failure_stage,
            report_path=partial_report_path,
        ) from stream_error

    report_path = (
        write_report_tree(final_state, ticker, output_dir)
        if output_dir is not None
        else None
    )
    return AnalysisExecution(
        ticker=ticker,
        analysis_date=analysis_date,
        final_state=final_state,
        rating=graph.process_signal(final_state.get("final_trade_decision", "")),
        report_path=report_path,
        duration_seconds=duration_seconds,
        stats=stats,
        graph=graph,
        fallback_warnings=fallback_count(),
        base_seed=base_seed,
    )


def run_analysis(checkpoint: bool | None = None, portfolio=None):
    # First get all user selections
    selections = get_user_selections()

    config = _build_run_config(selections, checkpoint)

    stats_handler = StatsCallbackHandler()

    # Normalize analyst selection to predefined order (selection is a 'set', order is fixed)
    selected_set = {analyst.value for analyst in selections["analysts"]}
    selected_analyst_keys = [a for a in ANALYST_ORDER if a in selected_set]
    analyst_execution_plan = build_analyst_execution_plan(selected_analyst_keys)
    analyst_wall_time_tracker = AnalystWallTimeTracker(analyst_execution_plan)

    message_buffer.init_for_analysis(selected_analyst_keys)

    # Track start time for elapsed display
    start_time = time.time()

    results_dir = _run_directory(config, selections["ticker"], selections["analysis_date"])
    results_dir.mkdir(parents=True, exist_ok=True)
    report_dir = results_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    log_file = results_dir / "message_tool.log"
    log_file.touch(exist_ok=True)

    def save_message_decorator(obj, func_name):
        func = getattr(obj, func_name)

        @wraps(func)
        def wrapper(*args, **kwargs):
            func(*args, **kwargs)
            timestamp, message_type, content = obj.messages[-1]
            content = content.replace("\n", " ")  # Replace newlines with spaces
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"{timestamp} [{message_type}] {content}\n")
        return wrapper

    def save_tool_call_decorator(obj, func_name):
        func = getattr(obj, func_name)

        @wraps(func)
        def wrapper(*args, **kwargs):
            func(*args, **kwargs)
            timestamp, tool_name, args = obj.tool_calls[-1]
            args_str = ", ".join(f"{k}={v}" for k, v in args.items())
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"{timestamp} [Tool Call] {tool_name}({args_str})\n")
        return wrapper

    def save_report_section_decorator(obj, func_name):
        func = getattr(obj, func_name)

        @wraps(func)
        def wrapper(section_name, content):
            func(section_name, content)
            if section_name in obj.report_sections and obj.report_sections[section_name] is not None:
                content = obj.report_sections[section_name]
                if content:
                    file_name = f"{section_name}.md"
                    text = "\n".join(str(item) for item in content) if isinstance(content, list) else content
                    with open(report_dir / file_name, "w", encoding="utf-8") as f:
                        f.write(text)
        return wrapper

    message_buffer.add_message = save_message_decorator(message_buffer, "add_message")
    message_buffer.add_tool_call = save_tool_call_decorator(message_buffer, "add_tool_call")
    message_buffer.update_report_section = save_report_section_decorator(message_buffer, "update_report_section")

    layout = create_layout()

    def handle_chunk(chunk: dict) -> None:
        for message in chunk.get("messages", []):
            msg_id = getattr(message, "id", None)
            if msg_id is not None:
                if msg_id in message_buffer._processed_message_ids:
                    continue
                message_buffer._processed_message_ids.add(msg_id)

            msg_type, content = classify_message_type(message)
            if content and content.strip():
                message_buffer.add_message(msg_type, content)

            if hasattr(message, "tool_calls") and message.tool_calls:
                for tool_call in message.tool_calls:
                    if isinstance(tool_call, dict):
                        message_buffer.add_tool_call(tool_call["name"], tool_call["args"])
                    else:
                        message_buffer.add_tool_call(tool_call.name, tool_call.args)

        update_analyst_statuses(
            message_buffer,
            chunk,
            wall_time_tracker=analyst_wall_time_tracker,
        )

        if chunk.get("investment_debate_state"):
            debate_state = chunk["investment_debate_state"]
            bull_hist = debate_state.get("bull_history", "").strip()
            bear_hist = debate_state.get("bear_history", "").strip()
            judge = debate_state.get("judge_decision", "").strip()

            if bull_hist or bear_hist:
                update_research_team_status("in_progress")
            if bull_hist:
                message_buffer.update_report_section(
                    "investment_plan", f"### Bull Researcher Analysis\n{bull_hist}"
                )
            if bear_hist:
                message_buffer.update_report_section(
                    "investment_plan", f"### Bear Researcher Analysis\n{bear_hist}"
                )
            if judge:
                message_buffer.update_report_section(
                    "investment_plan", f"### Research Manager Decision\n{judge}"
                )
                update_research_team_status("completed")
                message_buffer.update_agent_status("Trader", "in_progress")

        if chunk.get("trader_investment_plan"):
            message_buffer.update_report_section(
                "trader_investment_plan", chunk["trader_investment_plan"]
            )
            if message_buffer.agent_status.get("Trader") != "completed":
                message_buffer.update_agent_status("Trader", "completed")
                message_buffer.update_agent_status("Aggressive Analyst", "in_progress")

        if chunk.get("risk_debate_state"):
            risk_state = chunk["risk_debate_state"]
            agg_hist = risk_state.get("aggressive_history", "").strip()
            con_hist = risk_state.get("conservative_history", "").strip()
            neu_hist = risk_state.get("neutral_history", "").strip()
            judge = risk_state.get("judge_decision", "").strip()

            for history, agent in (
                (agg_hist, "Aggressive Analyst"),
                (con_hist, "Conservative Analyst"),
                (neu_hist, "Neutral Analyst"),
            ):
                if history:
                    if message_buffer.agent_status.get(agent) != "completed":
                        message_buffer.update_agent_status(agent, "in_progress")
                    message_buffer.update_report_section(
                        "final_trade_decision", f"### {agent} Analysis\n{history}"
                    )
            if judge and message_buffer.agent_status.get("Portfolio Manager") != "completed":
                message_buffer.update_agent_status("Portfolio Manager", "in_progress")
                message_buffer.update_report_section(
                    "final_trade_decision", f"### Portfolio Manager Decision\n{judge}"
                )
                for agent in (
                    "Aggressive Analyst",
                    "Conservative Analyst",
                    "Neutral Analyst",
                    "Portfolio Manager",
                ):
                    message_buffer.update_agent_status(agent, "completed")

        update_display(layout, stats_handler=stats_handler, start_time=start_time)

    # The alternate screen keeps a layout taller than the window from redrawing
    # by scrolling; the final report prints after this block, on the normal screen.
    with Live(layout, refresh_per_second=4, screen=True):
        # Initial display
        update_display(layout, stats_handler=stats_handler, start_time=start_time)

        message_buffer.add_message("System", f"Selected ticker: {selections['ticker']}")
        if selections["asset_type"] != "stock":
            message_buffer.add_message("System", f"Detected asset type: {selections['asset_type']}")
        message_buffer.add_message(
            "System", f"Analysis date: {selections['analysis_date']}"
        )
        message_buffer.add_message(
            "System",
            f"Selected analysts: {', '.join(analyst.value for analyst in selections['analysts'])}",
        )
        update_display(layout, stats_handler=stats_handler, start_time=start_time)

        first_analyst = analyst_execution_plan.specs[0].agent_node
        message_buffer.update_agent_status(first_analyst, "in_progress")
        analyst_wall_time_tracker.mark_started(selected_analyst_keys[0])
        update_display(layout, stats_handler=stats_handler, start_time=start_time)

        spinner_text = (
            f"Analyzing {selections['ticker']} on {selections['analysis_date']}..."
        )
        update_display(layout, spinner_text, stats_handler=stats_handler, start_time=start_time)

        execution = execute_analysis(
            ticker=selections["ticker"],
            analysis_date=selections["analysis_date"],
            selected_analysts=selected_analyst_keys,
            asset_type=selections["asset_type"],
            config=config,
            portfolio=portfolio,
            stats_handler=stats_handler,
            on_chunk=handle_chunk,
            on_checkpoint=lambda graph: _announce_checkpoint_state(
                graph, selections["ticker"], selections["analysis_date"]
            ),
        )
        final_state = execution.final_state

        for agent in message_buffer.agent_status:
            message_buffer.update_agent_status(agent, "completed")

        message_buffer.add_message(
            "System", f"Completed analysis for {selections['analysis_date']}"
        )
        message_buffer.add_message("System", analyst_wall_time_tracker.format_summary())

        for section in message_buffer.report_sections:
            if section in final_state:
                message_buffer.update_report_section(section, final_state[section])

        update_display(layout, stats_handler=stats_handler, start_time=start_time)

    # Post-analysis prompts (outside Live context for clean interaction)
    console.print("\n[bold cyan]Analysis Complete![/bold cyan]\n")

    # A decision nobody can read is not a position. Say so here rather than
    # leaving the run to look like a normal result.
    if is_review(execution.rating):
        console.print(
            "[yellow]No rating could be read from the final decision, so this run "
            "is recorded for review rather than as a position. Re-run, or read the "
            "decision text below and judge it yourself.[/yellow]\n"
        )
    console.print(f"[dim]{analyst_wall_time_tracker.format_summary()}[/dim]")

    # Prompt to save report
    save_choice = typer.prompt("Save report?", default="Y").strip().upper()
    if save_choice in ("Y", "YES", ""):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # Under results_dir, not the working directory: in Docker the working
        # directory is inside the container and the report goes with it, while
        # results_dir is the mounted volume the rest of the run already writes to.
        default_path = (Path(config["results_dir"]) / "reports"
                        / f"{safe_ticker_component(selections['ticker'])}_{timestamp}")
        save_path_str = typer.prompt(
            "Save path (press Enter for default)",
            default=str(default_path)
        ).strip()
        save_path = Path(save_path_str)
        try:
            report_file = write_report_tree(final_state, selections["ticker"], save_path)
            console.print(f"\n[green]✓ Report saved to:[/green] {save_path.resolve()}")
            console.print(f"  [dim]Complete report:[/dim] {report_file.name}")
        except Exception as e:
            console.print(f"[red]Error saving report: {e}[/red]")

    # Prompt to display full report
    display_choice = typer.prompt("\nDisplay full report on screen?", default="Y").strip().upper()
    if display_choice in ("Y", "YES", ""):
        display_complete_report(final_state)
