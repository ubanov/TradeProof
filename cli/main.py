import os
import sys
from datetime import datetime
from pathlib import Path

import typer

from cli.display import ANALYST_ORDER, console
from cli.headless import (
    format_duration,
    parse_analysts,
    parse_depth,
    parse_tickers,
    run_batch_analysis,
    run_single_analysis,
    timestamped_echo,
)
from cli.run import run_analysis
from tradingagents.backtest import iter_grid, run_backtest, summarize
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.portfolio import load_portfolio
from tradingagents.run_seed import resolve_base_seed, validate_configured_seed

# Batch defaults so only --tickers/--tickers-file is required: each is an
# explicit flag, then an env var, then a hardcoded fallback -- in that order.
_DEFAULT_ANALYSTS = ",".join(ANALYST_ORDER)
_DEFAULT_DEPTH = "medium"

# prompt_toolkit's win32 output module is importable only on Windows (it asserts
# the platform at import time), so gate on the platform rather than catching the
# failure — that way a genuinely broken prompt_toolkit on Windows still surfaces
# instead of silently disabling the handler below. Off Windows this stays an
# empty tuple, which `except` accepts and never matches (#1138).
if sys.platform == "win32":  # pragma: no cover - platform dependent
    from prompt_toolkit.output.win32 import NoConsoleScreenBufferError

    _NO_CONSOLE_ERRORS: tuple[type[BaseException], ...] = (NoConsoleScreenBufferError,)
else:
    _NO_CONSOLE_ERRORS = ()

app = typer.Typer(
    name="TradingAgents",
    help="TradingAgents CLI: Multi-Agents LLM Financial Trading Framework",
    add_completion=True,  # Enable shell completion
)


@app.callback(invoke_without_command=True)
def analyze(
    ctx: typer.Context,
    checkpoint: bool | None = typer.Option(
        None,
        "--checkpoint/--no-checkpoint",
        help="Enable/disable checkpoint-resume (save state after each node so a "
        "crashed run can resume). Omit to honor TRADINGAGENTS_CHECKPOINT_ENABLED.",
    ),
    clear_checkpoints: bool = typer.Option(
        False,
        "--clear-checkpoints",
        help="Delete all saved checkpoints before running (force fresh start).",
    ),
    portfolio: str = typer.Option(
        None,
        "--portfolio",
        help="JSON file with current holdings and cash, so the trader, risk and "
        "portfolio agents size against your actual position.",
    ),
):
    """Run an analysis. This is what a bare `tradingagents` does."""
    if ctx.invoked_subcommand is not None:
        return
    if clear_checkpoints:
        from tradingagents.graph.checkpointer import clear_all_checkpoints
        n = clear_all_checkpoints(DEFAULT_CONFIG["data_cache_dir"])
        console.print(f"[yellow]Cleared {n} checkpoint(s).[/yellow]")
    portfolio_context = None
    if portfolio:
        try:
            portfolio_context = load_portfolio(portfolio)
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1) from None

    try:
        run_analysis(checkpoint=checkpoint, portfolio=portfolio_context)
    except _NO_CONSOLE_ERRORS:
        # A terminal with no console buffer cannot host the interactive prompts.
        # Emit one actionable line on stderr instead of a prompt_toolkit
        # traceback; plain text, since rich may not render here either (#1138).
        typer.echo(
            "Error: no Windows console available. The interactive CLI needs a real "
            "console buffer — run it from Windows Terminal, PowerShell, or cmd.exe "
            "rather than a piped or embedded terminal.",
            err=True,
        )
        raise typer.Exit(code=1) from None


@app.command("run")
def run_non_interactive(
    ticker: str = typer.Option(..., "--ticker", help="Ticker symbol to analyze"),
    date: str = typer.Option(..., "--date", help="Analysis date, YYYY-MM-DD"),
    analysts: str = typer.Option(
        ..., "--analysts", help="Comma-separated analysts"
    ),
    depth: str = typer.Option(
        ..., "--depth", help="Research depth: shallow, medium, or deep"
    ),
    language: str = typer.Option(..., "--language", help="Output language"),
    output_dir: Path = typer.Option(  # noqa: B008 - Typer option declaration
        ..., "--output-dir", help="Directory for the report and result.json"
    ),
    generation_llm: str | None = typer.Option(
        None, "--TRADINGAGENTS_GENERATION_LLM", "--generation-llm",
        help="Override the generation-role model for this run only (see "
        "README: 'Model roles: Generation vs Review'). With no --review-llm "
        "override, the whole run uses this model (review inherits it too).",
    ),
    review_llm: str | None = typer.Option(
        None, "--TRADINGAGENTS_REVIEW_LLM", "--review-llm",
        help="Override the review-role model for this run only.",
    ),
    deep_think_llm: str | None = typer.Option(
        None, "--TRADINGAGENTS_DEEP_THINK_LLM", "--deep-think-llm",
        help="Deprecated alias for --review-llm (see README: 'Legacy "
        "naming'); an explicit --review-llm wins if both are given.",
    ),
    quick_think_llm: str | None = typer.Option(
        None, "--TRADINGAGENTS_QUICK_THINK_LLM", "--quick-think-llm",
        help="Deprecated alias for --generation-llm (see README: 'Legacy "
        "naming'); an explicit --generation-llm wins if both are given.",
    ),
    generation_reasoning_effort: str | None = typer.Option(
        None, "--TRADINGAGENTS_GENERATION_REASONING_EFFORT", "--generation-reasoning-effort",
        help="Reasoning effort for the generation-role model (see README: "
        "'Model roles: Generation vs Review'). Valid levels are model/backend-"
        "specific (e.g. low/medium/high, or false/low/high/max) -- not "
        "validated here. Unset sends no override, so the backend/model's own "
        "default applies.",
    ),
    review_reasoning_effort: str | None = typer.Option(
        None, "--TRADINGAGENTS_REVIEW_REASONING_EFFORT", "--review-reasoning-effort",
        help="Reasoning effort for the review-role model, independent of "
        "--generation-reasoning-effort. See README: 'Model roles: Generation "
        "vs Review'.",
    ),
    quick_reasoning_effort: str | None = typer.Option(
        None, "--TRADINGAGENTS_QUICK_REASONING_EFFORT", "--quick-reasoning-effort",
        help="Deprecated alias for --generation-reasoning-effort (see "
        "README: 'Legacy naming').",
    ),
    deep_reasoning_effort: str | None = typer.Option(
        None, "--TRADINGAGENTS_DEEP_REASONING_EFFORT", "--deep-reasoning-effort",
        help="Deprecated alias for --review-reasoning-effort (see README: "
        "'Legacy naming').",
    ),
    validator_llm: str | None = typer.Option(
        None, "--TRADINGAGENTS_VALIDATOR_LLM", "--validator-llm",
        help="Model for the Validator role (see README: 'Generation / "
        "Validation / Review') -- a lightweight contract/format/arithmetic "
        "check, not another analyst opinion. Required if --validation is on.",
    ),
    validator_reasoning_effort: str | None = typer.Option(
        None, "--TRADINGAGENTS_VALIDATOR_REASONING_EFFORT", "--validator-reasoning-effort",
        help="Reasoning effort for the Validator model, independent of "
        "generation/review. See README: 'Generation / Validation / Review'.",
    ),
    validation_enabled: bool | None = typer.Option(
        None, "--validation/--no-validation",
        help="Enable the optional Validator LLM pass (see README: "
        "'Generation / Validation / Review'). Off by default -- deterministic "
        "checks always run regardless; this only gates the extra LLM calls. "
        "Requires --validator-llm (or TRADINGAGENTS_VALIDATOR_LLM) when on; "
        "there is no silent fallback to the generation model.",
    ),
    trace_llm: bool = typer.Option(
        False, "--trace-llm",
        help="Save a complete, auditable trace of every LLM call (rendered "
        "prompts, raw/parsed responses, timing, tokens) under <output-dir>/debug/. "
        "Off by default; traces are not transmitted, only written locally.",
    ),
    seed: int | None = typer.Option(
        None, "--TRADINGAGENTS_SEED", "--seed",
        help="Best-effort deterministic replay base seed (see README). 0 "
        "(the default when omitted, unless TRADINGAGENTS_SEED is set) "
        "auto-generates a base seed and prints it; a non-zero value reuses "
        "it, deriving a stable per-call seed per ticker/stage/call -- never "
        "guaranteed bit-for-bit identical output.",
    ),
):
    """Run one analysis without interactive prompts."""
    echo = timestamped_echo()
    try:
        _, depth_value = parse_depth(depth)
        analyst_names = parse_analysts(analysts)
        configured_seed = validate_configured_seed(
            seed if seed is not None else DEFAULT_CONFIG.get("seed", 0)
        )
        base_seed = resolve_base_seed(configured_seed)
        echo(f"Run seed: {base_seed}")
        echo(f"Analysis date: {date}")
        result = run_single_analysis(
            ticker=ticker,
            analysis_date=date,
            analysts=analyst_names,
            depth_value=depth_value,
            language=language,
            output_dir=output_dir,
            progress=echo,
            progress_prefix=ticker,
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
        # A setup-time failure that never reached graph execution (bad config,
        # invalid ticker, ...). A graph-execution failure does not raise here
        # any more -- run_single_analysis already turned it into an "error"
        # result with its partial artifacts intact, handled below.
        echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from None

    echo(f"Result: {output_dir.resolve() / 'result.json'}")
    if result["status"] != "ok":
        stage = result.get("failure_stage") or "unknown"
        echo(f"FAILED at {stage} after {format_duration(result['run_duration_seconds'])}: {result['error']}", err=True)
        if result.get("report_path"):
            echo(f"Partial report: {output_dir.resolve() / result['report_path']}")
        stats = result.get("stats") or {}
        if stats:
            echo(
                f"LLM {stats.get('llm_calls', 0)}; tools {stats.get('tool_calls', 0)}; "
                f"tokens {stats.get('tokens_in', 0)} in/{stats.get('tokens_out', 0)} out"
            )
        raise typer.Exit(code=1)

    fallback_warnings = result.get("fallback_warnings", 0)
    suffix = f" (with {fallback_warnings} fallback warning(s))" if fallback_warnings else ""
    echo(f"Completed {result['ticker']}{suffix}")
    echo(f"Report: {output_dir.resolve() / 'complete_report.md'}")
    stats = result["stats"]
    research = result["research_execution"]
    echo(
        f"Summary: {format_duration(result['run_duration_seconds'])}; "
        f"LLM {stats['llm_calls']}; tools {stats['tool_calls']}; "
        f"tokens {stats['tokens_in']} in/{stats['tokens_out']} out; "
        f"verifier {research['verification_status']} "
        f"({research['verifier_passes']} pass(es), "
        f"{research['repair_rounds']} repair(s))"
    )
    validation = result.get("validation") or {}
    if validation.get("checked"):
        echo(
            f"Validation: stages checked: {validation['checked']} / "
            f"pass: {validation['pass']} / warn: {validation['warn']} / "
            f"fail: {validation['fail']} / repairs: {validation['repairs']} / "
            f"unresolved: {validation['unresolved']}"
        )


@app.command("batch")
def batch_non_interactive(
    tickers: str | None = typer.Option(
        None, "--tickers", help="Comma-separated ticker symbols"
    ),
    tickers_file: Path | None = typer.Option(  # noqa: B008 - Typer option declaration
        None, "--tickers-file", help="UTF-8 file with one ticker per line"
    ),
    date: str | None = typer.Option(
        None, "--date",
        help="Analysis date, YYYY-MM-DD. Defaults to today (fixed at startup, so a "
        "run crossing midnight stays on the day it started).",
    ),
    analysts: str | None = typer.Option(
        None, "--analysts",
        help="Comma-separated analysts. Defaults to $TRADINGAGENTS_ANALYSTS, or all four.",
    ),
    depth: str | None = typer.Option(
        None, "--depth",
        help="Research depth: shallow, medium, or deep. Defaults to $TRADINGAGENTS_DEPTH, "
        "or medium.",
    ),
    language: str | None = typer.Option(
        None, "--language",
        help="Output language. Defaults to $TRADINGAGENTS_OUTPUT_LANGUAGE, or English.",
    ),
    output_dir: Path | None = typer.Option(  # noqa: B008 - Typer option declaration
        None, "--output-dir", help=r"Batch output directory. Defaults to .\test\<date>."
    ),
    generation_llm: str | None = typer.Option(
        None, "--TRADINGAGENTS_GENERATION_LLM", "--generation-llm",
        help="Override the generation-role model for this batch only (see "
        "README: 'Model roles: Generation vs Review'). With no --review-llm "
        "override, the whole batch uses this model (review inherits it too).",
    ),
    review_llm: str | None = typer.Option(
        None, "--TRADINGAGENTS_REVIEW_LLM", "--review-llm",
        help="Override the review-role model for this batch only.",
    ),
    deep_think_llm: str | None = typer.Option(
        None, "--TRADINGAGENTS_DEEP_THINK_LLM", "--deep-think-llm",
        help="Deprecated alias for --review-llm (see README: 'Legacy "
        "naming'); an explicit --review-llm wins if both are given.",
    ),
    quick_think_llm: str | None = typer.Option(
        None, "--TRADINGAGENTS_QUICK_THINK_LLM", "--quick-think-llm",
        help="Deprecated alias for --generation-llm (see README: 'Legacy "
        "naming'); an explicit --generation-llm wins if both are given.",
    ),
    generation_reasoning_effort: str | None = typer.Option(
        None, "--TRADINGAGENTS_GENERATION_REASONING_EFFORT", "--generation-reasoning-effort",
        help="Reasoning effort for the generation-role model (see README: "
        "'Model roles: Generation vs Review'). Valid levels are model/backend-"
        "specific -- not validated here. Unset sends no override, so the "
        "backend/model's own default applies.",
    ),
    review_reasoning_effort: str | None = typer.Option(
        None, "--TRADINGAGENTS_REVIEW_REASONING_EFFORT", "--review-reasoning-effort",
        help="Reasoning effort for the review-role model, independent of "
        "--generation-reasoning-effort. See README: 'Model roles: Generation "
        "vs Review'.",
    ),
    quick_reasoning_effort: str | None = typer.Option(
        None, "--TRADINGAGENTS_QUICK_REASONING_EFFORT", "--quick-reasoning-effort",
        help="Deprecated alias for --generation-reasoning-effort (see "
        "README: 'Legacy naming').",
    ),
    deep_reasoning_effort: str | None = typer.Option(
        None, "--TRADINGAGENTS_DEEP_REASONING_EFFORT", "--deep-reasoning-effort",
        help="Deprecated alias for --review-reasoning-effort (see README: "
        "'Legacy naming').",
    ),
    validator_llm: str | None = typer.Option(
        None, "--TRADINGAGENTS_VALIDATOR_LLM", "--validator-llm",
        help="Model for the Validator role (see README: 'Generation / "
        "Validation / Review') -- a lightweight contract/format/arithmetic "
        "check, not another analyst opinion. Required if --validation is on.",
    ),
    validator_reasoning_effort: str | None = typer.Option(
        None, "--TRADINGAGENTS_VALIDATOR_REASONING_EFFORT", "--validator-reasoning-effort",
        help="Reasoning effort for the Validator model, independent of "
        "generation/review. See README: 'Generation / Validation / Review'.",
    ),
    validation_enabled: bool | None = typer.Option(
        None, "--validation/--no-validation",
        help="Enable the optional Validator LLM pass for this batch (see "
        "README: 'Generation / Validation / Review'). Off by default -- "
        "deterministic checks always run regardless; this only gates the "
        "extra LLM calls. Requires --validator-llm (or "
        "TRADINGAGENTS_VALIDATOR_LLM) when on; there is no silent fallback "
        "to the generation model.",
    ),
    trace_llm: bool = typer.Option(
        False, "--trace-llm",
        help="Save a complete, auditable trace of every LLM call (rendered "
        "prompts, raw/parsed responses, timing, tokens) under each ticker's "
        "debug/ directory. Off by default.",
    ),
    seed: int | None = typer.Option(
        None, "--TRADINGAGENTS_SEED", "--seed",
        help="Best-effort deterministic replay base seed (see README). 0 "
        "(the default when omitted, unless TRADINGAGENTS_SEED is set) "
        "auto-generates ONE base seed for the whole batch and prints it; a "
        "non-zero value reuses it. Every ticker derives its own stable "
        "per-stage/per-call seeds from the same base seed -- never "
        "guaranteed bit-for-bit identical output.",
    ),
):
    """Run multiple analyses sequentially without interactive prompts.

    Only --tickers (or --tickers-file) is required; every other option falls
    back to an environment variable and then a hardcoded default.
    """
    echo = timestamped_echo()
    date = date or datetime.now().strftime("%Y-%m-%d")
    analysts = analysts or os.environ.get("TRADINGAGENTS_ANALYSTS") or _DEFAULT_ANALYSTS
    depth = depth or os.environ.get("TRADINGAGENTS_DEPTH") or _DEFAULT_DEPTH
    language = (
        language
        or os.environ.get("TRADINGAGENTS_OUTPUT_LANGUAGE")
        or DEFAULT_CONFIG["output_language"]
    )
    output_dir = output_dir or Path("test") / date
    try:
        depth_name, depth_value = parse_depth(depth)
        analyst_names = parse_analysts(analysts)
        ticker_names = parse_tickers(tickers, tickers_file)
        configured_seed = validate_configured_seed(
            seed if seed is not None else DEFAULT_CONFIG.get("seed", 0)
        )
    except (OSError, ValueError) as exc:
        echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from None

    # ONE base seed for the whole batch, resolved here, before any ticker
    # runs -- never regenerated per ticker (see run_seed.resolve_base_seed).
    base_seed = resolve_base_seed(configured_seed)
    echo(f"Run seed: {base_seed}")
    echo(f"Analysis date: {date}")
    echo(f"Output directory: {output_dir.resolve()}")
    try:
        results, summary_path, _ = run_batch_analysis(
            tickers=ticker_names,
            analysis_date=date,
            analysts=analyst_names,
            depth_name=depth_name,
            depth_value=depth_value,
            language=language,
            output_dir=output_dir,
            progress=echo,
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
        echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from None

    successful = sum(result["status"] == "ok" for result in results)
    failed = len(results) - successful
    echo("Batch completed")
    echo(f"Successful: {successful}")
    echo(f"Failed: {failed}")
    echo(f"Summary: {summary_path}")
    successful_results = [result for result in results if result["status"] == "ok"]
    total_stats = {
        key: sum(result["stats"][key] for result in successful_results)
        for key in ("llm_calls", "tool_calls", "tokens_in", "tokens_out")
    }
    total_research = {
        key: sum(result["research_execution"][key] for result in successful_results)
        for key in (
            "bull_initial_calls",
            "bear_initial_calls",
            "bull_review_calls",
            "bear_review_calls",
            "verifier_passes",
            "repair_rounds",
        )
    }
    echo(
        "Execution: "
        f"elapsed {format_duration(sum(result['run_duration_seconds'] for result in results))}; "
        f"LLM {total_stats['llm_calls']}; tools {total_stats['tool_calls']}; "
        f"tokens {total_stats['tokens_in']} in/{total_stats['tokens_out']} out"
    )
    echo(
        "Research calls: "
        f"Bull initial {total_research['bull_initial_calls']}; "
        f"Bear initial {total_research['bear_initial_calls']}; "
        f"Bull reviews {total_research['bull_review_calls']}; "
        f"Bear reviews {total_research['bear_review_calls']}; "
        f"Verifier {total_research['verifier_passes']}; "
        f"repairs {total_research['repair_rounds']}"
    )
    if failed:
        raise typer.Exit(code=1)


@app.command()
def backtest(
    tickers: str = typer.Argument(..., help="Comma-separated tickers, e.g. NVDA,AAPL"),
    start: str = typer.Option(..., "--start", help="First analysis date, YYYY-MM-DD"),
    end: str = typer.Option(..., "--end", help="Last analysis date, YYYY-MM-DD"),
    every: int = typer.Option(7, "--every", help="Days between analysis dates"),
    analysts: str = typer.Option(
        None, "--analysts", help="Comma-separated analysts to run; omit for all four"
    ),
    asset_type: str = typer.Option("stock", "--asset-type", help="stock or crypto"),
    portfolio: str = typer.Option(
        None, "--portfolio", help="JSON file with holdings and cash, held constant across the grid"
    ),
    run_id: str = typer.Option(
        None, "--run-id", help="Continue an earlier sweep: its cells are skipped and its log reused"
    ),
):
    """Score past decisions over a grid of tickers and dates."""

    try:
        dates = iter_grid(start, end, every)
        book = load_portfolio(portfolio) if portfolio else None
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None

    names = [t.strip() for t in tickers.split(",") if t.strip()]
    if not names:
        console.print("[red]No ticker to analyze; pass them comma-separated, e.g. NVDA,AAPL[/red]")
        raise typer.Exit(code=1)

    kwargs = {"asset_type": asset_type, "portfolio": book, "run_id": run_id}
    if analysts:
        kwargs["selected_analysts"] = [a.strip().lower() for a in analysts.split(",") if a.strip()]

    try:
        result = run_backtest(names, dates, DEFAULT_CONFIG, **kwargs)
    except Exception as exc:  # a missing key or an unknown analyst is a setup error
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None
    console.print(summarize(result).render())
    console.print(f"\nRan {result.cells_run} cells, skipped {result.skipped}. Log: {result.log_path}")
    for ticker, date, reason in result.failures:
        console.print(f"[yellow]failed:[/yellow] {ticker} {date}: {reason}")
    for ticker, reason in result.settlement_failures:
        console.print(f"[yellow]unsettled:[/yellow] {ticker}: {reason}")


if __name__ == "__main__":
    app()
