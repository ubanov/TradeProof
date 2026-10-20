"""run2's CLI registration and structural independence from `run` (fork
README: "run2"). See test_run2_pipeline.py for the behavioural pipeline
tests and test_pm_replay.py for the ``_imported_modules`` AST pattern this
file's import-boundary test follows.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from typer.testing import CliRunner

import cli.main as m

RUN2_PACKAGE_DIR = Path(__file__).resolve().parent.parent / "tradingagents" / "run2"

# Modules run2 must never import, directly or transitively through its own
# files (a transitive check would also need to walk every module run2 DOES
# import; this file checks run2's own source only, which is where a
# violation of the fork spec's "no reutilices... Bull/Bear -> Manager ->
# Trader -> Risk -> PM" constraint would actually be introduced).
_FORBIDDEN_MODULE_PREFIXES = (
    "tradingagents.graph",
    "tradingagents.agents.researchers",
    "tradingagents.agents.risk_mgmt",
    "tradingagents.agents.managers",
    "tradingagents.agents.trader",
    "tradingagents.agents.validator",
    "tradingagents.decision_log",
    "tradingagents.dataflows.vendors.yahoo.news",
    "tradingagents.dataflows.vendors.alpha_vantage.news",
    "tradingagents.dataflows.vendors.reddit",
    "tradingagents.dataflows.vendors.stocktwits",
    "tradingagents.recommendation_record",
    "tradingagents.pre_pm_snapshot",
    "tradingagents.pm_replay",
    "cli.run",
    "cli.main",
    "cli.headless",
)


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {
        alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names
    }
    names |= {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module
    }
    return names


@pytest.mark.unit
def test_1_run2_command_is_registered_and_independent():
    result = CliRunner().invoke(m.app, ["run2", "--help"])
    assert result.exit_code == 0, result.output
    assert "--ticker" in result.output
    assert "--date" in result.output
    assert "--generation-llm" in result.output
    assert "--validator-llm" in result.output
    assert "--seed" in result.output
    assert "--trace-llm" in result.output
    assert "--horizon-sessions" in result.output
    assert "--barrier-pct" in result.output
    # run2 explicitly has no --analysts flag (V1 is market-only by design).
    assert "--analysts" not in result.output
    # No --objective: run2 only ever implements the First Barrier
    # experiment, there is no second mode to select (fork spec: "run2"
    # simplification, "NO quiero anadir --objective").
    assert "--objective" not in result.output
    # Challenger and its confidence thresholds are gone entirely.
    assert "--review-llm" not in result.output
    assert "--min-confidence" not in result.output
    assert "--weaken-min-confidence" not in result.output


@pytest.mark.unit
def test_2_existing_commands_are_all_still_registered_alongside_run2():
    result = CliRunner().invoke(m.app, ["--help"])
    assert result.exit_code == 0, result.output
    for name in ("run2", "batch", "backtest", "archive-evidence", "replay-pm", "benchmark-models"):
        assert name in result.output, f"{name!r} missing from top-level CLI help"


@pytest.mark.unit
def test_3_run_single_analysis_entry_point_is_untouched_by_run2():
    """Structural: run2's own module never imports run_single_analysis /
    TradingAgentsGraph / execute_analysis -- it has its own, separate
    pipeline entry point (tradingagents.run2.pipeline.run_run2_analysis)."""
    import tradingagents.run2.pipeline as run2_pipeline

    assert not hasattr(run2_pipeline, "run_single_analysis")
    assert not hasattr(run2_pipeline, "TradingAgentsGraph")
    assert not hasattr(run2_pipeline, "execute_analysis")


@pytest.mark.unit
def test_4_run2_source_never_imports_the_main_pipeline_or_news_social_decision_memory():
    run2_files = sorted(RUN2_PACKAGE_DIR.glob("*.py"))
    assert run2_files, "expected tradingagents/run2/*.py to exist"
    for path in run2_files:
        imported = _imported_modules(path)
        violations = {
            name for name in imported
            if any(name == prefix or name.startswith(prefix + ".") for prefix in _FORBIDDEN_MODULE_PREFIXES)
        }
        assert not violations, f"{path.name} imports forbidden module(s): {violations}"


@pytest.mark.unit
def test_4b_run2_inference_modules_never_import_the_offline_evaluator():
    """evaluator.py is the ONE place in run2 allowed to look at data after
    analysis_date (fork spec: "run2" First Barrier, "La inferencia de run2
    jamas debe importar ni llamar al evaluador futuro"). Every LIVE
    INFERENCE module must never import it. ``quant_benchmark.py`` is
    deliberately excluded here: it is itself an offline, after-the-fact
    utility (same category as evaluator.py, never called from a live
    analysis run) that explicitly reuses ``evaluate_first_barrier`` by
    design (fork spec: "run2" quant benchmark, "Reutilizar...
    evaluate_first_barrier")."""
    inference_files = [
        p for p in RUN2_PACKAGE_DIR.glob("*.py")
        if p.name not in ("evaluator.py", "quant_benchmark.py")
    ]
    assert inference_files, "expected run2 inference modules to exist"
    for path in inference_files:
        imported = _imported_modules(path)
        violations = {name for name in imported if name == "tradingagents.run2.evaluator" or name.endswith(".evaluator")}
        assert not violations, f"{path.name} imports the offline evaluator: {violations}"


@pytest.mark.unit
def test_5_run2_cli_error_path_does_not_crash_the_app(monkeypatch):
    """An unresolvable model (no --generation-llm and no configured default)
    must surface as a clean CLI error, not a traceback -- same contract as
    replay-pm's own error handling."""
    import tradingagents.run2.pipeline as run2_pipeline

    monkeypatch.setattr(run2_pipeline, "DEFAULT_CONFIG", {})
    result = CliRunner().invoke(m.app, ["run2", "--ticker", "NVDA", "--date", "2026-01-07"])
    assert result.exit_code == 1
    assert "Error" in result.output
