"""`tradingagents run --depth` defaults to "medium" (TradeProof prioritizes
quality over speed; shallow stays available explicitly for smoke tests),
matching the precedent `batch --depth` already set -- see fork README:
"Non-interactive and batch CLI execution". --depth is no longer a required
flag for `run`, and run_provenance.execution.depth reflects whatever value
was actually used, including the new default.
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

import cli.main as m

RUN_RESULT = {
    "status": "ok",
    "ticker": "NVDA",
    "run_duration_seconds": 1.0,
    "fallback_warnings": 0,
    "stats": {"llm_calls": 1, "tool_calls": 0, "tokens_in": 1, "tokens_out": 1},
    "research_execution": {
        "verification_status": "PASS", "verifier_passes": 1, "repair_rounds": 0,
        "bull_initial_calls": 1, "bear_initial_calls": 1,
        "bull_review_calls": 0, "bear_review_calls": 0,
    },
}


@pytest.fixture
def runner():
    return CliRunner()


# --- depth defaults to medium when --depth is omitted ----------------------


@pytest.mark.unit
def test_run_without_depth_flag_uses_medium(runner, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(m, "run_single_analysis", lambda **kw: calls.append(kw) or RUN_RESULT)

    result = runner.invoke(m.app, [
        "run", "--ticker", "NVDA", "--date", "2026-09-30", "--analysts", "market",
        "--language", "English", "--output-dir", str(tmp_path),
    ])
    assert result.exit_code == 0, result.output
    assert calls[0]["depth_value"] == 3  # medium


@pytest.mark.unit
def test_run_explicit_depth_shallow_still_overrides_the_default(runner, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(m, "run_single_analysis", lambda **kw: calls.append(kw) or RUN_RESULT)

    result = runner.invoke(m.app, [
        "run", "--ticker", "NVDA", "--date", "2026-09-30", "--analysts", "market",
        "--depth", "shallow", "--language", "English", "--output-dir", str(tmp_path),
    ])
    assert result.exit_code == 0, result.output
    assert calls[0]["depth_value"] == 1


@pytest.mark.unit
def test_run_explicit_depth_deep_still_works(runner, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(m, "run_single_analysis", lambda **kw: calls.append(kw) or RUN_RESULT)

    result = runner.invoke(m.app, [
        "run", "--ticker", "NVDA", "--date", "2026-09-30", "--analysts", "market",
        "--depth", "deep", "--language", "English", "--output-dir", str(tmp_path),
    ])
    assert result.exit_code == 0, result.output
    assert calls[0]["depth_value"] == 5


# --- run_provenance reflects the effective depth, including the default ---


@pytest.mark.unit
def test_run_provenance_reflects_medium_when_depth_flag_is_omitted(
    runner, monkeypatch, tmp_path
):
    from types import SimpleNamespace

    import cli.run as cli_run
    from tests.test_llm_trace_integration import _FakeTradingAgentsGraph

    monkeypatch.setattr(cli_run, "TradingAgentsGraph", _FakeTradingAgentsGraph)
    monkeypatch.setattr(
        cli_run, "build_analyst_execution_plan", lambda analysts: SimpleNamespace(specs=[])
    )

    result = runner.invoke(m.app, [
        "run", "--ticker", "NVDA", "--date", "2026-09-30", "--analysts", "market",
        "--language", "English", "--output-dir", str(tmp_path),
    ])
    assert result.exit_code == 0, result.output
    saved = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert saved["run_provenance"]["execution"]["depth"] == 3


# --- --help no longer marks --depth as required, and shows the default ----


@pytest.mark.unit
def test_help_no_longer_marks_depth_as_required_and_shows_default(runner):
    result = runner.invoke(m.app, ["run", "--help"])
    assert result.exit_code == 0, result.output
    # Rich's markup parser treats a literal "[default: ...]" in help text as
    # a style tag and silently drops it -- unlike typer's OWN auto-rendered
    # default tag, a hand-written one must avoid square brackets. The
    # default is communicated in prose instead (see cli/main.py).
    assert "or medium" in result.output
    depth_line = next(line for line in result.output.splitlines() if "--depth" in line)
    assert "*" not in depth_line.split("--depth")[0]
