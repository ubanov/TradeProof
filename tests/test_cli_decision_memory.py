"""`tradingagents run`/`batch --no-decision-memory` / `--decision-memory-root`
(see fork README: "Decision memory isolation").

Mirrors the existing CLI-flag-threading test style (test_cli_run_defaults.py/
test_cli_seed.py): `run_single_analysis`/`run_batch_analysis` are mocked to
capture exactly what cli/main.py passes down, so these tests exercise Typer
option parsing and precedence without running a real analysis.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

import cli.main as m

RUN_RESULT = {
    "status": "ok",
    "ticker": "LLY",
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


@pytest.fixture
def mocked_run(monkeypatch):
    calls = []
    monkeypatch.setattr(m, "run_single_analysis", lambda **kw: calls.append(kw) or RUN_RESULT)
    return calls


@pytest.fixture
def mocked_batch(monkeypatch, tmp_path):
    calls = []

    def _fake_batch(**kwargs):
        calls.append(kwargs)
        return [], tmp_path / "summary.md", tmp_path / "summary.json"

    monkeypatch.setattr(m, "run_batch_analysis", _fake_batch)
    return calls


# --- `run`: defaults are unchanged -------------------------------------------


@pytest.mark.unit
def test_run_defaults_to_normal_memory(runner, mocked_run):
    result = runner.invoke(m.app, ["run", "--ticker", "LLY"])
    assert result.exit_code == 0, result.output
    assert mocked_run[0]["no_decision_memory"] is False
    assert mocked_run[0]["decision_memory_root"] is None


@pytest.mark.unit
def test_run_no_decision_memory_flag_threads_through(runner, mocked_run):
    result = runner.invoke(m.app, ["run", "--ticker", "LLY", "--no-decision-memory"])
    assert result.exit_code == 0, result.output
    assert mocked_run[0]["no_decision_memory"] is True
    assert mocked_run[0]["decision_memory_root"] is None


@pytest.mark.unit
def test_run_decision_memory_root_flag_threads_through(runner, mocked_run, tmp_path):
    root = str(tmp_path / "experiment-001" / "memory")
    result = runner.invoke(m.app, ["run", "--ticker", "LLY", "--decision-memory-root", root])
    assert result.exit_code == 0, result.output
    assert mocked_run[0]["no_decision_memory"] is False
    assert mocked_run[0]["decision_memory_root"] == root


@pytest.mark.unit
def test_run_env_style_flag_name_also_works(runner, mocked_run, tmp_path):
    root = str(tmp_path / "experiment-001")
    result = runner.invoke(
        m.app, ["run", "--ticker", "LLY", "--TRADINGAGENTS_DECISION_MEMORY_ROOT", root]
    )
    assert result.exit_code == 0, result.output
    assert mocked_run[0]["decision_memory_root"] == root


# --- `batch`: same precedence, threaded for the whole batch -----------------


@pytest.mark.unit
def test_batch_defaults_to_normal_memory(runner, mocked_batch):
    result = runner.invoke(m.app, ["batch", "--tickers", "NVDA"])
    assert result.exit_code == 0, result.output
    assert mocked_batch[0]["no_decision_memory"] is False
    assert mocked_batch[0]["decision_memory_root"] is None


@pytest.mark.unit
def test_batch_no_decision_memory_flag_threads_through(runner, mocked_batch):
    result = runner.invoke(m.app, ["batch", "--tickers", "NVDA", "--no-decision-memory"])
    assert result.exit_code == 0, result.output
    assert mocked_batch[0]["no_decision_memory"] is True


@pytest.mark.unit
def test_batch_decision_memory_root_flag_threads_through(runner, mocked_batch, tmp_path):
    root = str(tmp_path / "experiment-002")
    result = runner.invoke(
        m.app, ["batch", "--tickers", "NVDA", "--decision-memory-root", root]
    )
    assert result.exit_code == 0, result.output
    assert mocked_batch[0]["decision_memory_root"] == root


# --- Incompatible combination fails immediately, with a clear message ------


@pytest.mark.unit
def test_run_rejects_both_flags_at_once(runner, tmp_path):
    """Not mocked: exercises the real build_headless_config validation, so
    this is the actual CLI behavior a user hitting both flags would see."""
    root = str(tmp_path / "experiment-001")
    result = runner.invoke(
        m.app,
        ["run", "--ticker", "LLY", "--no-decision-memory", "--decision-memory-root", root],
    )
    assert result.exit_code == 1
    assert "--no-decision-memory" in result.output
    assert "--decision-memory-root" in result.output
    assert "incompatible" in result.output.lower()


@pytest.mark.unit
def test_batch_rejects_both_flags_at_once(runner, tmp_path):
    """Not mocked: run_batch_analysis's own upfront build_headless_config
    call (see cli/headless.py) must catch this before any ticker runs."""
    root = str(tmp_path / "experiment-001")
    result = runner.invoke(
        m.app,
        ["batch", "--tickers", "NVDA", "--no-decision-memory", "--decision-memory-root", root],
    )
    assert result.exit_code == 1
    assert "incompatible" in result.output.lower()
