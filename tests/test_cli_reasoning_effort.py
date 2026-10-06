"""CLI precedence for --quick-reasoning-effort / --deep-reasoning-effort on
`run`/`batch` (see fork README: "Quick vs Deep reasoning effort"). Same
precedence convention as every other non-interactive flag in this project:
an explicit CLI flag wins, otherwise TRADINGAGENTS_QUICK/DEEP_REASONING_EFFORT
(already overlaid onto DEFAULT_CONFIG at import time) stands, otherwise None
(no override sent)."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

import cli.headless as headless
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


@pytest.mark.unit
def test_run_default_sends_no_reasoning_effort_override(runner, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(m, "run_single_analysis", lambda **kw: calls.append(kw) or RUN_RESULT)

    result = runner.invoke(m.app, [
        "run", "--ticker", "NVDA", "--date", "2026-09-30", "--analysts", "market",
        "--depth", "shallow", "--language", "English", "--output-dir", str(tmp_path),
    ])
    assert result.exit_code == 0, result.output
    assert calls[0]["quick_reasoning_effort"] is None
    assert calls[0]["deep_reasoning_effort"] is None


@pytest.mark.unit
def test_run_explicit_flags_set_independent_values(runner, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(m, "run_single_analysis", lambda **kw: calls.append(kw) or RUN_RESULT)

    result = runner.invoke(m.app, [
        "run", "--ticker", "NVDA", "--date", "2026-09-30", "--analysts", "market",
        "--depth", "shallow", "--language", "English", "--output-dir", str(tmp_path),
        "--quick-reasoning-effort", "low", "--deep-reasoning-effort", "max",
    ])
    assert result.exit_code == 0, result.output
    assert calls[0]["quick_reasoning_effort"] == "low"
    assert calls[0]["deep_reasoning_effort"] == "max"


@pytest.mark.unit
def test_run_passes_none_through_when_flag_omitted(runner, monkeypatch, tmp_path):
    """The CLI flag itself defaults to None when omitted; the actual CLI>env
    precedence resolution happens one layer down in build_headless_config
    (see test_headless_config_falls_back_to_default_config_when_no_cli_value
    below) -- this just pins that run_non_interactive forwards the raw flag
    value unchanged, not an env-resolved one, so that layering stays intact."""
    calls = []
    monkeypatch.setattr(m, "run_single_analysis", lambda **kw: calls.append(kw) or RUN_RESULT)

    result = runner.invoke(m.app, [
        "run", "--ticker", "NVDA", "--date", "2026-09-30", "--analysts", "market",
        "--depth", "shallow", "--language", "English", "--output-dir", str(tmp_path),
    ])
    assert result.exit_code == 0, result.output
    assert calls[0]["quick_reasoning_effort"] is None
    assert calls[0]["deep_reasoning_effort"] is None


@pytest.mark.unit
def test_batch_explicit_flags_set_independent_values(runner, monkeypatch, tmp_path):
    calls = []

    def fake_batch(**kw):
        calls.append(kw)
        return [RUN_RESULT], tmp_path / "summary.md", tmp_path / "summary.json"

    monkeypatch.setattr(m, "run_batch_analysis", fake_batch)
    result = runner.invoke(m.app, [
        "batch", "--tickers", "NVDA", "--date", "2026-09-30", "--output-dir", str(tmp_path),
        "--quick-reasoning-effort", "low", "--deep-reasoning-effort", "max",
    ])
    assert result.exit_code == 0, result.output
    assert calls[0]["quick_reasoning_effort"] == "low"
    assert calls[0]["deep_reasoning_effort"] == "max"


# --- resolution happens one layer down, in build_headless_config -----------


@pytest.mark.unit
def test_headless_config_explicit_cli_value_wins_over_env(monkeypatch):
    # Canonical keys, as DEFAULT_CONFIG actually holds them once
    # resolve_role_config has normalized it (which it always has by the time
    # any real caller sees it) -- see fork README: "Legacy naming".
    patched = dict(
        headless.DEFAULT_CONFIG,
        generation_reasoning_effort="medium", review_reasoning_effort="xhigh",
    )
    monkeypatch.setattr(headless, "DEFAULT_CONFIG", patched)
    config = headless.build_headless_config(
        1, "English", Path("."),
        cli_quick_reasoning_effort="low", cli_deep_reasoning_effort="max",
    )
    assert config["quick_reasoning_effort"] == "low"
    assert config["deep_reasoning_effort"] == "max"
    assert config["generation_reasoning_effort"] == "low"
    assert config["review_reasoning_effort"] == "max"


@pytest.mark.unit
def test_headless_config_falls_back_to_default_config_when_no_cli_value(monkeypatch):
    patched = dict(
        headless.DEFAULT_CONFIG,
        generation_reasoning_effort="medium", review_reasoning_effort="xhigh",
    )
    monkeypatch.setattr(headless, "DEFAULT_CONFIG", patched)
    config = headless.build_headless_config(1, "English", Path("."))
    assert config["quick_reasoning_effort"] == "medium"
    assert config["deep_reasoning_effort"] == "xhigh"
    assert config["generation_reasoning_effort"] == "medium"
    assert config["review_reasoning_effort"] == "xhigh"
