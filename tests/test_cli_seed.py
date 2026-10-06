"""CLI seed precedence for `run`/`batch` (best-effort deterministic replay,
see tradingagents/run_seed.py and fork README). Same precedence convention as
every other non-interactive flag in this project: an explicit CLI flag wins,
otherwise the flag's own default falls through to DEFAULT_CONFIG (which
TRADINGAGENTS_SEED already overlays at import time, see
tests/test_env_overrides.py), otherwise 0 (auto-generate)."""

from __future__ import annotations

from unittest import mock

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


# --- 1, 3, 14: default is 0, explicit --seed is reused exactly -------------


@pytest.mark.unit
def test_run_default_seed_is_zero_and_generates_a_base_seed(runner, monkeypatch, tmp_path):
    monkeypatch.delenv("TRADINGAGENTS_SEED", raising=False)
    calls = []
    monkeypatch.setattr(m, "run_single_analysis", lambda **kw: calls.append(kw) or RUN_RESULT)

    result = runner.invoke(m.app, [
        "run", "--ticker", "NVDA", "--date", "2026-09-30", "--analysts", "market",
        "--depth", "shallow", "--language", "English", "--output-dir", str(tmp_path),
    ])
    assert result.exit_code == 0, result.output
    assert calls[0]["configured_seed"] == 0
    assert calls[0]["base_seed"] != 0
    assert isinstance(calls[0]["base_seed"], int)
    assert "Run seed:" in result.output


@pytest.mark.unit
def test_run_explicit_seed_is_reused_exactly(runner, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(m, "run_single_analysis", lambda **kw: calls.append(kw) or RUN_RESULT)

    result = runner.invoke(m.app, [
        "run", "--ticker", "NVDA", "--date", "2026-09-30", "--analysts", "market",
        "--depth", "shallow", "--language", "English", "--output-dir", str(tmp_path),
        "--seed", "184736291",
    ])
    assert result.exit_code == 0, result.output
    assert calls[0]["configured_seed"] == 184736291
    assert calls[0]["base_seed"] == 184736291
    assert "Run seed: 184736291" in result.output


@pytest.mark.unit
def test_run_rejects_a_negative_seed(runner, tmp_path):
    result = runner.invoke(m.app, [
        "run", "--ticker", "NVDA", "--date", "2026-09-30", "--analysts", "market",
        "--depth", "shallow", "--language", "English", "--output-dir", str(tmp_path),
        "--seed", "-5",
    ])
    assert result.exit_code == 1
    assert "Error" in result.output


# --- 14: CLI explicit > env/config > default 0 -------------------------------


@pytest.mark.unit
def test_run_env_seed_is_used_when_flag_omitted(runner, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(m, "run_single_analysis", lambda **kw: calls.append(kw) or RUN_RESULT)
    patched = dict(m.DEFAULT_CONFIG, seed=777)
    with mock.patch.object(m, "DEFAULT_CONFIG", patched):
        result = runner.invoke(m.app, [
            "run", "--ticker", "NVDA", "--date", "2026-09-30", "--analysts", "market",
            "--depth", "shallow", "--language", "English", "--output-dir", str(tmp_path),
        ])
    assert result.exit_code == 0, result.output
    assert calls[0]["configured_seed"] == 777
    assert calls[0]["base_seed"] == 777


@pytest.mark.unit
def test_run_explicit_flag_wins_over_env_config(runner, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(m, "run_single_analysis", lambda **kw: calls.append(kw) or RUN_RESULT)
    patched = dict(m.DEFAULT_CONFIG, seed=777)
    with mock.patch.object(m, "DEFAULT_CONFIG", patched):
        result = runner.invoke(m.app, [
            "run", "--ticker", "NVDA", "--date", "2026-09-30", "--analysts", "market",
            "--depth", "shallow", "--language", "English", "--output-dir", str(tmp_path),
            "--seed", "42",
        ])
    assert result.exit_code == 0, result.output
    assert calls[0]["configured_seed"] == 42
    assert calls[0]["base_seed"] == 42


# --- batch: one base seed for the whole run, printed exactly once ----------


@pytest.mark.unit
def test_batch_prints_run_seed_exactly_once_and_reuses_it(runner, monkeypatch, tmp_path):
    calls = []

    def fake_batch(**kw):
        calls.append(kw)
        return [RUN_RESULT], tmp_path / "summary.md", tmp_path / "summary.json"

    monkeypatch.setattr(m, "run_batch_analysis", fake_batch)
    result = runner.invoke(m.app, [
        "batch", "--tickers", "NVDA,AMD", "--date", "2026-09-30",
        "--output-dir", str(tmp_path), "--seed", "12345",
    ])
    assert result.exit_code == 0, result.output
    assert result.output.count("Run seed: 12345") == 1
    assert calls[0]["base_seed"] == 12345
    assert calls[0]["configured_seed"] == 12345
