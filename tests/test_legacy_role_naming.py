"""Legacy naming migration / backward compatibility (see fork README:
"Legacy naming" / "Model roles: Generation vs Review").

quick_think_llm/deep_think_llm (-> generation/review) and
quick_reasoning_effort/deep_reasoning_effort (-> generation_reasoning_effort/
review_reasoning_effort) must keep working unmigrated, with an explicit new
name always winning over its legacy equivalent when both are set --
``resolve_role_config`` is the single compatibility layer this all funnels
through.
"""
from __future__ import annotations

import importlib

import pytest
from typer.testing import CliRunner

import cli.main as m
import tradingagents.default_config as default_config_module
from tradingagents.default_config import resolve_role_config

# --- 31: legacy only ---------------------------------------------------------


@pytest.mark.unit
def test_legacy_only_model_names_resolve_to_canonical():
    config = resolve_role_config({"quick_think_llm": "A", "deep_think_llm": "B"})
    assert config["generation_llm"] == "A"
    assert config["review_llm"] == "B"


@pytest.mark.unit
def test_legacy_only_reasoning_effort_resolves_to_canonical():
    config = resolve_role_config(
        {"quick_reasoning_effort": "low", "deep_reasoning_effort": "max"}
    )
    assert config["generation_reasoning_effort"] == "low"
    assert config["review_reasoning_effort"] == "max"


# --- 32: new only -------------------------------------------------------------


@pytest.mark.unit
def test_new_only_model_names_resolve_unchanged():
    config = resolve_role_config({"generation_llm": "C", "review_llm": "D"})
    assert config["generation_llm"] == "C"
    assert config["review_llm"] == "D"
    # Mirrored onto the legacy keys too, for any code still reading them.
    assert config["quick_think_llm"] == "C"
    assert config["deep_think_llm"] == "D"


@pytest.mark.unit
def test_new_only_reasoning_effort_resolves_unchanged():
    config = resolve_role_config(
        {"generation_reasoning_effort": "low", "review_reasoning_effort": "max"}
    )
    assert config["generation_reasoning_effort"] == "low"
    assert config["review_reasoning_effort"] == "max"


# --- 33: both -- new wins -----------------------------------------------------


@pytest.mark.unit
def test_both_given_new_model_name_wins():
    config = resolve_role_config({"generation_llm": "C", "quick_think_llm": "A"})
    assert config["generation_llm"] == "C"
    assert config["quick_think_llm"] == "C"  # mirrored to the winner, not "A"


@pytest.mark.unit
def test_both_given_new_reasoning_effort_wins():
    config = resolve_role_config(
        {"generation_reasoning_effort": "low", "quick_reasoning_effort": "medium"}
    )
    assert config["generation_reasoning_effort"] == "low"
    assert config["quick_reasoning_effort"] == "low"


@pytest.mark.unit
def test_both_given_review_llm_wins_over_deep_think_llm():
    config = resolve_role_config({"review_llm": "D", "deep_think_llm": "B"})
    assert config["review_llm"] == "D"
    assert config["deep_think_llm"] == "D"


# --- hardcoded default when neither canonical nor legacy is set -------------


@pytest.mark.unit
def test_neither_set_falls_back_to_hardcoded_default():
    config = resolve_role_config({})
    assert config["generation_llm"]
    assert config["review_llm"]
    assert config["generation_reasoning_effort"] is None
    assert config["review_reasoning_effort"] is None


# --- 33 (env precedence): TRADINGAGENTS_GENERATION_LLM wins over
# TRADINGAGENTS_QUICK_THINK_LLM -----------------------------------------------


def _reload_with_env(monkeypatch, **overrides):
    for key in list(default_config_module._ENV_OVERRIDES):
        monkeypatch.delenv(key, raising=False)
    for key, val in overrides.items():
        monkeypatch.setenv(key, val)
    return importlib.reload(default_config_module)


@pytest.mark.unit
def test_env_legacy_only_resolves_to_canonical(monkeypatch):
    dc = _reload_with_env(
        monkeypatch, TRADINGAGENTS_QUICK_THINK_LLM="A", TRADINGAGENTS_DEEP_THINK_LLM="B"
    )
    assert dc.DEFAULT_CONFIG["generation_llm"] == "A"
    assert dc.DEFAULT_CONFIG["review_llm"] == "B"


@pytest.mark.unit
def test_env_new_name_wins_over_legacy(monkeypatch):
    dc = _reload_with_env(
        monkeypatch,
        TRADINGAGENTS_QUICK_THINK_LLM="A", TRADINGAGENTS_GENERATION_LLM="C",
        TRADINGAGENTS_DEEP_THINK_LLM="B", TRADINGAGENTS_REVIEW_LLM="D",
    )
    assert dc.DEFAULT_CONFIG["generation_llm"] == "C"
    assert dc.DEFAULT_CONFIG["review_llm"] == "D"
    assert dc.DEFAULT_CONFIG["quick_think_llm"] == "C"
    assert dc.DEFAULT_CONFIG["deep_think_llm"] == "D"


@pytest.mark.unit
def test_env_reasoning_effort_new_name_wins_over_legacy(monkeypatch):
    dc = _reload_with_env(
        monkeypatch,
        TRADINGAGENTS_QUICK_REASONING_EFFORT="medium",
        TRADINGAGENTS_GENERATION_REASONING_EFFORT="low",
    )
    assert dc.DEFAULT_CONFIG["generation_reasoning_effort"] == "low"


# --- 34: CLI precedence -- new flag wins over legacy flag -------------------


@pytest.mark.unit
def test_cli_new_model_flag_wins_over_legacy_flag(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(m, "run_batch_analysis", lambda **kw: calls.append(kw) or ([], tmp_path, tmp_path))
    result = CliRunner().invoke(m.app, [
        "batch", "--tickers", "NVDA", "--output-dir", str(tmp_path),
        "--quick-think-llm", "A", "--generation-llm", "C",
        "--deep-think-llm", "B", "--review-llm", "D",
    ])
    assert result.exit_code == 0, result.output
    assert calls[0]["quick_think_llm"] == "A"
    assert calls[0]["deep_think_llm"] == "B"
    assert calls[0]["generation_llm"] == "C"
    assert calls[0]["review_llm"] == "D"
    # The actual precedence resolution happens one layer down, in
    # build_headless_config -- see test_cli_new_model_flag_resolves_correctly.


@pytest.mark.unit
def test_cli_new_model_flag_resolves_correctly_over_legacy(tmp_path):
    from cli.headless import build_headless_config

    config = build_headless_config(
        1, "English", tmp_path,
        cli_quick_think_llm="A", cli_generation_llm="C",
        cli_deep_think_llm="B", cli_review_llm="D",
    )
    assert config["generation_llm"] == "C"
    assert config["review_llm"] == "D"


@pytest.mark.unit
def test_cli_legacy_model_flag_still_works_alone(tmp_path):
    from cli.headless import build_headless_config

    config = build_headless_config(
        1, "English", tmp_path,
        cli_quick_think_llm="A", cli_deep_think_llm="B",
    )
    assert config["generation_llm"] == "A"
    assert config["review_llm"] == "B"


@pytest.mark.unit
def test_cli_new_reasoning_effort_flag_wins_over_legacy(tmp_path):
    from cli.headless import build_headless_config

    config = build_headless_config(
        1, "English", tmp_path,
        cli_quick_reasoning_effort="medium", cli_generation_reasoning_effort="low",
    )
    assert config["generation_reasoning_effort"] == "low"
