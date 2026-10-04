"""Effective deep/quick model resolution: a single source of truth
(`resolve_effective_models`) used by the CLI, so --TRADINGAGENTS_DEEP_THINK_LLM
and --TRADINGAGENTS_QUICK_THINK_LLM have one, centrally-tested precedence/
inheritance rule (see fork README: "Model override CLI parameters")."""

from __future__ import annotations

import pytest

from tradingagents.default_config import resolve_effective_models


def _cfg(deep="A", quick="B"):
    return {"deep_think_llm": deep, "quick_think_llm": quick}


# --- 23-28: precedence/inheritance cases -------------------------------------


@pytest.mark.unit
def test_no_cli_overrides_with_non_empty_configured_quick():
    assert resolve_effective_models(_cfg("A", "B")) == ("A", "B")


@pytest.mark.unit
def test_no_cli_overrides_with_empty_configured_quick_inherits_deep():
    assert resolve_effective_models(_cfg("A", "")) == ("A", "A")


@pytest.mark.unit
def test_cli_deep_only_makes_quick_follow_it_regardless_of_configured_quick():
    assert resolve_effective_models(_cfg("A", "B"), cli_deep="C") == ("C", "C")


@pytest.mark.unit
def test_cli_deep_and_quick_both_given():
    assert resolve_effective_models(_cfg("A", "B"), cli_deep="C", cli_quick="D") == ("C", "D")


@pytest.mark.unit
def test_cli_quick_only_leaves_deep_as_configured():
    assert resolve_effective_models(_cfg("A", "B"), cli_quick="D") == ("A", "D")


@pytest.mark.unit
def test_explicit_empty_or_whitespace_cli_quick_inherits_effective_deep():
    assert resolve_effective_models(_cfg("A", "B"), cli_quick="") == ("A", "A")
    assert resolve_effective_models(_cfg("A", "B"), cli_quick="   ") == ("A", "A")


@pytest.mark.unit
def test_explicit_empty_cli_quick_combined_with_a_deep_override():
    assert resolve_effective_models(_cfg("A", "B"), cli_deep="C", cli_quick="") == ("C", "C")


@pytest.mark.unit
def test_existing_non_empty_quick_config_is_backward_compatible_with_no_overrides():
    """An existing user's distinct quick model keeps working unchanged when no
    CLI override is supplied at all -- this must not regress to "quick always
    inherits deep"."""
    assert resolve_effective_models(_cfg("gpt-6-sol", "gpt-6-luna")) == (
        "gpt-6-sol", "gpt-6-luna",
    )


# --- 29, 30: exact CLI syntax ------------------------------------------------


@pytest.mark.unit
def test_exact_deep_think_llm_cli_flag_syntax_is_accepted():
    from typer.testing import CliRunner

    import cli.main as m

    captured = {}

    def fake_batch(**kwargs):
        captured.update(kwargs)
        from pathlib import Path
        return [], Path("x"), Path("y")

    original = m.run_batch_analysis
    m.run_batch_analysis = fake_batch
    try:
        result = CliRunner().invoke(
            m.app, ["batch", "--tickers", "NVDA", "--TRADINGAGENTS_DEEP_THINK_LLM=gpt-oss:20b"]
        )
        assert result.exit_code == 0, result.output
        assert captured["deep_think_llm"] == "gpt-oss:20b"
    finally:
        m.run_batch_analysis = original


@pytest.mark.unit
def test_exact_quick_think_llm_cli_flag_syntax_is_accepted():
    from typer.testing import CliRunner

    import cli.main as m

    captured = {}

    def fake_batch(**kwargs):
        captured.update(kwargs)
        from pathlib import Path
        return [], Path("x"), Path("y")

    original = m.run_batch_analysis
    m.run_batch_analysis = fake_batch
    try:
        result = CliRunner().invoke(
            m.app,
            ["batch", "--tickers", "NVDA",
             "--TRADINGAGENTS_DEEP_THINK_LLM=modelA",
             "--TRADINGAGENTS_QUICK_THINK_LLM=modelB"],
        )
        assert result.exit_code == 0, result.output
        assert captured["deep_think_llm"] == "modelA"
        assert captured["quick_think_llm"] == "modelB"
    finally:
        m.run_batch_analysis = original


@pytest.mark.unit
def test_friendly_alias_flags_also_work():
    from typer.testing import CliRunner

    import cli.main as m

    captured = {}

    def fake_batch(**kwargs):
        captured.update(kwargs)
        from pathlib import Path
        return [], Path("x"), Path("y")

    original = m.run_batch_analysis
    m.run_batch_analysis = fake_batch
    try:
        result = CliRunner().invoke(
            m.app,
            ["batch", "--tickers", "NVDA", "--deep-think-llm", "modelA",
             "--quick-think-llm", "modelB"],
        )
        assert result.exit_code == 0, result.output
        assert captured["deep_think_llm"] == "modelA"
        assert captured["quick_think_llm"] == "modelB"
    finally:
        m.run_batch_analysis = original


@pytest.mark.unit
def test_trace_llm_flag_is_off_by_default_and_settable():
    from typer.testing import CliRunner

    import cli.main as m

    captured = {}

    def fake_batch(**kwargs):
        captured.update(kwargs)
        from pathlib import Path
        return [], Path("x"), Path("y")

    original = m.run_batch_analysis
    m.run_batch_analysis = fake_batch
    try:
        CliRunner().invoke(m.app, ["batch", "--tickers", "NVDA"])
        assert captured["trace_llm"] is False
        CliRunner().invoke(m.app, ["batch", "--tickers", "NVDA", "--trace-llm"])
        assert captured["trace_llm"] is True
    finally:
        m.run_batch_analysis = original


# --- build_headless_config threads the resolved models end to end -----------


@pytest.mark.unit
def test_build_headless_config_resolves_cli_overrides(tmp_path):
    from cli.headless import build_headless_config

    config = build_headless_config(
        3, "English", tmp_path, cli_deep_think_llm="gpt-oss:20b", cli_quick_think_llm=None,
    )
    assert config["deep_think_llm"] == "gpt-oss:20b"
    assert config["quick_think_llm"] == "gpt-oss:20b"


@pytest.mark.unit
def test_build_headless_config_without_overrides_uses_default_config_values(tmp_path):
    from cli.headless import build_headless_config
    from tradingagents.default_config import DEFAULT_CONFIG

    config = build_headless_config(3, "English", tmp_path)
    assert config["deep_think_llm"] == DEFAULT_CONFIG["deep_think_llm"]
    assert config["quick_think_llm"] == DEFAULT_CONFIG["quick_think_llm"]
