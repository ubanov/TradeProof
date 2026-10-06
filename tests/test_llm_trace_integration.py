"""End-to-end LLM trace integration through execute_analysis: off by default,
creates the expected debug/ root when enabled, and the manifest/startup
display agree on the same effective models (see fork README: "LLM trace
mode" / "Model override CLI parameters")."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import cli.run as cli_run
from cli.stats_handler import StatsCallbackHandler


class _FakeGraphCore:
    def stream(self, state, **kwargs):
        yield {
            "market_report": "MARKET", "sentiment_report": "S", "news_report": "N",
            "fundamentals_report": "F",
            "investment_debate_state": {
                "bull_initial": "b", "bear_initial": "r",
                "verification_history": [{"status": "PASS", "findings": []}],
                "judge_decision": "RM plan",
            },
            "trader_investment_plan": "Trader plan",
            "risk_debate_state": {
                "count": 6,
                "aggressive_initial": "a1", "conservative_initial": "c1", "neutral_initial": "n1",
                "aggressive_review": "a2", "conservative_review": "c2", "neutral_review": "n2",
                "judge_decision": "PM decision",
            },
            "final_trade_decision": "**Rating**: Hold",
        }


class _FakeTradingAgentsGraph:
    last_instance = None

    def __init__(self, selected_analysts, config=None, debug=False, callbacks=None):
        self.config = config or {}
        self.callbacks = callbacks or []
        self.graph = _FakeGraphCore()
        self.propagator = SimpleNamespace(get_graph_args=lambda callbacks=None: {})
        type(self).last_instance = self

    def create_run_state(self, ticker, date, asset_type, portfolio):
        return {}

    def checkpoint_input(self, state):
        return state

    def begin_checkpoint(self, *a, **k):
        return None

    def record_decision(self, *a, **k):
        pass

    def clear_checkpoint_on_success(self, *a, **k):
        pass

    def end_checkpoint(self):
        pass

    def process_signal(self, text):
        from tradingagents.agents.rating import parse_rating
        return parse_rating(text)


@pytest.fixture
def fake_graph(monkeypatch):
    monkeypatch.setattr(cli_run, "TradingAgentsGraph", _FakeTradingAgentsGraph)
    monkeypatch.setattr(
        cli_run, "build_analyst_execution_plan", lambda analysts: SimpleNamespace(specs=[])
    )
    return _FakeTradingAgentsGraph


def _run(tmp_path, *, trace_llm, deep="A", quick="B"):
    stats_handler = StatsCallbackHandler()
    return cli_run.execute_analysis(
        ticker="NVDA", analysis_date="2026-09-30", selected_analysts=["market"],
        asset_type="stock",
        config={"max_risk_discuss_rounds": 3, "deep_think_llm": deep, "quick_think_llm": quick,
                "llm_provider": "ollama", "output_language": "Spanish"},
        output_dir=tmp_path, stats_handler=stats_handler, trace_llm=trace_llm,
    )


# --- 1, 2: default off, no trace directory ----------------------------------


@pytest.mark.unit
def test_trace_mode_defaults_off_in_execute_analysis_signature():
    import inspect
    assert inspect.signature(cli_run.execute_analysis).parameters["trace_llm"].default is False


@pytest.mark.unit
def test_normal_run_creates_no_debug_directory(fake_graph, tmp_path):
    _run(tmp_path, trace_llm=False)
    assert not (tmp_path / "debug").exists()


# --- 3: enabling creates the expected trace root -----------------------------


@pytest.mark.unit
def test_enabling_trace_creates_the_expected_root(fake_graph, tmp_path, capsys):
    _run(tmp_path, trace_llm=True)
    debug_dir = tmp_path / "debug"
    assert debug_dir.is_dir()
    assert (debug_dir / "run_manifest.json").exists()
    assert (debug_dir / "prompts").is_dir()
    assert (debug_dir / "prompts" / "global_policy.txt").exists()
    captured = capsys.readouterr()
    assert f"[DEBUG] LLM trace enabled: {debug_dir}" in captured.out


@pytest.mark.unit
def test_trace_summary_is_written_with_aggregate_counts(fake_graph, tmp_path):
    _run(tmp_path, trace_llm=True)
    summary = json.loads((tmp_path / "debug" / "trace_summary.json").read_text())
    assert "llm_calls" in summary
    assert "deep_model" in summary and "quick_model" in summary


# --- 32, 33: effective models consistent across manifest/graph config -------


@pytest.mark.unit
def test_manifest_effective_models_match_the_config_used_to_build_the_graph(fake_graph, tmp_path):
    _run(tmp_path, trace_llm=True, deep="qwen3.8-flash-next:125b-a6b-q4_K_M", quick="gpt-oss:20b")
    manifest = json.loads((tmp_path / "debug" / "run_manifest.json").read_text())
    graph_config = fake_graph.last_instance.config
    assert manifest["effective_deep_model"] == graph_config["deep_think_llm"] == (
        "qwen3.8-flash-next:125b-a6b-q4_K_M"
    )
    assert manifest["effective_quick_model"] == graph_config["quick_think_llm"] == "gpt-oss:20b"


@pytest.mark.unit
def test_startup_display_shows_the_same_effective_models_as_the_manifest(fake_graph, tmp_path):
    import cli.headless as headless

    lines = []
    headless.run_single_analysis(
        ticker="NVDA", analysis_date="2026-09-30", analysts=["market"], depth_value=3,
        language="English", output_dir=tmp_path, progress=lines.append,
        deep_think_llm="gpt-oss:20b", trace_llm=True,
    )
    assert any(
        "Generation model: gpt-oss:20b" in line and "Review model: gpt-oss:20b" in line
        for line in lines
    )
    manifest = json.loads((tmp_path / "debug" / "run_manifest.json").read_text())
    assert manifest["effective_deep_model"] == "gpt-oss:20b"
    assert manifest["effective_quick_model"] == "gpt-oss:20b"
    assert manifest["generation_model"] == "gpt-oss:20b"
    assert manifest["review_model"] == "gpt-oss:20b"


@pytest.mark.unit
def test_tracer_is_isolated_per_ticker_not_leaked_across_sequential_runs(fake_graph, tmp_path):
    """A batch processes tickers sequentially in one process/thread; a run
    with trace off must not inherit a tracer left active by a prior traced
    run (contextvar persists across calls unless explicitly reset)."""
    from tradingagents.llm_trace import _active_tracer

    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    _run(first_dir, trace_llm=True)
    assert _active_tracer.get() is not None

    _run(second_dir, trace_llm=False)
    assert _active_tracer.get() is None
    assert not (second_dir / "debug").exists()
