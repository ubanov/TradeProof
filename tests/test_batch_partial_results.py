"""Batch/single-run robustness: a late-stage failure must not lose completed
work, must record where it happened, and must never crash on a missing
derived/optional value (the real GLD incident: 2h43m of completed work lost
to an uncaught KeyError immediately before Portfolio Manager)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import cli.run as cli_run
from cli.headless import run_batch_analysis, run_single_analysis
from cli.run import PartialExecutionError, infer_failure_stage
from cli.stats_handler import StatsCallbackHandler

# ---------------------------------------------------------------------------
# 1-4: risk_stance_summary / risk-level state is always safe to read
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_missing_risk_level_keys_do_not_raise_building_the_prompt():
    """An older/default risk_debate_state with none of the risk_level keys
    (a pre-upgrade checkpoint, or a bare hand-built state) must not KeyError
    when the Portfolio Manager renders its prompt."""
    from tradingagents.agents.managers.portfolio_manager import create_portfolio_manager
    from tradingagents.agents.schemas import PortfolioDecision, PortfolioRating

    class _PMLLM:
        def with_structured_output(self, _schema):
            return self

        def invoke(self, prompt):
            self.prompt = prompt
            return PortfolioDecision(
                rating=PortfolioRating.HOLD, disposition="KEEP",
                executive_summary="x", investment_thesis="y",
            )

    state = {
        "company_of_interest": "NVDA", "asset_type": "stock",
        "instrument_context": "", "portfolio_context": "", "investment_plan": "p",
        "trader_investment_plan": "t", "past_context": "",
        "risk_debate_state": {
            "history": "h", "aggressive_history": "", "conservative_history": "",
            "neutral_history": "", "current_aggressive_response": "",
            "current_conservative_response": "", "current_neutral_response": "",
            "count": 9,
            # Deliberately no aggressive_risk_level/conservative_risk_level/
            # neutral_risk_level keys at all.
        },
    }
    llm = _PMLLM()
    result = create_portfolio_manager(llm)(state)  # must not raise
    assert "not recorded" in llm.prompt
    assert result["final_trade_decision"]


@pytest.mark.unit
def test_structured_risk_success_initialises_all_three_risk_level_fields():
    from tradingagents.graph.propagation import Propagator

    state = Propagator().create_initial_state("NVDA", "2026-09-01")
    risk = state["risk_debate_state"]
    assert risk["aggressive_risk_level"] == ""
    assert risk["conservative_risk_level"] == ""
    assert risk["neutral_risk_level"] == ""


@pytest.mark.unit
def test_risk_free_text_fallback_leaves_risk_level_as_empty_string_not_missing():
    from tradingagents.agents.risk_mgmt.aggressive_debator import create_aggressive_debator
    from tradingagents.graph.propagation import Propagator

    state = Propagator().create_initial_state("NVDA", "2026-09-01")
    state.update(
        {"market_report": "m", "sentiment_report": "s", "news_report": "n",
         "fundamentals_report": "f", "trader_investment_plan": "plan"}
    )

    class _PlainLLM:
        def invoke(self, prompt):
            return SimpleNamespace(content="argument")

    llm = _PlainLLM()
    result = create_aggressive_debator(llm, llm)(state)
    risk = result["risk_debate_state"]
    assert risk["aggressive_risk_level"] == ""  # present, not absent
    assert "aggressive_risk_level" in risk


# ---------------------------------------------------------------------------
# infer_failure_stage
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_infer_failure_stage_portfolio_manager_when_all_risk_turns_done():
    final_state = {
        "risk_debate_state": {"count": 6, "neutral_review": "Neutral Reviewer (Cross-Review): ..."},
        "investment_debate_state": {"judge_decision": "plan"},
        "trader_investment_plan": "plan",
    }
    assert infer_failure_stage(final_state) == "portfolio_manager"


@pytest.mark.unit
def test_infer_failure_stage_risk_when_some_turns_done():
    final_state = {"risk_debate_state": {"count": 4}, "trader_investment_plan": "plan"}
    assert infer_failure_stage(final_state) == "risk"


@pytest.mark.unit
def test_infer_failure_stage_trader_when_manager_done_but_no_risk_yet():
    final_state = {
        "investment_debate_state": {"judge_decision": "plan"},
        "risk_debate_state": {"count": 0},
    }
    assert infer_failure_stage(final_state) == "trader"


@pytest.mark.unit
def test_infer_failure_stage_research_manager_after_verifier_passed():
    final_state = {
        "investment_debate_state": {
            "verification_history": [{"status": "PASS"}],
        },
    }
    assert infer_failure_stage(final_state) == "research_manager"


@pytest.mark.unit
def test_infer_failure_stage_repair_after_verifier_fail_not_yet_repaired():
    final_state = {
        "investment_debate_state": {
            "verification_history": [{"status": "FAIL"}],
            "repair_triggered": False,
        },
    }
    assert infer_failure_stage(final_state) == "repair"


@pytest.mark.unit
def test_infer_failure_stage_verifier_after_reviews_done():
    final_state = {
        "investment_debate_state": {
            "bull_initial": "b", "bear_initial": "r", "debate_round": 2,
        },
    }
    assert infer_failure_stage(final_state) == "verifier"


@pytest.mark.unit
def test_infer_failure_stage_research_review_after_one_initial():
    final_state = {"investment_debate_state": {"bull_initial": "b"}}
    assert infer_failure_stage(final_state) == "research_review"


@pytest.mark.unit
def test_infer_failure_stage_next_missing_analyst_report():
    final_state = {"market_report": "m", "sentiment_report": "s"}
    assert infer_failure_stage(final_state) == "news_analyst"


@pytest.mark.unit
def test_infer_failure_stage_research_initial_once_all_analysts_done():
    final_state = {
        "market_report": "m", "sentiment_report": "s",
        "news_report": "n", "fundamentals_report": "f",
    }
    assert infer_failure_stage(final_state) == "research_initial"


@pytest.mark.unit
def test_infer_failure_stage_complete_when_final_decision_exists():
    assert infer_failure_stage({"final_trade_decision": "x"}) == "complete"


@pytest.mark.unit
def test_infer_failure_stage_handles_empty_or_none_state():
    assert infer_failure_stage(None) == "market_analyst"
    assert infer_failure_stage({}) == "market_analyst"


# ---------------------------------------------------------------------------
# execute_analysis: a late failure preserves everything the stream yielded
# ---------------------------------------------------------------------------

_FULL_STATE_BEFORE_PM = {
    "market_report": "MARKET", "sentiment_report": "SENTIMENT",
    "news_report": "NEWS", "fundamentals_report": "FUNDAMENTALS",
    "investment_debate_state": {
        "bull_initial": "Bull Analyst (Initial): b",
        "bear_initial": "Bear Analyst (Initial): r",
        "bull_history": "bull history", "bear_history": "bear history",
        "debate_round": 2, "bull_rebuttal_count": 2, "bear_rebuttal_count": 2,
        "verification_history": [{"status": "PASS", "findings": []}],
        "verifier_pass_count": 1, "verification_status": "PASS",
        "repair_rounds": 0, "repaired_agents": [],
        "bull_repair_structured_status": "NOT_RUN", "bear_repair_structured_status": "NOT_RUN",
        "judge_decision": "Research Manager plan",
        "manager_integrity_status": "WARN",
        "manager_integrity_findings": [{"category": "NEW_UNSUPPORTED_THRESHOLD", "detail": "x"}],
    },
    "investment_plan": "Research Manager plan",
    "trader_investment_plan": "Trader plan",
    "risk_debate_state": {
        "history": "risk history", "count": 6, "latest_speaker": "Neutral",
        "aggressive_history": "agg", "conservative_history": "con", "neutral_history": "neu",
        "aggressive_risk_level": "LOW", "conservative_risk_level": "HIGH",
        "neutral_risk_level": "MEDIUM",
        "aggressive_disposition": "KEEP", "conservative_disposition": "REDUCE_RISK",
        "neutral_disposition": "KEEP",
        "aggressive_initial": "Aggressive Reviewer (Independent Assessment): ...",
        "conservative_initial": "Conservative Reviewer (Independent Assessment): ...",
        "neutral_initial": "Neutral Reviewer (Independent Assessment): ...",
        "aggressive_review": "Aggressive Reviewer (Cross-Review): ...",
        "conservative_review": "Conservative Reviewer (Cross-Review): ...",
        "neutral_review": "Neutral Reviewer (Cross-Review): ...",
        "current_aggressive_response": "", "current_conservative_response": "",
        "current_neutral_response": "",
        # Deliberately no judge_decision -- Portfolio Manager never completed.
    },
}


class _FakeGraphCore:
    def __init__(self, chunk, error):
        self.chunk = chunk
        self.error = error

    def stream(self, state, **kwargs):
        yield dict(self.chunk)
        raise self.error


class _FakeTradingAgentsGraph:
    last_instance = None

    def __init__(self, selected_analysts, config=None, debug=False, callbacks=None, run_provenance=None):
        self.config = config or {}
        self.run_provenance = run_provenance
        self.graph = _FakeGraphCore(_FULL_STATE_BEFORE_PM, KeyError("risk_stance_summary"))
        self.propagator = SimpleNamespace(get_graph_args=lambda callbacks=None: {})
        self.record_decision_called = False
        self.clear_checkpoint_called = False
        self.end_checkpoint_called = False
        type(self).last_instance = self

    def create_run_state(self, ticker, date, asset_type, portfolio):
        return {}

    def checkpoint_input(self, state):
        return state

    def begin_checkpoint(self, *a, **k):
        return None

    def record_decision(self, *a, **k):
        self.record_decision_called = True

    def clear_checkpoint_on_success(self, *a, **k):
        self.clear_checkpoint_called = True

    def end_checkpoint(self):
        self.end_checkpoint_called = True

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


@pytest.mark.unit
def test_late_failure_raises_partial_execution_error_with_everything_preserved(fake_graph, tmp_path):
    stats_handler = StatsCallbackHandler()
    stats_handler.llm_calls = 38
    stats_handler.tool_calls = 27

    with pytest.raises(PartialExecutionError) as excinfo:
        cli_run.execute_analysis(
            ticker="GLD", analysis_date="2026-09-29", selected_analysts=["market"],
            asset_type="stock", config={"max_risk_discuss_rounds": 3},
            output_dir=tmp_path, stats_handler=stats_handler,
        )

    exc = excinfo.value
    assert exc.failure_stage == "portfolio_manager"
    debate = exc.final_state["investment_debate_state"]
    risk = exc.final_state["risk_debate_state"]
    # 5-10: every stage that actually completed is still there.
    assert debate["bull_initial"] and debate["bear_initial"]
    assert debate["verifier_pass_count"] == 1 and debate["verification_status"] == "PASS"
    assert debate["bull_repair_structured_status"] == "NOT_RUN"
    assert debate["judge_decision"] == "Research Manager plan"
    assert debate["manager_integrity_status"] == "WARN"
    assert debate["manager_integrity_findings"]
    assert exc.final_state["trader_investment_plan"] == "Trader plan"
    assert risk["count"] == 6
    assert risk["aggressive_risk_level"] == "LOW"
    # 11: stats preserved even though the run failed.
    assert exc.stats["llm_calls"] == 38
    assert exc.stats["tool_calls"] == 27
    # A partial report was written.
    assert exc.report_path is not None
    assert exc.report_path.name == "partial_report.md"
    assert "INCOMPLETE RUN" in exc.report_path.read_text()
    assert "Research Manager plan" in exc.report_path.read_text()
    # record_decision/clear_checkpoint must NOT run for a failed analysis.
    graph = fake_graph.last_instance
    assert graph.record_decision_called is False
    assert graph.clear_checkpoint_called is False
    assert graph.end_checkpoint_called is True  # always runs, success or failure


@pytest.mark.unit
def test_run_single_analysis_returns_error_result_instead_of_raising(fake_graph, tmp_path):
    result = run_single_analysis(
        ticker="GLD", analysis_date="2026-09-29", analysts=["market"],
        depth_value=3, language="English", output_dir=tmp_path,
    )
    assert result["status"] == "error"
    assert result["failure_stage"] == "portfolio_manager"
    assert result["partial"]["research_manager_completed"] is True
    assert result["partial"]["trader_completed"] is True
    assert result["partial"]["risk_discussion_turns"] == 6
    assert result["partial"]["manager_integrity_status"] == "WARN"
    assert result["report_path"] == "partial_report.md"
    assert (tmp_path / "result.json").exists()
    assert (tmp_path / "partial_report.md").exists()


@pytest.mark.unit
def test_run_batch_analysis_preserves_partial_results_and_links_report(fake_graph, tmp_path):
    results, markdown_path, _ = run_batch_analysis(
        tickers=["GLD"], analysis_date="2026-09-29", analysts=["market"],
        depth_name="medium", depth_value=3, language="English",
        output_dir=tmp_path, progress=lambda line: None,
    )
    result = results[0]
    assert result["status"] == "error"
    assert result["failure_stage"] == "portfolio_manager"
    summary_text = markdown_path.read_text()
    assert "partial report" in summary_text
    assert "GLD/partial_report.md" in summary_text
    assert "(at portfolio_manager)" in summary_text


@pytest.mark.unit
def test_batch_progress_reports_stage_partial_artifacts_and_stats(fake_graph, tmp_path):
    lines = []
    run_batch_analysis(
        tickers=["GLD"], analysis_date="2026-09-29", analysts=["market"],
        depth_name="medium", depth_value=3, language="English",
        output_dir=tmp_path, progress=lines.append,
    )
    joined = "\n".join(lines)
    assert "FAILED at portfolio_manager after" in joined
    assert "Partial artifacts saved: GLD/partial_report.md" in joined


# ---------------------------------------------------------------------------
# Output-path correctness (relative / absolute / nested)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("make_output_dir", [
    lambda tmp_path: Path("batchout"),
    lambda tmp_path: tmp_path / "batchout",
    lambda tmp_path: Path("nested") / "deeper" / "batchout",
])
def test_batch_summary_path_is_never_duplicated(monkeypatch, tmp_path, make_output_dir):
    monkeypatch.chdir(tmp_path)
    output_dir = make_output_dir(tmp_path)

    def fake_single(*, ticker, analysis_date, analysts, depth_value, language,
                     output_dir, report_root=None, progress=None, progress_prefix=None):
        return {
            "ticker": ticker, "analysis_date": analysis_date, "status": "ok",
            "rating": "Hold", "rating_parse_status": "parsed", "action": "Hold",
            "entry_price": None, "stop_loss": None, "summary": None,
            "research_recommendation": "Hold", "portfolio_manager_rating": "Hold",
            "raw_final_decision": "x", "report_path": "complete_report.md",
            "run_duration_seconds": 1.0,
            "stats": {"llm_calls": 1, "tool_calls": 0, "tokens_in": 1, "tokens_out": 1},
            "research_execution": {
                "bull_initial_calls": 1, "bear_initial_calls": 1, "bull_review_calls": 0,
                "bear_review_calls": 0, "verifier_passes": 1, "verification_status": "PASS",
                "repair_rounds": 0, "repaired_agents": [],
            },
        }

    import cli.headless as headless_module
    monkeypatch.setattr(headless_module, "run_single_analysis", fake_single)

    results, markdown_path, json_path = run_batch_analysis(
        tickers=["AAPL"], analysis_date="2026-09-29", analysts=["market"],
        depth_name="medium", depth_value=3, language="English",
        output_dir=output_dir, progress=lambda line: None,
    )

    expected = output_dir.resolve() / "summary.md"
    assert markdown_path == expected
    assert markdown_path.exists()
    # The resolved path must not contain its own final segment twice in a row.
    parts = expected.parts
    for i in range(len(parts) - 1):
        assert not (parts[i] == parts[i + 1] and parts[i] not in ("\\", "/")), expected
    assert json_path == output_dir.resolve() / "summary.json"
    assert json_path.exists()
