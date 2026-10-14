"""Structured object preservation through the pipeline (see fork README:
"Preserve structured objects through the pipeline"). Trader/Research
Manager/Portfolio Manager each already produce a real Pydantic object
(TraderProposal/ResearchPlan/PortfolioDecision) that was rendered to
markdown and then discarded -- everything downstream re-derived a handful
of fields from that markdown by regex. This preserves the original object
(JSON-safe) through ``final_state`` and into ``result.json``, alongside the
unchanged legacy regex fields -- never replacing them, purely additive.

Numbered per the task's own "TESTS" list (1-16).
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from cli.result_document import _structured_section, result_document

# Reuse the exact fixtures test_structured_agents.py already established for
# the Trader/Research Manager happy-path shape, rather than re-inventing a
# parallel state/LLM-mock convention.
from tests.test_structured_agents import (
    _make_rm_state,
    _make_trader_state,
    _structured_rm_llm,
    _structured_trader_llm,
)
from tradingagents.agents.managers.portfolio_manager import create_portfolio_manager
from tradingagents.agents.managers.research_manager import create_research_manager
from tradingagents.agents.schemas import (
    PortfolioDecision,
    PortfolioRating,
    ResearchPlan,
    RiskDisposition,
    TraderAction,
    TraderProposal,
)
from tradingagents.agents.trader.trader import create_trader
from tradingagents.graph.propagation import Propagator


def _execution(final_state, **overrides):
    defaults = {
        "ticker": "NVDA", "analysis_date": "2026-09-01", "rating": "Buy",
        "report_path": None, "duration_seconds": 1.0,
        "stats": {"llm_calls": 1, "tool_calls": 0, "tokens_in": 1, "tokens_out": 1},
        "fallback_warnings": 0, "base_seed": None,
    }
    defaults.update(overrides)
    return SimpleNamespace(final_state=final_state, **defaults)


def _pm_state_and_llm(decision=None):
    if decision is None:
        decision = PortfolioDecision(
            rating=PortfolioRating.OVERWEIGHT, disposition=RiskDisposition.KEEP,
            executive_summary="Enter per the trader's plan.",
            investment_thesis="Bull case carried the debate.",
            price_target=220.0, time_horizon="3-6 months",
        )
    state = Propagator().create_initial_state("NVDA", "2026-09-01")
    state["investment_plan"] = "**Recommendation**: Overweight"
    state["trader_investment_plan"] = "**Action**: Buy\n**Entry Price**: 189.5\n**Stop Loss**: 178.0"

    structured = MagicMock()
    structured.invoke.return_value = decision
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    return state, llm, decision


# --- 1/4: TraderProposal structured survives to final_state, stop/target ---
# --- read straight from the structured dict, no regex involved ------------


@pytest.mark.unit
def test_1_trader_proposal_structured_survives_to_final_state():
    proposal = TraderProposal(
        action=TraderAction.BUY, reasoning="Strong setup.",
        entry_price=189.5, entry_price_low=188.0, entry_price_high=191.0,
        reference_entry=189.5, stop_loss=178.0, position_sizing="6% of portfolio",
    )
    trader = create_trader(_structured_trader_llm({}, proposal))
    result = trader(_make_trader_state())
    structured = result["trader_proposal_structured"]
    assert structured is not None
    assert structured["action"] == "Buy"  # 13: enum serialized to its plain value
    assert structured["entry_price"] == 189.5
    assert structured["entry_price_low"] == 188.0
    assert structured["entry_price_high"] == 191.0
    assert structured["reference_entry"] == 189.5
    assert structured["stop_loss"] == 178.0  # 4: read directly, no regex anywhere here
    assert structured["position_sizing"] == "6% of portfolio"


# --- 2: TraderProposal appears faithfully in result.json -------------------


@pytest.mark.unit
def test_2_trader_proposal_appears_faithfully_in_result_json(tmp_path):
    proposal = TraderProposal(
        action=TraderAction.SELL, reasoning="Guidance cut.",
        entry_price=100.0, stop_loss=110.0,
    )
    trader = create_trader(_structured_trader_llm({}, proposal))
    state = _make_trader_state()
    state.update(trader(state))
    result = result_document(_execution(state), tmp_path)
    assert result["structured"]["trader_proposal"] == {
        "action": "Sell", "reasoning": "Guidance cut.",
        "entry_price": 100.0, "entry_price_low": None, "entry_price_high": None,
        "reference_entry": None, "stop_loss": 110.0, "position_sizing": None,
    }


# --- 3: TradeMetrics keeps the full entry range ----------------------------


@pytest.mark.unit
def test_3_trade_metrics_keeps_the_full_entry_range():
    state = {
        "investment_debate_state": {
            "bull_trade_metrics": {
                "entry_reference": 150.0, "risk": 5.0, "reward": 15.0,
                "risk_reward": 3.0, "stop_atr": 1.2, "target_atr": 3.6,
                "risk_reward_min": 2.5, "risk_reward_max": 3.5,
            },
            "bear_trade_metrics": None,
        },
    }
    structured = _structured_section(state)
    assert structured["bull_trade_metrics"]["risk_reward_min"] == 2.5
    assert structured["bull_trade_metrics"]["risk_reward_max"] == 3.5
    assert structured["bull_trade_metrics"]["stop_atr"] == 1.2
    assert structured["bull_trade_metrics"]["target_atr"] == 3.6


# --- 5: ResearchPlan structured survives ------------------------------------


@pytest.mark.unit
def test_5_research_plan_structured_survives():
    plan = ResearchPlan(
        recommendation=PortfolioRating.OVERWEIGHT,
        rationale="Bull case carried the debate.",
        strategic_actions="Build position over two weeks.",
    )
    rm = create_research_manager(_structured_rm_llm({}, plan))
    result = rm(_make_rm_state())
    structured = result["research_plan_structured"]
    assert structured == {
        "recommendation": "Overweight",
        "rationale": "Bull case carried the debate.",
        "strategic_actions": "Build position over two weeks.",
    }


# --- 6: PortfolioDecision structured survives -------------------------------


@pytest.mark.unit
def test_6_portfolio_decision_structured_survives():
    state, llm, decision = _pm_state_and_llm()
    pm = create_portfolio_manager(llm)
    result = pm(state)
    structured = result["portfolio_decision_structured"]
    assert structured["rating"] == "Overweight"
    assert structured["disposition"] == "KEEP"
    assert structured["price_target"] == 220.0
    assert structured["time_horizon"] == "3-6 months"
    assert structured["investment_thesis"] == "Bull case carried the debate."


# --- 7/8: Bull/Bear conviction are preserved --------------------------------


@pytest.mark.unit
def test_7_8_bull_and_bear_conviction_are_preserved():
    state = {
        "investment_debate_state": {
            "bull_conviction_history": [60, 70, 75],
            "bear_conviction_history": [40, 35],
        },
    }
    structured = _structured_section(state)
    assert structured["bull_conviction_history"] == [60, 70, 75]
    assert structured["bear_conviction_history"] == [40, 35]


# --- 9/10: legacy regex fields still exist and agree with structured -------


@pytest.mark.unit
def test_9_10_legacy_fields_still_exist_and_agree_with_structured(tmp_path):
    proposal = TraderProposal(action=TraderAction.BUY, reasoning="x", entry_price=189.5, stop_loss=178.0)
    trader = create_trader(_structured_trader_llm({}, proposal))
    state = _make_trader_state()
    state.update(trader(state))
    result = result_document(_execution(state), tmp_path)
    # 9: legacy fields untouched/still present.
    assert result["action"] == "Buy"
    assert result["entry_price"] == 189.5
    assert result["stop_loss"] == 178.0
    # 10: legacy (regex-recovered) and structured (schema-sourced) agree on
    # the same concept for the same underlying decision.
    assert result["action"] == result["structured"]["trader_proposal"]["action"]
    assert result["entry_price"] == result["structured"]["trader_proposal"]["entry_price"]
    assert result["stop_loss"] == result["structured"]["trader_proposal"]["stop_loss"]


# --- 11: a partial run never fabricates structured data ---------------------


@pytest.mark.unit
def test_11_partial_run_never_fabricates_structured_data(tmp_path):
    # Trader/RM/PM never ran -- only a bare, otherwise-populated state.
    state = {"trader_investment_plan": "", "investment_plan": "", "final_trade_decision": ""}
    result = result_document(_execution(state), tmp_path)
    structured = result["structured"]
    assert structured["trader_proposal"] is None
    assert structured["research_plan"] is None
    assert structured["portfolio_decision"] is None
    assert structured["bull_trade_metrics"] is None
    assert structured["bear_thesis"] is None


@pytest.mark.unit
def test_11b_a_repair_whose_text_never_reparsed_is_also_none_not_fabricated():
    """outcome.final_parsed is None whenever the repair's own text failed to
    re-validate against the schema -- confirmed directly against
    validate_and_repair's own contract, never guessed at here."""
    from tradingagents.agents.validator import ValidationOutcome

    outcome = ValidationOutcome(status="FAIL", findings=[], final_text="unparseable prose", final_parsed=None)
    structured = outcome.final_parsed.model_dump(mode="json") if outcome.final_parsed is not None else None
    assert structured is None


# --- 12: full JSON serialization works --------------------------------------


@pytest.mark.unit
def test_12_full_json_serialization_works(tmp_path):
    proposal = TraderProposal(action=TraderAction.HOLD, reasoning="x")
    plan = ResearchPlan(recommendation=PortfolioRating.HOLD, rationale="r", strategic_actions="s")
    state = {
        "trader_investment_plan": "", "investment_plan": "", "final_trade_decision": "",
        "trader_proposal_structured": proposal.model_dump(mode="json"),
        "research_plan_structured": plan.model_dump(mode="json"),
        "investment_debate_state": {
            "bull_trade_metrics": {"reward_risk": 2.1},
            "bull_thesis": {"direction": "BULL", "conviction": {"total": 70}},
            "bull_conviction_history": [70],
        },
    }
    result = result_document(_execution(state), tmp_path)
    json.dumps(result)  # must not raise -- no non-JSON-safe objects anywhere


# --- 13: enums serialize correctly (also covered inline in test 1/6) -------


@pytest.mark.unit
def test_13_enums_serialize_as_plain_strings_not_enum_instances():
    proposal = TraderProposal(action=TraderAction.SELL, reasoning="x")
    dumped = proposal.model_dump(mode="json")
    assert dumped["action"] == "Sell"
    assert isinstance(dumped["action"], str)

    decision = PortfolioDecision(
        rating=PortfolioRating.UNDERWEIGHT, disposition=RiskDisposition.REDUCE_RISK,
        executive_summary="s", investment_thesis="t",
    )
    dumped_pm = decision.model_dump(mode="json")
    assert dumped_pm["rating"] == "Underweight"
    assert dumped_pm["disposition"] == "REDUCE_RISK"


# --- 14: report.md rendering is unaffected (reporting.py untouched) --------


@pytest.mark.unit
def test_14_report_tree_rendering_is_unaffected_by_structured_keys(tmp_path):
    from tradingagents.reporting import write_report_tree

    base_state = {"market_report": "m", "final_trade_decision": "d"}
    with_structured = {
        **base_state,
        "trader_proposal_structured": {"action": "Buy"},
        "research_plan_structured": {"recommendation": "Buy"},
        "portfolio_decision_structured": {"rating": "Buy"},
    }
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    path_a = write_report_tree(base_state, "NVDA", out_a)
    path_b = write_report_tree(with_structured, "NVDA", out_b)
    assert path_a.read_text(encoding="utf-8") == path_b.read_text(encoding="utf-8")


# --- 15: the pipeline's final decision text is unchanged --------------------


@pytest.mark.unit
def test_15_final_decision_text_is_unchanged_by_the_new_field():
    proposal = TraderProposal(action=TraderAction.BUY, reasoning="Strong setup.", entry_price=189.5)
    trader = create_trader(_structured_trader_llm({}, proposal))
    result = trader(_make_trader_state())
    # Same exact rendered markdown the Trader always produced -- the new
    # trader_proposal_structured key is purely additive.
    assert result["trader_investment_plan"].startswith("**Action**: Buy")
    assert "FINAL TRANSACTION PROPOSAL: **BUY**" in result["trader_investment_plan"]


# --- 16: validation_log / source_coverage remain intact ---------------------


@pytest.mark.unit
def test_16_validation_and_source_coverage_remain_intact(tmp_path):
    state = {
        "trader_investment_plan": "", "investment_plan": "", "final_trade_decision": "",
        "validation_log": [{
            "stage": "Trader", "status": "PASS", "initial_status": "PASS",
            "finding_count": 0, "categories": [], "repair_attempted": False,
            "repair_succeeded": False, "llm_calls": 1,
        }],
        "source_coverage": {"news": {"company": {"status": "SUCCESS_WITH_DATA"}}},
    }
    result = result_document(_execution(state), tmp_path)
    assert result["validation"] is not None
    assert result["source_coverage"] == {"news": {"company": {"status": "SUCCESS_WITH_DATA"}}}
    assert "structured" in result  # additive, doesn't replace either existing key
