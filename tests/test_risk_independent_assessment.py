"""Risk is redesigned as three independent assessments of the frozen Trader
plan, followed by exactly one cross-review round -- not a competitive,
repeated debate (see fork README: "Risk as independent assessment + one
cross-review"). This covers the independence/cross-review properties
themselves; structured-state mechanics are in test_risk_stance_structured.py
and the deterministic integrity check is in test_risk_integrity.py."""

from __future__ import annotations

import pytest

from tradingagents.agents.risk_mgmt.aggressive_debator import create_aggressive_debator
from tradingagents.agents.risk_mgmt.conservative_debator import create_conservative_debator
from tradingagents.agents.risk_mgmt.neutral_debator import create_neutral_debator
from tradingagents.agents.schemas import RiskStanceAssessment
from tradingagents.graph.conditional_logic import ConditionalLogic
from tradingagents.graph.propagation import Propagator


class _SeqLLM:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.prompts: list[str] = []

    def with_structured_output(self, schema):
        return self

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return self.outputs.pop(0)


def _assessment(rationale="ok", **overrides):
    base = {"risk_level": "LOW", "disposition": "KEEP", "rationale": rationale}
    base.update(overrides)
    return RiskStanceAssessment(**base)


def _state():
    state = Propagator().create_initial_state("NVDA", "2026-09-01")
    state.update(
        {
            "market_report": "MARKET", "sentiment_report": "SENTIMENT",
            "news_report": "NEWS", "fundamentals_report": "FUNDAMENTALS",
            "trader_investment_plan": "Buy, entry 100, stop 95.",
        }
    )
    return state


def _run_full_risk_phase(state, agg_llm, con_llm, neu_llm):
    nodes = {
        "Aggressive Analyst": create_aggressive_debator(agg_llm, agg_llm),
        "Conservative Analyst": create_conservative_debator(con_llm, con_llm),
        "Neutral Analyst": create_neutral_debator(neu_llm, neu_llm),
    }
    logic = ConditionalLogic()
    target = logic.should_continue_risk_analysis(state)
    steps = 0
    while target != "Portfolio Manager":
        result = nodes[target](state)
        state["risk_debate_state"] = result["risk_debate_state"]
        target = logic.should_continue_risk_analysis(state)
        steps += 1
        assert steps <= 6, "risk phase did not terminate within six turns"
    return steps


# --- A: independence ---------------------------------------------------------


@pytest.mark.unit
def test_aggressive_initial_cannot_see_conservative_or_neutral_initial():
    state = _state()
    # Even if the other two reviewers had already produced a distinctive
    # independent assessment in state (as they will have by the time the
    # real graph reaches Aggressive, since node order is fixed), Aggressive's
    # own INITIAL prompt must not reference it.
    state["risk_debate_state"]["conservative_initial"] = "Conservative distinctive marker CONMARK"
    state["risk_debate_state"]["neutral_initial"] = "Neutral distinctive marker NEUMARK"

    agg_llm = _SeqLLM([_assessment("agg says X")])
    create_aggressive_debator(agg_llm, agg_llm)(state)

    assert "CONMARK" not in agg_llm.prompts[0]
    assert "NEUMARK" not in agg_llm.prompts[0]


@pytest.mark.unit
def test_conservative_initial_cannot_see_aggressive_or_neutral_initial():
    state = _state()
    agg_llm = _SeqLLM([_assessment("agg says a distinctive AGGMARK")])
    agg_result = create_aggressive_debator(agg_llm, agg_llm)(state)
    state["risk_debate_state"] = agg_result["risk_debate_state"]
    # Neutral has not run yet in the real sequence, but even if it somehow had,
    # Conservative's independent prompt must not reference it either.
    state["risk_debate_state"]["neutral_initial"] = "Neutral distinctive marker NEUMARK"

    con_llm = _SeqLLM([_assessment("con says Y")])
    create_conservative_debator(con_llm, con_llm)(state)
    assert "AGGMARK" not in con_llm.prompts[0]
    assert "NEUMARK" not in con_llm.prompts[0]


@pytest.mark.unit
def test_neutral_initial_cannot_see_aggressive_or_conservative_initial():
    state = _state()
    agg_llm = _SeqLLM([_assessment("agg distinctive")])
    agg_result = create_aggressive_debator(agg_llm, agg_llm)(state)
    state["risk_debate_state"] = agg_result["risk_debate_state"]
    con_llm = _SeqLLM([_assessment("con distinctive")])
    con_result = create_conservative_debator(con_llm, con_llm)(state)
    state["risk_debate_state"] = con_result["risk_debate_state"]

    neu_llm = _SeqLLM([_assessment("neu says Z")])
    create_neutral_debator(neu_llm, neu_llm)(state)
    assert "agg distinctive" not in neu_llm.prompts[0]
    assert "con distinctive" not in neu_llm.prompts[0]


# --- B: cross-review sees all three, exactly one round ----------------------


@pytest.mark.unit
def test_all_three_cross_reviews_see_the_same_three_initial_assessments():
    state = _state()
    agg_llm = _SeqLLM([_assessment("agg initial unique marker AGG1"), _assessment("agg review")])
    con_llm = _SeqLLM([_assessment("con initial unique marker CON1"), _assessment("con review")])
    neu_llm = _SeqLLM([_assessment("neu initial unique marker NEU1"), _assessment("neu review")])

    steps = _run_full_risk_phase(state, agg_llm, con_llm, neu_llm)
    assert steps == 6

    # Each reviewer's SECOND prompt (the cross-review) must contain the other
    # two's initial markers.
    assert "CON1" in agg_llm.prompts[1] and "NEU1" in agg_llm.prompts[1]
    assert "AGG1" in con_llm.prompts[1] and "NEU1" in con_llm.prompts[1]
    assert "AGG1" in neu_llm.prompts[1] and "CON1" in neu_llm.prompts[1]


@pytest.mark.unit
def test_no_second_or_third_full_cross_review_round_occurs():
    """Exactly six turns total -- three independent assessments, one
    cross-review round -- never the old repeated-debate structure."""
    state = _state()
    agg_llm = _SeqLLM([_assessment() for _ in range(2)])
    con_llm = _SeqLLM([_assessment() for _ in range(2)])
    neu_llm = _SeqLLM([_assessment() for _ in range(2)])
    steps = _run_full_risk_phase(state, agg_llm, con_llm, neu_llm)
    assert steps == 6
    assert state["risk_debate_state"]["count"] == 6
    # Every LLM was called exactly twice -- not three, not more.
    assert len(agg_llm.prompts) == len(con_llm.prompts) == len(neu_llm.prompts) == 2
