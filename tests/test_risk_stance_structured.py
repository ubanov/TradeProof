"""Structured output for the Risk reviewers (Aggressive/Conservative/Neutral):
risk_level and disposition ride alongside the structured assessment, with a
free-text fallback. Independence/cross-review phase behavior is covered in
``test_risk_independent_assessment.py``."""

from __future__ import annotations

import pytest

from tradingagents.agents.risk_mgmt.aggressive_debator import create_aggressive_debator
from tradingagents.agents.risk_mgmt.conservative_debator import create_conservative_debator
from tradingagents.agents.risk_mgmt.neutral_debator import create_neutral_debator
from tradingagents.agents.schemas import RiskStanceAssessment
from tradingagents.graph.propagation import Propagator


class _StructuredLLM:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.prompts: list[str] = []

    def with_structured_output(self, schema):
        return self

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return self.outputs.pop(0)


def _assessment(**overrides):
    base = {"risk_level": "LOW", "disposition": "KEEP", "rationale": "Plan looks adequate."}
    base.update(overrides)
    return RiskStanceAssessment(**base)


def _state():
    state = Propagator().create_initial_state("NVDA", "2026-09-01")
    state.update(
        {
            "market_report": "MARKET",
            "sentiment_report": "SENTIMENT",
            "news_report": "NEWS",
            "fundamentals_report": "FUNDAMENTALS",
            "trader_investment_plan": "Buy, entry 100, stop 95.",
        }
    )
    return state


def _apply(state, node):
    state["risk_debate_state"] = node(state)["risk_debate_state"]


@pytest.mark.unit
def test_aggressive_initial_turn_sets_risk_level_disposition_and_marker():
    state = _state()
    llm = _StructuredLLM([_assessment(risk_level="LOW", disposition="DEFER")])
    _apply(state, create_aggressive_debator(llm, llm))

    debate = state["risk_debate_state"]
    assert debate["aggressive_risk_level"] == "LOW"
    assert debate["aggressive_disposition"] == "DEFER"
    assert debate["aggressive_initial"]  # completion marker set
    assert not debate["aggressive_review"]
    assert debate["aggressive_initial_assessment"]["disposition"] == "DEFER"


@pytest.mark.unit
def test_conservative_and_neutral_each_track_their_own_risk_level():
    state = _state()
    con_llm = _StructuredLLM([_assessment(risk_level="HIGH")])
    _apply(state, create_conservative_debator(con_llm, con_llm))
    assert state["risk_debate_state"]["conservative_risk_level"] == "HIGH"
    assert state["risk_debate_state"]["aggressive_risk_level"] == ""

    neu_llm = _StructuredLLM([_assessment(risk_level="MEDIUM")])
    _apply(state, create_neutral_debator(neu_llm, neu_llm))
    assert state["risk_debate_state"]["neutral_risk_level"] == "MEDIUM"
    # Earlier speakers' levels survive an unrelated turn.
    assert state["risk_debate_state"]["conservative_risk_level"] == "HIGH"


@pytest.mark.unit
def test_second_turn_is_the_cross_review_not_another_independent_assessment():
    state = _state()
    llm = _StructuredLLM(
        [
            _assessment(risk_level="LOW", disposition="KEEP"),
            _assessment(risk_level="MEDIUM", disposition="REDUCE_RISK"),
        ]
    )
    aggressive = create_aggressive_debator(llm, llm)
    _apply(state, aggressive)
    first_initial = state["risk_debate_state"]["aggressive_initial"]

    _apply(state, aggressive)
    debate = state["risk_debate_state"]
    assert debate["aggressive_risk_level"] == "MEDIUM"
    assert debate["aggressive_disposition"] == "REDUCE_RISK"
    assert debate["aggressive_initial"] == first_initial  # untouched by the review turn
    assert debate["aggressive_review"]  # now set


@pytest.mark.unit
def test_cross_review_updates_only_review_fields_not_the_initial_record():
    """The frozen initial assessment (Phase A) must survive unchanged once
    the cross-review (Phase B) runs -- the review is a new, separate record,
    not an edit of the independent one."""
    state = _state()
    llm = _StructuredLLM(
        [
            _assessment(risk_level="LOW", disposition="KEEP", rationale="initial take"),
            _assessment(risk_level="HIGH", disposition="REJECT_PLAN", rationale="review take"),
        ]
    )
    aggressive = create_aggressive_debator(llm, llm)
    _apply(state, aggressive)
    initial_text = state["risk_debate_state"]["aggressive_initial"]
    initial_assessment = state["risk_debate_state"]["aggressive_initial_assessment"]

    _apply(state, aggressive)
    debate = state["risk_debate_state"]
    assert debate["aggressive_initial"] == initial_text
    assert debate["aggressive_initial_assessment"] == initial_assessment
    assert debate["aggressive_review_assessment"]["rationale"] == "review take"
    assert debate["aggressive_review_assessment"] != initial_assessment


@pytest.mark.unit
def test_unsupported_claims_seen_during_cross_review_are_recorded():
    state = _state()
    llm = _StructuredLLM(
        [
            _assessment(rationale="initial take"),
            _assessment(
                rationale="review take",
                unsupported_claims_seen=["Conservative cited an unverified 1.5% equity-loss rule"],
            ),
        ]
    )
    aggressive = create_aggressive_debator(llm, llm)
    _apply(state, aggressive)
    _apply(state, aggressive)
    review = state["risk_debate_state"]["aggressive_review_assessment"]
    assert review["unsupported_claims_seen"] == [
        "Conservative cited an unverified 1.5% equity-loss rule"
    ]


@pytest.mark.unit
def test_falls_back_to_freetext_when_structured_output_unsupported():
    from types import SimpleNamespace

    class _PlainLLM:
        def invoke(self, prompt):
            return SimpleNamespace(content="plain prose assessment")

    state = _state()
    llm = _PlainLLM()
    _apply(state, create_aggressive_debator(llm, llm))
    debate = state["risk_debate_state"]
    assert "plain prose assessment" in debate["aggressive_initial"]
    assert debate["aggressive_risk_level"] == ""  # no signal available, not invented
    assert debate["aggressive_disposition"] == ""
    assert debate["aggressive_initial_assessment"] == {}


@pytest.mark.unit
def test_portfolio_manager_receives_the_latest_risk_stance_summary():
    from tradingagents.agents.managers.portfolio_manager import create_portfolio_manager
    from tradingagents.agents.schemas import PortfolioDecision, PortfolioRating

    class _PMLLM:
        def __init__(self):
            self.prompts = []

        def with_structured_output(self, _schema):
            return self

        def invoke(self, prompt):
            self.prompts.append(prompt)
            return PortfolioDecision(
                rating=PortfolioRating.HOLD,
                disposition="DEFER",
                executive_summary="Hold for now.",
                investment_thesis="Balanced risk views.",
            )

    state = _state()
    state["investment_plan"] = "Buy plan"
    state["risk_debate_state"].update(
        {
            "aggressive_risk_level": "LOW",
            "conservative_risk_level": "HIGH",
            "neutral_risk_level": "MEDIUM",
            "aggressive_disposition": "KEEP",
            "conservative_disposition": "REDUCE_RISK",
            "neutral_disposition": "DEFER",
        }
    )
    llm = _PMLLM()
    create_portfolio_manager(llm)(state)
    prompt = llm.prompts[0]
    assert "Aggressive: risk=LOW, disposition=KEEP" in prompt
    assert "Conservative: risk=HIGH, disposition=REDUCE_RISK" in prompt
    assert "Neutral: risk=MEDIUM, disposition=DEFER" in prompt


@pytest.mark.unit
def test_portfolio_manager_output_preserves_risk_level_fields():
    """Regression: Portfolio Manager used to build its returned
    risk_debate_state by listing every key explicitly, silently dropping the
    three risk_level fields (and anything else added later) from the final
    saved state/report."""
    from tradingagents.agents.managers.portfolio_manager import create_portfolio_manager
    from tradingagents.agents.schemas import PortfolioDecision, PortfolioRating

    class _PMLLM:
        def with_structured_output(self, _schema):
            return self

        def invoke(self, prompt):
            return PortfolioDecision(
                rating=PortfolioRating.HOLD, disposition="DEFER",
                executive_summary="x", investment_thesis="y",
            )

    state = _state()
    state["investment_plan"] = "Buy plan"
    state["risk_debate_state"].update(
        {
            "aggressive_risk_level": "LOW",
            "conservative_risk_level": "HIGH",
            "neutral_risk_level": "MEDIUM",
        }
    )
    result = create_portfolio_manager(_PMLLM())(state)
    new_risk_state = result["risk_debate_state"]
    assert new_risk_state["aggressive_risk_level"] == "LOW"
    assert new_risk_state["conservative_risk_level"] == "HIGH"
    assert new_risk_state["neutral_risk_level"] == "MEDIUM"
    assert new_risk_state["judge_decision"]  # still set correctly
