"""Structured Bull/Bear initial thesis and review flow, end to end with fake LLMs.

Complements ``test_research_debate.py`` (free-text fallback path, unchanged)
by exercising the structured-output path: a fake LLM that implements
``with_structured_output`` and returns parsed schema instances, the way a
real provider does.
"""

from __future__ import annotations

import pytest

from tradingagents.agents.researchers.bear_researcher import create_bear_researcher
from tradingagents.agents.researchers.bull_researcher import create_bull_researcher
from tradingagents.agents.schemas import (
    ConvictionScore,
    EntryLevel,
    EntryLevelType,
    InitialResearchThesis,
    ResearchDirection,
    ResearchReviewOutcome,
    SuggestedRiskUnit,
)
from tradingagents.graph.propagation import Propagator


class _StructuredLLM:
    """Returns queued schema instances; mimics ``with_structured_output``.

    A node binds two schemas (initial thesis, review) from the same LLM
    instance; both bindings share this one queue/prompt log so outputs pop in
    the exact order the node invokes them.
    """

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.prompts: list[str] = []

    def with_structured_output(self, schema):
        return self

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return self.outputs.pop(0)


def _thesis(direction, total=75, entry_low=82900, entry_high=83300, tp=87360, sl=80920):
    quarter = total // 4
    remainder = total - quarter * 4
    return InitialResearchThesis(
        direction=direction,
        conviction=ConvictionScore(
            evidence_quality=quarter + remainder,
            internal_consistency=quarter,
            robustness_to_challenge=quarter,
            trade_plan_coherence=quarter,
        ),
        horizon="5-10 trading sessions",
        entry=EntryLevel(type=EntryLevelType.RANGE, low=entry_low, high=entry_high),
        take_profit=tp,
        stop_loss=sl,
        thesis="A structured initial thesis.",
        evidence=["Price above SMA50"],
        risks=["Macro event risk"],
        data_gaps=["No options-flow data supplied"],
    )


def _state():
    state = Propagator().create_initial_state(
        "NVDA", "2026-09-01", setup_tags={"RSI_BUCKET": "RSI_50_65"}, atr_reference=2192.41
    )
    state.update(
        {
            "instrument_context": "INSTRUMENT_REPORT",
            "market_report": "MARKET_REPORT",
            "sentiment_report": "SENTIMENT_REPORT",
            "news_report": "NEWS_REPORT",
            "fundamentals_report": "FUNDAMENTALS_REPORT",
        }
    )
    return state


def _apply(state, node):
    state["investment_debate_state"] = node(state)["investment_debate_state"]


@pytest.mark.unit
def test_initial_thesis_populates_structured_fields_and_deterministic_metrics():
    state = _state()
    bull_llm = _StructuredLLM([_thesis(ResearchDirection.BULL)])
    _apply(state, create_bull_researcher(bull_llm, bull_llm))

    debate = state["investment_debate_state"]
    assert debate["bull_thesis"]["direction"] == "BULL"
    assert debate["bull_conviction_history"] == [75]
    assert debate["bull_thesis"]["suggested_risk_unit"] == "MEDIUM"

    metrics = debate["bull_trade_metrics"]
    assert metrics["entry_reference"] == pytest.approx(83100)
    assert metrics["risk"] == pytest.approx(2180)
    assert metrics["reward"] == pytest.approx(4260)
    assert metrics["stop_atr"] == pytest.approx(2180 / 2192.41)

    # The rendered markdown (what the rest of the system reads as prose) still
    # carries the structured content, so downstream free-text consumers keep working.
    assert "BULL" in debate["bull_initial"]
    assert "NOT a probability" in debate["bull_initial"]


@pytest.mark.unit
def test_bull_and_bear_initial_theses_are_independent_even_when_structured():
    state = _state()
    bull_llm = _StructuredLLM([_thesis(ResearchDirection.BULL)])
    bear_llm = _StructuredLLM([_thesis(ResearchDirection.BEAR, total=40)])

    _apply(state, create_bull_researcher(bull_llm, bull_llm))
    assert "BEAR" not in bull_llm.prompts[0]

    _apply(state, create_bear_researcher(bear_llm, bear_llm))
    assert "BULL" not in bear_llm.prompts[0]

    debate = state["investment_debate_state"]
    assert debate["bull_thesis"]["direction"] == "BULL"
    assert debate["bear_thesis"]["direction"] == "BEAR"
    assert debate["bull_conviction_history"] == [75]
    assert debate["bear_conviction_history"] == [40]


@pytest.mark.unit
def test_both_sides_receive_the_identical_shared_setup_tags_and_horizon():
    state = _state()
    bull_llm = _StructuredLLM([_thesis(ResearchDirection.BULL)])
    bear_llm = _StructuredLLM([_thesis(ResearchDirection.BEAR)])
    _apply(state, create_bull_researcher(bull_llm, bull_llm))
    _apply(state, create_bear_researcher(bear_llm, bear_llm))

    for prompt in (bull_llm.prompts[0], bear_llm.prompts[0]):
        assert "RSI_BUCKET: RSI_50_65" in prompt
        assert "5-10 trading sessions" in prompt


def _review(conviction_before, conviction_after, sl_action="KEEP", sl_new=None, tp_action="KEEP", tp_new=None):
    trade_plan_changes = {
        "entry": {"action": "KEEP"},
        "take_profit": {"action": tp_action, "new": tp_new, "reason": "test"},
        "stop_loss": {"action": sl_action, "new": sl_new, "reason": "test"},
    }
    return ResearchReviewOutcome(
        accepted=["A valid correction"],
        conviction_before=conviction_before,
        conviction_after=conviction_after,
        conviction_reason="Material evidence changed.",
        trade_plan_changes=trade_plan_changes,
    )


@pytest.mark.unit
def test_review_can_revise_stop_loss_and_recompute_deterministic_metrics():
    state = _state()
    bull_llm = _StructuredLLM(
        [_thesis(ResearchDirection.BULL), _review(75, 68, sl_action="REVISE", sl_new=76860)]
    )
    bear_llm = _StructuredLLM([_thesis(ResearchDirection.BEAR)])
    bull = create_bull_researcher(bull_llm, bull_llm)
    bear = create_bear_researcher(bear_llm, bear_llm)

    _apply(state, bull)
    _apply(state, bear)
    _apply(state, bull)  # review round 1

    debate = state["investment_debate_state"]
    assert debate["bull_thesis"]["stop_loss"] == 76860
    assert debate["bull_conviction_history"] == [75, 68]
    metrics = debate["bull_trade_metrics"]
    assert metrics["risk"] == pytest.approx(abs(83100 - 76860))


@pytest.mark.unit
def test_review_can_withdraw_take_profit():
    state = _state()
    bull_llm = _StructuredLLM(
        [_thesis(ResearchDirection.BULL), _review(75, 75, tp_action="WITHDRAW")]
    )
    bear_llm = _StructuredLLM([_thesis(ResearchDirection.BEAR)])
    _apply(state, create_bull_researcher(bull_llm, bull_llm))
    _apply(state, create_bear_researcher(bear_llm, bear_llm))
    _apply(state, create_bull_researcher(bull_llm, bull_llm))

    debate = state["investment_debate_state"]
    assert debate["bull_thesis"]["take_profit"] is None
    assert debate["bull_trade_metrics"]["reward"] is None


@pytest.mark.unit
def test_withdrawn_trade_plan_level_is_recorded_in_debate_state():
    state = _state()
    bull_llm = _StructuredLLM(
        [_thesis(ResearchDirection.BULL), _review(75, 75, tp_action="WITHDRAW")]
    )
    bear_llm = _StructuredLLM([_thesis(ResearchDirection.BEAR)])
    _apply(state, create_bull_researcher(bull_llm, bull_llm))
    _apply(state, create_bear_researcher(bear_llm, bear_llm))
    _apply(state, create_bull_researcher(bull_llm, bull_llm))

    assert state["investment_debate_state"]["withdrawn_values"] == [87360]


@pytest.mark.unit
def test_review_can_keep_all_trade_plan_fields_unchanged():
    state = _state()
    bull_llm = _StructuredLLM([_thesis(ResearchDirection.BULL), _review(75, 75)])
    bear_llm = _StructuredLLM([_thesis(ResearchDirection.BEAR)])
    _apply(state, create_bull_researcher(bull_llm, bull_llm))
    _apply(state, create_bear_researcher(bear_llm, bear_llm))
    _apply(state, create_bull_researcher(bull_llm, bull_llm))

    debate = state["investment_debate_state"]
    assert debate["bull_thesis"]["stop_loss"] == 80920
    assert debate["bull_thesis"]["take_profit"] == 87360
    assert debate["bull_conviction_history"] == [75, 75]


@pytest.mark.unit
def test_nvda_2026_01_07_regression_invalid_short_stop_revision_is_rejected():
    """End-to-end replay of the real NVDA 2026-01-07 bug through the actual
    bear_node, not just the underlying function: a BEAR thesis (valid SHORT
    geometry, stop above entry) whose second review round proposes REVISE-ing
    the stop to a level BELOW entry (the exact LONG-side "2x ATR below the
    close" reasoning mistakenly applied to a SHORT) must never reach
    bear_thesis.stop_loss, and the rejection must be visible in state."""
    state = _state()
    bear_llm = _StructuredLLM([
        _thesis(ResearchDirection.BEAR, entry_low=189.0, entry_high=190.5, tp=180.6, sl=195.2),
        _review(62, 58),  # review round 1: no trade-plan change
        _review(58, 62, sl_action="REVISE", sl_new=177.8),  # review round 2: the bug
    ])
    bear = create_bear_researcher(bear_llm, bear_llm)

    _apply(state, bear)  # initial
    _apply(state, bear)  # review round 1
    _apply(state, bear)  # review round 2 -- the invalid REVISE

    debate = state["investment_debate_state"]
    assert debate["bear_thesis"]["stop_loss"] == 195.2
    assert debate["bear_thesis"]["stop_loss"] != 177.8
    assert debate["bear_thesis"]["take_profit"] == 180.6
    rejections = debate["trade_geometry_rejections"]
    assert len(rejections) == 1
    assert rejections[0]["side"] == "bear"
    assert "177.8" in rejections[0]["claim"]
    # The conviction update from the SAME review round is unaffected by the
    # trade-plan rejection -- only entry/stop_loss/take_profit revert. (75 is
    # _thesis()'s own default initial conviction total.)
    assert debate["bear_conviction_history"] == [75, 58, 62]


@pytest.mark.unit
def test_new_data_exception_is_recorded_and_tagged_by_side_and_round():
    state = _state()
    review_with_exception = ResearchReviewOutcome(
        conviction_before=70,
        conviction_after=70,
        new_data_exceptions=[
            {
                "datum": "Unscheduled macro print",
                "reason": "Corrects a material misreading of the prior CPI figure",
                "source": "News report",
            }
        ],
    )
    bull_llm = _StructuredLLM([_thesis(ResearchDirection.BULL, total=70), review_with_exception])
    bear_llm = _StructuredLLM([_thesis(ResearchDirection.BEAR)])
    _apply(state, create_bull_researcher(bull_llm, bull_llm))
    _apply(state, create_bear_researcher(bear_llm, bear_llm))
    _apply(state, create_bull_researcher(bull_llm, bull_llm))

    exceptions = state["investment_debate_state"]["new_data_exceptions"]
    assert len(exceptions) == 1
    assert exceptions[0]["side"] == "bull"
    assert exceptions[0]["round"] == 1
    assert exceptions[0]["datum"] == "Unscheduled macro print"


@pytest.mark.unit
def test_review_outcome_lines_still_feed_the_existing_review_outcomes_extractor():
    """The structured path must stay compatible with review_outcomes.py's regex-based
    extraction, since the Research Manager and Verifier still read review_outcomes as text."""
    state = _state()
    bull_llm = _StructuredLLM([_thesis(ResearchDirection.BULL), _review(75, 68)])
    bear_llm = _StructuredLLM(
        [_thesis(ResearchDirection.BEAR), _review(60, 60)]
    )
    _apply(state, create_bull_researcher(bull_llm, bull_llm))
    _apply(state, create_bear_researcher(bear_llm, bear_llm))
    _apply(state, create_bull_researcher(bull_llm, bull_llm))
    _apply(state, create_bear_researcher(bear_llm, bear_llm))

    assert "ACCEPTED: A valid correction" in state["investment_debate_state"]["review_outcomes"]


@pytest.mark.unit
def test_suggested_risk_unit_is_not_chosen_by_the_llm_but_derived():
    thesis = _thesis(ResearchDirection.BULL, total=30)
    assert thesis.suggested_risk_unit is SuggestedRiskUnit.NO_TRADE


@pytest.mark.unit
def test_research_manager_receives_conviction_trail_setup_tags_and_trade_metrics():
    from tradingagents.agents.managers.research_manager import create_research_manager
    from tradingagents.agents.schemas import PortfolioRating, ResearchPlan

    class _ManagerLLM:
        def __init__(self):
            self.prompts = []

        def with_structured_output(self, _schema):
            return self

        def invoke(self, prompt):
            self.prompts.append(prompt)
            return ResearchPlan(
                recommendation=PortfolioRating.HOLD,
                rationale="Balanced.",
                strategic_actions="Hold.",
            )

    state = _state()
    bull_llm = _StructuredLLM([_thesis(ResearchDirection.BULL), _review(75, 68)])
    bear_llm = _StructuredLLM([_thesis(ResearchDirection.BEAR, total=40), _review(40, 40)])
    _apply(state, create_bull_researcher(bull_llm, bull_llm))
    _apply(state, create_bear_researcher(bear_llm, bear_llm))
    _apply(state, create_bull_researcher(bull_llm, bull_llm))
    _apply(state, create_bear_researcher(bear_llm, bear_llm))

    manager_llm = _ManagerLLM()
    create_research_manager(manager_llm)(state)
    prompt = manager_llm.prompts[0]

    assert "5-10 trading sessions" in prompt
    assert "RSI_BUCKET: RSI_50_65" in prompt
    assert "75 -> 68" in prompt
    assert "MEDIUM" in prompt  # Bull's suggested risk unit at conviction 75
    assert "NOT a probability of the market outcome" in prompt
    assert "not a portfolio percentage" in prompt
    assert "Entry (reference): 83100.00" in prompt
    # _thesis() uses a genuine entry range: the Research Manager must receive
    # the deterministic range metrics too, not just the reference entry's R/R.
    assert "R/R across entry range" in prompt


@pytest.mark.unit
def test_verifier_receives_deterministic_trade_metrics():
    from tradingagents.agents.researchers.research_verifier import create_research_verifier
    from tradingagents.agents.schemas import ResearchVerification, VerificationStatus

    class _VerifierLLM:
        def __init__(self):
            self.prompts = []

        def with_structured_output(self, _schema):
            return self

        def invoke(self, prompt):
            self.prompts.append(prompt)
            return ResearchVerification(status=VerificationStatus.PASS)

    state = _state()
    bull_llm = _StructuredLLM([_thesis(ResearchDirection.BULL)])
    bear_llm = _StructuredLLM([_thesis(ResearchDirection.BEAR)])
    _apply(state, create_bull_researcher(bull_llm, bull_llm))
    _apply(state, create_bear_researcher(bear_llm, bear_llm))

    verifier_llm = _VerifierLLM()
    _apply(state, create_research_verifier(verifier_llm))

    prompt = verifier_llm.prompts[0]
    assert "Entry (reference): 83100.00" in prompt
    # _thesis() uses a genuine entry range (82900-83300): the verifier must
    # see both the reference R/R and the deterministic range, never just the
    # midpoint's ratio presented as if it were "the plan's R/R".
    assert "Reference R/R (at reference entry)" in prompt
    assert "R/R across entry range" in prompt
