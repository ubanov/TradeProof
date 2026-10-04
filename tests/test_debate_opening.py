"""The first speaker in each debate must not rebut a nonexistent argument (#1176).

Each debate round's opening speaker receives an empty opponent response; the
prompt used to interpolate it into a "refute the opponent" instruction, so models
fabricated the other side's position. Bull/Bear now substitute an explicit
opening marker when the opponent has not spoken, and pass a real argument
through unchanged.

The three Risk reviewers no longer use this opponent/history mechanism at all
(see fork README: "Risk as independent assessment + one cross-review") --
their independence and cross-review properties are covered in
``test_risk_independent_assessment.py`` instead.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from tradingagents.agents.context import opponent_argument_or_opening
from tradingagents.agents.researchers.bear_researcher import create_bear_researcher
from tradingagents.agents.researchers.bull_researcher import create_bull_researcher

_REPORTS = {
    "company_of_interest": "AAPL", "asset_type": "stock",
    "market_report": "m", "sentiment_report": "s",
    "news_report": "n", "fundamentals_report": "f",
}


def _capturing_llm(captured: dict):
    """``llm.with_structured_output`` auto-succeeds on a bare ``MagicMock``
    (returning another mock rather than raising), so the structured attempt
    always "succeeds" into unparseable output and falls back to free text --
    which now also triggers one canonical-recovery call on this same object
    (see ``researchers/thesis_flow.py``). The real content-generation prompt
    is always the first call recorded here; recovery's own small JSON-only
    prompt follows it, also captured for tests that want to inspect it."""
    llm = MagicMock()
    llm.invoke.side_effect = lambda prompt: (
        captured.setdefault("prompts", []).append(prompt) or MagicMock(content="argument")
    )
    return llm


def _investment_state(current_response):
    return {
        **_REPORTS,
        "count": 0,
        "investment_debate_state": {
            "history": "", "bull_history": "", "bear_history": "",
            "current_response": current_response, "count": 0,
        },
    }


# --- shared helper ----------------------------------------------------------

@pytest.mark.unit
def test_helper_marks_empty_and_passes_through():
    assert "has not spoken yet" in opponent_argument_or_opening("", "bear analyst")
    assert opponent_argument_or_opening("  real point ", "bear") == "real point"


# --- researchers ------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize("factory", [create_bull_researcher, create_bear_researcher])
def test_researcher_initial_thesis_has_no_phantom_opponent(factory):
    captured = {}
    factory(_capturing_llm(captured))(_investment_state(""))
    assert "independent initial thesis" in captured["prompts"][0]


@pytest.mark.unit
def test_researcher_passes_real_opponent_argument():
    captured = {}
    state = _investment_state("Bear Analyst: valuation is stretched")
    state["investment_debate_state"].update({
        "bull_initial": "Bull Analyst (Initial): growth is durable",
        "bear_initial": "Bear Analyst (Initial): valuation is stretched",
        "current_bull_response": "Bull Analyst (Initial): growth is durable",
        "current_bear_response": "Bear Analyst (Initial): valuation is stretched",
        "rebuttal_history": "",
        "debate_round": 0,
    })
    create_bull_researcher(_capturing_llm(captured))(state)
    assert "valuation is stretched" in captured["prompts"][0]
