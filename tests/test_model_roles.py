"""Model roles: Generation vs Review -- routing verification (see fork
README: "Model roles: Generation vs Review").

These tests execute real nodes/flows with recording doubles and assert WHICH
LLM object actually received invoke() -- not just that the wiring exists
statically. ``_NamedLLM`` has no ``with_structured_output``, so every
structured-capable call site falls back to free text (the path every
provider without structured support already takes), and it answers
``bind_tools()`` by returning itself, so it also serves the three tool-loop
analysts. One simple fake covers the whole routing matrix.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from tradingagents.agents.analysts.fundamentals_analyst import create_fundamentals_analyst
from tradingagents.agents.analysts.market_analyst import create_market_analyst
from tradingagents.agents.analysts.news_analyst import create_news_analyst
from tradingagents.agents.analysts.sentiment_analyst import create_sentiment_analyst
from tradingagents.agents.managers.portfolio_manager import create_portfolio_manager
from tradingagents.agents.managers.research_manager import create_research_manager
from tradingagents.agents.researchers.bear_researcher import create_bear_researcher
from tradingagents.agents.researchers.bull_researcher import create_bull_researcher
from tradingagents.agents.researchers.research_repair import create_bear_repair, create_bull_repair
from tradingagents.agents.researchers.research_verifier import create_research_verifier
from tradingagents.agents.risk_mgmt.aggressive_debator import create_aggressive_debator
from tradingagents.agents.risk_mgmt.conservative_debator import create_conservative_debator
from tradingagents.agents.risk_mgmt.neutral_debator import create_neutral_debator
from tradingagents.agents.trader.trader import create_trader
from tradingagents.graph.propagation import Propagator


class _NamedLLM:
    """No with_structured_output -- forces the free-text fallback everywhere
    structured output would otherwise be tried; bind_tools() returns self,
    serving the tool-loop analysts too. Records how many times THIS object
    was invoked, so a test can tell generation_llm and review_llm apart."""

    def __init__(self, name="llm"):
        self.name = name
        self.calls = 0

    def bind_tools(self, tools, **kwargs):
        return self

    def invoke(self, prompt, **kwargs):
        self.calls += 1
        return SimpleNamespace(content=f"{self.name} response", tool_calls=[])


def _base_state(**overrides):
    state = Propagator().create_initial_state("NVDA", "2026-09-01")
    state.update(
        {
            "market_report": "MARKET", "sentiment_report": "SENTIMENT",
            "news_report": "NEWS", "fundamentals_report": "FUNDAMENTALS",
            "trader_investment_plan": "Buy, entry 100, stop 95.",
        }
    )
    state.update(overrides)
    return state


# --- 1-4: the four analysts -> generation -----------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "factory", [create_market_analyst, create_news_analyst, create_fundamentals_analyst]
)
def test_tool_loop_analyst_uses_generation(factory):
    generation = _NamedLLM("gen")
    factory(generation)(_base_state())
    assert generation.calls == 1


@pytest.mark.unit
def test_sentiment_analyst_uses_generation(monkeypatch):
    import tradingagents.agents.analysts.sentiment_analyst as sentiment_mod

    monkeypatch.setattr(sentiment_mod, "get_news", type("T", (), {"func": staticmethod(lambda *a, **k: "")}))
    monkeypatch.setattr(sentiment_mod, "fetch_stocktwits_messages", lambda *a, **k: "")
    monkeypatch.setattr(sentiment_mod, "fetch_reddit_posts", lambda *a, **k: "")
    monkeypatch.setattr(sentiment_mod, "jev_screen", lambda *a, **k: None)

    generation = _NamedLLM("gen")
    state = _base_state(company_of_interest="NVDA")
    create_sentiment_analyst(generation)(state)
    assert generation.calls == 1


# --- 5-8: Bull/Bear initial -> generation, review -> review -----------------


@pytest.mark.unit
@pytest.mark.parametrize("factory", [create_bull_researcher, create_bear_researcher])
def test_researcher_initial_uses_generation_not_review(factory):
    generation, review = _NamedLLM("gen"), _NamedLLM("rev")
    factory(generation, review)(_base_state())
    assert generation.calls >= 1
    assert review.calls == 0


@pytest.mark.unit
@pytest.mark.parametrize(
    "factory, side_key, initial_key, current_key",
    [
        (create_bull_researcher, "bull", "bull_initial", "current_bull_response"),
        (create_bear_researcher, "bear", "bear_initial", "current_bear_response"),
    ],
)
def test_researcher_review_uses_review_not_generation(factory, side_key, initial_key, current_key):
    generation, review = _NamedLLM("gen"), _NamedLLM("rev")
    state = _base_state()
    state["investment_debate_state"].update({
        initial_key: f"{side_key} initial", current_key: f"{side_key} initial",
        "current_bull_response": "bull initial", "current_bear_response": "bear initial",
    })
    factory(generation, review)(state)
    assert review.calls >= 1
    assert generation.calls == 0


# --- 9, 10: Research Verifier (structured + JSON fallback) -> review -------


@pytest.mark.unit
def test_research_verifier_uses_review_for_both_attempts():
    generation, review = _NamedLLM("gen"), _NamedLLM("rev")
    state = _base_state()
    state["investment_debate_state"].update({
        "bull_initial": "Bull Analyst (Initial): thesis", "bear_initial": "Bear Analyst (Initial): thesis",
        "current_bull_response": "Bull Analyst (Initial): thesis",
        "current_bear_response": "Bear Analyst (Initial): thesis",
    })
    create_research_verifier(review)(state)
    assert review.calls >= 1
    assert generation.calls == 0


# --- 11, 12: Research repair Bull/Bear -> review ----------------------------


@pytest.mark.unit
@pytest.mark.parametrize("factory", [create_bull_repair, create_bear_repair])
def test_research_repair_uses_review(factory):
    generation, review = _NamedLLM("gen"), _NamedLLM("rev")
    state = _base_state()
    state["investment_debate_state"].update({
        "bull_initial": "Bull Analyst (Initial): thesis", "bear_initial": "Bear Analyst (Initial): thesis",
        "current_bull_response": "Bull Analyst (Initial): thesis",
        "current_bear_response": "Bear Analyst (Initial): thesis",
        "verification_history": [{"status": "FAIL", "findings": []}],
    })
    factory(review)(state)
    assert review.calls >= 1
    assert generation.calls == 0


# --- 13, 14: Research Manager (structured/fallback) -> generation -----------


@pytest.mark.unit
def test_research_manager_uses_generation():
    generation, review = _NamedLLM("gen"), _NamedLLM("rev")
    state = _base_state()
    state["investment_debate_state"].update({
        "bull_initial": "Bull Analyst (Initial): thesis", "bear_initial": "Bear Analyst (Initial): thesis",
    })
    create_research_manager(generation)(state)
    assert generation.calls >= 1
    assert review.calls == 0


# --- 15: Trader (structured/fallback) -> generation -------------------------


@pytest.mark.unit
def test_trader_uses_generation():
    generation, review = _NamedLLM("gen"), _NamedLLM("rev")
    state = _base_state(investment_plan="Plan", company_of_interest="NVDA")
    create_trader(generation)(state)
    assert generation.calls >= 1
    assert review.calls == 0


# --- 16-18: Risk Aggressive/Conservative/Neutral initial -> generation -----


@pytest.mark.unit
@pytest.mark.parametrize(
    "factory", [create_aggressive_debator, create_conservative_debator, create_neutral_debator]
)
def test_risk_initial_assessment_uses_generation(factory):
    generation, review = _NamedLLM("gen"), _NamedLLM("rev")
    factory(generation, review)(_base_state())
    assert generation.calls >= 1
    assert review.calls == 0


# --- 19-20: Risk cross-review -> review -------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "factory, key",
    [
        (create_aggressive_debator, "aggressive_initial"),
        (create_conservative_debator, "conservative_initial"),
        (create_neutral_debator, "neutral_initial"),
    ],
)
def test_risk_cross_review_uses_review_not_generation(factory, key):
    generation, review = _NamedLLM("gen"), _NamedLLM("rev")
    state = _base_state()
    state["risk_debate_state"][key] = "already recorded"
    factory(generation, review)(state)
    assert review.calls >= 1
    assert generation.calls == 0


# --- 21: Portfolio Manager (structured/fallback) -> generation --------------


@pytest.mark.unit
def test_portfolio_manager_uses_generation():
    generation, review = _NamedLLM("gen"), _NamedLLM("rev")
    state = _base_state(investment_plan="Plan", trader_investment_plan="Trader plan")
    state["risk_debate_state"].update({
        "history": "h", "aggressive_history": "", "conservative_history": "", "neutral_history": "",
    })
    create_portfolio_manager(generation)(state)
    assert generation.calls >= 1
    assert review.calls == 0


# --- 22, 23, 24: canonical recovery inherits the role of the call it recovers


@pytest.mark.unit
def test_canonical_recovery_from_initial_fallback_uses_generation():
    generation, review = _NamedLLM("gen"), _NamedLLM("rev")
    create_bull_researcher(generation, review)(_base_state())
    # One fallback call + one canonical-recovery call, both on generation_llm.
    assert generation.calls == 2
    assert review.calls == 0


@pytest.mark.unit
def test_canonical_recovery_from_review_fallback_uses_review():
    generation, review = _NamedLLM("gen"), _NamedLLM("rev")
    state = _base_state()
    state["investment_debate_state"].update({
        "bull_initial": "Bull Analyst (Initial): thesis", "current_bull_response": "Bull Analyst (Initial): thesis",
        "current_bear_response": "Bear Analyst (Initial): thesis",
    })
    create_bull_researcher(generation, review)(state)
    assert generation.calls == 0
    assert review.calls == 2


@pytest.mark.unit
def test_canonical_recovery_from_repair_fallback_uses_review():
    generation, review = _NamedLLM("gen"), _NamedLLM("rev")
    state = _base_state()
    state["investment_debate_state"].update({
        "bull_initial": "Bull Analyst (Initial): thesis",
        "current_bull_response": "Bull Analyst (Initial): thesis",
        "current_bear_response": "Bear Analyst (Initial): thesis",
    })
    create_bull_repair(review)(state)
    assert generation.calls == 0
    assert review.calls == 2


# --- 27, 28: generation_llm and review_llm may be the same or different ----


@pytest.mark.unit
def test_generation_and_review_can_be_the_same_object():
    shared = _NamedLLM("shared")
    create_bull_researcher(shared, shared)(_base_state())
    assert shared.calls > 0


@pytest.mark.unit
def test_generation_and_review_can_be_different_objects():
    generation, review = _NamedLLM("gen"), _NamedLLM("rev")
    create_bull_researcher(generation, review)(_base_state())
    assert generation.calls > 0
    assert review.calls == 0
