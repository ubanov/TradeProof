"""Independent-thesis and paired cross-review semantics for the research debate."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tradingagents.agents.managers.research_manager import create_research_manager
from tradingagents.agents.researchers.bear_researcher import create_bear_researcher
from tradingagents.agents.researchers.bull_researcher import create_bull_researcher
from tradingagents.agents.researchers.canonical_recovery import RECOVERY_PROMPT_MARKER
from tradingagents.agents.schemas import PortfolioRating, ResearchPlan
from tradingagents.graph.conditional_logic import ConditionalLogic
from tradingagents.graph.propagation import Propagator


class _ResearchLLM:
    """No ``with_structured_output``, so every call here is a Research textual
    fallback: ``invoke_initial_thesis``/``invoke_review`` (see
    ``researchers/thesis_flow.py``) follow every such call with one canonical-
    recovery call on this same object, asking for plain JSON. This fake
    answers that JSON request with plain, non-JSON prose (it is never valid
    recovery JSON on purpose -- these tests are about the free-text content
    itself, not canonical recovery), and keeps it out of ``self.prompts``/
    ``self.responses`` indexing so every assertion below keeps addressing the
    real per-round content call it always has.
    """

    def __init__(self, prefix: str, responses: list[str] | None = None):
        self.prefix = prefix
        self.responses = responses or []
        self.prompts: list[str] = []
        self.recovery_prompts: list[str] = []

    def invoke(self, prompt):
        if RECOVERY_PROMPT_MARKER in prompt:
            self.recovery_prompts.append(prompt)
            return SimpleNamespace(content="not a recovery JSON object")
        self.prompts.append(prompt)
        index = len(self.prompts) - 1
        content = (
            self.responses[index]
            if index < len(self.responses)
            else f"{self.prefix}_{len(self.prompts)}"
        )
        return SimpleNamespace(content=content)


class _ManagerLLM:
    def __init__(self):
        self.prompts: list[str] = []

    def with_structured_output(self, _schema):
        return self

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return ResearchPlan(
            recommendation=PortfolioRating.HOLD,
            rationale="Balanced evidence.",
            strategic_actions="Maintain the current allocation.",
        )


def _state():
    state = Propagator().create_initial_state("NVDA", "2026-09-01")
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


def _run_initial_phase(state, bull_llm, bear_llm):
    _apply(state, create_bull_researcher(bull_llm, bull_llm))
    _apply(state, create_bear_researcher(bear_llm, bear_llm))


@pytest.mark.unit
def test_initial_contexts_are_independent_and_receive_all_reports():
    state = _state()
    bull_llm = _ResearchLLM("BULL_SECRET")
    bear_llm = _ResearchLLM("BEAR_SECRET")

    _apply(state, create_bull_researcher(bull_llm, bull_llm))
    assert "BEAR_SECRET" not in bull_llm.prompts[0]

    # Bull's output is now in graph state, but Bear's initial prompt builder does
    # not pass any Bull field to the LLM. Independence is enforced by context
    # construction rather than an instruction to ignore visible text.
    assert "BULL_SECRET_1" in str(state["investment_debate_state"])
    _apply(state, create_bear_researcher(bear_llm, bear_llm))
    assert "BULL_SECRET_1" not in bear_llm.prompts[0]

    for prompt in (bull_llm.prompts[0], bear_llm.prompts[0]):
        for report in (
            "INSTRUMENT_REPORT",
            "MARKET_REPORT",
            "SENTIMENT_REPORT",
            "NEWS_REPORT",
            "FUNDAMENTALS_REPORT",
        ):
            assert report in prompt


@pytest.mark.unit
def test_reviews_use_only_the_previous_completed_phase_or_round():
    state = _state()
    bull_llm = _ResearchLLM("BULL")
    bear_llm = _ResearchLLM("BEAR")
    bull = create_bull_researcher(bull_llm, bull_llm)
    bear = create_bear_researcher(bear_llm, bear_llm)
    _run_initial_phase(state, bull_llm, bear_llm)

    _apply(state, bull)
    assert "BEAR_1" in bull_llm.prompts[1]
    _apply(state, bear)
    assert "BULL_1" in bear_llm.prompts[1]
    assert "BULL_2" not in bear_llm.prompts[1]

    _apply(state, bull)
    assert "BEAR_2" in bull_llm.prompts[2]
    _apply(state, bear)
    assert "BULL_2" in bear_llm.prompts[2]
    assert "BULL_3" not in bear_llm.prompts[2]

    _apply(state, bull)
    assert "BEAR_3" in bull_llm.prompts[3]
    assert "BEAR_2" not in bull_llm.prompts[3]
    _apply(state, bear)
    assert "BULL_3" in bear_llm.prompts[3]
    assert "BULL_4" not in bear_llm.prompts[3]
    assert "BULL_2" not in bear_llm.prompts[3]


@pytest.mark.unit
def test_bull_and_bear_review_prompts_are_collaborative_but_distinct():
    state = _state()
    bull_llm = _ResearchLLM("BULL")
    bear_llm = _ResearchLLM("BEAR")
    bull = create_bull_researcher(bull_llm, bull_llm)
    bear = create_bear_researcher(bear_llm, bear_llm)
    _run_initial_phase(state, bull_llm, bear_llm)

    _apply(state, bull)
    _apply(state, bear)

    for prompt in (bull_llm.prompts[1], bear_llm.prompts[1]):
        assert "complementary researchers" in prompt
        assert "You are not trying to win" in prompt
        assert "Changing your mind is not losing" in prompt
        assert "lowering conviction after valid counterevidence" in prompt
        assert "Do not defend a weak claim" in prompt
        assert "genuinely unresolved" in prompt
        assert "unrelated mistake" in prompt
        assert "What did the other researcher identify" in prompt

    assert "bullish-perspective member" in bull_llm.prompts[1]
    assert "bearish/reduce-risk-perspective member" in bear_llm.prompts[1]


@pytest.mark.unit
def test_research_manager_receives_initial_theses_and_rebuttal_history():
    """``_ResearchLLM`` never achieves structured output or canonical
    recovery (see its docstring), so every one of its turns is an UNAPPLIED
    textual fallback: Research Manager's own context construction must show
    only the bare UNAPPLIED marker for them, never this raw, un-anchored
    prose (see test_canonical_recovery.py's acceptance tests for the
    APPLIED/structured-success counterexamples where real content does
    reach the manager)."""
    state = _state()
    bull_llm = _ResearchLLM("BULL")
    bear_llm = _ResearchLLM("BEAR")
    _run_initial_phase(state, bull_llm, bear_llm)
    _apply(state, create_bull_researcher(bull_llm, bull_llm))
    _apply(state, create_bear_researcher(bear_llm, bear_llm))

    manager_llm = _ManagerLLM()
    create_research_manager(manager_llm)(state)
    prompt = manager_llm.prompts[0]
    for marker in ("BULL_1", "BEAR_1", "BULL_2", "BEAR_2"):
        assert marker not in prompt
    assert "NON-CANONICAL FALLBACK UNAPPLIED" in prompt
    assert "complementary research roles" in prompt
    assert "reduced conviction after valid counterevidence" in prompt
    assert "unresolved disagreements as useful information" in prompt
    assert "not which side argued more aggressively" in prompt


@pytest.mark.unit
def test_accepted_correction_persists_in_later_reviews_and_manager_context():
    correction = "ACCEPTED: corrected subtraction is 3,126.56"
    state = _state()
    bull_llm = _ResearchLLM("BULL", ["bull initial", correction, "bull final"])
    bear_llm = _ResearchLLM("BEAR", ["bear initial", correction, "bear final"])
    bull = create_bull_researcher(bull_llm, bull_llm)
    bear = create_bear_researcher(bear_llm, bear_llm)
    _run_initial_phase(state, bull_llm, bear_llm)

    _apply(state, bull)
    _apply(state, bear)
    assert state["investment_debate_state"]["review_outcomes"] == correction

    _apply(state, bull)
    assert correction in bull_llm.prompts[2]
    _apply(state, bear)

    manager_llm = _ManagerLLM()
    create_research_manager(manager_llm)(state)
    manager_prompt = manager_llm.prompts[0]
    # The classified correction persists into the manager's context either
    # way (review_outcomes is a classification signal, not raw fallback
    # prose). But bull/bear's own initial/final text here never achieved
    # canonical recovery (_ResearchLLM, see its docstring), so Research
    # Manager's context construction shows only the UNAPPLIED marker for
    # those, never this raw, un-anchored prose.
    assert correction in manager_prompt
    assert "NON-CANONICAL FALLBACK UNAPPLIED" in manager_prompt
    assert "bull initial" not in manager_prompt
    assert "bear initial" not in manager_prompt
    assert "bull final" not in manager_prompt
    assert "bear final" not in manager_prompt


@pytest.mark.unit
@pytest.mark.parametrize("rounds", [1, 2, 3])
def test_configured_value_is_the_number_of_cross_review_rounds(rounds):
    state = _state()
    bull_llm = _ResearchLLM("BULL")
    bear_llm = _ResearchLLM("BEAR")
    nodes = {
        "Bull Researcher": create_bull_researcher(bull_llm, bull_llm),
        "Bear Researcher": create_bear_researcher(bear_llm, bear_llm),
    }
    logic = ConditionalLogic(max_debate_rounds=rounds)
    target = "Bull Researcher"

    while target != "Research Verifier":
        _apply(state, nodes[target])
        target = logic.should_continue_debate(state)

    debate = state["investment_debate_state"]
    assert len(bull_llm.prompts) == 1 + rounds
    assert len(bear_llm.prompts) == 1 + rounds
    assert debate["count"] == 2 + 2 * rounds
    assert debate["bull_rebuttal_count"] == rounds
    assert debate["bear_rebuttal_count"] == rounds
    assert debate["debate_round"] == rounds
