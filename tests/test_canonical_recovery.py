"""Canonical recovery: a Research (Bull/Bear) textual fallback must never
silently diverge from the canonical ``bull_thesis``/``bear_thesis`` state.

Before this, ``invoke_initial_thesis``/``invoke_review`` (see
``researchers/thesis_flow.py``) returned ``None`` for the structured dict on
any fallback, so a real correction the researcher stated in free text (a new
conviction, a revised stop) was invisible to everything downstream
(``recompute_trade_metrics``, the Research Manager, the Verifier, Risk's
frozen evidence, the Manager integrity check) -- while still being visible,
as prose, to a human or an LLM reading the debate history. Canonical recovery
(``researchers/canonical_recovery.py``) closes that gap with one small,
explicit, Pydantic-validated JSON patch request -- never a semantic regex
parser -- so a fallback either becomes real canonical state (APPLIED) or is
clearly marked as not having done so (UNAPPLIED), with the previous valid
canonical state preserved exactly.

Complements ``test_research_structured_thesis.py`` (structured happy path,
unaffected -- no recovery call is ever made there) and
``test_structured_repair.py`` (TEXT_ONLY repair non-degradation, unaffected
by this module except that a repair's fallback can now also become APPLIED
via a successful recovery, see the last two tests below).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tradingagents.agents.managers.research_manager import create_research_manager
from tradingagents.agents.researchers.bull_researcher import create_bull_researcher
from tradingagents.agents.researchers.canonical_recovery import RECOVERY_PROMPT_MARKER
from tradingagents.agents.researchers.research_repair import create_bull_repair
from tradingagents.agents.researchers.research_summary import render_research_context_summary
from tradingagents.agents.schemas import PortfolioRating, ResearchPlan
from tradingagents.graph.propagation import Propagator

_BULL_THESIS = {
    "direction": "BULL",
    "conviction": {
        "evidence_quality": 18, "internal_consistency": 21,
        "robustness_to_challenge": 17, "trade_plan_coherence": 19, "total": 75,
    },
    "horizon": "5-10 trading sessions",
    "entry": {"type": "point", "price": 100.0, "low": None, "high": None},
    "take_profit": 120.0,
    "stop_loss": 90.0,
    "suggested_risk_unit": "MEDIUM",
}


class _AlwaysFailsStructured:
    """Stands in for a real provider's structured-output attempt that comes
    back malformed (the Gemma/GPT-OSS cases this feature targets): the bind
    succeeds, but the call itself always raises, so the caller falls back to
    free text exactly like production does."""

    def invoke(self, prompt):
        raise ValueError("malformed structured output")


class _FallbackLLM:
    """A researcher whose structured attempt always fails. The free-text
    fallback and (when triggered) the canonical-recovery call both go through
    this same ``invoke``, like a real single-model provider would; the
    recovery request is distinguished only by its marker, never by position.
    """

    def __init__(self, fallback_text: str, recovery_json: str | None = None):
        self.fallback_text = fallback_text
        self.recovery_json = recovery_json
        self.prompts: list[str] = []

    def with_structured_output(self, schema):
        return _AlwaysFailsStructured()

    def invoke(self, prompt):
        self.prompts.append(prompt)
        if RECOVERY_PROMPT_MARKER in prompt:
            content = self.recovery_json if self.recovery_json is not None else "not a JSON object"
            return SimpleNamespace(content=content)
        return SimpleNamespace(content=self.fallback_text)


def _state():
    state = Propagator().create_initial_state("NVDA", "2026-09-01", atr_reference=100.0)
    state.update({
        "instrument_context": "INSTRUMENT", "market_report": "MARKET",
        "sentiment_report": "", "news_report": "", "fundamentals_report": "",
    })
    return state


def _state_with_thesis(**overrides):
    # Deep-ish copy: the nested conviction/entry dicts must not be the same
    # objects as _BULL_THESIS's, or an in-place mutation in one test (e.g.
    # bumping conviction.total for a regression scenario) would leak into
    # every other test that reuses this fixture.
    state = _state()
    thesis = {
        **_BULL_THESIS,
        "conviction": dict(_BULL_THESIS["conviction"]),
        "entry": dict(_BULL_THESIS["entry"]),
        **overrides.pop("thesis", {}),
    }
    state["investment_debate_state"].update({
        "bull_initial": "Bull Analyst (Initial): seed",
        "current_bull_response": "Bull Analyst (Initial): seed",
        "bull_thesis": thesis,
        "bull_conviction_history": [thesis["conviction"]["total"]],
        **overrides,
    })
    return state, thesis


def _apply(state, node):
    state["investment_debate_state"] = node(state)["investment_debate_state"]


# --- 1: INITIAL structured success -- no recovery, unaffected ---------------


@pytest.mark.unit
def test_structured_initial_success_sets_recovery_status_not_needed():
    from tradingagents.agents.schemas import InitialResearchThesis

    class _StructuredLLM:
        def with_structured_output(self, schema):
            return self

        def invoke(self, prompt):
            return InitialResearchThesis(
                direction="BULL",
                conviction={
                    "evidence_quality": 20, "internal_consistency": 20,
                    "robustness_to_challenge": 20, "trade_plan_coherence": 15,
                },
                horizon="5-10 sessions",
                entry={"type": "point", "price": 100.0},
                take_profit=120.0, stop_loss=90.0,
                thesis="Strong setup.",
            )

    state = _state()
    llm = _StructuredLLM()
    _apply(state, create_bull_researcher(llm, llm))
    debate = state["investment_debate_state"]
    assert debate["bull_thesis"]["conviction"]["total"] == 75
    assert debate["bull_initial_recovery_status"] == "NOT_NEEDED"


# --- 2: INITIAL fail + fallback + successful recovery -----------------------


@pytest.mark.unit
def test_initial_fallback_with_successful_recovery_creates_canonical_thesis():
    fallback_text = (
        "I believe this is a strong BULL setup. My conviction is 72. "
        "Entry around 100, take profit 120, stop loss 90."
    )
    recovery_json = (
        '{"direction": "BULL", "conviction_total": 72, '
        '"entry": {"type": "point", "price": 100, "low": null, "high": null}, '
        '"take_profit": 120, "stop_loss": 90}'
    )
    llm = _FallbackLLM(fallback_text, recovery_json)
    state = _state()
    _apply(state, create_bull_researcher(llm, llm))
    debate = state["investment_debate_state"]

    assert debate["bull_initial_recovery_status"] == "APPLIED"
    assert debate["bull_thesis"]["direction"] == "BULL"
    assert debate["bull_thesis"]["conviction"]["total"] == 72
    assert debate["bull_thesis"]["entry"]["price"] == 100
    assert debate["bull_thesis"]["take_profit"] == 120
    assert debate["bull_thesis"]["stop_loss"] == 90
    # Deterministic metrics now populate too, exactly as a real structured thesis would.
    assert debate["bull_trade_metrics"]["risk"] == pytest.approx(10)


# --- 3: INITIAL fail + fallback + failed recovery ----------------------------


@pytest.mark.unit
def test_initial_fallback_with_failed_recovery_leaves_canonical_unavailable():
    fallback_text = "Bullish overall, high conviction, but details to follow."
    llm = _FallbackLLM(fallback_text, recovery_json=None)
    state = _state()
    _apply(state, create_bull_researcher(llm, llm))
    debate = state["investment_debate_state"]

    assert debate["bull_initial_recovery_status"] == "UNAPPLIED"
    assert debate.get("bull_thesis") == {}
    # The full fallback text (with its narrative [NON-CANONICAL: ...] note)
    # is preserved in state for report/audit/debug -- see the Research
    # Manager/Verifier acceptance tests below for where it is actually kept
    # out of a downstream LLM's own context instead.
    assert "Bullish overall" in debate["bull_initial"]
    assert "NON-CANONICAL" in debate["bull_initial"]
    assert "Bullish overall" in debate["bull_history"]


# --- 4: REVIEW structured success -- current behavior intact ----------------


@pytest.mark.unit
def test_structured_review_success_sets_recovery_status_not_needed():
    from tradingagents.agents.schemas import ResearchReviewOutcome

    class _StructuredLLM:
        def with_structured_output(self, schema):
            return self

        def invoke(self, prompt):
            return ResearchReviewOutcome(
                conviction_before=75, conviction_after=70,
                trade_plan_changes={
                    "entry": {"action": "KEEP"}, "take_profit": {"action": "KEEP"},
                    "stop_loss": {"action": "KEEP"},
                },
            )

    state, _ = _state_with_thesis()
    llm = _StructuredLLM()
    _apply(state, create_bull_researcher(llm, llm))
    debate = state["investment_debate_state"]
    assert debate["bull_review_recovery_status"] == "NOT_NEEDED"
    assert debate["bull_conviction_history"] == [75, 70]


# --- 5 & 6: REVIEW fallback + successful recovery, twice in a row -----------


@pytest.mark.unit
def test_successive_review_fallback_recoveries_never_revert_to_an_earlier_conviction():
    state, _ = _state_with_thesis()

    # Round 1: canonical conviction starts at 57.
    state["investment_debate_state"]["bull_thesis"]["conviction"]["total"] = 57
    state["investment_debate_state"]["bull_conviction_history"] = [57]

    round1_llm = _FallbackLLM(
        "After reviewing the bear's point, I'm lowering my conviction from 57 to 52.",
        '{"conviction_after": 52, "trade_plan_changes": '
        '{"entry": {"action": "KEEP"}, "take_profit": {"action": "KEEP"}, '
        '"stop_loss": {"action": "KEEP"}}}',
    )
    _apply(state, create_bull_researcher(round1_llm, round1_llm))
    debate = state["investment_debate_state"]
    assert debate["bull_review_recovery_status"] == "APPLIED"
    assert debate["bull_conviction_history"] == [57, 52]

    round2_llm = _FallbackLLM(
        "Further lowering my conviction from 52 to 50 given weak follow-through.",
        '{"conviction_after": 50, "trade_plan_changes": '
        '{"entry": {"action": "KEEP"}, "take_profit": {"action": "KEEP"}, '
        '"stop_loss": {"action": "KEEP"}}}',
    )
    _apply(state, create_bull_researcher(round2_llm, round2_llm))
    debate = state["investment_debate_state"]
    assert debate["bull_review_recovery_status"] == "APPLIED"
    assert debate["bull_conviction_history"] == [57, 52, 50]
    assert debate["bull_conviction_history"][-1] == 50

    # Regression 11: the Research Context trail reflects the final value, not 57.
    summary = render_research_context_summary(debate)
    assert "57 -> 52 -> 50" in summary


# --- 7: REVIEW fallback patches entry/TP/SL exactly, no reconstruction ------


@pytest.mark.unit
def test_review_fallback_recovery_applies_trade_plan_patch_exactly():
    state, thesis = _state_with_thesis()

    fallback_text = (
        "Conviction stays at 75. I'm revising the stop loss down to 85 given "
        "the new support level, keeping entry and take profit unchanged."
    )
    recovery_json = (
        '{"conviction_after": 75, "trade_plan_changes": '
        '{"entry": {"action": "KEEP"}, "take_profit": {"action": "KEEP"}, '
        '"stop_loss": {"action": "REVISE", "new": 85}}}'
    )
    llm = _FallbackLLM(fallback_text, recovery_json)
    _apply(state, create_bull_researcher(llm, llm))
    debate = state["investment_debate_state"]

    assert debate["bull_review_recovery_status"] == "APPLIED"
    assert debate["bull_thesis"]["stop_loss"] == 85
    assert debate["bull_thesis"]["entry"]["price"] == thesis["entry"]["price"]
    assert debate["bull_thesis"]["take_profit"] == thesis["take_profit"]
    assert debate["bull_thesis"]["direction"] == thesis["direction"]
    assert 90 in debate["withdrawn_values"]


# --- 8: REVIEW fallback recovery invalid -- canonical stays identical -------


@pytest.mark.unit
def test_review_fallback_with_failed_recovery_leaves_canonical_unchanged():
    state, thesis = _state_with_thesis()

    fallback_text = "I'm adjusting my view somewhat based on the bear's points."
    llm = _FallbackLLM(fallback_text, recovery_json=None)
    _apply(state, create_bull_researcher(llm, llm))
    debate = state["investment_debate_state"]

    assert debate["bull_review_recovery_status"] == "UNAPPLIED"
    assert debate["bull_thesis"] == thesis
    assert debate["bull_conviction_history"] == [75]
    # The full fallback text is preserved in state (pending_bull_rebuttal,
    # bull_history) for report/audit/debug -- see the acceptance tests below
    # for where a downstream LLM consumer is kept from reading it.
    assert "adjusting my view" in debate["pending_bull_rebuttal"]
    assert "adjusting my view" in debate["bull_history"]


# --- 9: recovery introducing a value absent from the text is rejected ------


@pytest.mark.unit
def test_review_recovery_is_rejected_when_it_introduces_a_value_not_in_the_text():
    state, thesis = _state_with_thesis()

    fallback_text = "Conviction remains at 75, with stop loss unchanged, no material new evidence this round."
    # The recovered JSON hallucinates a stop-loss revision to a number that
    # never appears anywhere in the researcher's own text above.
    recovery_json = (
        '{"conviction_after": 75, "trade_plan_changes": '
        '{"entry": {"action": "KEEP"}, "take_profit": {"action": "KEEP"}, '
        '"stop_loss": {"action": "REVISE", "new": 123456}}}'
    )
    llm = _FallbackLLM(fallback_text, recovery_json)
    _apply(state, create_bull_researcher(llm, llm))
    debate = state["investment_debate_state"]

    assert debate["bull_review_recovery_status"] == "UNAPPLIED"
    assert debate["bull_thesis"]["stop_loss"] == thesis["stop_loss"]


# --- 10: research_repair's TEXT_ONLY semantics are not degraded ------------


@pytest.mark.unit
def test_repair_text_only_stays_non_degraded_when_recovery_fails():
    state, thesis = _state_with_thesis()

    fallback_text = "Still confident, stop now at 79000 given the new low."
    llm = _FallbackLLM(fallback_text, recovery_json=None)
    _apply(state, create_bull_repair(llm))
    debate = state["investment_debate_state"]

    assert debate["bull_repair_structured_status"] == "TEXT_ONLY"
    assert debate["bull_thesis"] == thesis


@pytest.mark.unit
def test_repair_becomes_applied_via_canonical_recovery_when_it_succeeds():
    """Rule 8's permitted improvement: repair shares ``invoke_review`` with a
    normal round, so a repair whose free text *can* be recovered legitimately
    becomes APPLIED -- it is not a degradation, since the old behavior never
    guessed either way; it just never got the chance to recover."""
    state, thesis = _state_with_thesis()

    fallback_text = "Conviction holds at 75. Revising stop loss to 85 per the verifier's correction."
    recovery_json = (
        '{"conviction_after": 75, "trade_plan_changes": '
        '{"entry": {"action": "KEEP"}, "take_profit": {"action": "KEEP"}, '
        '"stop_loss": {"action": "REVISE", "new": 85}}}'
    )
    llm = _FallbackLLM(fallback_text, recovery_json)
    _apply(state, create_bull_repair(llm))
    debate = state["investment_debate_state"]

    assert debate["bull_repair_structured_status"] == "APPLIED"
    assert debate["bull_thesis"]["stop_loss"] == 85
    assert thesis["stop_loss"] == 90  # the prior value, confirming it actually changed


# --- 12: regression -- a failed recovery never lets a new value leak in ----


@pytest.mark.unit
def test_regression_failed_review_recovery_keeps_research_context_at_prior_conviction():
    state, _ = _state_with_thesis()
    state["investment_debate_state"]["bull_thesis"]["conviction"]["total"] = 57
    state["investment_debate_state"]["bull_conviction_history"] = [57]

    fallback_text = "Lowering conviction to 50 given the bear's challenge."
    llm = _FallbackLLM(fallback_text, recovery_json=None)
    _apply(state, create_bull_researcher(llm, llm))
    debate = state["investment_debate_state"]

    assert debate["bull_review_recovery_status"] == "UNAPPLIED"
    assert debate["bull_conviction_history"] == [57]

    summary = render_research_context_summary(debate)
    assert "57" in summary
    assert "-> 50" not in summary
    assert "UNAPPLIED" in summary


# --- Fix 1: INITIAL canonical recovery must also recover suggested_risk_unit
# (and never infer it from conviction or ambiguous risk language -- see the
# Gemma3:26b/XOM regression tests further below)


@pytest.mark.unit
def test_initial_recovery_with_explicit_risk_unit_label_sets_canonical_medium():
    fallback_text = (
        "Strong BULL setup. Conviction: 70 (Medium-High). Entry 100, take "
        "profit 120, stop loss 90. Suggested Research Risk Unit: MEDIUM."
    )
    recovery_json = (
        '{"direction": "BULL", "conviction_total": 70, '
        '"entry": {"type": "point", "price": 100, "low": null, "high": null}, '
        '"take_profit": 120, "stop_loss": 90, "suggested_risk_unit": "MEDIUM"}'
    )
    llm = _FallbackLLM(fallback_text, recovery_json)
    state = _state()
    _apply(state, create_bull_researcher(llm, llm))
    debate = state["investment_debate_state"]

    assert debate["bull_initial_recovery_status"] == "APPLIED"
    assert debate["bull_thesis"]["suggested_risk_unit"] == "MEDIUM"


@pytest.mark.unit
def test_initial_recovery_with_qualitative_conviction_and_no_explicit_risk_unit():
    """Conviction expressed as "High" (plus its numeric self-assessment, as
    the system's own prompts require) recovers normally; with no explicit
    Suggested Research Risk Unit label anywhere, the canonical value stays
    unrecorded -- "high conviction" must never be read as "HIGH risk unit"."""
    fallback_text = (
        "Direction: Long. Conviction: High (85/100). Entry 160-163, take "
        "profit 168, stop loss 157."
    )
    recovery_json = (
        '{"direction": "BULL", "conviction_total": 85, '
        '"entry": {"type": "range", "price": null, "low": 160, "high": 163}, '
        '"take_profit": 168, "stop_loss": 157}'
    )
    llm = _FallbackLLM(fallback_text, recovery_json)
    state = _state()
    _apply(state, create_bull_researcher(llm, llm))
    debate = state["investment_debate_state"]

    assert debate["bull_initial_recovery_status"] == "APPLIED"
    assert debate["bull_thesis"]["conviction"]["total"] == 85
    assert debate["bull_thesis"]["suggested_risk_unit"] == "not recorded"


@pytest.mark.unit
def test_xom_regression_high_conviction_fallback_does_not_invent_high_risk_unit():
    """Regression for the Gemma3:26b nightly stress-test finding: an XOM Bull
    initial fallback stated direction/conviction/entry/TP/SL but no Suggested
    Research Risk Unit at all -- recovery must not invent HIGH from "high
    conviction"."""
    fallback_text = (
        "Direction: Long. Conviction: High (82/100). Entry: 160-163. "
        "TP: 168. SL: 157."
    )
    # The recovery LLM reports a numeric conviction derived from the
    # qualitative "High" judgment (as the normal structured path also would),
    # but the fallback text itself never labels a Suggested Research Risk
    # Unit -- the recovered JSON must not get to invent one either.
    recovery_json = (
        '{"direction": "BULL", "conviction_total": 82, '
        '"entry": {"type": "range", "price": null, "low": 160, "high": 163}, '
        '"take_profit": 168, "stop_loss": 157}'
    )
    llm = _FallbackLLM(fallback_text, recovery_json)
    state = _state()
    _apply(state, create_bull_researcher(llm, llm))
    debate = state["investment_debate_state"]

    assert debate["bull_initial_recovery_status"] == "APPLIED"
    assert debate["bull_thesis"]["direction"] == "BULL"
    assert debate["bull_thesis"]["entry"]["low"] == 160
    assert debate["bull_thesis"]["entry"]["high"] == 163
    assert debate["bull_thesis"]["take_profit"] == 168
    assert debate["bull_thesis"]["stop_loss"] == 157
    assert debate["bull_thesis"]["suggested_risk_unit"] != "HIGH"
    assert debate["bull_thesis"]["suggested_risk_unit"] == "not recorded"


@pytest.mark.unit
def test_initial_recovery_ambiguous_high_risk_wording_is_not_inferred_as_risk_unit():
    """A recovered JSON that hallucinates "HIGH" from ambiguous prose ("high
    risk given volatility") rather than an explicit Suggested Research Risk
    Unit label must still be rejected for that field specifically -- the rest
    of the initial recovery stays APPLIED."""
    fallback_text = (
        "Strong BULL setup with high risk given volatility. Conviction: 75. "
        "Entry 100, take profit 120, stop loss 90."
    )
    recovery_json = (
        '{"direction": "BULL", "conviction_total": 75, '
        '"entry": {"type": "point", "price": 100, "low": null, "high": null}, '
        '"take_profit": 120, "stop_loss": 90, "suggested_risk_unit": "HIGH"}'
    )
    llm = _FallbackLLM(fallback_text, recovery_json)
    state = _state()
    _apply(state, create_bull_researcher(llm, llm))
    debate = state["investment_debate_state"]

    assert debate["bull_initial_recovery_status"] == "APPLIED"
    assert debate["bull_thesis"]["conviction"]["total"] == 75
    assert debate["bull_thesis"]["suggested_risk_unit"] == "not recorded"


@pytest.mark.unit
def test_initial_recovery_with_invalid_risk_unit_is_rejected():
    fallback_text = (
        "Strong BULL setup. Conviction is 70. Entry 100, take profit 120, "
        "stop loss 90. Extremely aggressive sizing warranted here."
    )
    # "ULTRA_HIGH" is not a member of SuggestedRiskUnit -- the whole patch must
    # be rejected, exactly like any other malformed field.
    recovery_json = (
        '{"direction": "BULL", "conviction_total": 70, '
        '"entry": {"type": "point", "price": 100, "low": null, "high": null}, '
        '"take_profit": 120, "stop_loss": 90, "suggested_risk_unit": "ULTRA_HIGH"}'
    )
    llm = _FallbackLLM(fallback_text, recovery_json)
    state = _state()
    _apply(state, create_bull_researcher(llm, llm))
    debate = state["investment_debate_state"]

    assert debate["bull_initial_recovery_status"] == "UNAPPLIED"
    assert debate.get("bull_thesis") == {}


@pytest.mark.unit
def test_structured_initial_happy_path_suggested_risk_unit_still_derived_not_chosen():
    """Unaffected by the new recoverable field: the structured happy path
    already ignores whatever the LLM puts in ``suggested_risk_unit`` and
    derives it from conviction (see ``test_suggested_risk_unit_is_not_chosen_
    by_the_llm_but_derived`` in test_research_structured_thesis.py)."""
    from tradingagents.agents.schemas import InitialResearchThesis

    class _StructuredLLM:
        def with_structured_output(self, schema):
            return self

        def invoke(self, prompt):
            return InitialResearchThesis(
                direction="BULL",
                conviction={
                    "evidence_quality": 20, "internal_consistency": 20,
                    "robustness_to_challenge": 20, "trade_plan_coherence": 15,
                },
                horizon="5-10 sessions",
                entry={"type": "point", "price": 100.0},
                take_profit=120.0, stop_loss=90.0,
                thesis="Strong setup.",
                suggested_risk_unit="NO_TRADE",  # must be overridden by derivation
            )

    state = _state()
    llm = _StructuredLLM()
    _apply(state, create_bull_researcher(llm, llm))
    debate = state["investment_debate_state"]
    assert debate["bull_initial_recovery_status"] == "NOT_NEEDED"
    assert debate["bull_thesis"]["suggested_risk_unit"] == "MEDIUM"


# --- Fix 2: an UNAPPLIED fallback must not leak semantic content downstream


class _ManagerLLM:
    def __init__(self):
        self.prompts: list = []

    def with_structured_output(self, _schema):
        return self

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return ResearchPlan(recommendation=PortfolioRating.HOLD, rationale="x", strategic_actions="y")


@pytest.mark.unit
def test_acceptance_a_failed_review_recovery_hides_semantic_content_from_research_manager():
    state, _ = _state_with_thesis()
    state["investment_debate_state"]["bull_thesis"]["conviction"]["total"] = 57
    state["investment_debate_state"]["bull_conviction_history"] = [57]

    fallback_text = "Lowering conviction to 50, also raising take profit to 240 given momentum."
    llm = _FallbackLLM(fallback_text, recovery_json=None)  # recovery fails
    _apply(state, create_bull_researcher(llm, llm))
    debate = state["investment_debate_state"]

    assert debate["bull_review_recovery_status"] == "UNAPPLIED"
    assert debate["bull_conviction_history"] == [57]
    # State keeps the full fallback text verbatim (pending_bull_rebuttal,
    # bull_history) for report/audit/debug.
    assert "Lowering conviction to 50" in debate["bull_history"]
    assert "Lowering conviction to 50" in debate["pending_bull_rebuttal"]

    # Simulate Bear's turn promoting the round's (full) text into
    # current_bull_response, exactly as bear_researcher.py does
    # ("current_bull_response": pending_bull_rebuttal) -- the redaction this
    # test is about happens when Research Manager reads it, not before.
    debate["current_bull_response"] = debate["pending_bull_rebuttal"]

    manager_llm = _ManagerLLM()
    create_research_manager(manager_llm)(state)
    prompt = manager_llm.prompts[0]

    assert "50" not in prompt
    assert "240" not in prompt
    assert "NON-CANONICAL FALLBACK UNAPPLIED" in prompt
    # The canonical conviction trail (57, unchanged) is still the only
    # authoritative conviction value the manager's prompt presents.
    assert "57" in prompt


@pytest.mark.unit
def test_acceptance_b_successful_review_recovery_keeps_narrative_visible_downstream():
    state, _ = _state_with_thesis()
    state["investment_debate_state"]["bull_thesis"]["conviction"]["total"] = 57
    state["investment_debate_state"]["bull_conviction_history"] = [57]

    fallback_text = "Lowering my conviction from 57 to 50 given the bear's challenge."
    recovery_json = (
        '{"conviction_after": 50, "trade_plan_changes": '
        '{"entry": {"action": "KEEP"}, "take_profit": {"action": "KEEP"}, '
        '"stop_loss": {"action": "KEEP"}}}'
    )
    llm = _FallbackLLM(fallback_text, recovery_json)
    _apply(state, create_bull_researcher(llm, llm))
    debate = state["investment_debate_state"]

    assert debate["bull_review_recovery_status"] == "APPLIED"
    assert debate["bull_conviction_history"] == [57, 50]
    # APPLIED narrative may stay visible -- it is already represented by the
    # updated canonical state, so it is no longer an un-anchored claim.
    assert "Lowering my conviction" in debate["pending_bull_rebuttal"]

    debate["current_bull_response"] = debate["pending_bull_rebuttal"]

    manager_llm = _ManagerLLM()
    create_research_manager(manager_llm)(state)
    prompt = manager_llm.prompts[0]

    assert "Lowering my conviction" in prompt
