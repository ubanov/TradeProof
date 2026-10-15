"""Regression for a real NVDA/2026-09-30 incident: a Bull/Bear review LLM's
tool call used the key ``"trade_plan"`` (a string-encoded JSON blob) where
the contract -- unchanged in the tool schema actually sent to the model,
confirmed from the trace's own ``request.json`` -- names the field
``trade_plan_changes`` (a nested object). Pydantic's default
``extra="ignore"`` silently dropped the unrecognized key AND silently
defaulted the real ``trade_plan_changes`` to "nothing changed" (it has a
``default_factory``): the call reported total success while a genuine
KEEP/REVISE/WITHDRAW the researcher actually stated was lost with no error,
no warning, and nothing to recover from.

Root cause: the model deviated from an already-correct contract -- not a
prompt wording problem (neither ``researchers/review_first.txt`` nor
``review_followup.txt`` mentions "trade_plan"/"trade_plan_changes" in prose
at all; the only place the exact field name is communicated is the
auto-generated tool schema itself, and that schema already named it
correctly). No alias from ``trade_plan`` to ``trade_plan_changes`` was
added -- that would paper over a model deviation from a correct contract.

Fix: ``ResearchReviewOutcome``/``TradePlanChanges``/``TradePlanFieldChange``/
``ReviewCanonicalPatch`` now set ``extra="forbid"``. The SAME deviation that
used to validate silently now raises inside ``thesis_flow.invoke_review``'s
own structured-call try block, which already catches exactly this
(``except Exception``) and routes through the EXISTING fallback ->
canonical-recovery path (no new machinery): a fresh free-text call from the
same model, then ``canonical_recovery.recover_review_patch`` over that text.
The researcher's real intent gets one more, auditable chance
(``CanonicalRecoveryStatus.APPLIED``/``UNAPPLIED``) to become canonical
state -- never silently accepted as complete when it wasn't.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from tradingagents.agents.researchers.canonical_recovery import (
    RECOVERY_PROMPT_MARKER,
    CanonicalRecoveryStatus,
)
from tradingagents.agents.researchers.thesis_flow import invoke_review
from tradingagents.agents.schemas import (
    ReviewCanonicalPatch,
    TradePlanAction,
    TradePlanChanges,
    TradePlanFieldChange,
)

_PRIOR_THESIS = {
    "direction": "BULL",
    "conviction": {
        "evidence_quality": 20, "internal_consistency": 18,
        "robustness_to_challenge": 16, "trade_plan_coherence": 20, "total": 74,
    },
    "horizon": "5-10 trading sessions",
    "entry": {"type": "range", "price": None, "low": 225.45, "high": 229.0},
    "take_profit": 236.0,
    "stop_loss": 221.0,
    "suggested_risk_unit": "MEDIUM",
}

# The real malformed tool-call shape (abbreviated from the actual trace):
# a top-level "trade_plan" string where "trade_plan_changes" (a nested
# object) belongs.
_REAL_MALFORMED_ARGS = {
    "accepted": ["Accepted the cash-conversion critique."],
    "rejected": [],
    "unresolved": [],
    "arithmetic_corrections": [],
    "trade_plan": (
        '{"take_profit": {"action": "REVISE", "old": 217.23, "new": 211.56, '
        '"reason": "superseded"}, "stop_loss": {"action": "KEEP"}, '
        '"entry": {"action": "KEEP"}}'
    ),
    "conviction_before": 65,
    "conviction_after": 57,
    "conviction_reason": "Accepted several Bull points.",
    "new_data_exceptions": [],
    "remaining_disagreements": [],
}


# --- Schema-level: the wrong key is rejected, never silently accepted -------


@pytest.mark.unit
def test_1_trade_plan_wrong_key_is_rejected_not_silently_dropped():
    """This is the exact shape the real run's malformed tool call had --
    must raise, never validate as if trade_plan_changes had simply been
    omitted (KEEP everything)."""
    from tradingagents.agents.schemas import ResearchReviewOutcome

    with pytest.raises(ValidationError, match="trade_plan"):
        ResearchReviewOutcome.model_validate(_REAL_MALFORMED_ARGS)


@pytest.mark.unit
def test_2_a_correctly_keyed_call_is_unaffected():
    from tradingagents.agents.schemas import ResearchReviewOutcome

    valid_args = dict(_REAL_MALFORMED_ARGS)
    del valid_args["trade_plan"]
    valid_args["trade_plan_changes"] = {
        "entry": {"action": "KEEP"},
        "take_profit": {"action": "REVISE", "old": 217.23, "new": 211.56, "reason": "superseded"},
        "stop_loss": {"action": "KEEP"},
    }
    result = ResearchReviewOutcome.model_validate(valid_args)
    assert result.trade_plan_changes.take_profit.action is TradePlanAction.REVISE
    assert result.trade_plan_changes.take_profit.new == 211.56


@pytest.mark.unit
def test_3_sibling_contract_schemas_also_forbid_extra_fields():
    """The same contractual family -- a mis-keyed field one level deeper
    (inside trade_plan_changes itself) must be equally impossible to lose
    silently."""
    with pytest.raises(ValidationError):
        TradePlanFieldChange.model_validate({"action": "KEEP", "old_value": 221.0})
    with pytest.raises(ValidationError):
        TradePlanChanges.model_validate({"entry": {}, "stop": {}})  # "stop" instead of "stop_loss"
    with pytest.raises(ValidationError):
        ReviewCanonicalPatch.model_validate({"conviction_after": 50, "trade_plan": "bogus"})


# --- invoke_review-level: the chosen policy is explicit recovery, never a
# silent "validation succeeded" ------------------------------------------


class _WrongKeyStructuredLLM:
    """Stands in for generation_llm.with_structured_output(ResearchReviewOutcome):
    the FIRST (structured) attempt returns the real malformed shape -- a
    plain dict LangChain handed back for invoke_review's own explicit
    model_validate() to validate, exactly like thesis_flow.py's own
    defensive `if not isinstance(result, ResearchReviewOutcome)` branch
    anticipates."""

    def invoke(self, prompt, **kwargs):
        return dict(_REAL_MALFORMED_ARGS)


class _FallbackThenRecoverLLM:
    """The plain_llm used for BOTH the free-text fallback (once the
    structured attempt above fails validation) and the canonical-recovery
    bookkeeping call that follows it."""

    def __init__(self, fallback_text: str, recovery_json: str | None):
        self.fallback_text = fallback_text
        self.recovery_json = recovery_json
        self.calls = 0

    def invoke(self, prompt, **kwargs):
        self.calls += 1
        if RECOVERY_PROMPT_MARKER in prompt:
            return SimpleNamespace(content=self.recovery_json if self.recovery_json is not None else "not json")
        return SimpleNamespace(content=self.fallback_text)


@pytest.mark.unit
def test_4_wrong_key_falls_back_and_recovers_explicitly_and_auditably():
    """The chosen policy, end to end: a mis-keyed structured call is never
    silently accepted as complete -- it is caught, falls back to a fresh
    free-text generation, and that text's real intent (the take-profit
    revision) is recovered explicitly, with CanonicalRecoveryStatus.APPLIED
    making the recovery auditable (never a quiet default)."""
    fallback_text = (
        "I accept the cash-conversion critique. I'm revising my take-profit "
        "from 217.23 to 211.56, since that level is no longer defensible. "
        "Stop loss and entry are unchanged. Conviction: 65 -> 57."
    )
    recovery_json = (
        '{"conviction_after": 57, "trade_plan_changes": {'
        '"entry": {"action": "KEEP", "new": null}, '
        '"take_profit": {"action": "REVISE", "new": 211.56}, '
        '"stop_loss": {"action": "KEEP", "new": null}}}'
    )
    llm = _FallbackThenRecoverLLM(fallback_text, recovery_json)

    rendered, review_dict, status = invoke_review(
        _WrongKeyStructuredLLM(), llm, "prompt", "Bull Researcher", prior_thesis=_PRIOR_THESIS,
    )

    assert status == CanonicalRecoveryStatus.APPLIED.value
    assert review_dict is not None
    assert review_dict["trade_plan_changes"]["take_profit"]["action"] == "REVISE"
    assert review_dict["trade_plan_changes"]["take_profit"]["new"] == 211.56
    # Never the malformed structured call's own content trusted directly --
    # the canonical state came from the fresh fallback + recovery, not from
    # salvaging anything out of the rejected "trade_plan" string.
    assert llm.calls == 2  # one fallback generation, one recovery bookkeeping call


@pytest.mark.unit
def test_5_wrong_key_when_recovery_also_fails_is_explicitly_unapplied_never_silently_ok():
    """The other half of the policy: if recovery ALSO cannot reconstruct a
    trustworthy patch, the result is explicitly UNAPPLIED (and the
    rendered text carries the NON-CANONICAL marker) -- never silently
    treated as if the structured call had actually succeeded."""
    fallback_text = "Ambiguous free text with no clear structured intent."
    llm = _FallbackThenRecoverLLM(fallback_text, recovery_json=None)  # recovery returns invalid JSON

    rendered, review_dict, status = invoke_review(
        _WrongKeyStructuredLLM(), llm, "prompt", "Bull Researcher", prior_thesis=_PRIOR_THESIS,
    )

    assert status == CanonicalRecoveryStatus.UNAPPLIED.value
    assert review_dict is None
    assert "NON-CANONICAL" in rendered


@pytest.mark.unit
def test_6_the_wrong_key_text_itself_is_never_silently_promoted_to_canonical_state():
    """Structural guard against the original bug's exact shape: even though
    the first (rejected) attempt's raw args technically CONTAIN the
    researcher's real intent (as a string blob under "trade_plan"), that
    content must never be read directly as canonical state -- only the
    fresh fallback text, explicitly recovered, may become canonical."""
    llm = _FallbackThenRecoverLLM("Completely different fallback text.", recovery_json=None)
    _, review_dict, status = invoke_review(
        _WrongKeyStructuredLLM(), llm, "prompt", "Bull Researcher", prior_thesis=_PRIOR_THESIS,
    )
    assert status == CanonicalRecoveryStatus.UNAPPLIED.value
    assert review_dict is None  # the malformed call's own "211.56" is nowhere in canonical state
