"""Canonical structured recovery for a Validator-accepted repair (see
``tradingagents/agents/structured_recovery.py`` and ``validate_and_repair``
in ``validator.py``).

Real-run regression (NVDA/2026-09-30): the Research Manager's validation
repair (call 033) and the Portfolio Manager's free-text fallback (call 045)
both produced Markdown rather than their expected schema's JSON shape; both
were subsequently accepted (PASS) on recheck, yet
``structured.research_plan``/``structured.portfolio_decision`` ended up
``null`` in ``result.json`` -- and with ``portfolio_decision`` null,
``RecommendationRecord`` could never be written at all
(``build_recommendation_record`` requires it). ``validate_and_repair`` had
exactly two ways to populate ``final_parsed`` after a repair -- a direct
``model_validate_json``, then a dollar-escaped ``json.loads`` -- and no
further attempt once a repair genuinely came back as prose. This module
adds exactly one more attempt, generic across any ``expected_schema``:
canonical recovery, same architecture as
``researchers/canonical_recovery.py``.

Complements ``test_validator.py`` (the full PASS/WARN/FAIL/repair state
machine, unaffected except that a free-text repair can now also recover a
structured object) and ``test_canonical_recovery.py`` (the Bull/Bear-
specific sibling mechanism, untouched by this change).
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from tradingagents.agents.managers.portfolio_manager import create_portfolio_manager
from tradingagents.agents.managers.research_manager import create_research_manager
from tradingagents.agents.schemas import (
    PortfolioDecision,
    ResearchPlan,
    TraderProposal,
    render_pm_decision,
    render_research_plan,
    render_trader_proposal,
)
from tradingagents.agents.structured_recovery import (
    STRUCTURED_RECOVERY_PROMPT_MARKER,
    recover_structured_from_text,
)
from tradingagents.agents.validation_schemas import ValidationResult, ValidationStatus
from tradingagents.agents.validator import validate_and_repair
from tradingagents.graph.propagation import Propagator

# --- Shared fakes -------------------------------------------------------------


class _EnabledValidator:
    """validator_llm standing in for validation_enabled=True that always
    reports PASS, mirroring test_validator.py's own ``_enabled_validator``
    -- the deterministic_check_fn under test drives FAIL/repair, not this."""

    def with_structured_output(self, _schema, include_raw=False):
        return self

    def invoke(self, prompt, seed=None):
        return {
            "raw": SimpleNamespace(content=ValidationResult(status=ValidationStatus.PASS, findings=[]).model_dump_json()),
            "parsed": ValidationResult(status=ValidationStatus.PASS, findings=[]),
            "parsing_error": None,
        }


class _RepairThenRecoverLLM:
    """original_llm for a repair that comes back as free text, followed by
    a canonical-recovery call -- both routed through this one ``invoke``,
    like a real single-model provider. Distinguished only by the recovery
    marker, never by call position."""

    def __init__(self, repair_text: str, recovery_json: str | None):
        self.repair_text = repair_text
        self.recovery_json = recovery_json
        self.calls = 0

    def invoke(self, prompt, **kwargs):
        self.calls += 1
        if STRUCTURED_RECOVERY_PROMPT_MARKER in prompt:
            return SimpleNamespace(content=self.recovery_json if self.recovery_json is not None else "not json")
        return SimpleNamespace(content=self.repair_text)


# --- Unit-level: recover_structured_from_text ---------------------------------


@pytest.mark.unit
def test_recover_structured_from_text_recovers_a_research_plan():
    fallback_text = (
        "**Recommendation**: Overweight\n"
        "**Rationale**: Bull case stronger after verification.\n"
        "**Strategic Actions**: Buy on confirmation."
    )
    llm = _RepairThenRecoverLLM(fallback_text, json.dumps({
        "recommendation": "Overweight",
        "rationale": "Bull case stronger after verification.",
        "strategic_actions": "Buy on confirmation.",
    }))
    result = recover_structured_from_text(fallback_text, ResearchPlan, llm, "Research Manager")
    assert isinstance(result, ResearchPlan)
    assert result.recommendation.value == "Overweight"


@pytest.mark.unit
def test_recover_structured_from_text_returns_none_on_invalid_json():
    llm = _RepairThenRecoverLLM("free text", "not a JSON object")
    result = recover_structured_from_text("free text", ResearchPlan, llm, "Research Manager")
    assert result is None


@pytest.mark.unit
def test_recover_structured_from_text_returns_none_on_schema_mismatch():
    llm = _RepairThenRecoverLLM("free text", json.dumps({"not_a_real_field": 1}))
    result = recover_structured_from_text("free text", ResearchPlan, llm, "Research Manager")
    assert result is None


@pytest.mark.unit
def test_recover_structured_from_text_rejects_a_hallucinated_number():
    """A number in the recovered object that is NOT present in the fallback
    text itself must be rejected -- recovery can only bookkeep what the
    stage's own accepted text already says, never add a value."""
    fallback_text = "**Action**: Buy\n**Reasoning**: Clean plan, no explicit stop."
    llm = _RepairThenRecoverLLM(fallback_text, json.dumps({
        "action": "Buy", "reasoning": "Clean plan, no explicit stop.", "stop_loss": 90.0,
    }))
    result = recover_structured_from_text(fallback_text, TraderProposal, llm, "Trader")
    assert result is None


@pytest.mark.unit
def test_recover_structured_from_text_accepts_a_value_present_in_the_text():
    fallback_text = "**Action**: Buy\n**Reasoning**: Entry near 100.0, stop at 90.0."
    llm = _RepairThenRecoverLLM(fallback_text, json.dumps({
        "action": "Buy", "reasoning": "Entry near 100.0, stop at 90.0.",
        "entry_price": 100.0, "stop_loss": 90.0,
    }))
    result = recover_structured_from_text(fallback_text, TraderProposal, llm, "Trader")
    assert isinstance(result, TraderProposal)
    assert result.entry_price == 100.0
    assert result.stop_loss == 90.0


# --- validate_and_repair: Research Manager canonical recovery -----------------


@pytest.mark.unit
def test_a1_research_manager_repair_free_text_recovers_to_research_plan():
    """RM structured -> validator FAIL -> repair free text -> canonical
    recovery -> ResearchPlan final structured (task section A, case 1)."""
    repair_text = (
        "**Recommendation**: Overweight\n"
        "**Rationale**: Bull case stronger after verification.\n"
        "**Strategic Actions**: Buy on confirmation."
    )
    recovery_json = json.dumps({
        "recommendation": "Overweight",
        "rationale": "Bull case stronger after verification.",
        "strategic_actions": "Buy on confirmation.",
    })
    llm = _RepairThenRecoverLLM(repair_text, recovery_json)

    outcome = validate_and_repair(
        stage="Research Manager", validator_llm=_EnabledValidator(), original_llm=llm,
        original_role="generation", call_identity_base="research_manager",
        instructions="x", evidence_context="x",
        candidate_text="stale candidate text with an unsupported number 9999.99",
        candidate_parsed=None, expected_schema=ResearchPlan, render_fn=render_research_plan,
        # The deterministic check itself is irrelevant to this test (an
        # _EnabledValidator PASS alone forces FAIL to be deterministic-only,
        # which this stub never contributes) -- forced FAIL once, then PASS.
        deterministic_check_fn=_fail_once_then_pass(),
    )
    assert outcome.repair_attempted is True
    assert outcome.final_parsed is not None
    assert isinstance(outcome.final_parsed, ResearchPlan)
    assert outcome.final_parsed.recommendation.value == "Overweight"
    assert outcome.final_text == render_research_plan(outcome.final_parsed)
    assert outcome.status != ValidationStatus.FAIL.value  # the recovered text was accepted


def _fail_once_then_pass():
    from tradingagents.agents.validation_schemas import ValidationCategory, ValidationFinding

    calls = {"n": 0}

    def check(text, parsed):
        calls["n"] += 1
        if calls["n"] == 1:
            return [ValidationFinding(
                category=ValidationCategory.UNSUPPORTED_NUMBER, claim="9999.99", reason="not sourced",
            )]
        return []

    return check


@pytest.mark.unit
def test_a3_canonical_recovery_failure_leaves_final_parsed_null():
    """Canonical recovery failure (hallucinated/invalid) must still leave
    final_parsed null -- never a half-trusted structured object."""
    repair_text = "**Recommendation**: Overweight\n**Rationale**: x\n**Strategic Actions**: y"
    llm = _RepairThenRecoverLLM(repair_text, "not a JSON object at all")

    outcome = validate_and_repair(
        stage="Research Manager", validator_llm=_EnabledValidator(), original_llm=llm,
        original_role="generation", call_identity_base="research_manager",
        instructions="x", evidence_context="x",
        candidate_text="stale candidate text with an unsupported number 9999.99",
        candidate_parsed=None, expected_schema=ResearchPlan, render_fn=render_research_plan,
        deterministic_check_fn=_fail_once_then_pass(),
    )
    assert outcome.repair_attempted is True
    assert outcome.final_parsed is None
    assert outcome.final_text == repair_text  # the free text itself, never fabricated structure


# --- validate_and_repair: Portfolio Manager canonical recovery ----------------


@pytest.mark.unit
def test_a2_portfolio_manager_repair_free_text_recovers_to_portfolio_decision():
    """PM structured failure -> free-text fallback -> validator PASS ->
    canonical recovery -> PortfolioDecision final structured (task section
    A, case 2)."""
    repair_text = (
        "**Rating**: Overweight\n**Disposition**: KEEP\n"
        "**Executive Summary**: Clean summary after repair.\n"
        "**Investment Thesis**: Clean thesis after repair."
    )
    recovery_json = json.dumps({
        "rating": "Overweight", "disposition": "KEEP",
        "executive_summary": "Clean summary after repair.",
        "investment_thesis": "Clean thesis after repair.",
    })
    llm = _RepairThenRecoverLLM(repair_text, recovery_json)

    outcome = validate_and_repair(
        stage="Portfolio Manager", validator_llm=_EnabledValidator(), original_llm=llm,
        original_role="generation", call_identity_base="portfolio_manager",
        instructions="x", evidence_context="x",
        candidate_text="stale candidate with an unsupported number 9999.99",
        candidate_parsed=None, expected_schema=PortfolioDecision, render_fn=render_pm_decision,
        deterministic_check_fn=_fail_once_then_pass(),
    )
    assert outcome.final_parsed is not None
    assert isinstance(outcome.final_parsed, PortfolioDecision)
    assert outcome.final_parsed.rating.value == "Overweight"
    assert outcome.final_parsed.disposition.value == "KEEP"


# --- Trader: unaffected by this change ----------------------------------------


@pytest.mark.unit
def test_a4_trader_direct_json_repair_is_unaffected_no_extra_call():
    """The already-working case (a repair that comes back as valid JSON
    directly) must take exactly the same path as before this change --
    canonical recovery must never even be attempted, so no extra LLM call
    happens for a stage that was never broken."""
    repair_json_text = json.dumps({
        "action": "Buy", "reasoning": "Repaired reasoning.", "entry_price": 100.0, "stop_loss": 90.0,
    })
    llm = _RepairThenRecoverLLM(repair_json_text, recovery_json=None)

    outcome = validate_and_repair(
        stage="Trader", validator_llm=_EnabledValidator(), original_llm=llm,
        original_role="generation", call_identity_base="trader",
        instructions="x", evidence_context="x",
        candidate_text="stale candidate with an unsupported number 9999.99",
        candidate_parsed=None, expected_schema=TraderProposal, render_fn=render_trader_proposal,
        deterministic_check_fn=_fail_once_then_pass(),
    )
    assert llm.calls == 1  # only the repair call -- recovery never attempted
    assert isinstance(outcome.final_parsed, TraderProposal)
    assert outcome.final_parsed.entry_price == 100.0


# --- End-to-end node-level: structured.* is populated, not null --------------


_BAD_RM_PLAN = (
    "**Recommendation**: Buy\n"
    "**Rationale**: Reduce exposure if price exceeds 9999.99.\n"
    "**Strategic Actions**: Enter per plan."
)


class _NodeRepairThenRecoverLLM:
    """Candidate generation (free text, deliberately bad) -> repair (free
    text, clean but still not JSON) -> canonical recovery (JSON) -- all
    through the research/portfolio manager's own single ``generation_llm``,
    exactly like a real single-model provider."""

    def __init__(self, repair_text: str, recovery_json: str):
        self.repair_text = repair_text
        self.recovery_json = recovery_json
        self.calls = 0

    def invoke(self, prompt, **kwargs):
        self.calls += 1
        if STRUCTURED_RECOVERY_PROMPT_MARKER in prompt:
            return SimpleNamespace(content=self.recovery_json)
        if self.calls == 1:
            return SimpleNamespace(content=_BAD_RM_PLAN)
        return SimpleNamespace(content=self.repair_text)


@pytest.mark.unit
def test_research_plan_structured_is_populated_after_canonical_recovery():
    """End-to-end through create_research_manager: a repair that only
    produces free text must still leave research_plan_structured non-null
    once that text is accepted -- this is exactly what
    build_recommendation_record reads."""
    repair_text = (
        "**Recommendation**: Overweight\n"
        "**Rationale**: Bull case stronger after verification.\n"
        "**Strategic Actions**: Buy on confirmation."
    )
    recovery_json = json.dumps({
        "recommendation": "Overweight",
        "rationale": "Bull case stronger after verification.",
        "strategic_actions": "Buy on confirmation.",
    })
    llm = _NodeRepairThenRecoverLLM(repair_text, recovery_json)
    state = Propagator().create_initial_state("NVDA", "2026-09-01")
    state["investment_debate_state"]["bull_initial"] = "Bull thesis: target 200.0."
    state["investment_debate_state"]["bear_initial"] = "Bear thesis: risks noted."

    result = create_research_manager(llm, validator_llm=_EnabledValidator())(state)
    assert result["research_plan_structured"] is not None
    assert result["research_plan_structured"]["recommendation"] == "Overweight"


class _PMNodeRepairThenRecoverLLM:
    def __init__(self, repair_text: str, recovery_json: str):
        self.repair_text = repair_text
        self.recovery_json = recovery_json
        self.calls = 0

    def invoke(self, prompt, **kwargs):
        self.calls += 1
        if STRUCTURED_RECOVERY_PROMPT_MARKER in prompt:
            return SimpleNamespace(content=self.recovery_json)
        if self.calls == 1:
            return SimpleNamespace(content=(
                "**Rating**: Buy\n**Disposition**: KEEP\n"
                "**Executive Summary**: The level 216.48 is above 220.00, which supports the plan.\n"
                "**Investment Thesis**: x"
            ))
        return SimpleNamespace(content=self.repair_text)


@pytest.mark.unit
def test_portfolio_decision_structured_is_populated_after_canonical_recovery():
    """End-to-end through create_portfolio_manager: the real NVDA/2026-09-30
    regression -- a free-text repair, accepted on recheck, must leave
    portfolio_decision_structured non-null, since that is exactly what
    gates whether RecommendationRecord can be written at all."""
    repair_text = (
        "**Rating**: Overweight\n**Disposition**: KEEP\n"
        "**Executive Summary**: Clean summary after repair.\n"
        "**Investment Thesis**: Clean thesis after repair."
    )
    recovery_json = json.dumps({
        "rating": "Overweight", "disposition": "KEEP",
        "executive_summary": "Clean summary after repair.",
        "investment_thesis": "Clean thesis after repair.",
    })
    llm = _PMNodeRepairThenRecoverLLM(repair_text, recovery_json)
    state = Propagator().create_initial_state("NVDA", "2026-09-01")
    state["investment_plan"] = "Overweight plan"
    state["trader_investment_plan"] = "**Action**: Buy\n**Entry Price**: 100\n**Stop Loss**: 95"

    result = create_portfolio_manager(llm, validator_llm=_EnabledValidator())(state)
    assert result["portfolio_decision_structured"] is not None
    assert result["portfolio_decision_structured"]["rating"] == "Overweight"
    assert result["portfolio_decision_structured"]["disposition"] == "KEEP"
