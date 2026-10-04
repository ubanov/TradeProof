"""Bounded independent verification and repair for the research team."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tradingagents.agents.managers.research_manager import create_research_manager
from tradingagents.agents.researchers.arithmetic import check_arithmetic
from tradingagents.agents.researchers.research_repair import (
    create_bear_repair,
    create_bull_repair,
)
from tradingagents.agents.researchers.research_verifier import (
    create_research_verifier,
)
from tradingagents.agents.researchers.verification import (
    MAX_REPAIR_ROUNDS,
    normalize_verification,
)
from tradingagents.agents.schemas import (
    ArithmeticClaim,
    FindingCategory,
    FindingSeverity,
    PortfolioRating,
    ResearchPlan,
    ResearchVerification,
    VerificationFinding,
    VerificationStatus,
)
from tradingagents.graph.conditional_logic import ConditionalLogic
from tradingagents.graph.propagation import Propagator


class _VerifierLLM:
    def __init__(self, results):
        self.results = list(results)
        self.prompts = []

    def with_structured_output(self, _schema):
        return self

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return self.results.pop(0)


class _TextLLM:
    def __init__(self, text):
        self.text = text
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return SimpleNamespace(content=self.text)


class _ManagerLLM:
    def __init__(self):
        self.prompts = []

    def with_structured_output(self, _schema):
        return self

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return ResearchPlan(
            recommendation=PortfolioRating.HOLD,
            rationale="Evidence quality considered.",
            strategic_actions="Maintain allocation.",
        )


def _state():
    state = Propagator().create_initial_state("NVDA", "2026-09-01")
    state.update(
        {
            "instrument_context": "INSTRUMENT_REPORT",
            "market_report": "MARKET_EVIDENCE",
            "sentiment_report": "SENTIMENT_EVIDENCE",
            "news_report": "NEWS_EVIDENCE",
            "fundamentals_report": "FUNDAMENTALS_EVIDENCE",
        }
    )
    state["investment_debate_state"].update(
        {
            "bull_initial": "Bull Analyst (Initial): BULL_INITIAL",
            "bear_initial": "Bear Analyst (Initial): BEAR_INITIAL",
            "current_bull_response": "Bull Analyst (Review 2): BULL_FINAL",
            "current_bear_response": "Bear Analyst (Review 2): BEAR_FINAL",
            "review_outcomes": "UNRESOLVED: valuation interpretation",
            "debate_round": 2,
            "bull_rebuttal_count": 2,
            "bear_rebuttal_count": 2,
        }
    )
    return state


def _finding(
    severity=FindingSeverity.MEDIUM,
    category=FindingCategory.UNSUPPORTED_CLAIM,
    agent="bull_review_2",
    arithmetic=None,
):
    return VerificationFinding(
        category=category,
        severity=severity,
        agent=agent,
        claim="55-60% probability",
        evidence="No supplied sample supports the probability.",
        reason="The numerical probability is unsupported.",
        arithmetic=arithmetic,
    )


def _verification(status, findings=None):
    return ResearchVerification(status=status, findings=findings or [])


def _apply(state, node):
    state["investment_debate_state"] = node(state)["investment_debate_state"]


@pytest.mark.unit
@pytest.mark.parametrize("status", [VerificationStatus.PASS, VerificationStatus.WARN])
def test_pass_and_warn_route_directly_to_manager_without_repair(status):
    state = _state()
    findings = [_finding()] if status is VerificationStatus.WARN else []
    llm = _VerifierLLM([_verification(status, findings)])
    _apply(state, create_research_verifier(llm))

    assert ConditionalLogic().should_continue_verification(state) == "Research Manager"
    debate = state["investment_debate_state"]
    assert debate["repair_rounds"] == 0
    assert debate["verification_history"][0]["status"] == status.value
    assert len(debate["verification_history"][0]["findings"]) == len(findings)


@pytest.mark.unit
def test_fail_runs_one_repair_and_one_second_verification_then_stops():
    arithmetic = ArithmeticClaim(
        operation="subtract",
        operands=[86602.91, 83476.35],
        reported_result=3752,
    )
    first = _verification(
        VerificationStatus.WARN,
        [_finding(FindingSeverity.MEDIUM, arithmetic=arithmetic)],
    )
    second = _verification(
        VerificationStatus.FAIL,
        [_finding(FindingSeverity.HIGH, agent="bear_repair")],
    )
    state = _state()
    verifier_llm = _VerifierLLM([first, second])
    verifier = create_research_verifier(verifier_llm)
    bull_llm = _TextLLM("ACCEPTED: corrected subtraction; corrected Bull position")
    bear_llm = _TextLLM("UNRESOLVED: probability removed; corrected Bear position")
    logic = ConditionalLogic(max_debate_rounds=3)

    _apply(state, verifier)
    assert state["investment_debate_state"]["verification_status"] == "FAIL"
    assert logic.should_continue_verification(state) == "Bull Repair"
    _apply(state, create_bull_repair(bull_llm))
    _apply(state, create_bear_repair(bear_llm))
    _apply(state, verifier)

    debate = state["investment_debate_state"]
    assert logic.should_continue_verification(state) == "Research Manager"
    assert debate["verifier_pass_count"] == 2
    assert debate["repair_rounds"] == MAX_REPAIR_ROUNDS == 1
    assert debate["repaired_agents"] == ["bull", "bear"]
    assert debate["debate_round"] == 2
    # One repair content call each, plus one canonical-recovery attempt each
    # (the repair text above is not valid recovery JSON, so recovery fails and
    # bull_thesis/bear_thesis stay untouched -- see test_structured_repair.py
    # for that non-degradation guarantee; this test is only about repair
    # running exactly once, not twice).
    assert len(bull_llm.prompts) == len(bear_llm.prompts) == 2
    assert "corrected Bull position" in verifier_llm.prompts[1]
    assert "corrected Bear position" in verifier_llm.prompts[1]
    assert "Computed result: 3126.56" in bull_llm.prompts[0]


@pytest.mark.unit
def test_verifier_receives_final_positions_and_original_analyst_evidence():
    state = _state()
    llm = _VerifierLLM([_verification(VerificationStatus.PASS)])
    _apply(state, create_research_verifier(llm))

    prompt = llm.prompts[0]
    for marker in (
        "BULL_FINAL",
        "BEAR_FINAL",
        "MARKET_EVIDENCE",
        "SENTIMENT_EVIDENCE",
        "NEWS_EVIDENCE",
        "FUNDAMENTALS_EVIDENCE",
    ):
        assert marker in prompt
    assert "BULL_INITIAL" not in prompt
    assert "BEAR_INITIAL" not in prompt
    assert "rejecting a valid correction without evidence" in prompt
    assert "substituting rhetoric for evidence" in prompt
    assert "NEW_DATA_EXCEPTION" in prompt
    assert "probability" in prompt
    assert "Deterministic Trade Metrics" in prompt


@pytest.mark.unit
def test_manager_receives_findings_independent_theses_and_repaired_positions():
    state = _state()
    state["investment_debate_state"].update(
        {
            "verification_history": [
                _verification(
                    VerificationStatus.FAIL,
                    [_finding(FindingSeverity.HIGH)],
                ).model_dump(mode="json")
            ],
            "bull_repair": "BULL_REPAIRED",
            "bear_repair": "BEAR_REPAIRED",
            "repaired_agents": ["bull", "bear"],
        }
    )
    llm = _ManagerLLM()
    create_research_manager(llm)(state)

    prompt = llm.prompts[0]
    for marker in (
        "BULL_INITIAL",
        "BEAR_INITIAL",
        "UNSUPPORTED_CLAIM",
        "55-60% probability",
        "BULL_REPAIRED",
        "BEAR_REPAIRED",
    ):
        assert marker in prompt


@pytest.mark.unit
def test_medium_unsupported_probability_warns_without_repair():
    state = _state()
    llm = _VerifierLLM(
        [_verification(VerificationStatus.FAIL, [_finding(FindingSeverity.MEDIUM)])]
    )
    _apply(state, create_research_verifier(llm))

    debate = state["investment_debate_state"]
    assert debate["verification_status"] == "WARN"
    assert debate["verification_history"][0]["repair_required"] is False
    assert ConditionalLogic().should_continue_verification(state) == "Research Manager"


@pytest.mark.unit
def test_arithmetic_helper_catches_subtraction_and_handles_ratio_percentages():
    subtraction = check_arithmetic("subtract", [86602.91, 83476.35], 3752)
    assert subtraction.matches_reported is False
    assert subtraction.computed_result is not None
    assert float(subtraction.computed_result) == pytest.approx(3126.56)

    assert check_arithmetic("ratio", [10, 4], 2.5).matches_reported is True
    assert check_arithmetic("percentage", [1, 4], 25).matches_reported is True
    assert (
        check_arithmetic("percentage_change", [80, 100], 25).matches_reported
        is True
    )


@pytest.mark.unit
def test_missing_or_ambiguous_arithmetic_does_not_crash():
    check = check_arithmetic("subtract", [1], 1)
    assert check.computed_result is None
    assert check.matches_reported is None
    assert "requires two operands" in check.error

    state = _state()
    llm = _VerifierLLM(
        [_verification(VerificationStatus.WARN, [_finding(arithmetic=None)])]
    )
    _apply(state, create_research_verifier(llm))
    assert state["investment_debate_state"]["verification_status"] == "WARN"


@pytest.mark.unit
def test_live_model_schema_variants_are_normalised_without_fallback():
    result = ResearchVerification.model_validate(
        {
            "status": "warn",
            "findings": [
                {
                    "category": "ARITHMETIC_ERROR",
                    "severity": "MEDIUM",
                    "agent": "bull_review_2",
                    "claim": "70 - 56.71 = 13",
                    "arithmetic": [
                        {
                            "operation": "subtract",
                            "operands": [70, 56.71],
                            "reported_result": 13,
                        }
                    ],
                }
            ],
            "verified_points": [
                {
                    "claim": "Price rose 3.0%",
                    "evidence": "Market report values 101.48 and 104.54",
                }
            ],
            "notes": ["First note", "Second note"],
            "repair_targets": ["Bull Analyst"],
        }
    )

    normalised = normalize_verification(result)
    assert normalised.status is VerificationStatus.FAIL
    assert normalised.findings[0].correction == "Computed result: 13.29"
    assert normalised.verified_points == [
        "Price rose 3.0% | Market report values 101.48 and 104.54"
    ]
    assert normalised.notes == "First note\nSecond note"
    assert normalised.repair_targets == ["bull"]


@pytest.mark.unit
def test_single_arithmetic_object_is_normalised_to_a_list():
    finding = VerificationFinding.model_validate(
        {
            "category": "ARITHMETIC_ERROR",
            "severity": "HIGH",
            "claim": "10 / 4 = 3",
            "arithmetic": {
                "operation": "divide",
                "operands": [10, 4],
                "reported_result": 3,
            },
        }
    )
    assert len(finding.arithmetic) == 1
    assert finding.arithmetic[0].operation.value == "divide"
