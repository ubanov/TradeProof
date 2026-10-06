"""Bounded independent verification and repair for the research team."""

from __future__ import annotations

import json
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


class _AlwaysRaisesStructured:
    def invoke(self, prompt):
        raise ValueError("malformed structured output")


class _FailsStructuredThenJSON:
    """Mimics a real provider whose structured tool-call attempt raises (the
    qwen3.8-flash-next:125b-a6b-q4_K_M incident this regression is based on),
    falling back to a plain ``.invoke`` call that returns ``json_text`` as
    free-text content -- exactly what ``_parse_freetext_result`` must
    validate."""

    def __init__(self, json_text: str):
        self.json_text = json_text
        self.prompts: list = []

    def with_structured_output(self, schema):
        return _AlwaysRaisesStructured()

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return SimpleNamespace(content=self.json_text)


# --- JSON-fallback validation must not discard a real FAIL+findings result --


@pytest.mark.unit
def test_json_fallback_preserves_a_valid_fail_with_a_high_finding():
    payload = json.dumps(
        {
            "status": "FAIL",
            "findings": [
                {
                    "category": "ARITHMETIC_ERROR",
                    "severity": "HIGH",
                    "agent": "bull_review_2",
                    "claim": "70 - 56.71 = 13",
                    "evidence": "Market report values 70 and 56.71.",
                    "reason": "The subtraction is wrong.",
                }
            ],
        }
    )
    state = _state()
    llm = _FailsStructuredThenJSON(payload)
    _apply(state, create_research_verifier(llm))

    debate = state["investment_debate_state"]
    assert debate["verification_status"] == "FAIL"
    assert len(debate["verification_findings"]) == 1
    assert debate["verification_findings"][0]["category"] == "ARITHMETIC_ERROR"
    assert debate["verification_findings"][0]["severity"] == "HIGH"
    assert ConditionalLogic().should_continue_verification(state) == "Bull Repair"


@pytest.mark.unit
def test_json_fallback_normalises_lowercase_enum_casing_without_fuzzy_matching():
    payload = json.dumps(
        {
            "status": "fail",
            "findings": [
                {
                    "category": "arithmetic_error",
                    "severity": "high",
                    "agent": "bull_review_2",
                    "claim": "70 - 56.71 = 13",
                }
            ],
        }
    )
    state = _state()
    llm = _FailsStructuredThenJSON(payload)
    _apply(state, create_research_verifier(llm))

    debate = state["investment_debate_state"]
    assert debate["verification_status"] == "FAIL"
    assert len(debate["verification_findings"]) == 1
    assert debate["verification_findings"][0]["category"] == "ARITHMETIC_ERROR"
    assert debate["verification_findings"][0]["severity"] == "HIGH"


@pytest.mark.unit
def test_json_fallback_accepts_a_missing_optional_field_with_its_safe_default():
    # "agent" is omitted entirely; the schema already has a safe, explicit
    # default for it ("unknown_research_output") -- nothing is invented here
    # that the schema itself doesn't already represent.
    payload = json.dumps(
        {
            "status": "WARN",
            "findings": [
                {
                    "category": "UNSUPPORTED_CLAIM",
                    "severity": "MEDIUM",
                    "claim": "55-60% probability",
                }
            ],
        }
    )
    state = _state()
    llm = _FailsStructuredThenJSON(payload)
    _apply(state, create_research_verifier(llm))

    debate = state["investment_debate_state"]
    assert debate["verification_status"] == "WARN"
    assert debate["verification_findings"][0]["agent"] == "unknown_research_output"


@pytest.mark.unit
def test_json_fallback_rejects_an_unknown_category_instead_of_fuzzy_mapping():
    payload = json.dumps(
        {
            "status": "FAIL",
            "findings": [
                {
                    "category": "BAD_MATH_THING",
                    "severity": "HIGH",
                    "claim": "70 - 56.71 = 13",
                }
            ],
        }
    )
    state = _state()
    llm = _FailsStructuredThenJSON(payload)
    _apply(state, create_research_verifier(llm))

    debate = state["investment_debate_state"]
    # Rejected wholesale -- never fuzzy-mapped to ARITHMETIC_ERROR or any
    # other category -- and conservative, not a confirmed "no findings".
    assert debate["verification_status"] == "WARN"
    assert debate["verification_findings"] == []
    assert "unavailable" in debate["verification_history"][0]["notes"].lower() or (
        "unvalidated" in debate["verification_history"][0]["notes"].lower()
    )


@pytest.mark.unit
def test_json_fallback_non_json_prose_is_conservative_not_semantically_parsed():
    state = _state()
    llm = _FailsStructuredThenJSON("I looked this over and it seems mostly fine.")
    _apply(state, create_research_verifier(llm))

    debate = state["investment_debate_state"]
    assert debate["verification_status"] == "WARN"
    assert debate["verification_findings"] == []
    notes = debate["verification_history"][0]["notes"].lower()
    assert "unvalidated" in notes or "unavailable" in notes


@pytest.mark.unit
def test_regression_qwen_fail_high_arithmetic_error_never_degrades_to_warn_no_findings():
    """The exact real-world shape that triggered this fix: a structured call
    failed, the JSON fallback contained a genuine FAIL with one HIGH
    ARITHMETIC_ERROR finding, and the pipeline must never present this as
    WARN with no findings."""
    payload = json.dumps(
        {
            "status": "FAIL",
            "findings": [
                {
                    "category": "ARITHMETIC_ERROR",
                    "severity": "HIGH",
                    "agent": "bull_review_2",
                    "claim": "86,602.91 - 83,476.35 = 3,752",
                    "evidence": "Market report values 86,602.91 and 83,476.35.",
                    "reason": "The subtraction does not match the supplied figures.",
                }
            ],
        }
    )
    state = _state()
    llm = _FailsStructuredThenJSON(payload)
    _apply(state, create_research_verifier(llm))

    debate = state["investment_debate_state"]
    assert debate["verification_status"] != "WARN"
    assert debate["verification_status"] == "FAIL"
    assert debate["verification_findings"] != []


@pytest.mark.unit
def test_json_fallback_failure_notes_never_claim_no_findings_were_verified():
    from tradingagents.agents.researchers.verification import render_verification_history

    state = _state()
    llm = _FailsStructuredThenJSON("not valid json at all")
    _apply(state, create_research_verifier(llm))

    debate = state["investment_debate_state"]
    assert debate["verification_status"] == "WARN"
    rendered = render_verification_history(debate["verification_history"])
    assert "- No findings" in rendered  # the data-level count is honestly empty...
    # ...but the note right below it makes clear this was never verified,
    # not a confirmed absence of findings.
    assert "unvalidated" in rendered.lower() or "unavailable" in rendered.lower()
    assert "NOTE:" in rendered


# --- \$ dollar-escape normalization + optional-arithmetic entry sanitizing --
# (real gemma4:26b batch: AMZN's JSON fallback used "\$101.82B" -- an invalid
# JSON escape -- and AAPL's otherwise-valid FAIL+HIGH-finding fallback had one
# finding's optional "arithmetic" as a positional ["percentage_change", [a,
# b]] instead of the named-field object the schema expects.)


@pytest.mark.unit
def test_dollar_escape_is_normalized_before_parsing():
    from tradingagents.agents.researchers.research_verifier import _parse_freetext_result

    content = (
        '{"status": "PASS", "findings": [], '
        '"verified_points": ["Efectivo de \\$101.82B y Deuda de \\$50B"]}'
    )
    result = _parse_freetext_result(content)
    assert result.status == VerificationStatus.PASS
    assert result.verified_points == ["Efectivo de $101.82B y Deuda de $50B"]


@pytest.mark.unit
def test_only_the_dollar_escape_is_touched_other_valid_escapes_survive():
    from tradingagents.agents.researchers.research_verifier import (
        _normalize_dollar_escape,
    )

    text = '{"notes": "line1\\nline2\\ttabbed \\"quoted\\" \\\\backslash \\u00e9 \\$5"}'
    normalized = _normalize_dollar_escape(text)
    data = json.loads(normalized)
    assert data["notes"] == 'line1\nline2\ttabbed "quoted" \\backslash é $5'


@pytest.mark.unit
def test_aapl_like_malformed_arithmetic_preserves_the_high_finding():
    from tradingagents.agents.researchers.research_verifier import _parse_freetext_result

    content = json.dumps(
        {
            "status": "FAIL",
            "findings": [
                {
                    "category": "CONTRADICTION",
                    "severity": "HIGH",
                    "agent": "Bear Analyst",
                    "claim": "Entry equals Take Profit, zero reward.",
                },
                {
                    "category": "ARITHMETIC_ERROR",
                    "severity": "LOW",
                    "agent": "Bear Analyst",
                    "claim": "FCF declined ~9.3%.",
                    "arithmetic": ["percentage_change", [108.807, 98.767]],
                },
            ],
        }
    )
    result = _parse_freetext_result(content)
    assert result.status == VerificationStatus.FAIL
    assert len(result.findings) == 2
    assert result.findings[0].category == FindingCategory.CONTRADICTION
    assert result.findings[0].severity == FindingSeverity.HIGH
    # The second finding survives too -- only its malformed arithmetic is gone.
    assert result.findings[1].arithmetic == []


@pytest.mark.unit
def test_mixed_valid_and_invalid_arithmetic_entries_keeps_only_the_valid_one():
    from tradingagents.agents.researchers.research_verifier import _parse_freetext_result

    content = json.dumps(
        {
            "status": "WARN",
            "findings": [
                {
                    "category": "ARITHMETIC_ERROR",
                    "severity": "MEDIUM",
                    "claim": "mixed arithmetic list",
                    "arithmetic": [
                        {"operation": "subtract", "operands": [10, 4], "reported_result": 6},
                        ["percentage_change", [108.807, 98.767]],
                    ],
                }
            ],
        }
    )
    result = _parse_freetext_result(content)
    assert len(result.findings[0].arithmetic) == 1
    assert result.findings[0].arithmetic[0].operation.value == "subtract"


@pytest.mark.unit
def test_arithmetic_missing_reported_result_is_dropped_not_invented():
    from tradingagents.agents.researchers.research_verifier import _parse_freetext_result

    content = json.dumps(
        {
            "status": "WARN",
            "findings": [
                {
                    "category": "ARITHMETIC_ERROR",
                    "severity": "LOW",
                    "claim": "incomplete arithmetic",
                    "arithmetic": [{"operation": "percentage_change", "operands": [108.807, 98.767]}],
                }
            ],
        }
    )
    result = _parse_freetext_result(content)
    assert result.findings[0].arithmetic == []


@pytest.mark.unit
def test_invalid_category_is_still_rejected_not_bypassed():
    """The arithmetic sanitization must not become a general validation
    bypass -- an unrecognised category elsewhere in the same finding still
    fails the whole document, falling back to the conservative WARN."""
    from tradingagents.agents.researchers.research_verifier import _parse_freetext_result

    content = json.dumps(
        {
            "status": "FAIL",
            "findings": [{"category": "BAD_MATH_THING", "severity": "HIGH", "claim": "x"}],
        }
    )
    result = _parse_freetext_result(content)
    assert result.status == VerificationStatus.WARN
    assert result.findings == []
    assert "unvalidated" in result.notes.lower() or "unavailable" in result.notes.lower()


@pytest.mark.unit
def test_aapl_like_fail_with_sanitized_arithmetic_still_reaches_repair():
    """End to end through create_research_verifier: the AAPL-like payload's
    FAIL + HIGH finding must still route to Bull/Bear Repair exactly like any
    other recovered FAIL."""
    payload = json.dumps(
        {
            "status": "FAIL",
            "findings": [
                {
                    "category": "CONTRADICTION",
                    "severity": "HIGH",
                    "agent": "bear_review_2",
                    "claim": "Entry equals Take Profit, zero reward.",
                },
                {
                    "category": "ARITHMETIC_ERROR",
                    "severity": "LOW",
                    "agent": "bear_review_2",
                    "claim": "FCF declined ~9.3%.",
                    "arithmetic": ["percentage_change", [108.807, 98.767]],
                },
            ],
        }
    )
    state = _state()
    llm = _FailsStructuredThenJSON(payload)
    _apply(state, create_research_verifier(llm))

    debate = state["investment_debate_state"]
    assert debate["verification_status"] == "FAIL"
    assert ConditionalLogic().should_continue_verification(state) == "Bull Repair"


@pytest.mark.unit
def test_lly_like_yaml_prose_is_still_not_semantically_recovered():
    """The YAML/prose shape (no JSON at all) from the real LLY run must stay
    on the conservative path -- neither normalization applies to prose, and
    neither is a JSON/YAML parser introduced to rescue it."""
    from tradingagents.agents.researchers.research_verifier import _parse_freetext_result

    content = (
        "status: FAIL\n"
        "findings:\n"
        "  - category: ARITHMETIC_ERROR\n"
        "    severity: HIGH\n"
        "    claim: some claim\n"
    )
    result = _parse_freetext_result(content)
    assert result.status == VerificationStatus.WARN
    assert result.findings == []
    assert "unvalidated" in result.notes.lower() or "unavailable" in result.notes.lower()


@pytest.mark.unit
def test_recovered_fail_triggers_exactly_one_bounded_repair_round_like_before():
    payload = json.dumps(
        {
            "status": "FAIL",
            "findings": [
                {
                    "category": "ARITHMETIC_ERROR",
                    "severity": "HIGH",
                    "agent": "bear_repair",
                    "claim": "70 - 56.71 = 13",
                }
            ],
        }
    )
    state = _state()
    verifier_llm_round1 = _FailsStructuredThenJSON(payload)
    verifier = create_research_verifier(verifier_llm_round1)
    logic = ConditionalLogic(max_debate_rounds=3)

    _apply(state, verifier)
    assert state["investment_debate_state"]["verification_status"] == "FAIL"
    assert logic.should_continue_verification(state) == "Bull Repair"

    bull_llm = _TextLLM("ACCEPTED: corrected; corrected Bull position")
    bear_llm = _TextLLM("ACCEPTED: corrected; corrected Bear position")
    _apply(state, create_bull_repair(bull_llm))
    _apply(state, create_bear_repair(bear_llm))

    # Second pass: structured succeeds cleanly (PASS) -- the repair resolved it.
    second_verifier = create_research_verifier(
        _VerifierLLM([_verification(VerificationStatus.PASS)])
    )
    _apply(state, second_verifier)

    debate = state["investment_debate_state"]
    assert logic.should_continue_verification(state) == "Research Manager"
    assert debate["repair_rounds"] == MAX_REPAIR_ROUNDS == 1
    assert debate["verifier_pass_count"] == 2


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
