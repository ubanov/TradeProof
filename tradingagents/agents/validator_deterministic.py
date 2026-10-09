"""Deterministic (LLM-free) half of the Validator layer (see fork README:
"Generation / Validation / Review"). Always safe to run -- no API call, no
cost -- so it runs regardless of ``validation_enabled`` (which only gates
the optional Validator LLM pass).

Reuses the exact same primitives the Research Manager/Risk integrity checks
already use (``integrity_shared.py``) and the Research Verifier's own
arithmetic checker (``researchers/arithmetic.py``) -- this module is a thin
adapter mapping their findings onto ``ValidationFinding``, not a second
implementation of the same regex/arithmetic.
"""

from __future__ import annotations

import json

from pydantic import BaseModel, ValidationError

from tradingagents.agents.integrity_shared import (
    find_relational_errors,
    find_unit_errors,
)
from tradingagents.agents.managers.integrity_check import (
    ManagerFindingCategory,
    check_manager_integrity,
)
from tradingagents.agents.researchers.arithmetic import check_arithmetic
from tradingagents.agents.schemas import ArithmeticClaim
from tradingagents.agents.validation_schemas import ValidationCategory, ValidationFinding


def check_json_syntax(text: str) -> ValidationFinding | None:
    """The candidate is supposed to be JSON (schema/contract-only stages,
    e.g. a JSON-fallback reconstruction) but doesn't even parse."""
    try:
        json.loads(text)
    except (json.JSONDecodeError, TypeError) as exc:
        return ValidationFinding(
            category=ValidationCategory.JSON_SYNTAX_ERROR,
            claim="Candidate response is not valid JSON.",
            evidence=[text[:200]],
            reason=str(exc),
        )
    return None


def check_schema(candidate: dict | BaseModel, schema: type[BaseModel]) -> ValidationFinding | None:
    """The candidate (already-parsed dict, or an instance) does not satisfy
    its expected Pydantic contract -- required fields, enum validity, types."""
    if isinstance(candidate, schema):
        return None
    try:
        schema.model_validate(candidate)
        return None
    except ValidationError as exc:
        return ValidationFinding(
            category=ValidationCategory.SCHEMA_ERROR,
            claim=f"Candidate does not satisfy the {schema.__name__} contract.",
            evidence=[],
            reason="; ".join(
                f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()[:5]
            ),
        )


def check_arithmetic_claims(claims: list[ArithmeticClaim]) -> list[ValidationFinding]:
    """Reuses ``check_arithmetic`` (the same function the Research Verifier's
    normalization already relies on) -- never a second arithmetic engine."""
    findings = []
    for claim in claims:
        result = check_arithmetic(claim.operation, claim.operands, claim.reported_result)
        if result.matches_reported is False:
            findings.append(
                ValidationFinding(
                    category=ValidationCategory.ARITHMETIC_ERROR,
                    claim=f"{claim.operation.value}({claim.operands}) reported as {claim.reported_result}",
                    evidence=[str(v) for v in claim.operands],
                    reason=f"Computed result is {result.computed_result}, not {claim.reported_result}.",
                    suggested_correction=str(result.computed_result),
                )
            )
    return findings


def check_text_relational_and_unit_errors(text: str) -> list[ValidationFinding]:
    """Reuses ``integrity_shared``'s sentence-level scanners -- the same ones
    the Research Manager/Risk integrity checks already run -- rather than a
    third regex implementation of the same checks."""
    findings = []
    for excerpt in find_relational_errors(text):
        findings.append(
            ValidationFinding(
                category=ValidationCategory.RELATIONAL_ERROR,
                claim=excerpt,
                reason="Stated comparison between two explicit numbers does not hold.",
            )
        )
    for excerpt in find_unit_errors(text):
        findings.append(
            ValidationFinding(
                category=ValidationCategory.UNIT_ERROR,
                claim=excerpt,
                reason="A price/level field is stated as a bare percentage.",
            )
        )
    return findings


def check_entry_stop_direction(
    action: str, entry: float | None, stop_loss: float | None,
) -> ValidationFinding | None:
    """Buy must risk DOWN (stop below entry); Sell must risk UP (stop above
    entry) -- a structural direction check on the Trader's own structured
    fields, not a semantic judgment of the plan itself."""
    if entry is None or stop_loss is None:
        return None
    action = (action or "").upper()
    if action == "BUY" and stop_loss >= entry:
        wrong = "at or above"
    elif action == "SELL" and stop_loss <= entry:
        wrong = "at or below"
    else:
        return None
    return ValidationFinding(
        category=ValidationCategory.RELATIONAL_ERROR,
        claim=f"{action} with stop_loss={stop_loss} {wrong} entry={entry}",
        evidence=[str(entry), str(stop_loss)],
        reason=f"A {action} stop-loss must be on the losing side of entry.",
    )


def check_missing_requirement(candidate_text: str, required_markers: list[str]) -> list[ValidationFinding]:
    """Stage-specific required sections/fields the original prompt demanded
    (e.g. a labelled header) that are simply absent from the rendered text."""
    findings = []
    for marker in required_markers:
        if marker.lower() not in (candidate_text or "").lower():
            findings.append(
                ValidationFinding(
                    category=ValidationCategory.MISSING_REQUIREMENT,
                    claim=f"Required element '{marker}' is missing.",
                    reason="The prompt's contract requires this element; it was not found in the output.",
                )
            )
    return findings


# Research Manager's own finding categories map onto Validator's narrower
# taxonomy where the defect really is the same thing; anything reserved/
# never actually produced (INTERNAL_INCONSISTENCY) still gets a safe mapping.
_MANAGER_CATEGORY_MAP = {
    ManagerFindingCategory.NEW_UNSUPPORTED_THRESHOLD.value: ValidationCategory.UNSUPPORTED_NUMBER,
    ManagerFindingCategory.NEW_UNSUPPORTED_LEVEL.value: ValidationCategory.UNSUPPORTED_NUMBER,
    ManagerFindingCategory.UNSUPPORTED_SIZING_RULE.value: ValidationCategory.UNSUPPORTED_NUMBER,
    ManagerFindingCategory.ARITHMETIC_ERROR.value: ValidationCategory.ARITHMETIC_ERROR,
    ManagerFindingCategory.RELATIONAL_ERROR.value: ValidationCategory.RELATIONAL_ERROR,
    # A self-assessed conviction score stated as if it were a market
    # probability violates the prompt's own explicit instruction, not a
    # missing/unsupported number as such.
    ManagerFindingCategory.CONVICTION_AS_PROBABILITY.value: ValidationCategory.PROMPT_REQUIREMENT_VIOLATION,
    # A value the review round explicitly withdrew/corrected away is, by
    # definition, no longer supported if reused.
    ManagerFindingCategory.WITHDRAWN_VALUE_REUSED.value: ValidationCategory.UNSUPPORTED_NUMBER,
    ManagerFindingCategory.INTERNAL_INCONSISTENCY.value: ValidationCategory.INTERNAL_CONTRADICTION,
}


def check_manager_integrity_findings(
    manager_output_text: str, debate: dict, *, extra_sources: tuple[str, ...] = (),
) -> list[ValidationFinding]:
    """Thin adapter over the EXISTING Research Manager integrity check
    (``managers/integrity_check.py::check_manager_integrity``) -- reused
    as-is (known-number pool, arithmetic/relational/withdrawn-value
    scanning), never a second implementation of the same checks.

    Deliberately takes NO raw analyst ``reports`` parameter at all -- unlike
    ``check_manager_integrity``'s own pre-existing call site (which still
    feeds the Trader-facing integrity notice with the full reports), this
    adapter is Validator's provenance check for what Research Manager's OWN
    prompt actually renders to it, which never includes the raw reports.
    ``extra_sources`` is that allowed universe's free-text fields (bull/bear
    initial theses, review outcomes, verification summary, repair outputs,
    final bull/bear positions) -- passing a raw analyst report here would
    reintroduce the exact provenance leak this signature prevents structurally
    (a number known only from a report Research Manager never saw being
    treated as "supported" merely because it was reachable in AgentState)."""
    result = check_manager_integrity(manager_output_text, debate, extra_sources=extra_sources)
    return [
        ValidationFinding(
            category=_MANAGER_CATEGORY_MAP.get(
                finding.category, ValidationCategory.UNCLASSIFIED_VALIDATION_ERROR,
            ),
            claim=finding.detail,
            reason=f"Research Manager integrity check: {finding.category}",
        )
        for finding in result.findings
    ]
