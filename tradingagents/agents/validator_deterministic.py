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
