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
import re

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
from tradingagents.agents.researchers.trade_metrics import (
    entry_reference,
    trade_geometry_violations,
)
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


def check_relational_errors(text: str) -> list[ValidationFinding]:
    """RELATIONAL_ERROR only -- no unit-error scanning. Used by Trader in
    place of ``check_text_relational_and_unit_errors`` (see
    ``check_price_field_percentage`` below for the narrower, field-line-
    based UNIT_ERROR replacement): the generic sentence-wide ``find_unit_
    errors`` false-positived on a descriptive stop-distance percentage in
    narrative prose (real-run regression), and is deliberately NOT reused
    here for Trader. ``check_text_relational_and_unit_errors`` itself is
    left completely unchanged for its other caller (Portfolio Manager)."""
    return [
        ValidationFinding(
            category=ValidationCategory.RELATIONAL_ERROR,
            claim=excerpt,
            reason="Stated comparison between two explicit numbers does not hold.",
        )
        for excerpt in find_relational_errors(text)
    ]


# Trader's own contractual price/level fields (see schemas.py::
# render_trader_proposal and the free-text fallback's equivalent prose) --
# deliberately a short, explicit, line-anchored list, never a scan of every
# LEVEL_WORDS occurrence in narrative text (see task: "porcentaje
# descriptivo de distancia confundido con price field"). Longer labels
# first so "Entry Price" matches before the shorter "Entry" alternative can.
_TRADER_PRICE_FIELD_LABELS = ("Entry Price", "Entry", "Stop Loss", "Stop", "Take Profit", "TP")
_TRADER_FIELD_LINE_RE = re.compile(
    r"(?im)^\s*\**(" + "|".join(re.escape(label) for label in _TRADER_PRICE_FIELD_LABELS) + r")\**\s*:\s*(.+?)\s*$"
)
# The field's ENTIRE value, not a substring -- "2.16%" is a bare percentage
# standing in for an absolute price; "1145.00 - 1165.00" or "not provided"
# never match this, regardless of what percentages appear elsewhere in the
# surrounding sentence/paragraph.
_BARE_PERCENT_VALUE_RE = re.compile(r"^[+-]?\$?\d+(?:\.\d+)?\s*%$")


def check_price_field_percentage(text: str) -> list[ValidationFinding]:
    """UNIT_ERROR only when an EXPLICIT Trader contract field line (Entry
    Price/Entry/Stop Loss/Stop/Take Profit/TP) states a bare percentage as
    its entire value (e.g. "**Stop Loss**: 3%") -- never a general scan for
    "%" anywhere near a level word in narrative/reasoning text. A descriptive
    stop-distance percentage ("25 pts, approximately 2.16% of current
    price"), a relative-change percentage, or a financial-metric percentage
    (revenue growth, FCF margin, position sizing "% of portfolio") never
    appears on one of these labeled lines and is therefore never flagged,
    no matter where else in the response it appears. The real contractual
    rule -- entry_price/stop_loss must be absolute prices -- is unaffected:
    a genuine "Entry Price: 2.5%" or "Stop Loss: 3%" line is still caught.
    """
    findings = []
    for line in (text or "").splitlines():
        match = _TRADER_FIELD_LINE_RE.match(line)
        if match is None:
            continue
        label, value = match.group(1), match.group(2).strip().strip("*").strip()
        if _BARE_PERCENT_VALUE_RE.match(value):
            findings.append(
                ValidationFinding(
                    category=ValidationCategory.UNIT_ERROR,
                    claim=f"{label}: {value}",
                    reason="A price/level field is stated as a bare percentage instead of an absolute price.",
                )
            )
    return findings


def check_trade_geometry(
    direction: str | None,
    entry_price: float | None = None,
    entry_low: float | None = None,
    entry_high: float | None = None,
    reference_entry: float | None = None,
    stop_loss: float | None = None,
    take_profit: float | None = None,
) -> ValidationFinding | None:
    """A directional plan's stop-loss AND take-profit must each be on the
    correct side of entry for its own stated direction -- superseding the
    narrower ``check_entry_stop_direction`` (stop-loss only, point entry
    only), which this replaces (see fork README / NVDA 2026-01-07
    regression: a SHORT thesis whose stop-loss ended up below entry instead
    of above produced a plausible-looking but structurally non-executable
    plan, undetected because nothing checked direction, only arithmetic).

    When ``entry_low``/``entry_high`` are BOTH given (a genuine entry
    range), geometry is checked against the FULL range -- the worst-case
    bound a real fill could land on, not just a single reference point --
    regardless of whether ``reference_entry`` is also given (that value
    still matters for metrics, never for this check once a real range
    exists; see ``trade_geometry_violations``). Otherwise, ``reference_entry``
    takes precedence when given; failing that, the entry is resolved via
    ``entry_reference`` exactly like ``trade_metrics.py`` does. A direction
    that isn't LONG/BUY/BULL or SHORT/SELL/BEAR (Hold, unrecognized, or
    absent), a missing/unresolvable entry, or a missing stop_loss/
    take_profit individually all mean nothing is checked for that
    field -- never invented, never corrected.
    """
    entry = reference_entry if reference_entry is not None else entry_reference(
        entry_price, entry_low, entry_high
    )
    violations = trade_geometry_violations(
        direction, entry, stop_loss, take_profit,
        entry_low=entry_low, entry_high=entry_high,
    )
    if not violations:
        return None
    return ValidationFinding(
        category=ValidationCategory.TRADE_GEOMETRY_ERROR,
        claim="; ".join(violations),
        evidence=[
            str(v) for v in (entry, entry_low, entry_high, stop_loss, take_profit) if v is not None
        ],
        reason=(
            "A directional plan's stop-loss and take-profit must be on the "
            "correct side of entry for the trade's direction -- a structural "
            "execution error, not a minor arithmetic discrepancy."
        ),
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
