"""Canonical recovery for a Research (Bull/Bear) textual fallback.

When the structured Bull/Bear initial-thesis or review call fails, the
researcher still produces free text -- but that text used to simply bypass
``bull_thesis``/``bear_thesis`` (the canonical state every later stage reads:
``recompute_trade_metrics``, the Research Manager, the Verifier, Risk's
frozen evidence, the Manager integrity check). This module is the one place
that free text is given a chance to become canonical state again, so a
researcher's real correction (conviction, entry, TP, SL, KEEP/REVISE/
WITHDRAW) is never silently invisible to everything downstream.

Deliberately NOT a semantic regex parser: recovery asks the researcher's own
LLM one small, plain-JSON question ("what does the text above already say,
in this minimal shape?"), validates the answer with Pydantic
(``InitialThesisCanonicalPatch``/``ReviewCanonicalPatch`` -- see
``schemas.py``), and then rejects the whole patch unless every number in it
is already present in the fallback text (checked with the existing,
generic ``integrity_shared.extract_numbers``/``numbers_match`` -- the same
membership check used elsewhere, not a new semantic extractor). No LLM call
happens on the structured happy path; this only runs once a fallback has
already occurred (see ``researchers/thesis_flow.py``).

A failed recovery (invalid JSON, failed validation, or a hallucinated value)
must never guess -- the caller keeps the last valid canonical state exactly
as it was and marks the fallback UNAPPLIED (see ``CanonicalRecoveryStatus``).
"""

from __future__ import annotations

import json
import logging
import re
from enum import Enum

from pydantic import ValidationError

from tradingagents.agents.integrity_shared import extract_numbers, numbers_match
from tradingagents.agents.schemas import (
    EntryLevelType,
    InitialThesisCanonicalPatch,
    ReviewCanonicalPatch,
    SuggestedRiskUnit,
    TradePlanAction,
)
from tradingagents.llm_trace import record_structured_parse_result, traced_invoke

logger = logging.getLogger(__name__)

# The researcher's own LLM answers this exact small question, so test doubles
# (and anyone reading a trace) can recognise a recovery call unambiguously.
RECOVERY_PROMPT_MARKER = "CANONICAL_RECOVERY_REQUEST"


class CanonicalRecoveryStatus(str, Enum):
    """Whether a Research call's canonical state came from a direct
    structured success, a successful textual-fallback recovery, or neither."""

    NOT_NEEDED = "NOT_NEEDED"
    APPLIED = "APPLIED"
    UNAPPLIED = "UNAPPLIED"


# What a downstream LLM consumer (Research Manager, Verifier, the same
# researcher's own next-round prompt) sees in place of an UNAPPLIED
# fallback's full text: a bare, unambiguous marker, never the failed
# fallback's semantic content (a stray "conviction 57 -> 50" or "TP 240" in
# that prose must not be readable as if it were live evidence). The full text
# is still preserved verbatim in the audit-facing fields (``history``,
# ``bull_history``/``bear_history``) for report/debug -- see
# ``bull_researcher.py``/``bear_researcher.py``.
UNAPPLIED_DOWNSTREAM_STUB = "[NON-CANONICAL FALLBACK UNAPPLIED — canonical state unchanged]"


def downstream_safe_text(text: str, recovery_status: str) -> str:
    """What a downstream LLM consumer (Research Manager, Verifier) should
    read in place of ``text`` for a given recovery status.

    UNAPPLIED replaces ``text`` with the bare stub -- the failed fallback's
    semantic content (a stray "conviction 57 -> 50" or "TP 240") must never
    be readable downstream as if it were live, authoritative evidence. Every
    other status (NOT_NEEDED/APPLIED/NOT_RUN) passes ``text`` through
    unchanged: APPLIED's narrative is no longer an un-anchored claim, since
    the canonical state has already picked up its content. An empty ``text``
    passes through unchanged too, so callers chaining this with ``or`` still
    fall through correctly instead of short-circuiting on a stub.
    """
    if not text:
        return text
    if recovery_status == CanonicalRecoveryStatus.UNAPPLIED.value:
        return UNAPPLIED_DOWNSTREAM_STUB
    return text


def latest_side_recovery_status(debate: dict, side: str) -> str:
    """Whichever recovery status governs ``{side}_current_*_response``'s
    current content: the latest review's status once at least one review
    round has run, otherwise the initial thesis call's status. Mirrors the
    same "review overrides initial" precedence the researchers themselves use
    when they promote a review's text over the initial one (see
    ``bull_researcher.py``/``bear_researcher.py``).
    """
    review_status = debate.get(f"{side}_review_recovery_status") or "NOT_RUN"
    if review_status != "NOT_RUN":
        return review_status
    return debate.get(f"{side}_initial_recovery_status") or "NOT_RUN"


def _initial_recovery_prompt(fallback_text: str) -> str:
    return (
        f"{RECOVERY_PROMPT_MARKER}\n"
        "You are doing a narrow bookkeeping task, not giving a new opinion or "
        "improving the thesis. Below is a researcher's own free-text statement "
        "of an initial trade thesis. Translate ONLY what it already says into "
        "the minimal JSON object below. Do not add, infer, or introduce any "
        "value that is not already stated in the text.\n\n"
        "Respond with ONLY a single JSON object, no markdown fences, no "
        "commentary, matching exactly this shape:\n"
        '{"direction": "BULL or BEAR", "conviction_total": <0-100 integer>, '
        '"entry": {"type": "point or range", "price": <number or null>, '
        '"low": <number or null>, "high": <number or null>}, '
        '"take_profit": <number or null>, "stop_loss": <number or null>, '
        '"suggested_risk_unit": "NO_TRADE or LOW or MEDIUM or HIGH or null"}\n\n'
        "Only set suggested_risk_unit if the text explicitly labels a value as "
        'the Suggested (Research) Risk Unit (e.g. "Suggested Risk Unit: '
        'MEDIUM"). Do not set it based on conviction, general risk language, '
        "or trading vocabulary -- leave it null if it is not explicitly "
        "labeled.\n\n"
        "Researcher statement:\n"
        f"{fallback_text}\n"
    )


def _review_recovery_prompt(fallback_text: str, prior_thesis: dict | None) -> str:
    prior_text = json.dumps(prior_thesis, default=str) if prior_thesis else "not available"
    return (
        f"{RECOVERY_PROMPT_MARKER}\n"
        "You are doing a narrow bookkeeping task, not giving a new opinion or "
        "improving the thesis. Below is a researcher's own free-text review of "
        "their existing trade thesis, followed by the prior structured "
        "trade-plan state. Translate ONLY the changes the text already states "
        "into the minimal JSON patch below. Do not add, infer, or introduce "
        "any value that is not already stated in the text. If the text does "
        'not mention a field, use action "KEEP" for it.\n\n'
        "Respond with ONLY a single JSON object, no markdown fences, no "
        "commentary, matching exactly this shape:\n"
        '{"conviction_after": <0-100 integer>, "trade_plan_changes": '
        '{"entry": {"action": "KEEP or REVISE or WITHDRAW", "new": <number or null>}, '
        '"take_profit": {"action": "...", "new": <number or null>}, '
        '"stop_loss": {"action": "...", "new": <number or null>}}}\n\n'
        f"Prior structured trade plan:\n{prior_text}\n\n"
        "Researcher statement:\n"
        f"{fallback_text}\n"
    )


def _strip_json_fence(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        return "\n".join(lines[1:-1]) if len(lines) >= 3 else text
    return text


def _parse_patch(content: str, schema):
    try:
        data = json.loads(_strip_json_fence(content))
    except (json.JSONDecodeError, ValueError):
        return None
    try:
        return schema.model_validate(data)
    except ValidationError:
        return None


def _values_present_in_text(values: list[float], text: str) -> bool:
    """Every value the patch introduces must already be literally present in
    the fallback text it was derived from -- a deterministic membership check
    (reusing the existing generic number extractor), not a new semantic
    price/conviction parser. Rejects a patch that would let recovery "improve"
    or invent content beyond what the researcher actually wrote."""
    if not values:
        return True
    pool = extract_numbers(text)
    return all(numbers_match(value, pool) for value in values)


# A risk unit is only trusted when the fallback text itself explicitly labels
# a value with our own field's name -- "Suggested Risk Unit" or "Suggested
# Research Risk Unit" (the phrasing used across this codebase's prompts/
# schemas; see schemas.py/research_manager.txt) -- immediately followed by
# the value. This is deliberately NOT a general mapper for ambiguous risk
# language: "high conviction" or "high risk" alone must never match, only an
# explicit statement of this specific field.
_RISK_UNIT_LABEL_RE = re.compile(
    r"(?:suggested\s+)?research\s+risk\s+unit|suggested\s+risk\s+unit",
    re.IGNORECASE,
)
_RISK_UNIT_VALUE_RE = re.compile(
    r"\s*[:\-=]?\s*(NO_TRADE|NO\s+TRADE|LOW|MEDIUM|HIGH)\b", re.IGNORECASE
)


def _explicitly_labeled_risk_unit(
    fallback_text: str, declared: SuggestedRiskUnit | None
) -> SuggestedRiskUnit | None:
    """``declared`` (whatever the recovery JSON reported) is only trusted when
    the fallback text explicitly labels that exact value as the Suggested
    (Research) Risk Unit. Otherwise returns ``None`` -- a missing or
    unconfirmed risk unit must never fail the rest of the initial recovery,
    it just stays unrecorded (see module docstring and schemas.py)."""
    if declared is None:
        return None
    text = (fallback_text or "").replace("*", "")
    label_match = _RISK_UNIT_LABEL_RE.search(text)
    if label_match is None:
        return None
    value_match = _RISK_UNIT_VALUE_RE.match(text, label_match.end())
    if value_match is None:
        return None
    matched = value_match.group(1).upper().replace(" ", "_")
    if matched != declared.value:
        return None
    return declared


def _initial_patch_to_thesis_dict(patch: InitialThesisCanonicalPatch, fallback_text: str) -> dict:
    risk_unit = _explicitly_labeled_risk_unit(fallback_text, patch.suggested_risk_unit)
    return {
        "direction": patch.direction.value,
        "conviction": {
            "evidence_quality": 0,
            "internal_consistency": 0,
            "robustness_to_challenge": 0,
            "trade_plan_coherence": 0,
            "total": patch.conviction_total,
        },
        "horizon": "",
        "entry": patch.entry.model_dump(mode="json"),
        "take_profit": patch.take_profit,
        "stop_loss": patch.stop_loss,
        "suggested_risk_unit": risk_unit.value if risk_unit is not None else "not recorded",
    }


def _review_patch_to_review_dict(patch: ReviewCanonicalPatch) -> dict:
    return {
        "trade_plan_changes": patch.trade_plan_changes.model_dump(mode="json"),
        "conviction_after": patch.conviction_after,
    }


def recover_initial_thesis(
    fallback_text: str, llm, agent_name: str
) -> tuple[dict | None, CanonicalRecoveryStatus]:
    """Attempt to reconstruct a minimal canonical initial thesis from the
    free text ``agent_name`` just produced. Based exclusively on that text --
    no tools, no new evidence, no re-reading of analyst reports."""
    try:
        response = traced_invoke(
            llm, _initial_recovery_prompt(fallback_text), attempt="canonical_recovery"
        )
        patch = _parse_patch(response.content, InitialThesisCanonicalPatch)
        if patch is None:
            record_structured_parse_result(
                success=False, error="canonical recovery: invalid JSON/schema"
            )
            return None, CanonicalRecoveryStatus.UNAPPLIED

        values = [float(patch.conviction_total)]
        if patch.entry.type is EntryLevelType.POINT:
            if patch.entry.price is not None:
                values.append(patch.entry.price)
        else:
            if patch.entry.low is not None:
                values.append(patch.entry.low)
            if patch.entry.high is not None:
                values.append(patch.entry.high)
        if patch.take_profit is not None:
            values.append(patch.take_profit)
        if patch.stop_loss is not None:
            values.append(patch.stop_loss)

        if not _values_present_in_text(values, fallback_text):
            record_structured_parse_result(
                success=False,
                error="canonical recovery: a recovered value is not present in the fallback text",
            )
            return None, CanonicalRecoveryStatus.UNAPPLIED

        record_structured_parse_result(success=True, parsed=patch.model_dump(mode="json"))
        return _initial_patch_to_thesis_dict(patch, fallback_text), CanonicalRecoveryStatus.APPLIED
    except Exception as exc:
        record_structured_parse_result(success=False, error=str(exc))
        logger.debug("%s: canonical recovery (initial) raised: %s", agent_name, exc)
        return None, CanonicalRecoveryStatus.UNAPPLIED


def recover_review_patch(
    fallback_text: str, prior_thesis: dict | None, llm, agent_name: str
) -> tuple[dict | None, CanonicalRecoveryStatus]:
    """Attempt to reconstruct a minimal canonical review patch from the free
    text ``agent_name`` just produced, expressed as a mutation over
    ``prior_thesis`` -- never a full reconstruction of the thesis."""
    try:
        response = traced_invoke(
            llm,
            _review_recovery_prompt(fallback_text, prior_thesis),
            attempt="canonical_recovery",
        )
        patch = _parse_patch(response.content, ReviewCanonicalPatch)
        if patch is None:
            record_structured_parse_result(
                success=False, error="canonical recovery: invalid JSON/schema"
            )
            return None, CanonicalRecoveryStatus.UNAPPLIED

        values = [float(patch.conviction_after)]
        for change in (
            patch.trade_plan_changes.entry,
            patch.trade_plan_changes.take_profit,
            patch.trade_plan_changes.stop_loss,
        ):
            if change.action is TradePlanAction.REVISE and change.new is not None:
                values.append(change.new)

        if not _values_present_in_text(values, fallback_text):
            record_structured_parse_result(
                success=False,
                error="canonical recovery: a recovered value is not present in the fallback text",
            )
            return None, CanonicalRecoveryStatus.UNAPPLIED

        record_structured_parse_result(success=True, parsed=patch.model_dump(mode="json"))
        return _review_patch_to_review_dict(patch), CanonicalRecoveryStatus.APPLIED
    except Exception as exc:
        record_structured_parse_result(success=False, error=str(exc))
        logger.debug("%s: canonical recovery (review) raised: %s", agent_name, exc)
        return None, CanonicalRecoveryStatus.UNAPPLIED
