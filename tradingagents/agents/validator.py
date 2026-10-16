"""Central Validator infrastructure (see fork README: "Generation /
Validation / Review").

Validator is a lightweight quality-control layer, not another analyst or
investment opinion: it checks contract/format/arithmetic/obvious consistency
on a candidate response BEFORE it moves to the next stage, and -- only on an
objective FAIL -- sends the SAME original agent (same LLM, same role, same
reasoning effort) back to correct ONLY the reported problems. Validator never
rewrites content itself and never produces a BUY/HOLD/SELL/rating decision.

    Agent produces candidate output
            |
    deterministic validation (always, cheap -- recorded for observability
            |                 even when validation is disabled)
    optional Validator LLM (validation_enabled only)
            |
          PASS/WARN? --yes--> next stage
            | FAIL
    validation_enabled? --no--> next stage (FAIL recorded, nothing repaired)
            | yes
    SAME original agent repairs findings (max ONE attempt)
            |
    final validation status (observability only -- never a second repair)

One reusable entry point, ``validate_and_repair``, is meant to be called by
each integrated stage with its own deterministic-check function and schema --
no stage-specific validation logic is duplicated here.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ValidationError

from tradingagents.agents.fallback_log import warn_fallback
from tradingagents.agents.integrity_shared import split_sentences
from tradingagents.agents.structured_recovery import recover_structured_from_text
from tradingagents.agents.validation_schemas import (
    ValidationCategory,
    ValidationFinding,
    ValidationResult,
    ValidationStatus,
)
from tradingagents.agents.validator_deterministic import check_text_relational_and_unit_errors
from tradingagents.llm_trace import (
    record_structured_parse_result,
    record_validator_debug,
    traced_invoke,
)

logger = logging.getLogger(__name__)

# Mirrors research_verifier.py's own dollar-escape fix, kept as a small,
# independent copy here rather than importing from that module -- the
# Research Verifier's internals are explicitly out of scope for this change
# (see fork README: "Generation / Validation / Review" -- Validator
# complements Review, Review's own code is untouched).
_DOLLAR_ESCAPE_RE = re.compile(r"\\\$")


def _normalize_dollar_escape(text: str) -> str:
    return _DOLLAR_ESCAPE_RE.sub("$", text)


@dataclass(frozen=True)
class ValidationOutcome:
    """Everything a caller needs for observability, plus the (possibly
    repaired) content to actually use going forward."""

    status: str  # ValidationStatus value (PASS/WARN/FAIL) -- the FINAL status
    findings: list[dict] = field(default_factory=list)
    repair_attempted: bool = False
    repair_succeeded: bool = False
    final_text: str = ""
    final_parsed: BaseModel | None = None
    llm_calls: int = 0
    # The status BEFORE any repair -- equal to `status` whenever no repair
    # was attempted. Kept separately so a real "initial FAIL -> repaired ->
    # final PASS" cycle stays visible in the summary even though the FINAL
    # status alone (what `fail`/`unresolved` count) would show PASS.
    initial_status: str = ""

    def as_log_entry(self, stage: str) -> dict:
        """Compact dict for AgentState's ``validation_log`` -- see fork
        README; never the full findings' evidence payloads duplicated into
        the main graph state."""
        return {
            "stage": stage,
            "status": self.status,
            "initial_status": self.initial_status or self.status,
            "finding_count": len(self.findings),
            "categories": sorted({f["category"] for f in self.findings}),
            "repair_attempted": self.repair_attempted,
            "repair_succeeded": self.repair_succeeded,
            "llm_calls": self.llm_calls,
        }


def bind_validator(validator_llm: Any, agent_name: str) -> Any | None:
    """Like ``structured.bind_structured``, but requests ``include_raw=True``:
    a failed parse must still expose the model's raw text so recovery can
    work on THIS SAME attempt's output, never by making a fresh, independently
    -judged call (see task: "ORDEN INCORRECTO DEL RECOVERY"). Falls back to
    no structured binding at all if the provider rejects ``include_raw``."""
    if validator_llm is None:
        return None
    try:
        return validator_llm.with_structured_output(ValidationResult, include_raw=True)
    except (NotImplementedError, AttributeError, TypeError) as exc:
        warn_fallback(
            logger,
            f"{agent_name} Validator: structured output unavailable; "
            "continuing with free-text generation.",
            detail=str(exc),
        )
        return None


def _validator_prompt(stage: str, instructions: str, evidence_context: str, candidate_text: str) -> str:
    """Compact, stage-specific context only -- never the full graph state.
    Deliberately narrow instructions keep the Validator from re-analyzing
    the investment case (that is Review's job, not Validator's)."""
    return (
        "You are a strict, narrow quality-control validator. Check ONLY "
        "objective contract/format/arithmetic/consistency problems in the "
        "candidate response below. You are NOT an investment analyst: do "
        "not judge investment quality, do not suggest a different "
        "action/rating, do not second-guess reasonable inferences. Only "
        "flag a problem if it is objectively checkable (a number that "
        "doesn't add up, a stated relationship that contradicts the cited "
        "values, a unit mix-up, a required element that is missing, a "
        "structured field that contradicts the narrative, or a value with "
        "no support in the inputs below).\n\n"
        f"Stage: {stage}\n\n"
        f"Original instructions/contract:\n{instructions}\n\n"
        f"Relevant input/evidence:\n{evidence_context}\n\n"
        f"Candidate response to check:\n{candidate_text}\n\n"
        "Respond with status PASS (no issues), WARN (minor, does not need "
        "correction), or FAIL (an objective problem that should be "
        "corrected before this continues), and a findings list (empty for "
        "PASS). Never recommend an action, rating, or disposition."
    )


def _validator_normalize_prompt(raw_response: str) -> str:
    """FORMAT REPAIR only, never revalidation (see task: "JSON FALLBACK NO
    DEBE REVALIDAR"): converts the validator's OWN prior raw response into
    the required JSON schema. Deliberately does NOT resend candidate/
    evidence/instructions -- doing so would let the model re-judge the case
    from scratch and could change PASS/WARN/FAIL on pure sampling variance,
    exactly the bug this corrects (a real FAIL from the first attempt was
    overwritten by a fresh, different verdict from the old "fallback" call)."""
    return (
        "Convert your previous validator response below into the required "
        "JSON schema. Preserve its status and findings exactly as stated -- "
        "do not re-evaluate the candidate, do not change PASS/WARN/FAIL, "
        "and do not add or remove substantive findings.\n\n"
        f"Your previous response:\n{raw_response}\n\n"
        "Return ONLY valid JSON matching this schema, nothing else:\n"
        '{"status": "PASS|WARN|FAIL", "findings": [{"category": "...", '
        '"claim": "...", "evidence": [], "reason": "...", '
        '"suggested_correction": "..."}]}\n\n'
        "No markdown. No prose before or after the JSON."
    )


# --- Limited Validator Recovery ---------------------------------------------
#
# A small/local model frequently fails BOTH the structured tool call and the
# JSON-only retry, while still answering the actual question unambiguously in
# plain text: a bare "PASS", "status: FAIL" followed by a couple of short
# bullet reasons, or YAML-shaped "status: WARN\nfindings:\n- ...". Before this
# recovery existed, all of these collapsed to the same conservative
# ValidationResult(WARN, []) as genuinely unparseable noise -- which silently
# swallowed every FAIL the model actually reported, so the FAIL -> repair ->
# recheck cycle was never exercised against this kind of model at all.
#
# This recovery is deliberately mechanical, never semantic: it only accepts
# text whose FIRST non-blank line is an unambiguous status token, refuses
# outright on any contradiction or excess length, and -- when a FAIL/WARN
# comes with short reason lines -- records them under one honest category,
# UNCLASSIFIED_VALIDATION_ERROR, rather than guessing a specific one. It
# never invents a finding the model didn't actually write.

_STATUS_LINE_RE = re.compile(r"(?i)^\s*(?:status\s*[:=]?\s*)?(PASS|WARN|FAIL)\s*[:.]?\s*$")
_BULLET_PREFIX_RE = re.compile(r"^(?:[-*•]|\d+[.)])\s*")
_MAX_RECOVERY_TEXT_LEN = 2000
_MAX_REASON_LINE_LEN = 300
_MAX_REASON_LINES = 10

# A line like "category: ARITHMETIC_ERROR" or "- claim: ...\" (after bullet-
# prefix stripping) is a YAML-ish finding FIELD, not a short reason -- a
# structured multi-field finding (category/claim/evidence/reason/
# suggested_correction/severity, each its own key) must never be fragmented
# into one UNCLASSIFIED_VALIDATION_ERROR finding per line (real-world
# regression: a single structured finding became seven). Any such line means
# this response needs real YAML/JSON structure-awareness this mechanical
# recovery deliberately doesn't have -- refuse outright and let the
# normalization fallback (which DOES understand structure, via the model
# converting its own prior answer to JSON) handle it instead.
_FINDING_FIELD_KEYS = ("category", "claim", "evidence", "reason", "suggested_correction", "severity")
_FIELD_KEY_LINE_RE = re.compile(r"(?i)^(?:" + "|".join(_FINDING_FIELD_KEYS) + r")\s*:")


def _limited_validator_recovery(text: str) -> ValidationResult | None:
    """Returns ``None`` (never raises) the moment anything is ambiguous --
    the caller falls back to the existing conservative WARN default in that
    case, exactly as it did before this recovery existed."""
    stripped = (text or "").strip()
    if not stripped or len(stripped) > _MAX_RECOVERY_TEXT_LEN:
        return None
    lines = [ln for ln in (raw.strip() for raw in stripped.splitlines()) if ln]
    if not lines:
        return None

    first_match = _STATUS_LINE_RE.match(lines[0])
    if first_match is None:
        return None
    status_token = first_match.group(1).upper()

    reason_lines: list[str] = []
    for line in lines[1:]:
        if line.rstrip(":").strip().lower() == "findings":
            continue  # a bare YAML-ish "findings:" header, not a reason
        other_match = _STATUS_LINE_RE.match(line)
        if other_match is not None:
            if other_match.group(1).upper() != status_token:
                # A second, contradicting status line -- refuse rather than
                # guess which one the model meant.
                return None
            continue
        reason = _BULLET_PREFIX_RE.sub("", line).strip()
        if _FIELD_KEY_LINE_RE.match(reason):
            # A structured, multi-field finding (YAML/mapping-shaped) --
            # too much structure for this mechanical recovery; refuse so the
            # normalization fallback handles it instead of fragmenting it.
            return None
        if reason:
            reason_lines.append(reason)

    if len(reason_lines) > _MAX_REASON_LINES or any(len(r) > _MAX_REASON_LINE_LEN for r in reason_lines):
        # Looks like a dumped analysis/full response, not a short findings
        # list -- too much to safely interpret mechanically.
        return None

    if status_token == "PASS":
        if reason_lines:
            return None  # PASS shouldn't carry findings -- ambiguous, refuse
        return ValidationResult(status=ValidationStatus.PASS, findings=[])

    findings = [
        ValidationFinding(
            category=ValidationCategory.UNCLASSIFIED_VALIDATION_ERROR,
            claim="", reason=reason, evidence=[], suggested_correction="",
        )
        for reason in reason_lines
    ]
    if status_token == "WARN":
        return ValidationResult(status=ValidationStatus.WARN, findings=findings)

    # FAIL: never invent a specific finding, but an entirely empty findings
    # list on a FAIL would leave the repair prompt with "(no findings)" to
    # act on -- one honest, unclassified placeholder is more useful than none.
    if not findings:
        findings = [
            ValidationFinding(
                category=ValidationCategory.UNCLASSIFIED_VALIDATION_ERROR,
                claim="", reason="Validator returned FAIL with no machine-parseable finding detail.",
                evidence=[], suggested_correction="",
            )
        ]
    return ValidationResult(status=ValidationStatus.FAIL, findings=findings)


def _sanitize_validation_payload(payload: dict) -> dict:
    """Syntactic-only null sanitization, NEVER semantic (see task: "NORMALIZER
    PIERDE FINDINGS VALIDOS POR CAMPOS NULL"). A model (notably a
    normalize-only retry) sometimes writes an explicit JSON ``null`` for an
    optional/auxiliary finding field that the schema itself already treats
    as empty by default -- ``suggested_correction``/``reason``/``claim``
    default to ``""``, ``evidence`` defaults to ``[]`` -- so rejecting the
    whole otherwise-valid finding over that null is unnecessarily fragile.
    Converts null to that SAME already-defined empty equivalent for exactly
    those fields; never touches ``category`` or the top-level ``status``
    (no safe empty equivalent exists for either -- a null there must keep
    failing validation conservatively, never be guessed at), never invents
    text, and never touches anything this function doesn't explicitly name.
    """
    if not isinstance(payload, dict):
        return payload
    findings = payload.get("findings")
    if not isinstance(findings, list):
        return payload
    sanitized_findings = []
    for finding in findings:
        if not isinstance(finding, dict):
            sanitized_findings.append(finding)
            continue
        finding = dict(finding)  # never mutate the caller's own dict
        for key in ("claim", "reason", "suggested_correction"):
            if finding.get(key) is None and key in finding:
                finding[key] = ""
        if finding.get("evidence") is None and "evidence" in finding:
            finding["evidence"] = []
        sanitized_findings.append(finding)
    return {**payload, "findings": sanitized_findings}


def _try_json_parse(text: str) -> ValidationResult | None:
    """Direct JSON/schema parse only, no new LLM call -- shared by the
    same-attempt recovery below and by the (separate) normalize-call parse."""
    try:
        payload = json.loads(_normalize_dollar_escape(text))
    except (json.JSONDecodeError, TypeError):
        return None
    if isinstance(payload, dict):
        payload = _sanitize_validation_payload(payload)
    try:
        return ValidationResult.model_validate(payload)
    except ValidationError:
        return None


def _recover_same_attempt_response(raw_text: str) -> ValidationResult | None:
    """Recovery over the SAME attempt's raw text, no new LLM call: a direct
    JSON/schema parse first (the model wrote JSON directly instead of using
    the tool-call shape), then the mechanical PASS/WARN/FAIL recovery above.
    ``None`` means neither could make sense of it safely."""
    parsed = _try_json_parse(raw_text)
    if parsed is not None:
        return parsed
    return _limited_validator_recovery(raw_text)


def _parse_freetext_validation_result(text: str) -> tuple[ValidationResult, str]:
    """Returns ``(result, path)`` where ``path`` is one of "json_fallback" /
    "limited_recovery" / "conservative_default", for trace/debug (see task:
    "structured_parse_success / json_fallback_success / limited_recovery_success")."""
    parsed = _try_json_parse(text)
    if parsed is not None:
        return parsed, "json_fallback"
    recovered = _limited_validator_recovery(text)
    if recovered is not None:
        return recovered, "limited_recovery"
    return ValidationResult(status=ValidationStatus.WARN, findings=[]), "conservative_default"


def _finalize_validator_response(
    raw_text: str | None,
    validator_llm: Any,
    *,
    stage: str,
    instructions: str,
    evidence_context: str,
    candidate_text: str,
    call_identity: str,
) -> ValidationResult:
    """Shared tail once a structured attempt has failed (or structured output
    was never available at all): try recovery on ``raw_text`` from THE SAME
    attempt first (direct JSON parse, then mechanical PASS/WARN/FAIL
    recovery -- see ``_recover_same_attempt_response``); only call the model
    again -- a short FORMAT-REPAIR normalization of that same raw text, never
    a fresh verdict -- when recovery is impossible. ``raw_text`` is ``None``
    only in the rare case no raw output was captured at all (e.g. a transport
    error), which is the one case that must ask the question itself."""
    if raw_text:
        recovered = _recover_same_attempt_response(raw_text)
        if recovered is not None:
            record_validator_debug({
                "structured_parse_success": False, "limited_recovery_success": True,
                "json_fallback_called": False, "json_fallback_success": False,
                "validator_raw_response": raw_text[:2000],
            })
            return recovered
        fallback_prompt = _validator_normalize_prompt(raw_text)
    else:
        fallback_prompt = _validator_prompt(stage, instructions, evidence_context, candidate_text)

    response = traced_invoke(
        validator_llm, fallback_prompt, attempt="validator_json_fallback",
        call_identity=call_identity, llm_role="validator",
    )
    result, path = _parse_freetext_validation_result(response.content)
    record_validator_debug({
        "structured_parse_success": False, "limited_recovery_success": False,
        "json_fallback_called": True, "json_fallback_success": path != "conservative_default",
        "validator_raw_response": raw_text[:2000] if raw_text else None,
    })
    return result


def invoke_validator(
    validator_llm: Any,
    *,
    stage: str,
    call_identity: str,
    instructions: str,
    evidence_context: str,
    candidate_text: str,
) -> ValidationResult:
    """One Validator judgment. Order (see task: "ORDEN INCORRECTO DEL
    RECOVERY"): (1) structured ``ValidationResult`` parse; (2) on failure,
    limited mechanical recovery over THIS SAME attempt's raw text -- never a
    second, independently-judged LLM call; (3) only if recovery is impossible,
    a short normalization-only retry of that exact raw text (never
    candidate/evidence again, so it can't silently overturn a real verdict);
    (4) the existing conservative WARN default."""
    structured_prompt = _validator_prompt(stage, instructions, evidence_context, candidate_text)
    structured_llm = bind_validator(validator_llm, stage)

    if structured_llm is None:
        # No structured-output capability at all -- this first plain call IS
        # the one attempt to recover from, exactly like the structured path.
        response = traced_invoke(
            validator_llm, structured_prompt, attempt="freetext",
            call_identity=call_identity, llm_role="validator",
        )
        return _finalize_validator_response(
            response.content, validator_llm, stage=stage, instructions=instructions,
            evidence_context=evidence_context, candidate_text=candidate_text,
            call_identity=call_identity,
        )

    try:
        raw_result = traced_invoke(
            structured_llm, structured_prompt, attempt="structured",
            call_identity=call_identity, llm_role="validator",
        )
    except Exception as exc:
        # The include_raw=True wrapper itself raised (a transport/provider
        # error, not a parse failure) -- no raw text was ever captured.
        record_structured_parse_result(success=False, error=str(exc))
        warn_fallback(
            logger, f"{stage} Validator: structured call failed; attempting recovery.",
            detail=str(exc),
        )
        return _finalize_validator_response(
            None, validator_llm, stage=stage, instructions=instructions,
            evidence_context=evidence_context, candidate_text=candidate_text,
            call_identity=call_identity,
        )

    raw_message = raw_result.get("raw") if isinstance(raw_result, dict) else None
    raw_text = getattr(raw_message, "content", None) if raw_message is not None else None
    parsing_error = raw_result.get("parsing_error") if isinstance(raw_result, dict) else None
    parsed_candidate = raw_result.get("parsed") if isinstance(raw_result, dict) else raw_result

    parsed_result = None
    if parsing_error is None and parsed_candidate is not None:
        try:
            parsed_result = (
                parsed_candidate if isinstance(parsed_candidate, ValidationResult)
                else ValidationResult.model_validate(parsed_candidate)
            )
        except ValidationError:
            parsed_result = None

    if parsed_result is not None:
        record_structured_parse_result(success=True, parsed=parsed_result.model_dump(mode="json"))
        record_validator_debug({
            "structured_parse_success": True, "limited_recovery_success": False,
            "json_fallback_called": False, "json_fallback_success": False,
        })
        return parsed_result

    record_structured_parse_result(
        success=False, error=str(parsing_error) if parsing_error else "schema mismatch",
    )
    warn_fallback(
        logger, f"{stage} Validator: structured output unavailable; attempting recovery.",
        detail=str(parsing_error) if parsing_error else "schema mismatch",
    )
    return _finalize_validator_response(
        raw_text, validator_llm, stage=stage, instructions=instructions,
        evidence_context=evidence_context, candidate_text=candidate_text,
        call_identity=call_identity,
    )


# --- Mechanical post-filter for LLM findings we can also check ourselves ---
#
# A local model occasionally flags a value as UNSUPPORTED_NUMBER or
# UNKNOWN_USED_AS_FACT when that value is, in fact, right there in the
# evidence the Validator itself was given -- a provenance false positive,
# not a real defect. This filter only drops a finding when it can POSITIVELY
# confirm the model was wrong using the same evidence the model had; whenever
# it cannot determine that reliably, it keeps the finding (conservative by
# default, exactly like the recovery layer above).

_NUMBER_RE = re.compile(r"\$?-?\d[\d,]*(?:\.\d+)?")
_UNKNOWN_MARKERS = ("unknown", "n/a", "not available", "not provided", "undetermined", "not determinable")


def _extract_numbers(text: str) -> set[float]:
    numbers: set[float] = set()
    for match in _NUMBER_RE.finditer(text or ""):
        cleaned = match.group(0).replace("$", "").replace(",", "")
        try:
            numbers.add(float(cleaned))
        except ValueError:
            continue
    return numbers


def _finding_claimed_numbers(finding: ValidationFinding) -> set[float]:
    numbers = _extract_numbers(finding.claim)
    for item in finding.evidence:
        numbers |= _extract_numbers(item)
    return numbers


def _why_llm_finding_should_be_dropped(
    finding: ValidationFinding,
    det_findings: list[ValidationFinding],
    evidence_context: str,
) -> str | None:
    """Returns a human-readable drop reason, or ``None`` to keep the finding.

    ``evidence_context`` is the ONLY source checked for provenance --
    ``candidate_text`` never counts as evidence (see task: "UNSUPPORTED_NUMBER
    ... candidate_text NO cuenta como evidence"): the candidate is what's
    being checked, so a number appearing in it proves nothing about whether
    it's supported."""
    if finding.category == ValidationCategory.UNSUPPORTED_NUMBER:
        claimed = _finding_claimed_numbers(finding)
        # No number we can extract from the finding itself -- nothing to
        # positively confirm, so this one is left for Review's judgment.
        if claimed and claimed & _extract_numbers(evidence_context):
            return "unsupported number found in supplied evidence"
        return None

    if finding.category == ValidationCategory.UNKNOWN_USED_AS_FACT:
        claimed = _finding_claimed_numbers(finding)
        if not claimed:
            return None  # can't determine reliably -- keep
        for sentence in split_sentences(evidence_context):
            if claimed & _extract_numbers(sentence):
                if any(marker in sentence.lower() for marker in _UNKNOWN_MARKERS):
                    return None  # genuinely marked unknown nearby -- keep
                return "referenced value is not marked unknown in supplied evidence"
        return None  # value never appears in evidence -- can't determine -- keep

    if finding.category in (ValidationCategory.RELATIONAL_ERROR, ValidationCategory.UNIT_ERROR):
        # Corroborated by the stage's own deterministic pass -- never drop.
        if any(d.category == finding.category for d in det_findings):
            return None
        claim_text = (finding.claim or "").strip()
        if not claim_text:
            return None  # nothing to re-verify -- keep
        # Re-run the SAME deterministic scanner on the LLM's own cited claim
        # text (never a new/duplicate implementation of the check itself):
        # if it finds nothing wrong there, the deterministic result prevails.
        recheck_categories = {f.category for f in check_text_relational_and_unit_errors(claim_text)}
        if finding.category in recheck_categories:
            return None  # deterministic re-check reproduces the same problem
        return "contradicted by deterministic recheck; deterministic result prevails"

    return None


def _filter_llm_findings(
    llm_result: ValidationResult,
    det_findings: list[ValidationFinding],
    evidence_context: str,
) -> tuple[ValidationResult, list[dict]]:
    """Objective 3: drop LLM findings we can mechanically disprove, BEFORE
    ``_combine`` decides the overall status. Returns ``(filtered_result,
    dropped)`` where ``dropped`` is ``[{"finding": ..., "reason": ...}]`` for
    trace/debug -- a finding is never silently discarded without a reason."""
    if not llm_result.findings:
        return llm_result, []
    kept: list[ValidationFinding] = []
    dropped: list[dict] = []
    for finding in llm_result.findings:
        reason = _why_llm_finding_should_be_dropped(finding, det_findings, evidence_context)
        if reason is not None:
            dropped.append({"finding": finding.model_dump(mode="json"), "reason": reason})
        else:
            kept.append(finding)
    if not dropped:
        return llm_result, []
    # 0 findings left -> PASS (never keep an empty FAIL); any remaining
    # finding keeps the model's own original status -- this filter only
    # removes confirmed false positives, it never re-judges severity itself.
    new_status = llm_result.status if kept else ValidationStatus.PASS
    return ValidationResult(status=new_status, findings=kept), dropped


def _run_validation_pass(
    *,
    validator_llm: Any | None,
    stage: str,
    call_identity: str,
    instructions: str,
    evidence_context: str,
    candidate_text: str,
    det_findings: list[ValidationFinding],
) -> tuple[str, list[dict], int]:
    """One full validation pass (deterministic findings the caller already
    computed + optional Validator LLM + Objective-3 false-positive filter),
    shared by both the initial check and the post-repair recheck so neither
    duplicates this logic. Returns ``(status, findings, llm_calls)``."""
    llm_result = None
    llm_calls = 0
    if validator_llm is not None:
        llm_result = invoke_validator(
            validator_llm, stage=stage, call_identity=call_identity,
            instructions=instructions, evidence_context=evidence_context,
            candidate_text=candidate_text,
        )
        llm_calls += 1
        raw_findings = [f.model_dump(mode="json") for f in llm_result.findings]
        llm_result, dropped = _filter_llm_findings(llm_result, det_findings, evidence_context)
        record_validator_debug({
            "validator_findings_raw": raw_findings,
            "validator_findings_dropped": dropped,
            "validator_findings_final": [f.model_dump(mode="json") for f in llm_result.findings],
        })
    status, findings = _combine(det_findings, llm_result)
    return status, findings, llm_calls


def _combine(
    deterministic_findings: list[ValidationFinding], llm_result: ValidationResult | None,
) -> tuple[str, list[dict]]:
    """Deterministic findings are, by construction, objective defects --
    treated as FAIL regardless of what the (optional) Validator LLM says.
    The LLM result's own status is otherwise authoritative."""
    findings = [f.model_dump(mode="json") for f in deterministic_findings]
    if llm_result is not None:
        findings.extend(f.model_dump(mode="json") for f in llm_result.findings)

    if deterministic_findings or (llm_result is not None and llm_result.status is ValidationStatus.FAIL):
        return ValidationStatus.FAIL.value, findings
    if llm_result is not None and llm_result.status is ValidationStatus.WARN:
        return ValidationStatus.WARN.value, findings
    return ValidationStatus.PASS.value, findings


def _build_repair_prompt(
    stage: str, instructions: str, evidence_context: str,
    candidate_text: str, findings: list[dict],
) -> str:
    findings_text = "\n".join(
        f"- [{f['category']}] {f['claim']} -- {f.get('reason', '')}"
        + (f" (suggested: {f['suggested_correction']})" if f.get("suggested_correction") else "")
        for f in findings
    ) or "(no findings)"
    return (
        f"Stage: {stage}\n\n"
        f"Original instructions/contract:\n{instructions}\n\n"
        f"Relevant input/evidence:\n{evidence_context}\n\n"
        f"Your previous response:\n{candidate_text}\n\n"
        f"Validation findings to correct:\n{findings_text}\n\n"
        "Correct ONLY the reported validation problems above. Preserve "
        "every valid claim, conclusion, and number not involved in a "
        "finding. Do not introduce new evidence, levels, thresholds, or "
        "recommendations. Return the response using the exact same "
        "required contract/format as your previous response."
    )


def validate_and_repair(
    *,
    stage: str,
    validator_llm: Any | None,
    original_llm: Any,
    original_role: str,
    call_identity_base: str,
    instructions: str,
    evidence_context: str,
    candidate_text: str,
    candidate_parsed: BaseModel | None,
    expected_schema: type[BaseModel] | None,
    render_fn: Callable[[BaseModel], str] | None,
    deterministic_check_fn: Callable[[str, BaseModel | None], list[ValidationFinding]],
) -> ValidationOutcome:
    """The one central entry point every integrated stage calls.

    ``deterministic_check_fn(text, parsed) -> list[ValidationFinding]`` is
    supplied by the caller (it knows its own schema/fields) and is re-run
    after a repair, so the recheck reflects the actual repaired content --
    this function never guesses at a stage's structured fields itself.

    ``validator_llm=None`` means validation is disabled (the caller already
    resolved this from ``validation_enabled``): deterministic checks still
    run and their result is still returned for observability, but -- unlike
    an earlier version of this function -- a deterministic FAIL can NEVER
    trigger a repair, never calls the Validator LLM, and never mutates the
    candidate when validation is disabled (see task: "VALIDATION_ENABLED").
    Only when ``validator_llm`` is configured can a FAIL (deterministic or
    from the Validator LLM) trigger the one bounded repair below.
    """
    det_findings = deterministic_check_fn(candidate_text, candidate_parsed)
    status, findings, llm_calls = _run_validation_pass(
        validator_llm=validator_llm, stage=stage,
        call_identity=f"{call_identity_base}_validation",
        instructions=instructions, evidence_context=evidence_context,
        candidate_text=candidate_text, det_findings=det_findings,
    )
    if status != ValidationStatus.FAIL.value or validator_llm is None:
        final_parsed = candidate_parsed
        # Recovery applies only to content the system actually ACCEPTS as
        # PASS/WARN -- never to a FAIL that simply couldn't be repaired
        # because validation is disabled (validator_llm is None): that text
        # was never accepted, it is a known, unrepaired FAIL, and recovering
        # a structured object from it would make a flagged candidate look
        # as trustworthy as an accepted one.
        if final_parsed is None and expected_schema is not None and status != ValidationStatus.FAIL.value:
            # The accepted text never went through a repair cycle -- but the
            # caller's own initial structured attempt can itself have fallen
            # back to free text (e.g. the model answered without a tool
            # call), in which case candidate_parsed was already None on
            # arrival here, not something THIS function broke. Real-run
            # regression (2026-10-16, NVDA): Research Manager's structured
            # attempt produced no tool call, the free-text fallback was
            # accepted as PASS, and research_plan_structured stayed null
            # forever -- canonical recovery (commit 8bccba7) only ever ran
            # inside the repair branch below, never on this earlier, more
            # common path. One bounded attempt, same mechanism the repair
            # branch already uses, never a loop and never re-judging
            # PASS/WARN/FAIL.
            final_parsed = recover_structured_from_text(
                candidate_text, expected_schema, original_llm, stage, llm_role=original_role,
            )
        return ValidationOutcome(
            status=status, findings=findings, final_text=candidate_text,
            final_parsed=final_parsed, llm_calls=llm_calls, initial_status=status,
        )

    # Visible only on FAIL/repair -- never one line per PASS (see fork
    # README: "Generation / Validation / Review"). The trace/report carries
    # the full finding detail; this is just enough to notice from the console.
    print(
        f"[WARN] {stage} validation failed ({len(findings)} finding(s)); attempting one repair.",
        flush=True,
    )

    # Exactly one repair, by the SAME original LLM/role that produced the
    # candidate -- Validator never rewrites semantic content itself.
    repair_prompt = _build_repair_prompt(stage, instructions, evidence_context, candidate_text, findings)
    repaired = traced_invoke(
        original_llm, repair_prompt, attempt="validation_repair",
        llm_role=original_role, call_identity=f"{call_identity_base}_validation_repair",
    )
    repaired_text = repaired.content

    repaired_parsed = None
    if expected_schema is not None:
        try:
            repaired_parsed = expected_schema.model_validate_json(_normalize_dollar_escape(repaired_text))
        except (ValidationError, ValueError):
            try:
                repaired_parsed = expected_schema.model_validate(json.loads(_normalize_dollar_escape(repaired_text)))
            except (ValidationError, ValueError, json.JSONDecodeError):
                repaired_parsed = None
        if repaired_parsed is None:
            # The repair call returned free text, not expected_schema's own
            # JSON shape -- try canonical recovery (same architecture as
            # researchers/canonical_recovery.py, generalized to an
            # arbitrary schema; see structured_recovery.py) before giving
            # up on a structured object entirely. A response that goes on
            # to PASS/WARN on the recheck below must not still leave
            # final_parsed stuck at None just because THIS repair happened
            # to come back as prose (real-run regression: Research
            # Manager/Portfolio Manager repairs both did, and
            # research_plan_structured/portfolio_decision_structured ended
            # up null even though the repaired text was accepted).
            repaired_parsed = recover_structured_from_text(
                repaired_text, expected_schema, original_llm, stage, llm_role=original_role,
            )

    final_text = render_fn(repaired_parsed) if (repaired_parsed is not None and render_fn is not None) else repaired_text

    # Recheck only -- never another repair, no matter the outcome.
    recheck_det_findings = deterministic_check_fn(final_text, repaired_parsed)
    final_status, final_findings, recheck_llm_calls = _run_validation_pass(
        validator_llm=validator_llm, stage=stage,
        call_identity=f"{call_identity_base}_validation_recheck",
        instructions=instructions, evidence_context=evidence_context,
        candidate_text=final_text, det_findings=recheck_det_findings,
    )
    llm_calls += recheck_llm_calls
    if final_status == ValidationStatus.FAIL.value:
        print(f"[WARN] {stage} validation remains {final_status} after repair.", flush=True)
    return ValidationOutcome(
        status=final_status, findings=final_findings, repair_attempted=True,
        repair_succeeded=final_status != ValidationStatus.FAIL.value,
        final_text=final_text, final_parsed=repaired_parsed, llm_calls=llm_calls,
        initial_status=status,
    )


def summarize_validation_log(validation_log: list[dict]) -> dict:
    """Compact counters for a whole run's ``AgentState.validation_log`` (see
    fork README: "Generation / Validation / Review") -- used by both the
    saved report and the CLI/batch summary line, so neither recomputes this
    differently. ``unresolved`` is a stage still FAIL after its one repair
    attempt (or FAIL with no repair possible, e.g. no ``validator_llm``).

    ``fail``/``unresolved`` alone hide a real "initial FAIL -> repaired ->
    final PASS" cycle (the final status is PASS, so neither counts it) --
    ``initial_failures``/``repairs_attempted``/``repairs_succeeded``/
    ``final_unresolved`` make that visible without changing what the
    existing fields mean, for any caller already consuming them."""
    checked = len(validation_log)
    passed = sum(1 for e in validation_log if e["status"] == ValidationStatus.PASS.value)
    warned = sum(1 for e in validation_log if e["status"] == ValidationStatus.WARN.value)
    failed = sum(1 for e in validation_log if e["status"] == ValidationStatus.FAIL.value)
    repairs = sum(1 for e in validation_log if e["repair_attempted"])
    unresolved = sum(
        1 for e in validation_log
        if e["status"] == ValidationStatus.FAIL.value and not e["repair_succeeded"]
    )
    llm_calls = sum(e.get("llm_calls", 0) for e in validation_log)
    initial_failures = sum(
        1 for e in validation_log
        if e.get("initial_status", e["status"]) == ValidationStatus.FAIL.value
    )
    repairs_succeeded = sum(
        1 for e in validation_log if e["repair_attempted"] and e["repair_succeeded"]
    )
    return {
        "checked": checked,
        "pass": passed,
        "warn": warned,
        "fail": failed,
        "repairs": repairs,
        "unresolved": unresolved,
        "llm_calls": llm_calls,
        "initial_failures": initial_failures,
        "repairs_attempted": repairs,
        "repairs_succeeded": repairs_succeeded,
        "final_unresolved": unresolved,
    }


def render_validation_summary(validation_log: list[dict]) -> str:
    """Compact Markdown section for the saved report -- counts plus only the
    unresolved findings' detail, never a full dump of every PASS stage."""
    if not validation_log:
        return ""
    counts = summarize_validation_log(validation_log)
    lines = [
        "## Validation",
        "",
        (
            f"- Stages checked: {counts['checked']} / pass: {counts['pass']} / "
            f"warn: {counts['warn']} / fail: {counts['fail']} / "
            f"repairs: {counts['repairs']} / unresolved: {counts['unresolved']}"
        ),
        (
            f"- Initial failures: {counts['initial_failures']} / "
            f"repairs attempted: {counts['repairs_attempted']} / "
            f"repairs succeeded: {counts['repairs_succeeded']} / "
            f"final unresolved: {counts['final_unresolved']}"
        ),
    ]
    unresolved_entries = [
        e for e in validation_log
        if e["status"] == ValidationStatus.FAIL.value and not e["repair_succeeded"]
    ]
    if unresolved_entries:
        lines.append("")
        lines.append("Unresolved findings:")
        for entry in unresolved_entries:
            categories = ", ".join(entry["categories"]) or "(uncategorized)"
            lines.append(
                f"- **{entry['stage']}**: {entry['finding_count']} finding(s) "
                f"[{categories}]"
                + (" (repair attempted)" if entry["repair_attempted"] else "")
            )
    return "\n".join(lines) + "\n"


def repair_json_syntax(
    validator_llm: Any, *, stage: str, call_identity: str, raw_text: str, schema: type[BaseModel],
) -> BaseModel | None:
    """A narrow, separate repair for JSON/schema syntax ONLY -- used around
    the EXISTING structured/fallback mechanism (never replacing it): when
    deterministic JSON/schema validation fails, this asks the Validator LLM
    to fix syntax/shape without touching any substantive content, then
    re-parses deterministically. Returns ``None`` (never raises) if it can't
    be repaired safely, so the caller's existing fallback path still applies.
    """
    if validator_llm is None:
        return None
    prompt = (
        f"Stage: {stage}\n\n"
        f"The following response was supposed to be valid JSON matching this "
        f"schema: {schema.__name__}, but is not.\n\n"
        f"Response:\n{raw_text}\n\n"
        "Repair syntax/schema only. Preserve every number, claim, "
        "conclusion, and recommendation exactly. Do not add or remove "
        "substantive content. Return ONLY the corrected JSON object."
    )
    try:
        response = traced_invoke(
            validator_llm, prompt, attempt="syntax_repair",
            llm_role="validator", call_identity=f"{call_identity}_syntax_repair",
        )
        return schema.model_validate_json(_normalize_dollar_escape(response.content))
    except (ValidationError, ValueError, json.JSONDecodeError):
        return None
    except Exception:  # pragma: no cover - provider/network failure, never fatal
        logger.debug("%s: Validator syntax repair raised", stage, exc_info=True)
        return None
