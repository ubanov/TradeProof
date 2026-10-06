"""Independent quality-control agent for completed research debates."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from pydantic import ValidationError

from tradingagents.agents.context import (
    get_instrument_context_from_state,
    get_language_instruction,
    report_or_absent,
)
from tradingagents.agents.fallback_log import warn_fallback
from tradingagents.agents.researchers.canonical_recovery import (
    downstream_safe_text,
    latest_side_recovery_status,
)
from tradingagents.agents.researchers.trade_metrics import (
    render_trade_metrics,
    trade_metrics_from_dict,
)
from tradingagents.agents.schemas import ArithmeticClaim, ResearchVerification, VerificationStatus
from tradingagents.agents.structured import NO_EXTERNAL_TOOLS, bind_structured
from tradingagents.llm_trace import record_structured_parse_result, traced_invoke
from tradingagents.prompts.loader import render_agent_prompt

from .verification import normalize_verification

logger = logging.getLogger(__name__)


def _brief_error(exc: Exception) -> str:
    lines = [line.strip() for line in str(exc).splitlines() if line.strip()]
    return "; ".join(lines[:2])


# A backslash immediately followed by a dollar sign is never a valid JSON
# escape (JSON only allows \" \\ \/ \b \f \n \r \t \uXXXX) -- some models
# produce it anyway when writing a monetary amount in Markdown style
# (``\$101.82B``), which makes an otherwise semantically valid JSON fallback
# fail to parse at all. Narrowly undoing just this one two-character
# sequence before parsing cannot touch any other, genuinely valid escape.
_DOLLAR_ESCAPE_RE = re.compile(r"\\\$")


def _normalize_dollar_escape(text: str) -> str:
    return _DOLLAR_ESCAPE_RE.sub("$", text)


def _is_valid_arithmetic_claim(item: Any) -> bool:
    try:
        ArithmeticClaim.model_validate(item)
    except ValidationError:
        return False
    return True


def _sanitize_arithmetic_entries(data: Any) -> Any:
    """Drop only the individual ``finding.arithmetic`` list entries that do
    not validate as ``ArithmeticClaim`` -- never the finding, never the whole
    document. A malformed entry (e.g. a positional
    ``["percentage_change", [a, b]]`` instead of the named-field object the
    schema expects) is missing information (here, ``reported_result``) that
    must never be invented or guessed from a positional shape; it is simply
    removed, exactly like any other un-reconstructable optional value. If
    nothing in the list survives, the field becomes ``[]`` -- the schema's
    own existing representation of "no machine-checkable arithmetic".
    """
    if not isinstance(data, dict):
        return data
    findings = data.get("findings")
    if findings is None:
        return data
    finding_list = findings if isinstance(findings, list) else [findings]
    for finding in finding_list:
        if not isinstance(finding, dict):
            continue
        arithmetic = finding.get("arithmetic")
        if not isinstance(arithmetic, list):
            continue
        finding["arithmetic"] = [item for item in arithmetic if _is_valid_arithmetic_claim(item)]
    return data


def _parse_freetext_result(content: str) -> ResearchVerification:
    """Validate the JSON fallback's content, preserving whatever the model
    actually returned (including a genuine FAIL with findings) rather than
    collapsing it to an empty WARN the moment validation raises for any
    reason -- see ``VerificationFinding._normalise_enum_casing`` for the
    specific, narrow casing tolerance this validation already allows, and
    ``_normalize_dollar_escape``/``_sanitize_arithmetic_entries`` above for
    the two additional narrow, structural-only normalizations applied before
    validation. None of this attempts to parse non-JSON prose: if ``text``
    never becomes valid JSON, nothing below ever runs and the conservative
    fallback at the bottom of this function still applies unchanged.

    Validation failure is not proof the model found nothing: the conservative
    WARN returned here says so explicitly in ``notes``, it does not claim
    "No findings" as if that had been verified.
    """
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]) if len(lines) >= 3 else text
    text = _normalize_dollar_escape(text)
    try:
        data = _sanitize_arithmetic_entries(json.loads(text))
        result = ResearchVerification.model_validate(data)
    except (json.JSONDecodeError, ValueError) as exc:
        record_structured_parse_result(success=False, error=str(exc))
        warn_fallback(
            logger,
            "Research Verifier: JSON fallback could not be validated; "
            "continuing with WARN verification status.",
            detail=_brief_error(exc),
        )
        return ResearchVerification(
            status=VerificationStatus.WARN,
            notes=(
                "Verification output unavailable/unvalidated: the JSON "
                "fallback could not be validated into machine-identifiable "
                "findings (this is not confirmation that no findings exist). "
                f"Raw output excerpt: {content[:2000]}"
            ),
        )
    record_structured_parse_result(success=True, parsed=result.model_dump(mode="json"))
    logger.debug("Research Verifier: JSON fallback recovered successfully.")
    return result


def create_research_verifier(review_llm):
    """Entirely a review operation (see fork README: "Model roles: Generation
    vs Review") -- both the structured attempt and its JSON fallback audit
    the already-produced debate, never generate new content."""
    structured_llm = bind_structured(review_llm, ResearchVerification, "Research Verifier")

    def verifier_node(state) -> dict:
        debate = state["investment_debate_state"]
        # A fallback whose canonical recovery failed (UNAPPLIED) must not be
        # read here as live narrative evidence -- only its bare status is
        # shown, never the failed fallback's own semantic content. The full
        # text is still preserved verbatim in bull_history/bear_history for
        # report/audit/debug; this only affects what this prompt sees.
        bull_position = (
            debate.get("bull_repair")
            or downstream_safe_text(
                debate.get("current_bull_response", ""),
                latest_side_recovery_status(debate, "bull"),
            )
            or downstream_safe_text(
                debate.get("bull_initial", ""),
                debate.get("bull_initial_recovery_status", "NOT_RUN"),
            )
        )
        bear_position = (
            debate.get("bear_repair")
            or downstream_safe_text(
                debate.get("current_bear_response", ""),
                latest_side_recovery_status(debate, "bear"),
            )
            or downstream_safe_text(
                debate.get("bear_initial", ""),
                debate.get("bear_initial_recovery_status", "NOT_RUN"),
            )
        )
        prompt = (
            render_agent_prompt(
                "researchers/verifier.txt",
                instrument_context=get_instrument_context_from_state(state),
                market_report=report_or_absent(state.get("market_report", ""), "market"),
                sentiment_report=report_or_absent(state.get("sentiment_report", ""), "sentiment"),
                news_report=report_or_absent(state.get("news_report", ""), "news"),
                fundamentals_report=report_or_absent(
                    state.get("fundamentals_report", ""), "fundamentals"
                ),
                bull_position=bull_position,
                bear_position=bear_position,
                review_outcomes=debate.get("review_outcomes", "")
                or "No classified review outcomes were produced.",
                bull_trade_metrics=render_trade_metrics(
                    trade_metrics_from_dict(debate.get("bull_trade_metrics"))
                ),
                bear_trade_metrics=render_trade_metrics(
                    trade_metrics_from_dict(debate.get("bear_trade_metrics"))
                ),
                NO_EXTERNAL_TOOLS=NO_EXTERNAL_TOOLS,
            )
            + get_language_instruction()
        )

        pass_identity = f"pass_{debate.get('verifier_pass_count', 0) + 1}"
        result = None
        if structured_llm is not None:
            try:
                result = traced_invoke(
                    structured_llm, prompt, attempt="structured",
                    call_identity=pass_identity, llm_role="review",
                )
                if result is None:
                    raise ValueError("structured output returned no parsed result")
                record_structured_parse_result(
                    success=True,
                    parsed=result.model_dump(mode="json") if hasattr(result, "model_dump") else None,
                )
            except Exception as exc:
                record_structured_parse_result(success=False, error=str(exc))
                warn_fallback(
                    logger,
                    "Research Verifier: structured output unavailable; "
                    "retrying with JSON fallback.",
                    detail=_brief_error(exc),
                )
        if result is None:
            result = _parse_freetext_result(
                traced_invoke(
                    review_llm, prompt, attempt="verifier_json_fallback",
                    call_identity=pass_identity, llm_role="review",
                ).content
            )

        result = normalize_verification(
            result
            if isinstance(result, ResearchVerification)
            else ResearchVerification.model_validate(result)
        )
        history = list(debate.get("verification_history", []))
        history.append(result.model_dump(mode="json"))
        new_debate = {
            **debate,
            "verification_history": history,
            "verifier_pass_count": debate.get("verifier_pass_count", 0) + 1,
            "verification_status": result.status.value,
            "verification_findings": result.model_dump(mode="json")["findings"],
        }
        return {"investment_debate_state": new_debate}

    return verifier_node
