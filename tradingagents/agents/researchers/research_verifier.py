"""Independent quality-control agent for completed research debates."""

from __future__ import annotations

import json
import logging

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
from tradingagents.agents.schemas import ResearchVerification, VerificationStatus
from tradingagents.agents.structured import NO_EXTERNAL_TOOLS, bind_structured
from tradingagents.llm_trace import record_structured_parse_result, traced_invoke
from tradingagents.prompts.loader import render_agent_prompt

from .verification import normalize_verification

logger = logging.getLogger(__name__)


def _brief_error(exc: Exception) -> str:
    lines = [line.strip() for line in str(exc).splitlines() if line.strip()]
    return "; ".join(lines[:2])


def _parse_freetext_result(content: str) -> ResearchVerification:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]) if len(lines) >= 3 else text
    try:
        return ResearchVerification.model_validate(json.loads(text))
    except (json.JSONDecodeError, ValueError) as exc:
        warn_fallback(
            logger,
            "Research Verifier: JSON fallback could not be validated; "
            "continuing with WARN verification status.",
            detail=_brief_error(exc),
        )
        return ResearchVerification(
            status=VerificationStatus.WARN,
            notes=(
                "Verifier output could not be parsed into machine-identifiable "
                f"findings. Raw output excerpt: {content[:2000]}"
            ),
        )


def create_research_verifier(llm):
    structured_llm = bind_structured(llm, ResearchVerification, "Research Verifier")

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

        result = None
        if structured_llm is not None:
            try:
                result = traced_invoke(structured_llm, prompt, attempt="structured")
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
                traced_invoke(llm, prompt, attempt="verifier_json_fallback").content
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
