"""Shared structured-output flow for the Risk Management reviewers.

Mirrors ``researchers/thesis_flow.py``'s pattern: a structured assessment
rides alongside a rendered markdown block, with a graceful fallback to plain
text when the provider does not support structured output -- the same
degrade path already proven for the Trader, Research Manager, Portfolio
Manager, and Bull/Bear researchers.
"""

from __future__ import annotations

import logging

from tradingagents.agents.fallback_log import warn_fallback
from tradingagents.agents.schemas import RiskStanceAssessment, render_risk_assessment
from tradingagents.llm_trace import record_structured_parse_result, traced_invoke

logger = logging.getLogger(__name__)


def invoke_risk_assessment(
    structured_llm, plain_llm, prompt, agent_name: str,
    *, call_identity: str | None = None, llm_role: str | None = None,
):
    """Return ``(rendered_text, risk_level_or_None, disposition_or_None, assessment_dict_or_None)``.

    On free-text fallback, only ``rendered_text`` is available: the caller
    must not invent a risk_level/disposition/assessment that was never
    produced.

    ``call_identity`` is forwarded to ``traced_invoke`` for best-effort
    deterministic replay (see ``tradingagents/run_seed.py``). ``llm_role``
    ("generation"/"review", see fork README: "Model roles: Generation vs
    Review") is forwarded the same way.
    """
    if structured_llm is not None:
        try:
            result = traced_invoke(
                structured_llm, prompt, attempt="structured",
                call_identity=call_identity, llm_role=llm_role,
            )
            if result is None:
                raise ValueError("structured output returned no parsed result")
            if not isinstance(result, RiskStanceAssessment):
                result = RiskStanceAssessment.model_validate(result)
            record_structured_parse_result(success=True, parsed=result.model_dump(mode="json"))
            return (
                render_risk_assessment(result),
                result.risk_level,
                result.disposition.value,
                result.model_dump(mode="json"),
            )
        except Exception as exc:
            record_structured_parse_result(success=False, error=str(exc))
            warn_fallback(
                logger,
                f"{agent_name}: structured risk assessment unavailable; "
                "continuing with free-text fallback.",
                detail=str(exc),
            )
    return (
        traced_invoke(
            plain_llm, prompt, attempt="fallback", call_identity=call_identity, llm_role=llm_role,
        ).content,
        None,
        None,
        None,
    )
