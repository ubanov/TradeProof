"""Research Manager: turns the bull/bear debate into a structured investment plan for the trader."""

from __future__ import annotations

import logging

from tradingagents.agents.context import get_instrument_context_from_state, get_language_instruction
from tradingagents.agents.managers.integrity_check import check_manager_integrity
from tradingagents.agents.researchers.canonical_recovery import (
    downstream_safe_text,
    latest_side_recovery_status,
)
from tradingagents.agents.researchers.research_summary import render_repair_status_caveat
from tradingagents.agents.researchers.setup_tags import render_setup_tags
from tradingagents.agents.researchers.trade_metrics import (
    render_trade_metrics,
    trade_metrics_from_dict,
)
from tradingagents.agents.researchers.verification import render_verification_history
from tradingagents.agents.schemas import ResearchPlan, render_research_plan
from tradingagents.agents.structured import (
    NO_EXTERNAL_TOOLS,
    bind_structured,
    invoke_structured_or_freetext,
)
from tradingagents.prompts.loader import render_agent_prompt

logger = logging.getLogger(__name__)


def create_research_manager(llm):
    structured_llm = bind_structured(llm, ResearchPlan, "Research Manager")

    def research_manager_node(state) -> dict:
        instrument_context = get_instrument_context_from_state(state)
        investment_debate_state = state["investment_debate_state"]
        # A fallback whose canonical recovery failed (UNAPPLIED) must not be
        # read here as if it were live narrative evidence -- only its bare
        # status is shown, never the failed fallback's own semantic content
        # (a stray "conviction 57 -> 50" or "TP 240" in that prose). The full
        # text is still preserved verbatim in bull_history/bear_history for
        # report/audit/debug; this only affects what this prompt sees.
        bull_initial = downstream_safe_text(
            investment_debate_state.get("bull_initial")
            or investment_debate_state.get("bull_history", ""),
            investment_debate_state.get("bull_initial_recovery_status", "NOT_RUN"),
        )
        bear_initial = downstream_safe_text(
            investment_debate_state.get("bear_initial")
            or investment_debate_state.get("bear_history", ""),
            investment_debate_state.get("bear_initial_recovery_status", "NOT_RUN"),
        )
        bull_repair = investment_debate_state.get("bull_repair", "")
        bear_repair = investment_debate_state.get("bear_repair", "")
        repair_outputs = "\n\n".join(
            output
            for output in (bull_repair, bear_repair)
            if output
        ) or "No repair round was required."

        def _conviction_trail(history: list[int]) -> str:
            return " -> ".join(str(value) for value in history) if history else "not recorded"

        def _risk_unit(thesis: dict) -> str:
            return thesis.get("suggested_risk_unit", "not recorded") if thesis else "not recorded"

        # The prompt states: conflict alone is not a reason to Hold.
        prompt = render_agent_prompt(
            "managers/research_manager.txt",
            instrument_context=instrument_context,
            bull_initial=bull_initial,
            bear_initial=bear_initial,
            review_outcomes=investment_debate_state.get("review_outcomes", "")
            or "No classified review outcomes were produced.",
            verification_summary=render_verification_history(
                investment_debate_state.get("verification_history", []),
                repaired_agents=investment_debate_state.get("repaired_agents", []),
                include_details=True,
            ),
            repair_outputs=repair_outputs,
            repair_structured_note=render_repair_status_caveat(investment_debate_state)
            or "No repair fell back to free text.",
            final_bull_response=bull_repair
            or downstream_safe_text(
                investment_debate_state.get("current_bull_response", ""),
                latest_side_recovery_status(investment_debate_state, "bull"),
            )
            or bull_initial,
            final_bear_response=bear_repair
            or downstream_safe_text(
                investment_debate_state.get("current_bear_response", ""),
                latest_side_recovery_status(investment_debate_state, "bear"),
            )
            or bear_initial,
            research_horizon=investment_debate_state.get("research_horizon", "not recorded"),
            setup_tags=render_setup_tags(investment_debate_state.get("setup_tags") or {}),
            bull_conviction_trail=_conviction_trail(
                investment_debate_state.get("bull_conviction_history") or []
            ),
            bear_conviction_trail=_conviction_trail(
                investment_debate_state.get("bear_conviction_history") or []
            ),
            bull_suggested_risk_unit=_risk_unit(investment_debate_state.get("bull_thesis") or {}),
            bear_suggested_risk_unit=_risk_unit(investment_debate_state.get("bear_thesis") or {}),
            bull_trade_metrics=render_trade_metrics(
                trade_metrics_from_dict(investment_debate_state.get("bull_trade_metrics"))
            ),
            bear_trade_metrics=render_trade_metrics(
                trade_metrics_from_dict(investment_debate_state.get("bear_trade_metrics"))
            ),
            NO_EXTERNAL_TOOLS=NO_EXTERNAL_TOOLS,
        ) + get_language_instruction()

        investment_plan = invoke_structured_or_freetext(
            structured_llm,
            llm,
            prompt,
            render_research_plan,
            "Research Manager",
        )

        # Deterministic, LLM-free pass: the Verifier already checked the Bull/
        # Bear research, but the manager is itself a free-text-capable call and
        # could reintroduce an unsupported numeric threshold or sizing rule
        # verification never saw (fork README: "Research Manager integrity
        # check"). No LLM call, no correction loop -- see integrity_check.py.
        integrity_result = check_manager_integrity(
            investment_plan,
            investment_debate_state,
            state.get("market_report", ""),
            state.get("sentiment_report", ""),
            state.get("news_report", ""),
            state.get("fundamentals_report", ""),
        )
        if integrity_result.findings:
            logger.warning(
                "Manager integrity check: %s (%d finding(s))",
                integrity_result.status, len(integrity_result.findings),
            )
        else:
            logger.info("Manager integrity check: %s", integrity_result.status)

        new_investment_debate_state = {
            **investment_debate_state,
            "judge_decision": investment_plan,
            "current_response": investment_plan,
            "manager_integrity_status": integrity_result.status,
            "manager_integrity_findings": integrity_result.as_dict()["findings"],
        }

        return {
            "investment_debate_state": new_investment_debate_state,
            "investment_plan": investment_plan,
        }

    return research_manager_node
