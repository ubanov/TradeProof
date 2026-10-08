"""Portfolio Manager: synthesises the risk-analyst debate into the final decision.

Uses LangChain's ``with_structured_output`` so the LLM produces a typed
``PortfolioDecision`` directly, in a single call.  The result is rendered
back to markdown for storage in ``final_trade_decision`` so memory log,
CLI display, and saved reports continue to consume the same shape they do
today.  When a provider does not expose structured output, the agent falls
back gracefully to free-text generation.
"""

from __future__ import annotations

from tradingagents.agents.context import (
    get_instrument_context_from_state,
    get_language_instruction,
    get_portfolio_context_from_state,
)
from tradingagents.agents.rating import extract_choice
from tradingagents.agents.risk_mgmt.integrity_check import (
    check_risk_integrity,
    render_risk_integrity_notice,
)
from tradingagents.agents.schemas import (
    PortfolioDecision,
    RiskDisposition,
    render_pm_decision,
    render_risk_stance_summary,
)
from tradingagents.agents.structured import (
    NO_EXTERNAL_TOOLS,
    bind_structured,
    invoke_structured_or_freetext_parsed,
)
from tradingagents.agents.validator import validate_and_repair
from tradingagents.agents.validator_deterministic import check_text_relational_and_unit_errors
from tradingagents.prompts.loader import render_agent_prompt

# Compact, hand-written contract summary for the Validator (see fork README:
# "Generation / Validation / Review") -- not the full rendered prompt.
_PM_CONTRACT = (
    "Produce a final decision: rating (Buy/Overweight/Hold/Underweight/Sell), "
    "disposition toward the Trader's plan (KEEP/DEFER/REDUCE_RISK/REJECT_PLAN, "
    "a separate concept from rating), executive_summary, investment_thesis, "
    "optional price_target, optional time_horizon."
)


def _pm_deterministic_checks(text: str, _parsed):
    return check_text_relational_and_unit_errors(text)


def create_portfolio_manager(generation_llm, validator_llm=None):
    """The critical review already happened in Risk's cross-review; Portfolio
    Manager synthesizes and decides -- a generation operation (see fork
    README: "Model roles: Generation vs Review"). ``validator_llm`` is
    ``None`` unless validation is enabled; deterministic checks (unit/
    relational errors) run unconditionally either way."""
    structured_llm = bind_structured(generation_llm, PortfolioDecision, "Portfolio Manager")

    def portfolio_manager_node(state) -> dict:
        instrument_context = get_instrument_context_from_state(state)
        portfolio_context = get_portfolio_context_from_state(state)

        history = state["risk_debate_state"]["history"]
        risk_debate_state = state["risk_debate_state"]
        research_plan = state["investment_plan"]
        trader_plan = state["trader_investment_plan"]

        past_context = state.get("past_context", "")
        lessons_line = (
            f"- Lessons from prior decisions and outcomes:\n{past_context}\n"
            if past_context
            else ""
        )

        # Deterministic, LLM-free pass over the completed Risk phase (both
        # independent assessments and the cross-review), before this node's
        # own LLM call, so Portfolio Manager can see the findings in its own
        # prompt (fork README: "Risk integrity check").
        integrity_result = check_risk_integrity(history, state)
        risk_integrity_notice = render_risk_integrity_notice(
            integrity_result.status, integrity_result.as_dict()["findings"]
        )

        # The prompt states: conflict alone is not a reason to Hold.
        prompt = render_agent_prompt(
            "managers/portfolio_manager.txt",
            instrument_context=instrument_context,
            portfolio_context=portfolio_context,
            research_plan=research_plan,
            trader_plan=trader_plan,
            lessons_line=lessons_line,
            history=history,
            risk_stance_summary=render_risk_stance_summary(risk_debate_state),
            risk_integrity_notice=risk_integrity_notice
            or "No unsupported claims detected in the Risk discussion.",
            NO_EXTERNAL_TOOLS=NO_EXTERNAL_TOOLS,
        ) + get_language_instruction()

        final_trade_decision, parsed = invoke_structured_or_freetext_parsed(
            structured_llm,
            generation_llm,
            prompt,
            render_pm_decision,
            "Portfolio Manager",
            call_identity="portfolio_manager",
            llm_role="generation",
        )

        outcome = validate_and_repair(
            stage="Portfolio Manager",
            validator_llm=validator_llm,
            original_llm=generation_llm,
            original_role="generation",
            call_identity_base="portfolio_manager",
            instructions=_PM_CONTRACT,
            # Must cover the same sources Portfolio Manager's own prompt
            # actually gives it (see managers/portfolio_manager.txt) -- not
            # just the research/trader plans and Risk discussion, or
            # Validator flags a legitimately-cited portfolio/lessons number
            # as UNSUPPORTED_NUMBER simply for falling outside its own,
            # narrower evidence window.
            evidence_context=(
                f"Research plan:\n{research_plan}\n\nTrader plan:\n{trader_plan}\n\n"
                f"Risk discussion:\n{history}\n\nPortfolio context:\n{portfolio_context}"
                + (f"\n\nPrior lessons:\n{past_context}" if past_context else "")
            ),
            candidate_text=final_trade_decision,
            candidate_parsed=parsed,
            expected_schema=PortfolioDecision,
            render_fn=render_pm_decision,
            deterministic_check_fn=_pm_deterministic_checks,
        )
        final_trade_decision = outcome.final_text
        validation_log = [*state.get("validation_log", []), outcome.as_log_entry("Portfolio Manager")]

        # Real structured state, not inferred from `rating`: on a successful
        # structured call, `render_pm_decision` deterministically wrote
        # "**Disposition**: KEEP/DEFER/REDUCE_RISK/REJECT_PLAN" from the
        # model's own `disposition` field, so parsing it back out is exact,
        # not a guess. On free-text fallback it is parsed the same way only
        # if the model explicitly wrote it; otherwise this stays "" (not
        # recorded) rather than inventing a mapping from the rating.
        parsed_disposition = extract_choice(
            final_trade_decision,
            tuple(d.value for d in RiskDisposition),
            label_words=("disposition",),
        )
        portfolio_disposition = parsed_disposition.upper() if parsed_disposition else ""

        new_risk_debate_state = {
            **risk_debate_state,
            "risk_integrity_status": integrity_result.status,
            "risk_integrity_findings": integrity_result.as_dict()["findings"],
            "portfolio_disposition": portfolio_disposition,
            "judge_decision": final_trade_decision,
            "latest_speaker": "Judge",
        }

        return {
            "risk_debate_state": new_risk_debate_state,
            "final_trade_decision": final_trade_decision,
            "validation_log": validation_log,
        }

    return portfolio_manager_node
