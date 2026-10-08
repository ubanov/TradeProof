"""Trader: turns the Research Manager's investment plan into a concrete transaction proposal."""

from __future__ import annotations

import functools

from langchain_core.messages import AIMessage

from tradingagents.agents.context import (
    get_instrument_context_from_state,
    get_language_instruction,
    get_portfolio_context_from_state,
)
from tradingagents.agents.managers.integrity_check import render_integrity_notice_for_trader
from tradingagents.agents.researchers.research_summary import render_trade_plan_for_trader
from tradingagents.agents.schemas import TraderProposal, render_trader_proposal
from tradingagents.agents.structured import (
    NO_EXTERNAL_TOOLS,
    bind_structured,
    invoke_structured_or_freetext_parsed,
)
from tradingagents.agents.validator import validate_and_repair
from tradingagents.agents.validator_deterministic import (
    check_entry_stop_direction,
    check_text_relational_and_unit_errors,
)
from tradingagents.prompts.loader import load_prompt, render_agent_prompt

# Compact, hand-written contract summary for the Validator (see fork README:
# "Generation / Validation / Review") -- deliberately NOT the full rendered
# system/user prompt, which would be long and invite the Validator to
# re-litigate strategy instead of checking the contract.
_TRADER_CONTRACT = (
    "Produce a transaction proposal: action (Buy/Hold/Sell), reasoning, "
    "optional entry_price (or entry_price_low/entry_price_high for a "
    "range), optional stop_loss, optional position_sizing. entry_price and "
    "stop_loss must be absolute prices in the instrument's quote currency, "
    "never percentages. For a Buy, stop_loss must be below entry_price; "
    "for a Sell, stop_loss must be above entry_price."
)


def _trader_deterministic_checks(text: str, parsed):
    findings = check_text_relational_and_unit_errors(text)
    if parsed is not None and parsed.entry_price is not None:
        finding = check_entry_stop_direction(parsed.action.value, parsed.entry_price, parsed.stop_loss)
        if finding is not None:
            findings.append(finding)
    return findings


def create_trader(generation_llm, validator_llm=None):
    """Constructs the transaction proposal from the manager's plan -- a
    generation operation (see fork README: "Model roles: Generation vs
    Review"). ``validator_llm`` is ``None`` unless validation is enabled;
    deterministic checks (entry/stop direction, unit/relational errors) run
    unconditionally either way."""
    structured_llm = bind_structured(generation_llm, TraderProposal, "Trader")

    def trader_node(state, name):
        company_name = state["company_of_interest"]
        instrument_context = get_instrument_context_from_state(state)
        investment_plan = state["investment_plan"]
        # The research plan digests the debate but loses exact price structure;
        # give the Trader the technical market report so entry/stop levels are
        # grounded in real ATR / support-resistance / current price (#1167). The
        # report is empty when the user did not select the market analyst, so
        # only offer it (and the grounding instruction) when it has content.
        market_report = (state["market_report"] or "").strip()
        portfolio_context = get_portfolio_context_from_state(state)
        debate_state = state.get("investment_debate_state") or {}
        research_trade_plan = render_trade_plan_for_trader(debate_state)
        integrity_notice = render_integrity_notice_for_trader(
            debate_state.get("manager_integrity_status", ""),
            debate_state.get("manager_integrity_findings", []),
        )

        if market_report:
            grounding = (
                "Ground concrete price levels (entry, stop-loss, position sizing) in the technical "
                "market report's price structure -- current price, support/resistance, ATR, and "
                "volatility -- and use the research plan for direction and strategy. "
            )
            report_section = f"Technical Market Report:\n{market_report}\n\n"
        else:
            grounding = ""
            report_section = ""

        messages = [
            {
                "role": "system",
                "content": render_agent_prompt(
                    "trader/system.txt",
                    grounding=grounding,
                    NO_EXTERNAL_TOOLS=NO_EXTERNAL_TOOLS,
                ) + get_language_instruction(),
            },
            {
                "role": "user",
                "content": load_prompt("trader/user.txt").format(
                    company_name=company_name,
                    instrument_context=instrument_context,
                    report_section=report_section,
                    portfolio_context=portfolio_context,
                    investment_plan=investment_plan,
                    research_trade_plan=research_trade_plan,
                    integrity_notice=integrity_notice,
                ),
            },
        ]

        trader_plan, parsed = invoke_structured_or_freetext_parsed(
            structured_llm,
            generation_llm,
            messages,
            render_trader_proposal,
            "Trader",
            call_identity="trader",
            llm_role="generation",
        )

        outcome = validate_and_repair(
            stage="Trader",
            validator_llm=validator_llm,
            original_llm=generation_llm,
            original_role="generation",
            call_identity_base="trader",
            instructions=_TRADER_CONTRACT,
            # Must cover the same sources the Trader was actually allowed to
            # use (see trader/user.txt) -- the market report and portfolio
            # context, not just the research plan, or Validator flags the
            # Trader's own legitimately-cited price-structure numbers (current
            # price, SMA/EMA, RSI, ...) as UNSUPPORTED_NUMBER simply because
            # its own evidence window was narrower than the Trader's.
            evidence_context=(
                (f"Technical market report:\n{market_report}\n\n" if market_report else "")
                + f"Portfolio context:\n{portfolio_context}\n\n"
                f"Investment plan:\n{investment_plan}\n\n"
                f"Research trade plan:\n{research_trade_plan}"
            ),
            candidate_text=trader_plan,
            candidate_parsed=parsed,
            expected_schema=TraderProposal,
            render_fn=render_trader_proposal,
            deterministic_check_fn=_trader_deterministic_checks,
        )
        final_plan = outcome.final_text

        validation_log = [*state.get("validation_log", []), outcome.as_log_entry("Trader")]

        return {
            "messages": [AIMessage(content=final_plan)],
            "trader_investment_plan": final_plan,
            "sender": name,
            "validation_log": validation_log,
        }

    return functools.partial(trader_node, name="Trader")
