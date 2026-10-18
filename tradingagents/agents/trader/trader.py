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
    check_price_field_percentage,
    check_relational_errors,
    check_trade_geometry,
)
from tradingagents.prompts.loader import load_prompt, render_agent_prompt

# Compact, hand-written contract summary for the Validator (see fork README:
# "Generation / Validation / Review") -- deliberately NOT the full rendered
# system/user prompt, which would be long and invite the Validator to
# re-litigate strategy instead of checking the contract.
_TRADER_CONTRACT = (
    "Produce a transaction proposal: action (Buy/Hold/Sell), reasoning, "
    "optional entry_price (or entry_price_low/entry_price_high for a "
    "range), optional stop_loss, optional take_profit, optional "
    "position_sizing. entry_price, stop_loss and take_profit must be "
    "absolute prices in the instrument's quote currency, never percentages. "
    "For a Buy, stop_loss must be below and take_profit above entry_price; "
    "for a Sell, stop_loss must be above and take_profit below entry_price. "
    "take_profit is only set when a target is actually defensible -- never "
    "invented merely because the field exists."
)


def _trader_deterministic_checks(text: str, parsed):
    # Relational-error scanning stays as-is (narrative "X is below Y"
    # contradictions); unit-error scanning uses the narrow, field-line-based
    # check instead of the generic sentence-wide one -- a descriptive stop-
    # distance percentage in reasoning/prose ("25 pts, ~2.16% of current
    # price") is not a contractual Entry Price/Stop Loss stated as a bare
    # percentage, and must never be flagged as one (real-run regression).
    findings = check_relational_errors(text)
    findings += check_price_field_percentage(text)
    if parsed is not None:
        # KNOWN GAP (see fork README): "Sell" here is always read as
        # OPEN_SHORT. The Trader's 3-tier action (schemas.TraderAction) does
        # not distinguish that from closing/reducing an existing LONG
        # (sell-to-close), which would correctly have no "stop must be above
        # entry" requirement at all. Resolving that needs portfolio-position
        # awareness (PortfolioContext.position_in(ticker).quantity is signed
        # and already carries this information) threaded into this check,
        # which is deliberately out of scope here -- no new order-type model
        # is introduced by this change. In practice this was already the
        # exact scope ``check_entry_stop_direction`` (the function this
        # replaces) applied to stop_loss alone; this only extends the same,
        # already-accepted scope to take_profit and to entry ranges/
        # reference_entry.
        finding = check_trade_geometry(
            direction=parsed.action.value,
            entry_price=parsed.entry_price,
            entry_low=parsed.entry_price_low,
            entry_high=parsed.entry_price_high,
            reference_entry=parsed.reference_entry,
            stop_loss=parsed.stop_loss,
            take_profit=parsed.take_profit,
        )
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
            # Instrument identity (ticker + resolved company name) is also
            # rendered into the real prompt (see trader/user.txt's
            # {instrument_context}) -- omitting it here risks the same false
            # UNKNOWN_USED_AS_FACT-on-the-ticker regression caught on
            # Portfolio Manager.
            evidence_context=(
                f"Instrument identity:\n{instrument_context}\n\n"
                + (f"Technical market report:\n{market_report}\n\n" if market_report else "")
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

        # The structured object consistent with final_plan -- outcome.final_parsed
        # is the REPAIRED parse when validation triggered a repair, never the
        # stale pre-repair `parsed`, so this is never out of sync with the text
        # actually used downstream. None when no structured parse survived (a
        # free-text fallback with no parse) -- never fabricated from partial text.
        trader_proposal_structured = (
            outcome.final_parsed.model_dump(mode="json") if outcome.final_parsed is not None else None
        )

        return {
            "messages": [AIMessage(content=final_plan)],
            "trader_investment_plan": final_plan,
            "trader_proposal_structured": trader_proposal_structured,
            "sender": name,
            "validation_log": validation_log,
        }

    return functools.partial(trader_node, name="Trader")
