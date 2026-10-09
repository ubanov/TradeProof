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
    invoke_structured_or_freetext_parsed,
)
from tradingagents.agents.validator import validate_and_repair
from tradingagents.agents.validator_deterministic import check_manager_integrity_findings
from tradingagents.prompts.loader import render_agent_prompt

logger = logging.getLogger(__name__)

# Compact, hand-written contract summary for the Validator (see fork README:
# "Generation / Validation / Review") -- not the full rendered prompt, which
# would invite the Validator to re-litigate the debate instead of checking
# the contract. Research Manager is a SYNTHESIS step: Validator checks that
# synthesis didn't introduce a new objective error, never whether Bull or
# Bear "won" -- that stays the Research Verifier's and Manager's own job.
_RESEARCH_MANAGER_CONTRACT = (
    "Produce a recommendation (Buy/Overweight/Hold/Underweight/Sell), a "
    "rationale summarizing which side of the debate decided it, and "
    "strategic actions for the trader. Use only levels, thresholds, and "
    "numbers already present in the Bull/Bear theses, the verification "
    "summary, the deterministic trade metrics, the shared objective setup "
    "tags (technical indicator buckets -- e.g. RSI_BUCKET, MACD_SIGN, "
    "MACD_HISTOGRAM_SIGN, PRICE_VS_EMA10, PRICE_VS_SMA20, PRICE_VS_SMA50, "
    "SMA50_VS_SMA200, ATR_PCT_BUCKET, VOLUME_VS_AVERAGE, TREND_STRUCTURE), "
    "or the conviction trail below -- never invent a new price target, stop "
    "level, indicator threshold, probability, or sizing percentage. "
    "Restating a supplied setup tag in plain language (e.g. \"RSI between "
    "30 and 50\" for RSI_BUCKET: RSI_30_50, \"MACD negative\" for MACD_SIGN: "
    "NEGATIVE, or \"price below EMA10\" for PRICE_VS_EMA10: BELOW) is NOT an "
    "invented or unsupported claim -- it is a direct restatement of input "
    "already supplied. A self-assessed conviction score is not a market "
    "probability."
)


def create_research_manager(generation_llm, validator_llm=None):
    """Synthesizes the already-audited debate into a plan -- a generation
    operation, not an auditor role (see fork README: "Model roles:
    Generation vs Review"). ``validator_llm`` is ``None`` unless validation
    is enabled; deterministic checks (reusing the existing Research Manager
    integrity check) run unconditionally either way."""
    structured_llm = bind_structured(generation_llm, ResearchPlan, "Research Manager")

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

        # Named here (not inlined into render_agent_prompt below) so the
        # SAME rendered values can also build the Validator's evidence
        # context further down -- the exact sources this prompt gives the
        # Manager, no more (see fork README: "Generation / Validation /
        # Review" -- Validator evidence universe == what the agent saw).
        review_outcomes = (
            investment_debate_state.get("review_outcomes", "") or "No classified review outcomes were produced."
        )
        verification_summary = render_verification_history(
            investment_debate_state.get("verification_history", []),
            repaired_agents=investment_debate_state.get("repaired_agents", []),
            include_details=True,
        )
        final_bull_response = bull_repair or downstream_safe_text(
            investment_debate_state.get("current_bull_response", ""),
            latest_side_recovery_status(investment_debate_state, "bull"),
        ) or bull_initial
        final_bear_response = bear_repair or downstream_safe_text(
            investment_debate_state.get("current_bear_response", ""),
            latest_side_recovery_status(investment_debate_state, "bear"),
        ) or bear_initial
        bull_trade_metrics_rendered = render_trade_metrics(
            trade_metrics_from_dict(investment_debate_state.get("bull_trade_metrics"))
        )
        bear_trade_metrics_rendered = render_trade_metrics(
            trade_metrics_from_dict(investment_debate_state.get("bear_trade_metrics"))
        )
        research_horizon = investment_debate_state.get("research_horizon", "not recorded")
        setup_tags_rendered = render_setup_tags(investment_debate_state.get("setup_tags") or {})
        bull_conviction_trail = _conviction_trail(investment_debate_state.get("bull_conviction_history") or [])
        bear_conviction_trail = _conviction_trail(investment_debate_state.get("bear_conviction_history") or [])
        market_report = state.get("market_report", "")
        sentiment_report = state.get("sentiment_report", "")
        news_report = state.get("news_report", "")
        fundamentals_report = state.get("fundamentals_report", "")

        # The prompt states: conflict alone is not a reason to Hold.
        prompt = render_agent_prompt(
            "managers/research_manager.txt",
            instrument_context=instrument_context,
            bull_initial=bull_initial,
            bear_initial=bear_initial,
            review_outcomes=review_outcomes,
            verification_summary=verification_summary,
            repair_outputs=repair_outputs,
            repair_structured_note=render_repair_status_caveat(investment_debate_state)
            or "No repair fell back to free text.",
            final_bull_response=final_bull_response,
            final_bear_response=final_bear_response,
            research_horizon=research_horizon,
            setup_tags=setup_tags_rendered,
            bull_conviction_trail=bull_conviction_trail,
            bear_conviction_trail=bear_conviction_trail,
            bull_suggested_risk_unit=_risk_unit(investment_debate_state.get("bull_thesis") or {}),
            bear_suggested_risk_unit=_risk_unit(investment_debate_state.get("bear_thesis") or {}),
            bull_trade_metrics=bull_trade_metrics_rendered,
            bear_trade_metrics=bear_trade_metrics_rendered,
            NO_EXTERNAL_TOOLS=NO_EXTERNAL_TOOLS,
        ) + get_language_instruction()

        investment_plan, parsed = invoke_structured_or_freetext_parsed(
            structured_llm,
            generation_llm,
            prompt,
            render_research_plan,
            "Research Manager",
            call_identity="research_manager",
            llm_role="generation",
        )

        def _research_manager_deterministic_checks(text, _parsed):
            # Reuses the EXISTING Research Manager integrity check as-is
            # (fork README: "Research Manager integrity check") -- never a
            # second implementation of the same unsupported-number/
            # arithmetic/relational/withdrawn-value scanning. Re-run on
            # whatever text is passed in, so the post-repair recheck below
            # reflects the actual repaired content, not the original.
            #
            # Deliberately NOT given the raw analyst reports: Research
            # Manager's own prompt above never shows them, so a number known
            # only from a report would otherwise be wrongly treated as
            # "supported" for Validator's purposes (a provenance leak this
            # task corrected). `extra_sources` is exactly the free-text
            # universe the prompt above DOES render, matching the Validator's
            # own evidence_context below -- see fork README: "Validator
            # evidence universe == sources the original agent was allowed
            # to use."
            return check_manager_integrity_findings(
                text, investment_debate_state,
                extra_sources=(
                    bull_initial, bear_initial, review_outcomes, verification_summary,
                    repair_outputs, final_bull_response, final_bear_response,
                    # The setup tags' own bucket names embed their boundary
                    # numbers (e.g. "RSI_30_50" contains 30 and 50; "2_TO_3"
                    # contains 2 and 3) -- extract_numbers() picks those up
                    # the same way it already does for every other source
                    # here, with no new translation logic (real-run
                    # regression: "RSI between 30 and 50" / "ATR 2-3%"
                    # restating RSI_BUCKET/ATR_PCT_BUCKET wrongly flagged as
                    # unsupported). Research horizon and conviction trails
                    # carry no comparable numeric content worth scanning but
                    # are harmless to include for completeness.
                    setup_tags_rendered, research_horizon,
                    bull_conviction_trail, bear_conviction_trail,
                ),
            )

        outcome = validate_and_repair(
            stage="Research Manager",
            validator_llm=validator_llm,
            original_llm=generation_llm,
            original_role="generation",
            call_identity_base="research_manager",
            instructions=_RESEARCH_MANAGER_CONTRACT,
            # Full parity with the Manager's own prompt above -- every
            # source it renders, so the Validator LLM never flags a direct
            # restatement of supplied input (a setup tag, the research
            # horizon, a conviction trail) as unsupported/invented (real-run
            # regression: missing setup tags caused a false
            # PROMPT_REQUIREMENT_VIOLATION and an information-destroying
            # repair). Never the raw analyst reports (the Manager's prompt
            # never shows those directly) and never the candidate text itself.
            evidence_context=(
                f"Instrument identity:\n{instrument_context}\n\n"
                f"Bull initial thesis:\n{bull_initial}\n\n"
                f"Bear initial thesis:\n{bear_initial}\n\n"
                f"Review outcomes:\n{review_outcomes}\n\n"
                f"Verification summary:\n{verification_summary}\n\n"
                f"Repair outputs:\n{repair_outputs}\n\n"
                f"Final bull position:\n{final_bull_response}\n\n"
                f"Final bear position:\n{final_bear_response}\n\n"
                f"Research horizon: {research_horizon}\n\n"
                f"Objective setup tags (technical indicator buckets):\n{setup_tags_rendered}\n\n"
                f"Bull conviction trail: {bull_conviction_trail}\n"
                f"Bear conviction trail: {bear_conviction_trail}\n\n"
                f"Bull trade metrics:\n{bull_trade_metrics_rendered}\n\n"
                f"Bear trade metrics:\n{bear_trade_metrics_rendered}"
            ),
            candidate_text=investment_plan,
            candidate_parsed=parsed,
            expected_schema=ResearchPlan,
            render_fn=render_research_plan,
            deterministic_check_fn=_research_manager_deterministic_checks,
        )
        investment_plan = outcome.final_text
        validation_log = [*state.get("validation_log", []), outcome.as_log_entry("Research Manager")]

        # Deterministic, LLM-free pass over the FINAL (possibly Validator-
        # repaired) plan -- this is what actually continues downstream to
        # the Trader, so the integrity notice it feeds must describe THAT
        # text, never the pre-repair candidate (fork README: "Research
        # Manager integrity check"). No LLM call, no correction loop here.
        integrity_result = check_manager_integrity(
            investment_plan,
            investment_debate_state,
            market_report, sentiment_report, news_report, fundamentals_report,
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
            "validation_log": validation_log,
        }

    return research_manager_node
