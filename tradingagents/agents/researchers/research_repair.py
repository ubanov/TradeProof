"""One bounded correction pass for Bull and Bear research positions.

Reuses the exact same ``ResearchReviewOutcome`` schema and apply/recompute
mechanisms a normal cross-review round uses (see
``researchers/thesis_flow.py``): no separate parsing of free-form repair
prose. A successful structured repair updates the canonical
``bull_thesis``/``bear_thesis`` the same way a review would, so
``bull_trade_metrics``, the Research Manager's inputs, the post-manager
integrity check's provenance pool, and the Trader's trade-plan context all
pick up the repaired levels automatically -- they already read from these
same fields.

Falls back to free text exactly like every other structured agent in this
project when the provider does not support structured output; see
``{side}_repair_structured_status`` (APPLIED / TEXT_ONLY / NOT_RUN) for how
callers know which happened, so stale structured state is never mistaken for
a confirmed repair.
"""

from __future__ import annotations

from tradingagents.agents.context import (
    get_instrument_context_from_state,
    get_language_instruction,
    report_or_absent,
)
from tradingagents.agents.researchers.thesis_flow import (
    apply_trade_plan_changes,
    invoke_review,
    recompute_trade_metrics,
)
from tradingagents.agents.schemas import ResearchReviewOutcome
from tradingagents.agents.structured import bind_structured
from tradingagents.prompts.loader import render_agent_prompt

from .verification import render_verification_history


def _create_research_repair(review_llm, side: str):
    """Repair is a review operation end to end (correct a claim already
    flagged by the Verifier) -- see fork README: "Model roles: Generation vs
    Review". A single role, so ``create_bull_repair``/``create_bear_repair``
    take just the one LLM, unlike Bull/Bear's own initial-vs-review split."""
    side_key = side.lower()
    other_key = "bear" if side_key == "bull" else "bull"
    structured_llm = bind_structured(review_llm, ResearchReviewOutcome, f"{side} Repair")

    def repair_node(state) -> dict:
        debate = state["investment_debate_state"]
        own_position = debate.get(f"current_{side_key}_response") or debate.get(
            f"{side_key}_initial", ""
        )
        opponent_position = debate.get(
            f"current_{other_key}_response"
        ) or debate.get(f"{other_key}_initial", "")

        conviction_history = list(debate.get(f"{side_key}_conviction_history") or [])
        prior_conviction = (
            conviction_history[-1]
            if conviction_history
            else (debate.get(f"{side_key}_thesis") or {}).get("conviction", {}).get("total")
        )

        prompt = render_agent_prompt(
            "researchers/repair.txt",
            side=side,
            instrument_context=get_instrument_context_from_state(state),
            market_report=report_or_absent(state.get("market_report", ""), "market"),
            sentiment_report=report_or_absent(
                state.get("sentiment_report", ""), "sentiment"
            ),
            news_report=report_or_absent(state.get("news_report", ""), "news"),
            fundamentals_report=report_or_absent(
                state.get("fundamentals_report", ""), "fundamentals"
            ),
            own_position=own_position,
            opponent_position=opponent_position,
            verifier_findings=render_verification_history(
                debate.get("verification_history", [])[-1:], include_details=True
            ),
            prior_conviction=(
                str(prior_conviction)
                if prior_conviction is not None
                else "not previously recorded"
            ),
        ) + get_language_instruction()

        repair_round = debate.get("repair_rounds", 0) + 1
        rendered, review_dict, _recovery_status = invoke_review(
            structured_llm, review_llm, prompt, f"{side} Repair",
            prior_thesis=debate.get(f"{side_key}_thesis"),
            call_identity=f"repair_{repair_round}",
            llm_role="review",
        )
        repaired = f"{side} Analyst (Repair {repair_round}): {rendered}"

        new_debate = {
            **debate,
            f"{side_key}_repair": repaired,
            f"{side_key}_history": debate.get(f"{side_key}_history", "")
            + "\n"
            + repaired,
            "history": debate.get("history", "") + "\n" + repaired,
            "current_response": repaired,
            "count": debate.get("count", 0) + 1,
            "repair_triggered": True,
        }

        if review_dict is not None:
            new_debate[f"{side_key}_repair_structured_status"] = "APPLIED"
            current_thesis = debate.get(f"{side_key}_thesis") or {}
            if current_thesis:
                updated_thesis, retired = apply_trade_plan_changes(current_thesis, review_dict)
                new_debate[f"{side_key}_thesis"] = updated_thesis
                new_debate[f"{side_key}_trade_metrics"] = recompute_trade_metrics(
                    updated_thesis, debate.get("atr_reference")
                )
                if retired:
                    new_debate["withdrawn_values"] = (
                        debate.get("withdrawn_values", []) + retired
                    )
            new_debate[f"{side_key}_conviction_history"] = conviction_history + [
                review_dict["conviction_after"]
            ]
            # Repair must not introduce new external evidence -- stricter than a
            # normal review round, which allows a narrow NEW_DATA_EXCEPTION.
            # Any exception the model still attempted stays visible in the
            # rendered prose above (so the Verifier's next pass can flag it)
            # but is deliberately never persisted as an authorized one.
        else:
            # Structured repair failed; the free-text prose above is kept for
            # semantic context, but nothing here guesses at structured fields
            # from it -- bull_thesis/bear_thesis/withdrawn_values are left
            # exactly as they were, not silently mutated by regex parsing.
            new_debate[f"{side_key}_repair_structured_status"] = "TEXT_ONLY"

        if side_key == "bear":
            new_debate.update(
                {
                    "current_bull_response": debate.get("bull_repair")
                    or debate.get("current_bull_response", ""),
                    "current_bear_response": repaired,
                    "repair_rounds": repair_round,
                    "repaired_agents": ["bull", "bear"],
                }
            )
        return {"investment_debate_state": new_debate}

    return repair_node


def create_bull_repair(review_llm):
    return _create_research_repair(review_llm, "Bull")


def create_bear_repair(review_llm):
    return _create_research_repair(review_llm, "Bear")
