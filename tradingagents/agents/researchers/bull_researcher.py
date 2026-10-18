from tradingagents.agents.context import (
    get_instrument_context_from_state,
    get_language_instruction,
    report_or_absent,
)
from tradingagents.agents.researchers.setup_tags import render_setup_tags
from tradingagents.agents.researchers.thesis_flow import (
    apply_validated_trade_plan_changes,
    invoke_initial_thesis,
    invoke_review,
    recompute_trade_metrics,
)
from tradingagents.agents.schemas import InitialResearchThesis, ResearchReviewOutcome
from tradingagents.agents.structured import bind_structured
from tradingagents.prompts.loader import render_agent_prompt

from .review_outcomes import review_guidance


def create_bull_researcher(generation_llm, review_llm):
    """The initial thesis is a generation operation (produce/structure a
    position from the evidence); each review round is a review operation
    (challenge the opponent, audit and correct your own prior claims) -- see
    fork README: "Model roles: Generation vs Review". One researcher, one
    node, two LLMs: no agent duplication, since ``is_initial`` below already
    decides which phase a given turn is in and canonical recovery (on a
    fallback) naturally inherits whichever LLM its own call used.
    """
    structured_initial_llm = bind_structured(
        generation_llm, InitialResearchThesis, "Bull Researcher (initial)"
    )
    structured_review_llm = bind_structured(
        review_llm, ResearchReviewOutcome, "Bull Researcher (review)"
    )

    def bull_node(state) -> dict:
        investment_debate_state = state["investment_debate_state"]
        history = investment_debate_state.get("history", "")
        bull_history = investment_debate_state.get("bull_history", "")
        is_initial = not investment_debate_state.get("bull_initial")
        market_research_report = report_or_absent(state["market_report"], "market")
        sentiment_report = report_or_absent(state["sentiment_report"], "sentiment")
        news_report = report_or_absent(state["news_report"], "news")
        fundamentals_report = report_or_absent(state["fundamentals_report"], "fundamentals")
        instrument_context = get_instrument_context_from_state(state)
        asset_type = state.get("asset_type", "stock")
        target_label = "stock" if asset_type == "stock" else "asset"
        fundamentals_label = (
            "Company fundamentals report"
            if asset_type == "stock"
            else "Asset fundamentals report (may be unavailable for crypto)"
        )
        research_horizon = investment_debate_state.get("research_horizon", "")
        setup_tags_text = render_setup_tags(investment_debate_state.get("setup_tags") or {})

        prompt_values = {
            "target_label": target_label,
            "instrument_context": instrument_context,
            "market_research_report": market_research_report,
            "sentiment_report": sentiment_report,
            "news_report": news_report,
            "fundamentals_label": fundamentals_label,
            "fundamentals_report": fundamentals_report,
        }

        new_investment_debate_state = dict(investment_debate_state)

        if is_initial:
            prompt = render_agent_prompt(
                "researchers/bull_initial.txt",
                **prompt_values,
                research_horizon=research_horizon,
                setup_tags=setup_tags_text,
            )
            prompt += get_language_instruction()

            rendered, thesis_dict, initial_recovery_status = invoke_initial_thesis(
                structured_initial_llm, generation_llm, prompt, "Bull Researcher",
                call_identity="initial", llm_role="generation",
            )
            argument = f"Bull Analyst (Initial): {rendered}"

            new_investment_debate_state.update(
                {
                    "history": history + "\n" + argument,
                    "bull_history": bull_history + "\n" + argument,
                    "current_response": argument,
                    "count": investment_debate_state.get("count", 0) + 1,
                    "bull_initial": argument,
                    "current_bull_response": argument,
                    "bull_initial_recovery_status": initial_recovery_status,
                }
            )
            if thesis_dict is not None:
                new_investment_debate_state["bull_thesis"] = thesis_dict
                new_investment_debate_state["bull_conviction_history"] = [
                    thesis_dict["conviction"]["total"]
                ]
                new_investment_debate_state["bull_trade_metrics"] = recompute_trade_metrics(
                    thesis_dict, investment_debate_state.get("atr_reference")
                )
        else:
            review_round = investment_debate_state.get("debate_round", 0) + 1
            conviction_history = list(investment_debate_state.get("bull_conviction_history") or [])
            prior_conviction = (
                conviction_history[-1]
                if conviction_history
                else (investment_debate_state.get("bull_thesis") or {})
                .get("conviction", {})
                .get("total")
            )
            prompt = render_agent_prompt(
                "researchers/bull.txt",
                **prompt_values,
                review_round=review_round,
                own_previous_response=investment_debate_state.get(
                    "current_bull_response", investment_debate_state.get("bull_initial", "")
                ),
                opponent_previous_response=investment_debate_state.get(
                    "current_bear_response", investment_debate_state.get("bear_initial", "")
                ),
                review_outcomes=investment_debate_state.get("review_outcomes", "")
                or "No classified outcomes from prior rounds.",
                review_guidance=review_guidance(review_round),
                research_horizon=research_horizon,
                setup_tags=setup_tags_text,
                prior_conviction=(
                    str(prior_conviction)
                    if prior_conviction is not None
                    else "not previously recorded"
                ),
            )
            prompt += get_language_instruction()

            rendered, review_dict, review_recovery_status = invoke_review(
                structured_review_llm,
                review_llm,
                prompt,
                "Bull Researcher",
                prior_thesis=investment_debate_state.get("bull_thesis"),
                call_identity=f"review_{review_round}",
                llm_role="review",
            )
            argument = f"Bull Analyst (Review {review_round}): {rendered}"

            new_investment_debate_state.update(
                {
                    "history": history + "\n" + argument,
                    "bull_history": bull_history + "\n" + argument,
                    "current_response": argument,
                    "count": investment_debate_state.get("count", 0) + 1,
                    "pending_bull_rebuttal": argument,
                    "bull_rebuttal_count": investment_debate_state.get("bull_rebuttal_count", 0)
                    + 1,
                    "bull_review_recovery_status": review_recovery_status,
                }
            )
            if review_dict is not None:
                current_thesis = investment_debate_state.get("bull_thesis") or {}
                if current_thesis:
                    updated_thesis, withdrawn, geometry_finding = apply_validated_trade_plan_changes(
                        current_thesis, review_dict
                    )
                    new_investment_debate_state["bull_thesis"] = updated_thesis
                    new_investment_debate_state["bull_trade_metrics"] = recompute_trade_metrics(
                        updated_thesis, investment_debate_state.get("atr_reference")
                    )
                    if withdrawn:
                        new_investment_debate_state["withdrawn_values"] = (
                            investment_debate_state.get("withdrawn_values", []) + withdrawn
                        )
                    if geometry_finding is not None:
                        new_investment_debate_state["trade_geometry_rejections"] = (
                            investment_debate_state.get("trade_geometry_rejections", [])
                            + [{
                                "side": "bull",
                                "round": review_round,
                                "claim": geometry_finding.claim,
                                "reason": geometry_finding.reason,
                            }]
                        )
                new_investment_debate_state["bull_conviction_history"] = conviction_history + [
                    review_dict["conviction_after"]
                ]
                if review_dict.get("new_data_exceptions"):
                    tagged = [
                        {**exc, "side": "bull", "round": review_round}
                        for exc in review_dict["new_data_exceptions"]
                    ]
                    new_investment_debate_state["new_data_exceptions"] = (
                        investment_debate_state.get("new_data_exceptions", []) + tagged
                    )

        return {"investment_debate_state": new_investment_debate_state}

    return bull_node
