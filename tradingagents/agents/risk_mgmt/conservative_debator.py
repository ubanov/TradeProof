from tradingagents.agents.context import get_language_instruction
from tradingagents.agents.risk_mgmt.frozen_evidence import render_frozen_evidence
from tradingagents.agents.risk_mgmt.stance import invoke_risk_assessment
from tradingagents.agents.schemas import RiskStanceAssessment
from tradingagents.agents.structured import bind_structured
from tradingagents.prompts.loader import render_agent_prompt


def _initial_or_absent(risk_debate_state: dict, key: str) -> str:
    return risk_debate_state.get(key) or "(no independent assessment recorded)"


def create_conservative_debator(generation_llm, review_llm):
    """See ``create_aggressive_debator``'s docstring -- identical split:
    initial is generation, cross-review is review."""
    structured_generation_llm = bind_structured(
        generation_llm, RiskStanceAssessment, "Conservative Analyst (initial)"
    )
    structured_review_llm = bind_structured(
        review_llm, RiskStanceAssessment, "Conservative Analyst (review)"
    )

    def conservative_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        is_initial = not risk_debate_state.get("conservative_initial")
        trader_decision = state["trader_investment_plan"]
        frozen_evidence = render_frozen_evidence(state)

        if is_initial:
            prompt = render_agent_prompt(
                "risk_mgmt/conservative_initial.txt",
                trader_decision=trader_decision,
                frozen_evidence=frozen_evidence,
            ) + get_language_instruction()
        else:
            prompt = render_agent_prompt(
                "risk_mgmt/conservative_review.txt",
                trader_decision=trader_decision,
                frozen_evidence=frozen_evidence,
                own_initial=_initial_or_absent(risk_debate_state, "conservative_initial"),
                aggressive_initial=_initial_or_absent(risk_debate_state, "aggressive_initial"),
                neutral_initial=_initial_or_absent(risk_debate_state, "neutral_initial"),
            ) + get_language_instruction()

        llm_for_call = generation_llm if is_initial else review_llm
        structured_for_call = structured_generation_llm if is_initial else structured_review_llm

        rendered, risk_level, disposition, assessment = invoke_risk_assessment(
            structured_for_call, llm_for_call, prompt, "Conservative Analyst",
            call_identity="initial" if is_initial else "review",
            llm_role="generation" if is_initial else "review",
        )

        label = "Independent Assessment" if is_initial else "Cross-Review"
        argument = f"Conservative Reviewer ({label}): {rendered}"

        new_risk_debate_state = dict(risk_debate_state)
        new_risk_debate_state.update(
            {
                "history": risk_debate_state.get("history", "") + "\n" + argument,
                "conservative_history": risk_debate_state.get("conservative_history", "") + "\n" + argument,
                "latest_speaker": "Conservative",
                "current_conservative_response": argument,
                "count": risk_debate_state.get("count", 0) + 1,
            }
        )
        if risk_level:
            new_risk_debate_state["conservative_risk_level"] = risk_level
        if disposition:
            new_risk_debate_state["conservative_disposition"] = disposition

        if is_initial:
            new_risk_debate_state["conservative_initial"] = argument
            new_risk_debate_state["conservative_initial_assessment"] = assessment or {}
        else:
            new_risk_debate_state["conservative_review"] = argument
            new_risk_debate_state["conservative_review_assessment"] = assessment or {}

        return {"risk_debate_state": new_risk_debate_state}

    return conservative_node
