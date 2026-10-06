from tradingagents.agents.context import get_language_instruction
from tradingagents.agents.risk_mgmt.frozen_evidence import render_frozen_evidence
from tradingagents.agents.risk_mgmt.stance import invoke_risk_assessment
from tradingagents.agents.schemas import RiskStanceAssessment
from tradingagents.agents.structured import bind_structured
from tradingagents.prompts.loader import render_agent_prompt


def _initial_or_absent(risk_debate_state: dict, key: str) -> str:
    return risk_debate_state.get(key) or "(no independent assessment recorded)"


def create_aggressive_debator(generation_llm, review_llm):
    """Independent initial assessment is a generation operation; the
    cross-review round is a review operation (challenges the other two
    reviewers' assessments) -- see fork README: "Model roles: Generation vs
    Review"."""
    structured_generation_llm = bind_structured(
        generation_llm, RiskStanceAssessment, "Aggressive Analyst (initial)"
    )
    structured_review_llm = bind_structured(
        review_llm, RiskStanceAssessment, "Aggressive Analyst (review)"
    )

    def aggressive_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        is_initial = not risk_debate_state.get("aggressive_initial")
        trader_decision = state["trader_investment_plan"]
        frozen_evidence = render_frozen_evidence(state)

        if is_initial:
            # Independent assessment: deliberately does not reference the other
            # two reviewers' output, even though the graph runs this turn after
            # Trader -- mirrors the Bull/Bear independent-initial-thesis phase.
            prompt = render_agent_prompt(
                "risk_mgmt/aggressive_initial.txt",
                trader_decision=trader_decision,
                frozen_evidence=frozen_evidence,
            ) + get_language_instruction()
        else:
            prompt = render_agent_prompt(
                "risk_mgmt/aggressive_review.txt",
                trader_decision=trader_decision,
                frozen_evidence=frozen_evidence,
                own_initial=_initial_or_absent(risk_debate_state, "aggressive_initial"),
                conservative_initial=_initial_or_absent(risk_debate_state, "conservative_initial"),
                neutral_initial=_initial_or_absent(risk_debate_state, "neutral_initial"),
            ) + get_language_instruction()

        llm_for_call = generation_llm if is_initial else review_llm
        structured_for_call = structured_generation_llm if is_initial else structured_review_llm

        rendered, risk_level, disposition, assessment = invoke_risk_assessment(
            structured_for_call, llm_for_call, prompt, "Aggressive Analyst",
            call_identity="initial" if is_initial else "review",
            llm_role="generation" if is_initial else "review",
        )

        label = "Independent Assessment" if is_initial else "Cross-Review"
        argument = f"Aggressive Reviewer ({label}): {rendered}"

        new_risk_debate_state = dict(risk_debate_state)
        new_risk_debate_state.update(
            {
                "history": risk_debate_state.get("history", "") + "\n" + argument,
                "aggressive_history": risk_debate_state.get("aggressive_history", "") + "\n" + argument,
                "latest_speaker": "Aggressive",
                "current_aggressive_response": argument,
                "count": risk_debate_state.get("count", 0) + 1,
            }
        )
        if risk_level:
            new_risk_debate_state["aggressive_risk_level"] = risk_level
        if disposition:
            new_risk_debate_state["aggressive_disposition"] = disposition

        if is_initial:
            new_risk_debate_state["aggressive_initial"] = argument
            new_risk_debate_state["aggressive_initial_assessment"] = assessment or {}
        else:
            new_risk_debate_state["aggressive_review"] = argument
            new_risk_debate_state["aggressive_review_assessment"] = assessment or {}

        return {"risk_debate_state": new_risk_debate_state}

    return aggressive_node
