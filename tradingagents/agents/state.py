from typing import Annotated

from langgraph.graph import MessagesState
from typing_extensions import TypedDict


# Researcher team state
class InvestDebateState(TypedDict):
    bull_initial: Annotated[str, "Bull's independent initial thesis"]
    bear_initial: Annotated[str, "Bear's independent initial thesis"]
    rebuttal_history: Annotated[str, "Completed cross-review rounds"]
    review_outcomes: Annotated[str, "Classified cross-review outcomes"]
    current_bull_response: Annotated[str, "Latest completed Bull response"]
    current_bear_response: Annotated[str, "Latest completed Bear response"]
    pending_bull_rebuttal: Annotated[str, "Bull response awaiting Bear's review"]
    bull_rebuttal_count: Annotated[int, "Number of Bull cross-review calls"]
    bear_rebuttal_count: Annotated[int, "Number of Bear cross-review calls"]
    debate_round: Annotated[int, "Number of completed cross-review rounds"]
    verification_history: Annotated[list[dict], "Structured verifier pass results"]
    verifier_pass_count: Annotated[int, "Number of verifier calls"]
    verification_status: Annotated[str, "Status of the latest verifier pass"]
    verification_findings: Annotated[list[dict], "Findings from the latest verifier pass"]
    repair_triggered: Annotated[bool, "Whether the bounded repair phase ran"]
    repair_rounds: Annotated[int, "Number of completed repair rounds"]
    repaired_agents: Annotated[list[str], "Research agents repaired"]
    bull_repair: Annotated[str, "Bull's corrected final position"]
    bear_repair: Annotated[str, "Bear's corrected final position"]
    research_horizon: Annotated[str, "Shared analysis horizon used by both Bull and Bear"]
    setup_tags: Annotated[dict, "Deterministic objective setup tags shared by Bull and Bear"]
    atr_reference: Annotated[
        float | None, "Deterministic ATR value shared by both sides' trade-metric calculations"
    ]
    bull_thesis: Annotated[dict, "Bull's current structured trade hypothesis"]
    bear_thesis: Annotated[dict, "Bear's current structured trade hypothesis"]
    bull_conviction_history: Annotated[list[int], "Bull's conviction score after each phase"]
    bear_conviction_history: Annotated[list[int], "Bear's conviction score after each phase"]
    bull_trade_metrics: Annotated[dict, "Deterministic trade metrics for Bull's current plan"]
    bear_trade_metrics: Annotated[dict, "Deterministic trade metrics for Bear's current plan"]
    new_data_exceptions: Annotated[list[dict], "NEW_DATA_EXCEPTION entries raised during review"]
    trade_geometry_rejections: Annotated[
        list[dict],
        "Review/repair trade-plan changes rejected for violating directional "
        "geometry (stop/target on the wrong side of entry) -- the prior, "
        "valid levels were kept instead of consolidating these.",
    ]
    withdrawn_values: Annotated[
        list[float],
        "Trade-plan levels WITHDRAWN or genuinely REVISED away during review or repair; "
        "must never again be treated as active verified provenance",
    ]
    bull_repair_structured_status: Annotated[
        str, "APPLIED/TEXT_ONLY/NOT_RUN: whether Bull's repair updated bull_thesis"
    ]
    bear_repair_structured_status: Annotated[
        str, "APPLIED/TEXT_ONLY/NOT_RUN: whether Bear's repair updated bear_thesis"
    ]
    bull_initial_recovery_status: Annotated[
        str,
        "NOT_NEEDED/APPLIED/UNAPPLIED/NOT_RUN: whether Bull's initial-thesis "
        "textual fallback was reconstructed into canonical bull_thesis",
    ]
    bear_initial_recovery_status: Annotated[
        str,
        "NOT_NEEDED/APPLIED/UNAPPLIED/NOT_RUN: whether Bear's initial-thesis "
        "textual fallback was reconstructed into canonical bear_thesis",
    ]
    bull_review_recovery_status: Annotated[
        str,
        "NOT_NEEDED/APPLIED/UNAPPLIED/NOT_RUN: whether Bull's latest review "
        "textual fallback was reconstructed into a canonical trade-plan patch",
    ]
    bear_review_recovery_status: Annotated[
        str,
        "NOT_NEEDED/APPLIED/UNAPPLIED/NOT_RUN: whether Bear's latest review "
        "textual fallback was reconstructed into a canonical trade-plan patch",
    ]
    manager_integrity_status: Annotated[
        str, "PASS/WARN from the deterministic post-manager integrity check"
    ]
    manager_integrity_findings: Annotated[
        list[dict], "Findings from the deterministic post-manager integrity check"
    ]
    bull_history: Annotated[
        str, "Bullish Conversation history"
    ]  # Bullish Conversation history
    bear_history: Annotated[
        str, "Bearish Conversation history"
    ]  # Bullish Conversation history
    history: Annotated[str, "Conversation history"]  # Conversation history
    current_response: Annotated[str, "Latest response"]  # Last response
    judge_decision: Annotated[str, "Final judge decision"]  # Last response
    count: Annotated[int, "Length of the current conversation"]  # Conversation length


# Risk management team state
class RiskDebateState(TypedDict):
    aggressive_history: Annotated[
        str, "Aggressive reviewer's completed turns (independent + cross-review)"
    ]
    conservative_history: Annotated[
        str, "Conservative reviewer's completed turns (independent + cross-review)"
    ]
    neutral_history: Annotated[
        str, "Neutral reviewer's completed turns (independent + cross-review)"
    ]
    history: Annotated[str, "Conversation history"]  # Conversation history
    latest_speaker: Annotated[str, "Analyst that spoke last"]
    current_aggressive_response: Annotated[
        str, "Latest response by the aggressive analyst"
    ]  # Last response
    current_conservative_response: Annotated[
        str, "Latest response by the conservative analyst"
    ]  # Last response
    current_neutral_response: Annotated[
        str, "Latest response by the neutral analyst"
    ]  # Last response
    aggressive_risk_level: Annotated[
        str, "Aggressive analyst's latest LOW/MEDIUM/HIGH assessment of the trader's plan"
    ]
    conservative_risk_level: Annotated[
        str, "Conservative analyst's latest LOW/MEDIUM/HIGH assessment of the trader's plan"
    ]
    neutral_risk_level: Annotated[
        str, "Neutral analyst's latest LOW/MEDIUM/HIGH assessment of the trader's plan"
    ]
    aggressive_disposition: Annotated[
        str, "Aggressive reviewer's latest KEEP/DEFER/REDUCE_RISK/REJECT_PLAN disposition"
    ]
    conservative_disposition: Annotated[
        str, "Conservative reviewer's latest KEEP/DEFER/REDUCE_RISK/REJECT_PLAN disposition"
    ]
    neutral_disposition: Annotated[
        str, "Neutral reviewer's latest KEEP/DEFER/REDUCE_RISK/REJECT_PLAN disposition"
    ]
    aggressive_initial: Annotated[
        str, "Aggressive reviewer's independent initial assessment text (completion marker for Phase A)"
    ]
    conservative_initial: Annotated[
        str, "Conservative reviewer's independent initial assessment text (completion marker for Phase A)"
    ]
    neutral_initial: Annotated[
        str, "Neutral reviewer's independent initial assessment text (completion marker for Phase A)"
    ]
    aggressive_review: Annotated[
        str, "Aggressive reviewer's cross-review text (completion marker for Phase B)"
    ]
    conservative_review: Annotated[
        str, "Conservative reviewer's cross-review text (completion marker for Phase B)"
    ]
    neutral_review: Annotated[
        str, "Neutral reviewer's cross-review text (completion marker for Phase B)"
    ]
    aggressive_initial_assessment: Annotated[
        dict, "Aggressive reviewer's structured Phase A RiskStanceAssessment, or {} on free-text fallback"
    ]
    conservative_initial_assessment: Annotated[
        dict, "Conservative reviewer's structured Phase A RiskStanceAssessment, or {} on free-text fallback"
    ]
    neutral_initial_assessment: Annotated[
        dict, "Neutral reviewer's structured Phase A RiskStanceAssessment, or {} on free-text fallback"
    ]
    aggressive_review_assessment: Annotated[
        dict, "Aggressive reviewer's structured Phase B RiskStanceAssessment, or {} on free-text fallback"
    ]
    conservative_review_assessment: Annotated[
        dict, "Conservative reviewer's structured Phase B RiskStanceAssessment, or {} on free-text fallback"
    ]
    neutral_review_assessment: Annotated[
        dict, "Neutral reviewer's structured Phase B RiskStanceAssessment, or {} on free-text fallback"
    ]
    risk_integrity_status: Annotated[
        str, "PASS/WARN from the deterministic post-Risk integrity check"
    ]
    risk_integrity_findings: Annotated[
        list[dict], "Findings from the deterministic post-Risk integrity check"
    ]
    portfolio_disposition: Annotated[
        str,
        "Portfolio Manager's real structured KEEP/DEFER/REDUCE_RISK/REJECT_PLAN decision "
        "about the Trader plan -- a separate concept from the investment rating, never "
        "derived from it. Empty string when unparsed/not produced, not guessed.",
    ]
    judge_decision: Annotated[str, "Judge's decision"]
    count: Annotated[int, "Length of the current conversation"]  # Conversation length


class AgentState(MessagesState):
    company_of_interest: Annotated[str, "Company that we are interested in trading"]
    asset_type: Annotated[str, "Asset type under analysis such as stock or crypto"]
    instrument_context: Annotated[str, "Deterministic ticker identity resolved at run start"]
    trade_date: Annotated[str, "What date we are trading at"]

    sender: Annotated[str, "Agent that sent this message"]

    # research step
    market_report: Annotated[str, "Report from the Market Analyst"]
    sentiment_report: Annotated[str, "Report from the Sentiment Analyst"]
    news_report: Annotated[
        str, "Report from the News Researcher of current world affairs"
    ]
    fundamentals_report: Annotated[str, "Report from the Fundamentals Researcher"]

    # researcher team discussion step
    investment_debate_state: Annotated[
        InvestDebateState, "Current state of the debate on if to invest or not"
    ]
    investment_plan: Annotated[str, "Plan generated by the Analyst"]
    # The Research Manager's own structured output (ResearchPlan), model_dump
    # JSON-safe, as it was right before validate_and_repair's own render_fn
    # turned it into investment_plan's text -- None when no structured parse
    # survived (free-text fallback with no parse, or the stage never ran).
    # Structured object is the source of truth; investment_plan's markdown
    # is presentation only, rendered FROM this when it exists (fork README:
    # "Preserve structured objects through the pipeline").
    research_plan_structured: Annotated[
        dict | None, "ResearchPlan.model_dump(mode='json'), or None"
    ]

    trader_investment_plan: Annotated[str, "Plan generated by the Trader"]
    # The Trader's own structured output (TraderProposal), same rule as
    # research_plan_structured above.
    trader_proposal_structured: Annotated[
        dict | None, "TraderProposal.model_dump(mode='json'), or None"
    ]

    # risk management team discussion step
    risk_debate_state: Annotated[
        RiskDebateState, "Current state of the debate on evaluating risk"
    ]
    final_trade_decision: Annotated[str, "Final decision made by the Risk Analysts"]
    # The Portfolio Manager's own structured output (PortfolioDecision), same
    # rule as research_plan_structured above.
    portfolio_decision_structured: Annotated[
        dict | None, "PortfolioDecision.model_dump(mode='json'), or None"
    ]
    past_context: Annotated[str, "Memory log context injected at run start (same-ticker decisions + cross-ticker lessons)"]
    portfolio_context: Annotated[str, "Caller-supplied holdings and cash, rendered at run start; empty when not provided"]

    # Set by the Portfolio Manager node's snapshot-writing wrapper (see
    # pre_pm_snapshot.with_pre_pm_snapshot), never by the node itself --
    # absent/None when no PrePMSnapshot was written for this run (any
    # caller that did not supply run_provenance to TradingAgentsGraph, e.g.
    # backtest/propagate(), the same scoping RecommendationRecord already
    # has). See fork README: "PrePMSnapshot".
    pre_pm_snapshot_ref: Annotated[
        dict | None, "snapshot_id/content_hash/path of the PrePMSnapshot used, or None"
    ]

    # Compact per-stage Validator outcomes (see fork README: "Generation /
    # Validation / Review") -- status/finding-count/repair metadata only,
    # never the full findings' evidence payloads or a copy of the candidate
    # text, so this stays small regardless of how many stages are validated.
    validation_log: Annotated[
        list[dict], "One compact entry per validated stage: status, finding_count, "
        "categories, repair_attempted, repair_succeeded, llm_calls"
    ]
