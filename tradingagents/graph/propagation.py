from typing import Any

from tradingagents.agents.state import InvestDebateState, RiskDebateState
from tradingagents.dataflows.config import get_config
from tradingagents.default_config import DEFAULT_CONFIG


class Propagator:
    """Handles state initialization and propagation through the graph."""

    def __init__(self, max_recur_limit=100):
        """Initialize with configuration parameters."""
        self.max_recur_limit = max_recur_limit

    def create_initial_state(
        self,
        company_name: str,
        trade_date: str,
        asset_type: str = "stock",
        past_context: str = "",
        instrument_context: str = "",
        portfolio_context: str = "",
        setup_tags: dict | None = None,
        atr_reference: float | None = None,
    ) -> dict[str, Any]:
        """Create the initial state for the agent graph.

        ``instrument_context`` is the deterministic ticker-identity string
        resolved once at run start (see
        ``TradingAgentsGraph.resolve_instrument_context``). When empty, agents
        fall back to ticker-only context via
        ``get_instrument_context_from_state``.

        ``setup_tags``/``atr_reference`` are the deterministic objective setup
        tags (see ``researchers/setup_tags.py``), likewise resolved once at run
        start so Bull and Bear both see the identical values. Left empty/None
        when the caller does not supply them (e.g. a bare unit test building
        state directly), which never triggers a network call.
        """
        return {
            "messages": [("human", company_name)],
            "company_of_interest": company_name,
            "asset_type": asset_type,
            "instrument_context": instrument_context,
            "trade_date": str(trade_date),
            "past_context": past_context,
            "portfolio_context": portfolio_context,
            "investment_debate_state": InvestDebateState(
                {
                    "bull_initial": "",
                    "bear_initial": "",
                    "rebuttal_history": "",
                    "review_outcomes": "",
                    "current_bull_response": "",
                    "current_bear_response": "",
                    "pending_bull_rebuttal": "",
                    "bull_rebuttal_count": 0,
                    "bear_rebuttal_count": 0,
                    "debate_round": 0,
                    "verification_history": [],
                    "verifier_pass_count": 0,
                    "verification_status": "",
                    "verification_findings": [],
                    "repair_triggered": False,
                    "repair_rounds": 0,
                    "repaired_agents": [],
                    "bull_repair": "",
                    "bear_repair": "",
                    "bull_repair_structured_status": "NOT_RUN",
                    "bear_repair_structured_status": "NOT_RUN",
                    "bull_initial_recovery_status": "NOT_RUN",
                    "bear_initial_recovery_status": "NOT_RUN",
                    "bull_review_recovery_status": "NOT_RUN",
                    "bear_review_recovery_status": "NOT_RUN",
                    "research_horizon": get_config().get(
                        "research_horizon", DEFAULT_CONFIG["research_horizon"]
                    ),
                    "setup_tags": dict(setup_tags) if setup_tags else {},
                    "atr_reference": atr_reference,
                    "bull_thesis": {},
                    "bear_thesis": {},
                    "bull_conviction_history": [],
                    "bear_conviction_history": [],
                    "bull_trade_metrics": {},
                    "bear_trade_metrics": {},
                    "new_data_exceptions": [],
                    "trade_geometry_rejections": [],
                    "withdrawn_values": [],
                    "manager_integrity_status": "",
                    "manager_integrity_findings": [],
                    "bull_history": "",
                    "bear_history": "",
                    "history": "",
                    "current_response": "",
                    "judge_decision": "",
                    "count": 0,
                }
            ),
            "risk_debate_state": RiskDebateState(
                {
                    "aggressive_history": "",
                    "conservative_history": "",
                    "neutral_history": "",
                    "history": "",
                    "latest_speaker": "",
                    "current_aggressive_response": "",
                    "current_conservative_response": "",
                    "current_neutral_response": "",
                    "aggressive_risk_level": "",
                    "conservative_risk_level": "",
                    "neutral_risk_level": "",
                    "aggressive_disposition": "",
                    "conservative_disposition": "",
                    "neutral_disposition": "",
                    "aggressive_initial": "",
                    "conservative_initial": "",
                    "neutral_initial": "",
                    "aggressive_review": "",
                    "conservative_review": "",
                    "neutral_review": "",
                    "aggressive_initial_assessment": {},
                    "conservative_initial_assessment": {},
                    "neutral_initial_assessment": {},
                    "aggressive_review_assessment": {},
                    "conservative_review_assessment": {},
                    "neutral_review_assessment": {},
                    "risk_integrity_status": "",
                    "risk_integrity_findings": [],
                    "portfolio_disposition": "",
                    "judge_decision": "",
                    "count": 0,
                }
            ),
            "market_report": "",
            "fundamentals_report": "",
            "sentiment_report": "",
            "news_report": "",
        }

    def get_graph_args(self, callbacks: list | None = None) -> dict[str, Any]:
        """Get arguments for the graph invocation.

        Args:
            callbacks: Optional list of callback handlers for tool execution tracking.
                       Note: LLM callbacks are handled separately via LLM constructor.
        """
        config = {"recursion_limit": self.max_recur_limit}
        if callbacks:
            config["callbacks"] = callbacks
        return {
            "stream_mode": "values",
            "config": config,
        }
