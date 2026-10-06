from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from tradingagents.agents import (
    create_aggressive_debator,
    create_bear_repair,
    create_bear_researcher,
    create_bull_repair,
    create_bull_researcher,
    create_conservative_debator,
    create_fundamentals_analyst,
    create_market_analyst,
    create_msg_delete,
    create_neutral_debator,
    create_news_analyst,
    create_portfolio_manager,
    create_research_manager,
    create_research_verifier,
    create_sentiment_analyst,
    create_trader,
)
from tradingagents.agents.state import AgentState
from tradingagents.llm_trace import traced_node

from .analyst_execution import build_analyst_execution_plan
from .conditional_logic import ConditionalLogic

# Every target a shared conditional router can return. Each edge driven by the
# router maps all of them, so a fall-through return (e.g. under prompt/i18n/
# refactor drift in the speaker labels) can never hit a missing path_map entry
# and crash LangGraph mid-run (#1088).
DEBATE_PATH_MAP = {
    "Bull Researcher": "Bull Researcher",
    "Bear Researcher": "Bear Researcher",
    "Research Verifier": "Research Verifier",
}
VERIFICATION_PATH_MAP = {
    "Bull Repair": "Bull Repair",
    "Research Manager": "Research Manager",
}
RISK_ANALYSIS_PATH_MAP = {
    "Aggressive Analyst": "Aggressive Analyst",
    "Conservative Analyst": "Conservative Analyst",
    "Neutral Analyst": "Neutral Analyst",
    "Portfolio Manager": "Portfolio Manager",
}


def _tools_or_clear(spec):
    """Route an analyst's turn: run its tool calls, or finish its report."""
    def route(state) -> str:
        return spec.tool_node if state["messages"][-1].tool_calls else spec.clear_node
    return route


class GraphSetup:
    """Handles the setup and configuration of the agent graph.

    Wires each node to the LLM matching its cognitive function in this fork's
    architecture (see fork README: "Model roles: Generation vs Review"):

    - **generation** -- produce, extract, structure, synthesize, propose,
      decide: the four analysts, Bull/Bear initial thesis, Research Manager,
      Trader, Risk's independent initial assessments, Portfolio Manager.
    - **review** -- challenge, verify, audit, correct, repair: Bull/Bear
      review rounds, Research Verifier, research repair, Risk's cross-review.

    generation_llm and review_llm may be the same model or different ones;
    nothing here depends on which.
    """

    def __init__(
        self,
        generation_llm: Any,
        review_llm: Any,
        conditional_logic: ConditionalLogic,
    ):
        """Initialize with required components."""
        self.generation_llm = generation_llm
        self.review_llm = review_llm
        self.conditional_logic = conditional_logic

    def setup_graph(
        self, selected_analysts=("market", "social", "news", "fundamentals")
    ):
        """Set up and compile the agent workflow graph.

        Args:
            selected_analysts (list): List of analyst types to include. Options are:
                - "market": Market analyst
                - "social": Sentiment analyst
                - "news": News analyst
                - "fundamentals": Fundamentals analyst
        """
        plan = build_analyst_execution_plan(selected_analysts)

        analyst_factories = {
            "market": lambda: create_market_analyst(self.generation_llm),
            "social": lambda: create_sentiment_analyst(self.generation_llm),
            "news": lambda: create_news_analyst(self.generation_llm),
            "fundamentals": lambda: create_fundamentals_analyst(self.generation_llm),
        }

        # Bull/Bear and the three Risk reviewers each split initial (generation)
        # from review (review) within the same researcher/debator -- see each
        # factory's own docstring for why they take both LLMs explicitly
        # rather than being duplicated into separate agents.
        bull_researcher_node = create_bull_researcher(self.generation_llm, self.review_llm)
        bear_researcher_node = create_bear_researcher(self.generation_llm, self.review_llm)
        research_verifier_node = create_research_verifier(self.review_llm)
        bull_repair_node = create_bull_repair(self.review_llm)
        bear_repair_node = create_bear_repair(self.review_llm)
        research_manager_node = create_research_manager(self.generation_llm)
        trader_node = create_trader(self.generation_llm)

        aggressive_analyst = create_aggressive_debator(self.generation_llm, self.review_llm)
        neutral_analyst = create_neutral_debator(self.generation_llm, self.review_llm)
        conservative_analyst = create_conservative_debator(self.generation_llm, self.review_llm)
        portfolio_manager_node = create_portfolio_manager(self.generation_llm)

        workflow = StateGraph(AgentState)

        # Every LLM-invoking node is wrapped with traced_node: the only touch
        # point LLM trace mode needs here. It tags every call made while the
        # node runs with the node's own name (read by LLMTraceCallbackHandler
        # when attached) and is a no-op otherwise -- see llm_trace.py.
        for spec in plan.specs:
            workflow.add_node(spec.agent_node, traced_node(spec.agent_node, analyst_factories[spec.key]()))
            workflow.add_node(spec.clear_node, create_msg_delete())
            if spec.tools:
                workflow.add_node(spec.tool_node, ToolNode(list(spec.tools)))

        workflow.add_node("Bull Researcher", traced_node("Bull Researcher", bull_researcher_node))
        workflow.add_node("Bear Researcher", traced_node("Bear Researcher", bear_researcher_node))
        workflow.add_node("Research Verifier", traced_node("Research Verifier", research_verifier_node))
        workflow.add_node("Bull Repair", traced_node("Bull Repair", bull_repair_node))
        workflow.add_node("Bear Repair", traced_node("Bear Repair", bear_repair_node))
        workflow.add_node("Research Manager", traced_node("Research Manager", research_manager_node))
        workflow.add_node("Trader", traced_node("Trader", trader_node))
        workflow.add_node("Aggressive Analyst", traced_node("Aggressive Analyst", aggressive_analyst))
        workflow.add_node("Neutral Analyst", traced_node("Neutral Analyst", neutral_analyst))
        workflow.add_node("Conservative Analyst", traced_node("Conservative Analyst", conservative_analyst))
        workflow.add_node("Portfolio Manager", traced_node("Portfolio Manager", portfolio_manager_node))

        workflow.add_edge(START, plan.specs[0].agent_node)

        for i, spec in enumerate(plan.specs):
            if spec.tools:
                workflow.add_conditional_edges(
                    spec.agent_node, _tools_or_clear(spec), [spec.tool_node, spec.clear_node]
                )
                workflow.add_edge(spec.tool_node, spec.agent_node)
            else:
                workflow.add_edge(spec.agent_node, spec.clear_node)

            # The last analyst hands over to the research debate.
            following = plan.specs[i + 1].agent_node if i < len(plan.specs) - 1 else "Bull Researcher"
            workflow.add_edge(spec.clear_node, following)

        # Both research-debate edges share the complete DEBATE_PATH_MAP (#1088).
        for debate_node in ("Bull Researcher", "Bear Researcher"):
            workflow.add_conditional_edges(
                debate_node,
                self.conditional_logic.should_continue_debate,
                DEBATE_PATH_MAP,
            )
        workflow.add_conditional_edges(
            "Research Verifier",
            self.conditional_logic.should_continue_verification,
            VERIFICATION_PATH_MAP,
        )
        workflow.add_edge("Bull Repair", "Bear Repair")
        workflow.add_edge("Bear Repair", "Research Verifier")
        workflow.add_edge("Research Manager", "Trader")
        workflow.add_edge("Trader", "Aggressive Analyst")
        # All three risk edges share the complete RISK_ANALYSIS_PATH_MAP (#1088).
        for risk_node in ("Aggressive Analyst", "Conservative Analyst", "Neutral Analyst"):
            workflow.add_conditional_edges(
                risk_node,
                self.conditional_logic.should_continue_risk_analysis,
                RISK_ANALYSIS_PATH_MAP,
            )

        workflow.add_edge("Portfolio Manager", END)

        return workflow
