from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.context import get_instrument_context_from_state, get_language_instruction
from tradingagents.agents.tools import (
    get_company_events,
    get_global_news,
    get_macro_indicators,
    get_news,
    get_prediction_markets,
)
from tradingagents.llm_trace import traced_invoke
from tradingagents.prompts.loader import load_prompt, render_agent_prompt

# The tools this analyst is offered; its tool node is built from the same tuple.
TOOLS = (
    get_news,
    get_global_news,
    get_macro_indicators,
    get_prediction_markets,
    get_company_events,
)


def create_news_analyst(generation_llm):
    def news_analyst_node(state):
        current_date = state["trade_date"]
        asset_type = state.get("asset_type", "stock")
        asset_label = "company" if asset_type == "stock" else "asset"
        instrument_context = get_instrument_context_from_state(state)

        system_message = (
            render_agent_prompt("analysts/news.txt", asset_label=asset_label)
            + get_language_instruction()
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    # Keep the date guidance visible here: it anchors tool-call date ranges.
                    load_prompt("analysts/tool_system.txt"),
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in TOOLS]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        # Bind tools and invoke the bound LLM directly (not `prompt | bound_llm`
        # piped into one chain): a RunnableSequence only forwards invoke()'s
        # extra kwargs -- here, the best-effort replay `seed` -- to its FIRST
        # step. Piped, that step is the prompt template, which silently drops
        # the kwarg instead of passing it on to the LLM. Pre-formatting the
        # prompt and invoking the bound LLM directly produces the exact same
        # message list (verified: ChatPromptTemplate.invoke(list) and
        # .format_messages(messages=list) are equivalent) while letting the
        # kwarg reach the model.
        bound_llm = generation_llm.bind_tools(TOOLS)
        formatted_messages = prompt.format_messages(messages=state["messages"])

        # tool_round identity is the running message count for THIS ticker's
        # own conversation so far -- stable and available without depending
        # on a global call counter or other tickers' execution order (best-
        # effort deterministic replay, see tradingagents/run_seed.py).
        result = traced_invoke(
            bound_llm, formatted_messages,
            call_identity=f"tool_round_{len(state['messages'])}", llm_role="generation",
        )

        report = ""

        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "news_report": report,
        }

    return news_analyst_node
