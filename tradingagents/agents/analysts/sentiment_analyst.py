"""Sentiment analyst: one sentiment report from three sources.

The node fetches its sources before calling the model and puts them in the
prompt, so the model reports on data it was given rather than inventing posts:

  1. News headlines: Yahoo Finance
  2. StockTwits messages: the cashtag stream, with Bullish/Bearish tags
  3. Reddit posts: r/wallstreetbets, r/stocks, r/investing

Each source is trimmed to the analysis window. With a TypeSafe key, the social
posts are screened by Jev first (see post_screen). These feeds serve recent items
and are not archived, so a historical run's sentiment inputs are not
point-in-time.

The report is a SentimentReport through structured output where the provider
supports it and free text otherwise, so the band, score and confidence header
reads the same across providers.
"""

from datetime import datetime, timedelta

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.context import get_instrument_context_from_state, get_language_instruction
from tradingagents.agents.post_screen import jev_screen
from tradingagents.agents.schemas import (
    SentimentBand,
    SentimentDataStatus,
    SentimentReport,
    SentimentScoreSource,
    render_sentiment_report,
)
from tradingagents.agents.structured import (
    NO_EXTERNAL_TOOLS,
    bind_structured,
    invoke_structured_or_freetext,
)
from tradingagents.agents.tools import get_news
from tradingagents.dataflows.source_coverage import (
    SourceCoverageEntry,
    SourceStatus,
    classify_jev_note,
    classify_reddit_text,
    classify_router_text,
    classify_stocktwits_text,
    decide_data_status,
    record_source_coverage,
    source_coverage_category,
)
from tradingagents.dataflows.vendors.reddit import fetch_reddit_posts
from tradingagents.dataflows.vendors.stocktwits import fetch_stocktwits_messages
from tradingagents.prompts.loader import load_prompt, render_agent_prompt


def _seven_days_back(trade_date: str) -> str:
    return (datetime.strptime(trade_date, "%Y-%m-%d") - timedelta(days=7)).strftime("%Y-%m-%d")


def create_sentiment_analyst(generation_llm):
    """Create a sentiment analyst node for the trading graph.

    Pre-fetches news + StockTwits + Reddit data, injects them into the
    prompt as structured blocks, and produces a deterministic sentiment
    report via structured output (with a free-text fallback for providers
    that do not support it). A generation operation (see fork README:
    "Model roles: Generation vs Review").
    """
    structured_llm = bind_structured(generation_llm, SentimentReport, "Sentiment Analyst")

    def sentiment_analyst_node(state):
        ticker = state["company_of_interest"]
        end_date = state["trade_date"]
        start_date = _seven_days_back(end_date)
        instrument_context = get_instrument_context_from_state(state)

        # Pre-fetch all three sources. Each fetcher degrades gracefully and
        # returns a string (no exceptions surface from here), so the LLM
        # always sees something — either real data or a clear placeholder.
        # "sentiment" scopes this get_news call's source-coverage recording
        # to sentiment.news (flat), not news.company -- the same shared
        # get_news path the News Analyst's tool loop also goes through (see
        # tradingagents/dataflows/router.py).
        with source_coverage_category("sentiment"):
            news_block = get_news.func(ticker, start_date, end_date)
        # Pass the analysis window so a historical run trims social posts to it
        # instead of leaking today's chatter into a backtest (#1220).
        screen = jev_screen(ticker)
        stocktwits_block = fetch_stocktwits_messages(
            ticker, limit=30, start_date=start_date, end_date=end_date, screen=screen
        )
        reddit_block = fetch_reddit_posts(ticker, start_date=start_date, end_date=end_date, screen=screen)

        # News coverage was already recorded by the shared get_news/router
        # path above; classified again here (never re-recorded) purely to
        # feed this node's own short-circuit decision below.
        news_status, news_count = classify_router_text("get_news", news_block)
        news_entry = SourceCoverageEntry(source="yahoo_news", status=news_status, usable_count=news_count)
        stocktwits_entry = classify_stocktwits_text(stocktwits_block)
        reddit_entry = classify_reddit_text(reddit_block)
        record_source_coverage("stocktwits", "stocktwits", stocktwits_entry, top_level="sentiment")
        record_source_coverage("reddit", "reddit", reddit_entry, top_level="sentiment")
        _record_jev_screening_coverage(screen, stocktwits_block, reddit_block)

        data_status = decide_data_status([news_entry, stocktwits_entry, reddit_entry])
        if data_status != "HAS_DATA":
            report_text = _build_no_data_report(
                ticker=ticker, start_date=start_date, end_date=end_date, data_status=data_status,
                entries={"news": news_entry, "stocktwits": stocktwits_entry, "reddit": reddit_entry},
            )
            return {
                "messages": [AIMessage(content=report_text)],
                "sentiment_report": report_text,
            }

        system_message = _build_system_message(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            news_block=news_block,
            stocktwits_block=stocktwits_block,
            reddit_block=reddit_block,
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    load_prompt("analysts/prefetched_system.txt"),
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(current_date=end_date)
        prompt = prompt.partial(instrument_context=instrument_context)
        prompt = prompt.partial(NO_EXTERNAL_TOOLS=NO_EXTERNAL_TOOLS)

        # Format the template into a concrete message list so the structured
        # and free-text paths receive the same input. No bind_tools — the
        # data is already in the prompt.
        formatted_messages = prompt.format_messages(messages=state["messages"])

        report_text = invoke_structured_or_freetext(
            structured_llm,
            generation_llm,
            formatted_messages,
            render_sentiment_report,
            "Sentiment Analyst",
            call_identity="sentiment_analyst",
            llm_role="generation",
        )

        return {
            "messages": [AIMessage(content=report_text)],
            "sentiment_report": report_text,
        }

    return sentiment_analyst_node


def _record_jev_screening_coverage(screen, stocktwits_block: str, reddit_block: str) -> None:
    """Jev's own note is embedded in whichever block(s) it actually ran
    against (see post_screen.jev_screen) -- classified from there rather
    than calling it a second time. ``screen is None`` means no TypeSafe key
    is configured, so screening never ran at all for this ticker."""
    if screen is None:
        record_source_coverage(
            "jev_screening", "jev_screening",
            SourceCoverageEntry(source="jev_screening", status=SourceStatus.NOT_CONFIGURED),
            top_level="sentiment",
        )
        return
    entry = classify_jev_note(stocktwits_block) or classify_jev_note(reddit_block)
    if entry is not None:
        record_source_coverage("jev_screening", "jev_screening", entry, top_level="sentiment")


def _build_no_data_report(
    *, ticker: str, start_date: str, end_date: str, data_status: str,
    entries: dict[str, SourceCoverageEntry],
) -> str:
    """Deterministic SentimentReport, built with NO model call, for when no
    sentiment source returned usable content (see ``decide_data_status``).
    Never introduces an opinion: band/score stay at the schema's neutral
    default purely for contract compatibility, and ``render_sentiment_report``
    marks them unmistakably as placeholders whenever ``data_status`` is not
    ``HAS_DATA`` (see schemas.py; audit finding: "Neutral must never be
    presented as evidence of neutral sentiment").
    """
    coverage_lines = "\n".join(f"- {name}: {entry.status.value}" for name, entry in entries.items())
    if data_status == "NO_DATA":
        verdict = (
            "Every source legitimately had nothing to show for this window "
            "(point-in-time unsupported, a genuinely empty result, screened "
            "out, or not configured). This is a real absence-of-evidence "
            "verdict, not a retrieval failure."
        )
        status_enum = SentimentDataStatus.NO_DATA
    else:
        verdict = (
            "At least one source failed technically (timeout, rate limit, "
            "missing credentials, or a provider error) rather than "
            "legitimately returning nothing. This says nothing about whether "
            "sentiment existed -- retrieval simply did not succeed."
        )
        status_enum = SentimentDataStatus.SOURCE_UNAVAILABLE
    narrative = (
        f"No sentiment evidence could be retrieved for `{ticker}` over "
        f"{start_date}..{end_date}.\n\n"
        f"Source coverage:\n{coverage_lines}\n\n"
        f"{verdict}\n\n"
        "This report was generated deterministically, without a model call, "
        "because no source returned usable sentiment content for this window."
    )
    report = SentimentReport(
        overall_band=SentimentBand.NEUTRAL,
        overall_score=5.0,
        confidence="low",
        narrative=narrative,
        score_source=SentimentScoreSource.NO_DATA_DEFAULT,
        data_status=status_enum,
    )
    return render_sentiment_report(report)


def _build_system_message(
    *,
    ticker: str,
    start_date: str,
    end_date: str,
    news_block: str,
    stocktwits_block: str,
    reddit_block: str,
) -> str:
    """Assemble the sentiment-analyst system message with structured data blocks."""
    return render_agent_prompt(
        "analysts/sentiment.txt",
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        news_block=news_block,
        stocktwits_block=stocktwits_block,
        reddit_block=reddit_block,
    ) + get_language_instruction()
