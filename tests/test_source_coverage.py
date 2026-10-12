"""Source-coverage taxonomy, recording, and the Sentiment Analyst's
deterministic NO_DATA/SOURCE_UNAVAILABLE short-circuit (see fork README:
"News/Sentiment acquisition" and the audit it followed). Numbered per the
task's own "TESTS" list (1-20): classifiers 1-7, persistence 8-9, score_source
10, short-circuit 11-16, backward compatibility 17, LLY/MSFT-like cases
18-19, News Analyst unaffected 20.
"""

from __future__ import annotations

import copy
from unittest import mock

import pytest

import tradingagents.dataflows.config as config_module
import tradingagents.default_config as default_config
from tradingagents.agents.analysts.sentiment_analyst import create_sentiment_analyst
from tradingagents.agents.schemas import (
    SentimentDataStatus,
    SentimentReport,
    SentimentScoreSource,
    render_sentiment_report,
)
from tradingagents.dataflows import router
from tradingagents.dataflows.config import set_config
from tradingagents.dataflows.errors import (
    NoMarketDataError,
    VendorNotConfiguredError,
    VendorRateLimitError,
)
from tradingagents.dataflows.source_coverage import (
    SourceCoverageCollector,
    SourceCoverageEntry,
    SourceStatus,
    classify_jev_note,
    classify_reddit_text,
    classify_router_exception,
    classify_router_text,
    classify_stocktwits_text,
    decide_data_status,
    get_source_coverage_snapshot,
    record_news_router_outcome,
    set_active_source_coverage_collector,
)


def _reset_config():
    config_module._config = copy.deepcopy(default_config.DEFAULT_CONFIG)


@pytest.fixture(autouse=True)
def _clean_collector():
    set_active_source_coverage_collector(None)
    _reset_config()
    yield
    set_active_source_coverage_collector(None)
    _reset_config()


# --- 1: SUCCESS_WITH_DATA ------------------------------------------------


@pytest.mark.unit
def test_1_success_with_data_maps_correctly():
    fred_text = (
        "## FRED: Unemployment Rate (UNRATE)\n- Units: %\n- Frequency: Monthly\n"
        "- Window: 2026-01-01 to 2026-09-30\n\n**Latest:** 4.1 (2026-09-30)\n\n"
        "| Date | Value |\n| --- | --- |\n| 2026-08-30 | 4.0 |\n| 2026-09-30 | 4.1 |\n"
    )
    status, count = classify_router_text("get_macro_indicators", fred_text)
    assert status == SourceStatus.SUCCESS_WITH_DATA
    assert count == 2

    stocktwits_text = "Bullish: 3 (60%) · Bearish: 2 (40%) · Unlabeled: 0 · Total: 5 most-recent messages\n\n..."
    entry = classify_stocktwits_text(stocktwits_text)
    assert entry.status == SourceStatus.SUCCESS_WITH_DATA
    assert entry.usable_count == 5

    reddit_text = "r/stocks — 3 recent posts mentioning MSFT:\n  [2026-09-29] title"
    entry = classify_reddit_text(reddit_text)
    assert entry.status == SourceStatus.SUCCESS_WITH_DATA
    assert entry.usable_count == 3


# --- 2: SUCCESS_EMPTY -----------------------------------------------------


@pytest.mark.unit
def test_2_success_empty_maps_correctly():
    status, count = classify_router_text("get_news", "No news found for LLY between 2026-01-01 and 2026-01-07")
    assert status == SourceStatus.SUCCESS_EMPTY
    assert count == 0

    fred_empty = "## FRED: X\nNo observations for X in this window at the 2026-09-30 vintage."
    status, count = classify_router_text("get_macro_indicators", fred_empty)
    assert status == SourceStatus.SUCCESS_EMPTY

    entry = classify_stocktwits_text("<no StockTwits messages for $LLY within 2026-09-23..2026-09-30>")
    assert entry.status == SourceStatus.SUCCESS_EMPTY

    entry = classify_reddit_text("<no Reddit posts found mentioning LLY across r/stocks within 2026-09-23..2026-09-30>")
    assert entry.status == SourceStatus.SUCCESS_EMPTY


# --- 3: UNSUPPORTED_HISTORICAL --------------------------------------------


@pytest.mark.unit
def test_3_unsupported_historical_maps_correctly():
    text = "<Yahoo Finance news unavailable for 2026-09-23..2026-09-30: it only serves recent items, so this is not an absence of news for LLY>"
    status, _ = classify_router_text("get_news", text)
    assert status == SourceStatus.UNSUPPORTED_HISTORICAL

    polymarket_text = (
        "Prediction-market odds are withheld for 2026-09-30. Polymarket serves "
        "only live odds on open markets, with no historical vintage, so serving "
        "them would put post-decision information into a 2026-09-30 analysis."
    )
    status, _ = classify_router_text("get_prediction_markets", polymarket_text)
    assert status == SourceStatus.UNSUPPORTED_HISTORICAL

    entry = classify_stocktwits_text(
        "<StockTwits unavailable for 2026-09-23..2026-09-30: it only serves recent items, so this is not an absence of messages about $LLY>"
    )
    assert entry.status == SourceStatus.UNSUPPORTED_HISTORICAL

    entry = classify_reddit_text(
        "<Reddit search unavailable for 2026-09-23..2026-09-30: it only serves recent items, so this is not an absence of discussion of LLY>"
    )
    assert entry.status == SourceStatus.UNSUPPORTED_HISTORICAL


# --- 4: AUTH_MISSING -------------------------------------------------------


@pytest.mark.unit
def test_4_auth_missing_maps_correctly():
    status = classify_router_exception(VendorNotConfiguredError("FRED_API_KEY environment variable is not set."))
    assert status == SourceStatus.AUTH_MISSING


# --- 5: RATE_LIMITED --------------------------------------------------------


@pytest.mark.unit
def test_5_rate_limited_maps_correctly():
    status = classify_router_exception(VendorRateLimitError("FRED request failed: 429 Too Many Requests"))
    assert status == SourceStatus.RATE_LIMITED


# --- 6: TIMEOUT -------------------------------------------------------------


@pytest.mark.unit
def test_6_timeout_maps_correctly():
    status = classify_router_exception(TimeoutError("timed out after 30s"))
    assert status == SourceStatus.TIMEOUT

    status = classify_router_exception(NoMarketDataError("MSFT", "MSFT", "news unavailable: Read timeout"))
    assert status == SourceStatus.TIMEOUT


# --- 7: SCREENED_TO_ZERO -----------------------------------------------------


@pytest.mark.unit
def test_7_screened_to_zero_maps_correctly():
    entry = classify_stocktwits_text(
        "Screened by Jev: 0 of the 8 posts fetched are about Eli Lilly (LLY); their stance...\n\n"
        "<none of the 8 StockTwits messages is about $LLY>"
    )
    assert entry.status == SourceStatus.SCREENED_TO_ZERO

    entry = classify_reddit_text("r/stocks: <no posts about LLY after screening>\n\nr/investing: <no posts about LLY after screening>")
    assert entry.status == SourceStatus.SCREENED_TO_ZERO

    jev_entry = classify_jev_note("Screened by Jev: 0 of the 8 posts fetched are about Eli Lilly (LLY); their stance: 0 bullish, 0 bearish, 0 neutral, 0 unclear.")
    assert jev_entry is not None
    assert jev_entry.status == SourceStatus.SCREENED_TO_ZERO
    assert jev_entry.raw_count == 8
    assert jev_entry.usable_count == 0


# --- 8: serialization -------------------------------------------------------


@pytest.mark.unit
def test_8_source_coverage_serializes():
    collector = SourceCoverageCollector()
    collector.record("news", "company", "yfinance", SourceCoverageEntry(
        source="yfinance", status=SourceStatus.UNSUPPORTED_HISTORICAL, usable_count=0,
        requested_start="2026-09-23", requested_end="2026-09-30",
    ))
    collector.record("news", "macro", "fred", SourceCoverageEntry(
        source="fred", status=SourceStatus.SUCCESS_WITH_DATA, usable_count=15,
    ))
    snapshot = collector.snapshot()
    assert snapshot["news"]["company"]["status"] == "UNSUPPORTED_HISTORICAL"
    assert snapshot["news"]["company"]["usable_count"] == 0
    assert snapshot["news"]["company"]["requested_start"] == "2026-09-23"
    assert snapshot["news"]["macro"]["status"] == "SUCCESS_WITH_DATA"
    assert snapshot["news"]["macro"]["usable_count"] == 15
    # Compact: no null/absent fields written, no payload/secret-shaped keys.
    assert "reason" not in snapshot["news"]["macro"]
    assert all(k in {"status", "usable_count", "raw_count", "reason", "requested_start", "requested_end"}
               for k in snapshot["news"]["company"])


# --- 9: persisted result ----------------------------------------------------


@pytest.mark.unit
def test_9_source_coverage_appears_in_persisted_report(tmp_path):
    from tradingagents.reporting import write_report_tree

    final_state = {
        "market_report": "x",
        "source_coverage": {"sentiment": {"stocktwits": {"status": "UNSUPPORTED_HISTORICAL", "usable_count": 0}}},
    }
    write_report_tree(final_state, "LLY", tmp_path)
    coverage_path = tmp_path / "source_coverage.json"
    assert coverage_path.exists()
    import json
    data = json.loads(coverage_path.read_text(encoding="utf-8"))
    assert data["sentiment"]["stocktwits"]["status"] == "UNSUPPORTED_HISTORICAL"


# --- 10: score_source is no longer lost -------------------------------------


@pytest.mark.unit
def test_10_score_source_is_preserved_through_construction_and_render():
    report = SentimentReport(
        overall_band="Neutral", overall_score=5.0, confidence="low", narrative="n",
        score_source=SentimentScoreSource.NO_DATA_DEFAULT, data_status=SentimentDataStatus.NO_DATA,
    )
    assert report.score_source == SentimentScoreSource.NO_DATA_DEFAULT
    rendered = render_sentiment_report(report)
    assert "NO_DATA" in rendered


# --- Sentiment-node harness used by 11-19 ----------------------------------


def _build_sentiment_state():
    return {
        "company_of_interest": "LLY",
        "trade_date": "2026-09-30",
        "instrument_context": "LLY (Eli Lilly and Company)",
        "messages": [],
    }


def _patch_sources(monkeypatch, *, news, stocktwits, reddit, screen=None):
    import tradingagents.agents.analysts.sentiment_analyst as mod

    monkeypatch.setattr(mod.get_news, "func", lambda *a, **k: news)
    monkeypatch.setattr(mod, "fetch_stocktwits_messages", lambda *a, **k: stocktwits)
    monkeypatch.setattr(mod, "fetch_reddit_posts", lambda *a, **k: reddit)
    monkeypatch.setattr(mod, "jev_screen", lambda ticker: screen)


_UNAVAILABLE_NEWS = "<Yahoo Finance news unavailable for 2026-09-23..2026-09-30: it only serves recent items, so this is not an absence of news for LLY>"
_UNAVAILABLE_STOCKTWITS = "<StockTwits unavailable for 2026-09-23..2026-09-30: it only serves recent items, so this is not an absence of messages about $LLY>"
_UNAVAILABLE_REDDIT = "<Reddit search unavailable for 2026-09-23..2026-09-30: it only serves recent items, so this is not an absence of discussion of LLY>"


# --- 11: real data still calls the LLM --------------------------------------


@pytest.mark.unit
def test_11_sentiment_with_real_data_still_calls_the_llm(monkeypatch):
    import tradingagents.agents.analysts.sentiment_analyst as mod

    _patch_sources(
        monkeypatch, news=_UNAVAILABLE_NEWS,
        stocktwits="Bullish: 5 (100%) · Bearish: 0 (0%) · Unlabeled: 0 · Total: 5 most-recent messages\n\n...",
        reddit=_UNAVAILABLE_REDDIT,
    )
    called = mock.Mock(return_value="**Overall Sentiment:** **Bullish** (Score: 7.0/10)\n**Confidence:** High\n\nreal")
    monkeypatch.setattr(mod, "invoke_structured_or_freetext", called)
    node = create_sentiment_analyst(generation_llm=object())
    result = node(_build_sentiment_state())
    assert called.call_count == 1
    assert "Bullish" in result["sentiment_report"]


# --- 12: no historical data -> no LLM call ----------------------------------


@pytest.mark.unit
def test_12_sentiment_without_historical_data_never_calls_the_llm(monkeypatch):
    import tradingagents.agents.analysts.sentiment_analyst as mod

    _patch_sources(monkeypatch, news=_UNAVAILABLE_NEWS, stocktwits=_UNAVAILABLE_STOCKTWITS, reddit=_UNAVAILABLE_REDDIT)
    called = mock.Mock(side_effect=AssertionError("LLM must not be called"))
    monkeypatch.setattr(mod, "invoke_structured_or_freetext", called)
    node = create_sentiment_analyst(generation_llm=object())
    result = node(_build_sentiment_state())
    assert called.call_count == 0
    assert "sentiment_report" in result


# --- 13: short-circuit produces data_status=NO_DATA -------------------------


@pytest.mark.unit
def test_13_short_circuit_produces_no_data_status(monkeypatch):
    import tradingagents.agents.analysts.sentiment_analyst as mod

    _patch_sources(monkeypatch, news=_UNAVAILABLE_NEWS, stocktwits=_UNAVAILABLE_STOCKTWITS, reddit=_UNAVAILABLE_REDDIT)
    monkeypatch.setattr(mod, "invoke_structured_or_freetext", mock.Mock())
    node = create_sentiment_analyst(generation_llm=object())
    result = node(_build_sentiment_state())
    assert "**Sentiment data status:** **NO_DATA**" in result["sentiment_report"]


# --- 14: NO_DATA is never presented as neutral sentiment evidence -----------


@pytest.mark.unit
def test_14_no_data_is_never_presented_as_neutral_sentiment_evidence(monkeypatch):
    import tradingagents.agents.analysts.sentiment_analyst as mod

    _patch_sources(monkeypatch, news=_UNAVAILABLE_NEWS, stocktwits=_UNAVAILABLE_STOCKTWITS, reddit=_UNAVAILABLE_REDDIT)
    monkeypatch.setattr(mod, "invoke_structured_or_freetext", mock.Mock())
    node = create_sentiment_analyst(generation_llm=object())
    report_text = node(_build_sentiment_state())["sentiment_report"]
    assert "**Overall Sentiment:** **Neutral**" not in report_text
    assert "Observed sentiment:** UNKNOWN" in report_text
    assert "placeholders" in report_text.lower()


# --- 15: all-technical-failure -> SOURCE_UNAVAILABLE, not NO_DATA -----------


@pytest.mark.unit
def test_15_all_technical_failures_produce_source_unavailable_not_no_data(monkeypatch):
    import tradingagents.agents.analysts.sentiment_analyst as mod

    _patch_sources(
        monkeypatch,
        news="DATA_UNAVAILABLE: no configured vendor could serve get_news right now (rate limited).",
        stocktwits="<stocktwits unavailable: TimeoutError>",
        reddit="<Reddit unavailable: fetch failed (r/stocks, r/investing); this is not an absence of discussion>",
    )
    monkeypatch.setattr(mod, "invoke_structured_or_freetext", mock.Mock())
    node = create_sentiment_analyst(generation_llm=object())
    report_text = node(_build_sentiment_state())["sentiment_report"]
    assert "SOURCE_UNAVAILABLE" in report_text
    assert "**Sentiment data status:** **NO_DATA**" not in report_text


# --- 16: mixed sources (one with data) -> calls the LLM ---------------------


@pytest.mark.unit
def test_16_mixed_sources_with_one_having_data_calls_the_llm(monkeypatch):
    import tradingagents.agents.analysts.sentiment_analyst as mod

    _patch_sources(
        monkeypatch, news=_UNAVAILABLE_NEWS, stocktwits=_UNAVAILABLE_STOCKTWITS,
        reddit="r/stocks — 2 recent posts mentioning LLY:\n  [2026-09-29] title",
    )
    called = mock.Mock(return_value="**Overall Sentiment:** **Mixed** (Score: 5.0/10)\n**Confidence:** Low\n\nreal")
    monkeypatch.setattr(mod, "invoke_structured_or_freetext", called)
    node = create_sentiment_analyst(generation_llm=object())
    node(_build_sentiment_state())
    assert called.call_count == 1


# --- 17: backward compatibility with real data ------------------------------


@pytest.mark.unit
def test_17_backward_compatible_render_when_data_is_present():
    legacy_payload = {"overall_band": "Bullish", "overall_score": 7.5, "confidence": "high", "narrative": "x"}
    report = SentimentReport.model_validate(legacy_payload)
    assert report.data_status == SentimentDataStatus.HAS_DATA
    assert report.score_source == SentimentScoreSource.MODEL
    rendered = render_sentiment_report(report)
    assert rendered.startswith("**Overall Sentiment:** **Bullish** (Score: 7.5/10)")
    assert "Sentiment data status" not in rendered


# --- 18: LLY-like historical case -------------------------------------------


@pytest.mark.unit
def test_18_lly_like_historical_case_short_circuits(monkeypatch):
    import tradingagents.agents.analysts.sentiment_analyst as mod

    _patch_sources(monkeypatch, news=_UNAVAILABLE_NEWS, stocktwits=_UNAVAILABLE_STOCKTWITS, reddit=_UNAVAILABLE_REDDIT)
    monkeypatch.setattr(mod, "invoke_structured_or_freetext", mock.Mock(side_effect=AssertionError))
    node = create_sentiment_analyst(generation_llm=object())
    state = _build_sentiment_state()
    state["company_of_interest"] = "LLY"
    result = node(state)
    assert "NO_DATA" in result["sentiment_report"]


# --- 19: MSFT-like historical case ------------------------------------------


@pytest.mark.unit
def test_19_msft_like_historical_case_short_circuits(monkeypatch):
    import tradingagents.agents.analysts.sentiment_analyst as mod

    msft_news = _UNAVAILABLE_NEWS.replace("LLY", "MSFT")
    msft_stocktwits = _UNAVAILABLE_STOCKTWITS.replace("LLY", "MSFT")
    msft_reddit = _UNAVAILABLE_REDDIT.replace("LLY", "MSFT")
    _patch_sources(monkeypatch, news=msft_news, stocktwits=msft_stocktwits, reddit=msft_reddit)
    monkeypatch.setattr(mod, "invoke_structured_or_freetext", mock.Mock(side_effect=AssertionError))
    node = create_sentiment_analyst(generation_llm=object())
    state = _build_sentiment_state()
    state["company_of_interest"] = "MSFT"
    result = node(state)
    assert "NO_DATA" in result["sentiment_report"]


# --- 20: News Analyst behavior is unchanged ---------------------------------


@pytest.mark.unit
def test_20_news_router_return_value_is_unchanged_by_coverage_recording():
    """No active collector (the normal case outside a graph run, same as
    every pre-existing router test) -- recording is a complete no-op, and
    route_to_vendor's return value/control flow is byte-for-byte the same
    as before this task. Also covers a multi-vendor fallback chain (not
    exercised by the real default config, but must not have regressed)."""
    assert get_source_coverage_snapshot() is None

    def _vendor1(*a, **k):
        raise VendorRateLimitError("throttled")

    def _vendor2(*a, **k):
        return "## MSFT News, from 2026-09-23 to 2026-09-30:\n\n### Real headline\n"

    set_config({"data_vendors": {"news_data": "vendor1,vendor2"}})
    with mock.patch.dict(router.VENDOR_METHODS, {"get_news": {"vendor1": _vendor1, "vendor2": _vendor2}}, clear=False):
        result = router.route_to_vendor("get_news", "MSFT", "2026-09-23", "2026-09-30")
    assert result == "## MSFT News, from 2026-09-23 to 2026-09-30:\n\n### Real headline\n"
    # Still a no-op: nothing was recorded anywhere, since no run ever
    # activated a collector for this call.
    assert get_source_coverage_snapshot() is None


@pytest.mark.unit
def test_20b_coverage_recording_is_a_no_op_for_unrelated_methods():
    """get_stock_data/get_fundamentals/etc. are never touched by this
    task's instrumentation -- record_news_router_outcome must be a no-op
    for any method outside the four News-analyst ones, even with an active
    collector, so OHLCV/fundamentals routing is provably unaffected."""
    collector = SourceCoverageCollector()
    set_active_source_coverage_collector(collector)
    try:
        record_news_router_outcome("get_stock_data", "yfinance", ("MSFT",), text="some OHLCV text")
        record_news_router_outcome("get_fundamentals", "yfinance", ("MSFT",), text="some fundamentals text")
    finally:
        set_active_source_coverage_collector(None)
    assert collector.snapshot() == {}


@pytest.mark.unit
def test_decide_data_status_rules():
    unsupported = SourceCoverageEntry(source="x", status=SourceStatus.UNSUPPORTED_HISTORICAL, usable_count=0)
    empty = SourceCoverageEntry(source="x", status=SourceStatus.SUCCESS_EMPTY, usable_count=0)
    has_data = SourceCoverageEntry(source="x", status=SourceStatus.SUCCESS_WITH_DATA, usable_count=3)
    timeout = SourceCoverageEntry(source="x", status=SourceStatus.TIMEOUT, usable_count=0)

    assert decide_data_status([unsupported, empty, unsupported]) == "NO_DATA"
    assert decide_data_status([has_data, unsupported, empty]) == "HAS_DATA"
    assert decide_data_status([unsupported, timeout, empty]) == "SOURCE_UNAVAILABLE"
    assert decide_data_status([timeout, timeout, timeout]) == "SOURCE_UNAVAILABLE"
