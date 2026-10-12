"""Alpha Vantage as a historical fallback for News (``news_data =
yfinance,alpha_vantage``) -- see fork README: "News/Sentiment acquisition"
and the audit/evaluation that followed. Covers: the vendor chain actually
advancing past yfinance's UNSUPPORTED_HISTORICAL result (it didn't before
this task -- a successful return, even a placeholder, never reached the
router's except clauses), Alpha Vantage's own point-in-time post-check
(never trusting time_from/time_to alone), and per-vendor source_coverage.

Numbered per the task's own "TESTS" list (1-10).
"""

from __future__ import annotations

import copy
import json
from unittest import mock

import pytest

import tradingagents.dataflows.config as config_module
import tradingagents.default_config as default_config
from tradingagents.dataflows import router
from tradingagents.dataflows.config import set_config
from tradingagents.dataflows.source_coverage import (
    SourceCoverageCollector,
    SourceStatus,
    classify_router_exception,
    set_active_source_coverage_collector,
)
from tradingagents.dataflows.vendors.alpha_vantage import news as av_news
from tradingagents.dataflows.vendors.alpha_vantage.common import (
    AlphaVantageNotConfiguredError,
    AlphaVantageRateLimitError,
)


def _reset_config():
    config_module._config = copy.deepcopy(default_config.DEFAULT_CONFIG)


@pytest.fixture(autouse=True)
def _clean():
    set_active_source_coverage_collector(None)
    _reset_config()
    yield
    set_active_source_coverage_collector(None)
    _reset_config()


_YFINANCE_UNSUPPORTED = (
    "<Yahoo Finance news unavailable for 2026-09-23..2026-09-30: it only "
    "serves recent items, so this is not an absence of news for LLY>"
)


def _av_feed(*, title="Eli Lilly beats earnings", time_published="20260928T120000", ticker="LLY", relevance="0.8"):
    return json.dumps({
        "items": "1",
        "feed": [{
            "title": title,
            "url": "https://example.com/a",
            "time_published": time_published,
            "summary": "Eli Lilly reported quarterly results.",
            "source": "Example Wire",
            "ticker_sentiment": [{"ticker": ticker, "relevance_score": relevance}],
        }],
    })


# --- 1: yfinance unsupported -> chain can reach Alpha Vantage ---------------


@pytest.mark.unit
def test_1_yfinance_unsupported_historical_lets_the_chain_reach_alpha_vantage():
    set_config({"data_vendors": {"news_data": "yfinance,alpha_vantage"}})
    av_called = mock.Mock(return_value="## LLY News, from 2026-09-23 to 2026-09-30:\n\n### Real headline\n")
    with mock.patch.dict(
        router.VENDOR_METHODS, {"get_news": {"yfinance": lambda *a, **k: _YFINANCE_UNSUPPORTED, "alpha_vantage": av_called}},
        clear=False,
    ):
        result = router.route_to_vendor("get_news", "LLY", "2026-09-23", "2026-09-30")
    av_called.assert_called_once()
    assert result == "## LLY News, from 2026-09-23 to 2026-09-30:\n\n### Real headline\n"


@pytest.mark.unit
def test_1b_both_vendors_unsupported_historical_returns_the_last_ones_own_sentinel():
    set_config({"data_vendors": {"news_data": "yfinance,alpha_vantage"}})
    av_unsupported = "<Alpha Vantage news unavailable for 2026-09-23..2026-09-30: it only serves recent items, so this is not an absence of news for LLY>"
    with mock.patch.dict(
        router.VENDOR_METHODS, {"get_news": {"yfinance": lambda *a, **k: _YFINANCE_UNSUPPORTED, "alpha_vantage": lambda *a, **k: av_unsupported}},
        clear=False,
    ):
        result = router.route_to_vendor("get_news", "LLY", "2026-09-23", "2026-09-30")
    assert result == av_unsupported


# --- 2: AV item inside the window passes ------------------------------------


@pytest.mark.unit
def test_2_av_item_inside_window_passes(monkeypatch):
    monkeypatch.setattr(av_news, "_make_api_request", lambda fn, params: _av_feed(time_published="20260928T120000"))
    result = av_news.get_news("LLY", "2026-09-23", "2026-09-30")
    assert "Eli Lilly beats earnings" in result
    assert result.startswith("## LLY News, from 2026-09-23 to 2026-09-30:")


# --- 3: AV item after end_date is rejected ----------------------------------


@pytest.mark.unit
def test_3_av_item_after_end_date_is_rejected(monkeypatch):
    """Even though time_from/time_to were sent to the vendor, this project's
    own point-in-time post-check must independently reject an item the
    vendor served anyway with a time_published after the requested window --
    never trusting the vendor's own filtering blindly."""
    monkeypatch.setattr(
        av_news, "_make_api_request",
        lambda fn, params: _av_feed(title="Future leak", time_published="20261005T120000"),
    )
    result = av_news.get_news("LLY", "2026-09-23", "2026-09-30")
    assert "Future leak" not in result
    assert "unavailable" in result.lower() or "no news found" in result.lower()


@pytest.mark.unit
def test_3b_av_item_before_start_date_is_also_rejected(monkeypatch):
    monkeypatch.setattr(
        av_news, "_make_api_request",
        lambda fn, params: _av_feed(title="Too old", time_published="20260801T120000"),
    )
    result = av_news.get_news("LLY", "2026-09-23", "2026-09-30")
    assert "Too old" not in result


# --- 4: coverage keeps BOTH vendors' entries --------------------------------


@pytest.mark.unit
def test_4_coverage_keeps_both_vendor_entries_without_overwriting():
    set_config({"data_vendors": {"news_data": "yfinance,alpha_vantage"}})
    collector = SourceCoverageCollector()
    set_active_source_coverage_collector(collector)
    try:
        with mock.patch.dict(
            router.VENDOR_METHODS,
            {
                "get_news": {
                    "yfinance": lambda *a, **k: _YFINANCE_UNSUPPORTED,
                    "alpha_vantage": lambda *a, **k: "## LLY News, from 2026-09-23 to 2026-09-30:\n\n### Real (source: X)\n",
                },
            },
            clear=False,
        ):
            router.route_to_vendor("get_news", "LLY", "2026-09-23", "2026-09-30")
    finally:
        set_active_source_coverage_collector(None)
    snapshot = collector.snapshot()
    company = snapshot["news"]["company"]
    assert set(company.keys()) == {"yfinance", "alpha_vantage"}
    assert company["yfinance"]["status"] == "UNSUPPORTED_HISTORICAL"
    assert company["alpha_vantage"]["status"] == "SUCCESS_WITH_DATA"


# --- 5: Alpha Vantage AUTH_MISSING is recorded correctly --------------------


@pytest.mark.unit
def test_5_alpha_vantage_auth_missing_is_recorded():
    status = classify_router_exception(AlphaVantageNotConfiguredError("ALPHA_VANTAGE_API_KEY environment variable is not set."))
    assert status == SourceStatus.AUTH_MISSING

    set_config({"data_vendors": {"news_data": "yfinance,alpha_vantage"}})
    collector = SourceCoverageCollector()
    set_active_source_coverage_collector(collector)
    try:
        with mock.patch.dict(
            router.VENDOR_METHODS,
            {"get_news": {"yfinance": lambda *a, **k: _YFINANCE_UNSUPPORTED, "alpha_vantage": mock.Mock(side_effect=AlphaVantageNotConfiguredError("no key"))}},
            clear=False,
        ):
            router.route_to_vendor("get_news", "LLY", "2026-09-23", "2026-09-30")
    finally:
        set_active_source_coverage_collector(None)
    assert collector.snapshot()["news"]["company"]["alpha_vantage"]["status"] == "AUTH_MISSING"


# --- 6: RATE_LIMITED is recorded (also covers the real "premium endpoint" --
# response observed against the live key during this evaluation) -----------


@pytest.mark.unit
def test_6_alpha_vantage_rate_limited_is_recorded():
    # The real response seen with the configured key: NEWS_SENTIMENT requires
    # a paid plan, and the existing (unchanged) classifier in common.py
    # treats "premium" notices as a rate limit.
    premium_notice = (
        '{"Information": "Thank you for using Alpha Vantage! This is a premium '
        'endpoint. You may subscribe to any of the premium plans."}'
    )

    class _FakeResponse:
        status_code = 200
        text = premium_notice

        def raise_for_status(self):
            pass

    import tradingagents.dataflows.net as net
    with mock.patch.object(net.requests, "get", return_value=_FakeResponse()), pytest.raises(AlphaVantageRateLimitError):
        av_news.get_news("LLY", "2026-09-23", "2026-09-30")

    status = classify_router_exception(AlphaVantageRateLimitError("Alpha Vantage rate limit exceeded: ..."))
    assert status == SourceStatus.RATE_LIMITED

    set_config({"data_vendors": {"news_data": "yfinance,alpha_vantage"}})
    collector = SourceCoverageCollector()
    set_active_source_coverage_collector(collector)
    try:
        with mock.patch.dict(
            router.VENDOR_METHODS,
            {"get_news": {"yfinance": lambda *a, **k: _YFINANCE_UNSUPPORTED, "alpha_vantage": mock.Mock(side_effect=AlphaVantageRateLimitError("rate limited"))}},
            clear=False,
        ):
            router.route_to_vendor("get_news", "LLY", "2026-09-23", "2026-09-30")
    finally:
        set_active_source_coverage_collector(None)
    assert collector.snapshot()["news"]["company"]["alpha_vantage"]["status"] == "RATE_LIMITED"


# --- 7: valid Yahoo data is never discarded in favor of Alpha Vantage -------


@pytest.mark.unit
def test_7_valid_yahoo_data_is_not_discarded():
    set_config({"data_vendors": {"news_data": "yfinance,alpha_vantage"}})
    av_called = mock.Mock(return_value="## AV data\n")
    with mock.patch.dict(
        router.VENDOR_METHODS,
        {"get_news": {"yfinance": lambda *a, **k: "## LLY News, from 2026-09-23 to 2026-09-30:\n\n### Real yahoo headline\n", "alpha_vantage": av_called}},
        clear=False,
    ):
        result = router.route_to_vendor("get_news", "LLY", "2026-09-23", "2026-09-30")
    assert "Real yahoo headline" in result
    av_called.assert_not_called()  # SUCCESS_WITH_DATA never advances the chain


@pytest.mark.unit
def test_7b_yfinance_success_empty_also_does_not_advance_the_chain():
    """SUCCESS_EMPTY is a real answer (the window was covered; the vendor
    genuinely found nothing) -- only UNSUPPORTED_HISTORICAL triggers a
    fallback attempt, not every non-data outcome."""
    set_config({"data_vendors": {"news_data": "yfinance,alpha_vantage"}})
    av_called = mock.Mock(return_value="## AV data\n")
    with mock.patch.dict(
        router.VENDOR_METHODS,
        {"get_news": {"yfinance": lambda *a, **k: "No news found for LLY between 2026-09-23 and 2026-09-30", "alpha_vantage": av_called}},
        clear=False,
    ):
        router.route_to_vendor("get_news", "LLY", "2026-09-23", "2026-09-30")
    av_called.assert_not_called()


# --- 8: Sentiment is unaffected ----------------------------------------------


@pytest.mark.unit
def test_8_sentiment_short_circuit_unaffected_by_multi_vendor_news_chain(monkeypatch):
    """Sentiment's own decision only ever looks at the FINAL text get_news.func
    returns, regardless of which vendor in the chain produced it -- a
    two-vendor chain where Alpha Vantage ultimately serves real data must
    still be read by Sentiment's classifier as SUCCESS_WITH_DATA, not as the
    now-irrelevant fact that yfinance was unsupported first."""
    from tradingagents.dataflows.source_coverage import classify_router_text

    set_config({"data_vendors": {"news_data": "yfinance,alpha_vantage"}})
    with mock.patch.dict(
        router.VENDOR_METHODS,
        {"get_news": {"yfinance": lambda *a, **k: _YFINANCE_UNSUPPORTED, "alpha_vantage": lambda *a, **k: "## LLY News, from 2026-09-23 to 2026-09-30:\n\n### Real (source: X)\n"}},
        clear=False,
    ):
        final_text = router.route_to_vendor("get_news", "LLY", "2026-09-23", "2026-09-30")
    status, _ = classify_router_text("get_news", final_text)
    assert status == SourceStatus.SUCCESS_WITH_DATA


# --- 9: FRED is unaffected ----------------------------------------------------


@pytest.mark.unit
def test_9_fred_macro_routing_unaffected_by_the_news_chain_advance_logic():
    """get_macro_indicators's own SUCCESS_WITH_DATA/EMPTY outcomes still
    return immediately, exactly as before -- the new "continue past
    UNSUPPORTED_HISTORICAL" behavior only ever fires for that exact status,
    never for FRED's normal success/empty results."""
    set_config({"data_vendors": {"macro_data": "fred"}})
    called = mock.Mock(return_value="## FRED: Unemployment Rate (UNRATE)\n**Latest:** 4.1\n")
    with mock.patch.dict(router.VENDOR_METHODS, {"get_macro_indicators": {"fred": called}}, clear=False):
        result = router.route_to_vendor("get_macro_indicators", "unemployment", "2026-09-30", 365)
    called.assert_called_once()
    assert result == "## FRED: Unemployment Rate (UNRATE)\n**Latest:** 4.1\n"


# --- 10: no future leakage ----------------------------------------------------


@pytest.mark.unit
def test_10_no_future_leakage_across_the_window_boundary(monkeypatch):
    """An item published exactly at the half-open window's upper bound
    (midnight after end_date) must be excluded, matching in_window's own
    semantics everywhere else in the codebase -- never a looser rule just
    because this is a different vendor."""
    just_after_midnight = "20261001T000001"  # 1 second past end_date+1 day start
    monkeypatch.setattr(
        av_news, "_make_api_request",
        lambda fn, params: _av_feed(title="Just past the boundary", time_published=just_after_midnight),
    )
    result = av_news.get_news("LLY", "2026-09-23", "2026-09-30")
    assert "Just past the boundary" not in result

    on_end_date_itself = "20260930T235900"  # the analysis day itself -- must be INCLUDED
    monkeypatch.setattr(
        av_news, "_make_api_request",
        lambda fn, params: _av_feed(title="On the analysis day", time_published=on_end_date_itself),
    )
    result = av_news.get_news("LLY", "2026-09-23", "2026-09-30")
    assert "On the analysis day" in result
