"""The yahoo/news.py and stocktwits.py raw-fetch extractions (done for the
evidence archive) must not change either function's existing public
behavior, and each capture/render must cost exactly one network call --
never two for the same data (see their modules' own ``_fetch_raw_*`` docstrings).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from tradingagents.dataflows.vendors import stocktwits
from tradingagents.dataflows.vendors.yahoo import news as ynews


@pytest.mark.unit
class TestYahooNewsRefactorParity:
    def test_get_news_yfinance_still_renders_the_same_markdown_shape(self, monkeypatch):
        article = {
            "content": {
                "title": "NVDA beats", "summary": "Great quarter.",
                "provider": {"displayName": "Reuters"},
                "canonicalUrl": {"url": "https://example.com/a"},
                "pubDate": "2026-05-05T00:00:00Z",
            }
        }
        monkeypatch.setattr(ynews, "_fetch_raw_news_items", lambda canonical, count: [article])
        out = ynews.get_news_yfinance("NVDA", "2026-05-01", "2026-05-09")
        assert "### NVDA beats (source: Reuters)" in out
        assert "Great quarter." in out
        assert "Link: https://example.com/a" in out

    def test_get_news_yfinance_makes_exactly_one_raw_fetch_call(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            ynews, "_fetch_raw_news_items",
            lambda canonical, count: (calls.append((canonical, count)), [])[1],
        )
        ynews.get_news_yfinance("NVDA", "2026-05-01", "2026-05-09")
        assert len(calls) == 1

    def test_real_world_empty_result_still_degrades_to_the_pre_refactor_text_not_a_crash(self, monkeypatch):
        """Pins the exact real-world shape confirmed live on 2026-10-16:
        Yahoo's news endpoint 404s, yfinance's own get_news() swallows that
        into an empty list (no exception) -- confirmed identical for
        NVDA/AAPL/SPY and via the legacy .news property. The refactor is a
        byte-identical extraction (see git history), so the public wrapper
        must behave exactly as it did before: no crash, the project's
        standard "no news found" text, never the item-rendering loop
        (unreachable for yfinance's own empty-rather-than-raise behavior
        anyway, since there are 0 items to loop over)."""
        monkeypatch.setattr(ynews, "_fetch_raw_news_items", lambda canonical, count: [])
        out = ynews.get_news_yfinance("NVDA", "2026-01-01", "2026-01-07")
        assert "No news found for NVDA" in out or "unavailable" in out
        assert "###" not in out  # no article header ever rendered for 0 items

    def test_fetch_failure_still_raises_no_market_data_error(self, monkeypatch):
        from tradingagents.dataflows.errors import NoMarketDataError

        def boom(canonical, count):
            raise RuntimeError("yfinance down")

        monkeypatch.setattr(ynews, "_fetch_raw_news_items", boom)
        with pytest.raises(NoMarketDataError):
            ynews.get_news_yfinance("NVDA", "2026-05-01", "2026-05-09")

    def test_raw_fetch_helper_itself_calls_yf_ticker_get_news(self, monkeypatch):
        calls = {}

        class _FakeTicker:
            def __init__(self, symbol):
                calls["symbol"] = symbol

            def get_news(self, count):
                calls["count"] = count
                return ["raw"]

        monkeypatch.setattr(ynews.yf, "Ticker", _FakeTicker)
        monkeypatch.setattr(ynews, "yf_retry", lambda fn: fn())
        result = ynews._fetch_raw_news_items("NVDA", 20)
        assert result == ["raw"]
        assert calls == {"symbol": "NVDA", "count": 20}


def _stream_response(messages):
    import json as _json

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

        def read(self):
            return _json.dumps({"messages": messages}).encode()

    return _Resp()


@pytest.mark.unit
class TestStockTwitsRefactorParity:
    def test_fetch_stocktwits_messages_still_renders_the_same_summary_shape(self):
        messages = [{
            "created_at": "2026-05-05T00:00:00Z", "body": "NVDA to the moon",
            "user": {"username": "trader1"}, "entities": {"sentiment": {"basic": "Bullish"}},
        }]
        with patch.object(stocktwits, "urlopen", return_value=_stream_response(messages)):
            out = stocktwits.fetch_stocktwits_messages("NVDA")
        assert "Bullish: 1 (100%)" in out
        assert "NVDA to the moon" in out

    def test_fetch_stocktwits_messages_makes_exactly_one_raw_fetch_call(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            stocktwits, "_fetch_raw_messages",
            lambda ticker, timeout: (calls.append((ticker, timeout)), [])[1],
        )
        stocktwits.fetch_stocktwits_messages("NVDA")
        assert len(calls) == 1

    def test_transport_failure_still_degrades_to_placeholder(self):
        with patch.object(stocktwits, "urlopen", side_effect=OSError("down")):
            out = stocktwits.fetch_stocktwits_messages("NVDA")
        assert out.startswith("<stocktwits unavailable")
