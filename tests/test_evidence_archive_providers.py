"""Per-provider capture functions: SUCCESS/EMPTY/ERROR/UNSUPPORTED
classification, exactly one network call per capture, full raw-payload
fidelity, and secret scrubbing. No real network: every raw-fetch primitive
is monkeypatched at its call site inside tradingagents.evidence_archive.providers.
"""

from __future__ import annotations

import json

import pytest

from tradingagents.dataflows.vendors.alpha_vantage.common import AlphaVantageNotConfiguredError
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.evidence_archive import providers
from tradingagents.evidence_archive.schema import (
    EMPTY,
    EMPTY_UNVERIFIED,
    ERROR,
    SUCCESS,
    UNSUPPORTED,
    build_capture,
    verify_capture_integrity,
)


@pytest.mark.unit
class TestYfinanceNewsCapture:
    def test_success_normalizes_and_preserves_raw_payload(self, monkeypatch):
        raw_article = {
            "id": "story-1",
            "content": {
                "title": "NVDA surges", "summary": "Big quarter.",
                "provider": {"displayName": "Reuters"},
                "canonicalUrl": {"url": "https://example.com/a"},
                "pubDate": "2026-10-15T12:00:00Z",
            },
        }
        calls = []
        monkeypatch.setattr(
            providers, "_fetch_raw_news_items",
            lambda canonical, count: (calls.append((canonical, count)), [raw_article])[1],
        )

        result = providers.capture_news_yfinance("NVDA", DEFAULT_CONFIG)

        assert result.status == SUCCESS
        assert len(calls) == 1  # exactly one network call
        assert len(result.items) == 1
        item = result.items[0]
        assert item.normalized["provider_item_id"] == "story-1"
        assert item.normalized["title"] == "NVDA surges"
        assert item.normalized["author_or_source"] == "Reuters"
        assert item.normalized["url"] == "https://example.com/a"
        assert item.provider_payload == raw_article  # full fidelity

    def test_empty_when_provider_returns_no_items(self, monkeypatch):
        monkeypatch.setattr(providers, "_fetch_raw_news_items", lambda c, n: [])
        result = providers.capture_news_yfinance("NVDA", DEFAULT_CONFIG)
        assert result.status == EMPTY
        assert result.items == []

    def test_error_when_fetch_raises(self, monkeypatch):
        def boom(canonical, count):
            raise RuntimeError("yfinance down")
        monkeypatch.setattr(providers, "_fetch_raw_news_items", boom)
        result = providers.capture_news_yfinance("NVDA", DEFAULT_CONFIG)
        assert result.status == ERROR
        assert result.error.error_type == "RuntimeError"

    def test_dedupes_within_one_capture_by_exact_id_only(self, monkeypatch):
        a = {"id": "1", "content": {"title": "A", "summary": "", "provider": {}}}
        b = {"id": "1", "content": {"title": "A duplicate", "summary": "", "provider": {}}}
        monkeypatch.setattr(providers, "_fetch_raw_news_items", lambda c, n: [a, b])
        result = providers.capture_news_yfinance("NVDA", DEFAULT_CONFIG)
        assert len(result.items) == 1

    def test_real_world_empty_shape_yfinance_swallowing_an_upstream_http_error(self, monkeypatch):
        """Pins the EXACT real-world shape observed live on 2026-10-16:
        Yahoo's internal news endpoint (xhr/ncp) returned a plain HTTP 404
        with a valid, parseable JSON body (``{"message": "Not Found"}``).
        yfinance's own ``Ticker.get_news()`` does not treat that as an
        error (it only special-cases a "Will be right back" maintenance
        banner) -- it silently resolves to an empty ``data``/``tickerStream``
        chain and returns ``[]``, no exception raised. Confirmed via live
        reproduction across NVDA/AAPL/SPY and via the legacy ``.news``
        property (same underlying call) -- not ticker-specific, not
        something this project's refactor introduced (see git history:
        the pre-refactor code made the exact same call).

        At our layer, ``_fetch_raw_news_items`` only ever sees the final,
        validly-typed ``list`` ``get_news()`` returns -- we have no
        visibility into the swallowed HTTP response without reimplementing
        yfinance's private HTTP client, which this project deliberately
        never does (reuse existing dataflows, never reimplement a vendor).
        So an empty list here is, correctly, EMPTY -- not ERROR -- at this
        abstraction boundary.
        """
        monkeypatch.setattr(providers, "_fetch_raw_news_items", lambda c, n: [])
        result = providers.capture_news_yfinance("NVDA", DEFAULT_CONFIG)
        assert result.status == EMPTY
        assert result.items == []
        assert result.error is None


@pytest.mark.unit
class TestAlphaVantageNewsCapture:
    def test_success(self, monkeypatch):
        feed = {"feed": [{
            "title": "Fed holds rates", "url": "https://av.example/1",
            "time_published": "20261015T120000", "summary": "A summary.",
            "source": "Bloomberg", "overall_sentiment_score": 0.2,
        }]}
        monkeypatch.setattr(providers, "_make_api_request", lambda fn, params: json.dumps(feed))
        result = providers.capture_news_alpha_vantage("NVDA", DEFAULT_CONFIG)
        assert result.status == SUCCESS
        item = result.items[0]
        assert item.normalized["provider_item_id"] == "https://av.example/1"
        assert item.normalized["author_or_source"] == "Bloomberg"
        assert item.provider_payload["overall_sentiment_score"] == 0.2

    def test_missing_api_key_maps_to_error_with_not_configured_subtype(self, monkeypatch):
        def boom(fn, params):
            raise AlphaVantageNotConfiguredError("no key")
        monkeypatch.setattr(providers, "_make_api_request", boom)
        result = providers.capture_news_alpha_vantage("NVDA", DEFAULT_CONFIG)
        assert result.status == ERROR
        assert result.error.error_type == "NOT_CONFIGURED"

    def test_empty_feed(self, monkeypatch):
        monkeypatch.setattr(providers, "_make_api_request", lambda fn, params: json.dumps({"feed": []}))
        result = providers.capture_news_alpha_vantage("NVDA", DEFAULT_CONFIG)
        assert result.status == EMPTY

    def test_dedupes_by_url_not_by_title(self, monkeypatch):
        feed = {"feed": [
            {"title": "Same title", "url": "https://x/1", "time_published": "20261015T120000"},
            {"title": "Same title", "url": "https://x/2", "time_published": "20261015T120100"},
        ]}
        monkeypatch.setattr(providers, "_make_api_request", lambda fn, params: json.dumps(feed))
        result = providers.capture_news_alpha_vantage("NVDA", DEFAULT_CONFIG)
        # Distinct URLs are an unambiguous distinction even with identical
        # titles -- never collapsed on a fuzzy title match.
        assert len(result.items) == 2


@pytest.mark.unit
class TestStockTwitsCapture:
    def test_success(self, monkeypatch):
        messages = [{
            "id": 555, "created_at": "2026-10-15T12:00:00Z",
            "body": "NVDA to the moon", "user": {"username": "trader1"},
            "entities": {"sentiment": {"basic": "Bullish"}},
        }]
        monkeypatch.setattr(providers, "_fetch_raw_messages", lambda ticker, timeout=10.0: messages)
        result = providers.capture_social_stocktwits("NVDA", DEFAULT_CONFIG)
        assert result.status == SUCCESS
        item = result.items[0]
        assert item.normalized["provider_item_id"] == "555"
        assert item.normalized["engagement"] == {"sentiment": "Bullish"}
        assert item.normalized["author_or_source"] == "trader1"

    def test_unsupported_for_index_symbols_without_any_network_call(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            providers, "_fetch_raw_messages",
            lambda ticker, timeout=10.0: calls.append(ticker),
        )
        result = providers.capture_social_stocktwits("^GSPC", DEFAULT_CONFIG)
        assert result.status == UNSUPPORTED
        assert calls == []  # never attempted over the network

    def test_error_on_transport_failure(self, monkeypatch):
        def boom(ticker, timeout=10.0):
            raise OSError("network unreachable")
        monkeypatch.setattr(providers, "_fetch_raw_messages", boom)
        result = providers.capture_social_stocktwits("NVDA", DEFAULT_CONFIG)
        assert result.status == ERROR

    def test_empty_when_no_messages(self, monkeypatch):
        monkeypatch.setattr(providers, "_fetch_raw_messages", lambda ticker, timeout=10.0: [])
        result = providers.capture_social_stocktwits("NVDA", DEFAULT_CONFIG)
        assert result.status == EMPTY

    def test_message_without_id_is_never_deduped_against_another(self, monkeypatch):
        messages = [
            {"id": None, "created_at": "t1", "body": "a", "user": {}},
            {"id": None, "created_at": "t2", "body": "a", "user": {}},
        ]
        monkeypatch.setattr(providers, "_fetch_raw_messages", lambda ticker, timeout=10.0: messages)
        result = providers.capture_social_stocktwits("NVDA", DEFAULT_CONFIG)
        assert len(result.items) == 2


@pytest.mark.unit
class TestRedditCapture:
    def test_success_items_carry_no_id_and_are_never_deduped(self, monkeypatch):
        posts = [
            {"title": "NVDA earnings", "created_utc": 1760000000.0, "selftext": "body", "subreddit": "stocks"},
            {"title": "NVDA earnings", "created_utc": 1760000001.0, "selftext": "body2", "subreddit": "stocks"},
        ]
        monkeypatch.setattr(providers, "_fetch_subreddit_rss", lambda *a, **k: posts)
        result = providers.capture_social_reddit("NVDA", DEFAULT_CONFIG)
        assert result.status == SUCCESS
        assert len(result.items) == 2  # both kept: no unambiguous id, even with identical titles
        assert all(item.normalized["provider_item_id"] is None for item in result.items)
        assert all(item.normalized["url"] is None for item in result.items)

    def test_error_when_fetch_returns_none(self, monkeypatch):
        monkeypatch.setattr(providers, "_fetch_subreddit_rss", lambda *a, **k: None)
        result = providers.capture_social_reddit("NVDA", DEFAULT_CONFIG)
        assert result.status == ERROR

    def test_empty_when_fetch_returns_empty_list(self, monkeypatch):
        monkeypatch.setattr(providers, "_fetch_subreddit_rss", lambda *a, **k: [])
        result = providers.capture_social_reddit("NVDA", DEFAULT_CONFIG)
        assert result.status == EMPTY


@pytest.mark.unit
def test_yfinance_malformed_item_shape_is_never_silently_absorbed_into_empty(monkeypatch):
    """A raw item that does not match the expected dict shape (e.g. a
    future yfinance release changing what ``get_news()`` puts in its list)
    must surface as a failure, never get silently swallowed into an EMPTY
    capture that looks identical to a genuine absence of news. This
    function itself has no per-item try/except -- it raises, and
    ``orchestrator._run_one``'s own catch-all is what turns that into an
    explicit ERROR capture (see test_evidence_archive_orchestrator.py's
    equivalent end-to-end test)."""
    monkeypatch.setattr(providers, "_fetch_raw_news_items", lambda c, n: ["not-a-dict-item"])
    with pytest.raises(AttributeError):
        providers.capture_news_yfinance("NVDA", DEFAULT_CONFIG)


@pytest.mark.unit
class TestYfinanceNewsUnverifiedEmptySemantics:
    """yfinance's Ticker.get_news() can swallow a non-2xx upstream response
    into an empty list without raising (see providers.py's own
    ``_YFINANCE_NEWS_EMPTY_LIMITATION`` docstring) -- this EMPTY cannot be
    verified as a genuine absence of news without reimplementing yfinance's
    private HTTP client, which this project never does. The capture must
    say so explicitly, scoped to exactly this one case."""

    def test_empty_yfinance_news_is_flagged_unverified(self, monkeypatch):
        monkeypatch.setattr(providers, "_fetch_raw_news_items", lambda c, n: [])
        result = providers.capture_news_yfinance("NVDA", DEFAULT_CONFIG)
        assert result.status == EMPTY
        assert result.provider_metadata["empty_semantics"] == EMPTY_UNVERIFIED
        assert "observability_limitation" in result.provider_metadata

    def test_the_limitation_text_never_asserts_a_specific_status_code(self, monkeypatch):
        # Real investigation found a 404 on one specific day -- the capture
        # must describe the CLASS of failure, never assert that THIS
        # capture definitely saw a 404 (we have no way to know).
        monkeypatch.setattr(providers, "_fetch_raw_news_items", lambda c, n: [])
        result = providers.capture_news_yfinance("NVDA", DEFAULT_CONFIG)
        text = result.provider_metadata["observability_limitation"]
        assert "404" not in text
        assert "status_code" not in text.lower()

    def test_success_is_never_flagged(self, monkeypatch):
        article = {"id": "1", "content": {"title": "A", "summary": "", "provider": {}}}
        monkeypatch.setattr(providers, "_fetch_raw_news_items", lambda c, n: [article])
        result = providers.capture_news_yfinance("NVDA", DEFAULT_CONFIG)
        assert result.status == SUCCESS
        assert result.provider_metadata == {}

    def test_real_error_is_never_flagged(self, monkeypatch):
        def boom(c, n):
            raise RuntimeError("yfinance down")
        monkeypatch.setattr(providers, "_fetch_raw_news_items", boom)
        result = providers.capture_news_yfinance("NVDA", DEFAULT_CONFIG)
        assert result.status == ERROR
        assert result.provider_metadata == {}

    def test_other_providers_empty_is_never_flagged_as_unverified(self, monkeypatch):
        """Alpha Vantage, StockTwits, and Reddit each raise/return a
        distinct failure signal on a real transport or HTTP error (verified
        separately in their own TestXCapture classes above) -- their own
        EMPTY is a genuinely verified absence, so none of them get this
        marker."""
        monkeypatch.setattr(providers, "_make_api_request", lambda fn, params: json.dumps({"feed": []}))
        av_result = providers.capture_news_alpha_vantage("NVDA", DEFAULT_CONFIG)
        assert av_result.status == EMPTY
        assert av_result.provider_metadata == {}

        monkeypatch.setattr(providers, "_fetch_raw_messages", lambda ticker, timeout=10.0: [])
        st_result = providers.capture_social_stocktwits("NVDA", DEFAULT_CONFIG)
        assert st_result.status == EMPTY
        assert st_result.provider_metadata == {}

        monkeypatch.setattr(providers, "_fetch_subreddit_rss", lambda *a, **k: [])
        reddit_result = providers.capture_social_reddit("NVDA", DEFAULT_CONFIG)
        assert reddit_result.status == EMPTY
        assert reddit_result.provider_metadata == {}

    def test_verify_capture_integrity_still_works_with_the_new_metadata(self, monkeypatch):
        monkeypatch.setattr(providers, "_fetch_raw_news_items", lambda c, n: [])
        result = providers.capture_news_yfinance("NVDA", DEFAULT_CONFIG)
        capture = build_capture(
            ticker="NVDA", evidence_type="NEWS", provider="yfinance",
            requested_at="t", retrieved_at="t", requested_query=result.requested_query,
            status=result.status, items=result.items, error=result.error,
            provider_metadata=result.provider_metadata,
        )
        assert verify_capture_integrity(capture)
        assert capture["provider_metadata"]["empty_semantics"] == EMPTY_UNVERIFIED
        assert "retrieved_at" in capture["point_in_time_notice"]  # still present, unchanged


@pytest.mark.unit
def test_no_secrets_end_up_serialized_in_any_capture(monkeypatch):
    raw_article = {
        "id": "1",
        "content": {
            "title": "t", "summary": "s", "provider": {},
            "api_key": "SHOULD-NOT-LEAK", "auth_token": "ALSO-SHOULD-NOT-LEAK",
        },
    }
    monkeypatch.setattr(providers, "_fetch_raw_news_items", lambda c, n: [raw_article])
    result = providers.capture_news_yfinance("NVDA", DEFAULT_CONFIG)
    capture = build_capture(
        ticker="NVDA", evidence_type="NEWS", provider="yfinance",
        requested_at="t", retrieved_at="t", requested_query={}, status=result.status,
        items=result.items,
    )
    serialized = json.dumps(capture)
    assert "SHOULD-NOT-LEAK" not in serialized
    assert "ALSO-SHOULD-NOT-LEAK" not in serialized
