"""Per-provider capture functions: one network call, reusing the EXISTING
raw-fetch primitive for that provider, normalized into the common capture
contract plus the untouched raw payload.

Each function has the same shape ``(ticker, config) -> _RawCapture`` so the
orchestrator can dispatch to any of them uniformly. None of them render
Markdown/text, screen content (``jev_screen``), or call an LLM -- they stop
exactly at "what did the provider return."
"""

from __future__ import annotations

import http.client
import json
from dataclasses import dataclass
from datetime import datetime, timezone

from tradingagents.dataflows.symbols import crypto_base, normalize_symbol
from tradingagents.dataflows.vendors.alpha_vantage.common import (
    AlphaVantageNotConfiguredError,
    _make_api_request,
)
from tradingagents.dataflows.vendors.alpha_vantage.news import _feed, _parse_time_published
from tradingagents.dataflows.vendors.reddit import (
    _FEED_PAGE,
    DEFAULT_SUBREDDITS,
    _fetch_subreddit_rss,
)
from tradingagents.dataflows.vendors.stocktwits import _fetch_raw_messages, _stocktwits_symbol
from tradingagents.dataflows.vendors.yahoo.news import _extract_article_data, _fetch_raw_news_items
from tradingagents.evidence_archive.schema import (
    EMPTY,
    ERROR,
    SUCCESS,
    UNSUPPORTED,
    CaptureError,
    CaptureItem,
    dedupe_items,
    normalized_item,
    unverifiable_empty_metadata,
)

YFINANCE = "yfinance"
ALPHA_VANTAGE = "alpha_vantage"
STOCKTWITS = "stocktwits"
REDDIT = "reddit"

# yfinance's Ticker.get_news() can itself return [] when Yahoo's internal
# news endpoint returned a non-2xx response -- it only raises for a 429
# rate limit, never for any other HTTP error (confirmed via source reading
# and live reproduction, 2026-10-16: a 404 with a valid-but-empty JSON body
# parses without exception and resolves to an empty item list). This
# project never reimplements yfinance's private HTTP client to recover that
# status, so an EMPTY yfinance-news capture cannot be verified as a genuine
# absence of news -- it is flagged, never silently presented as equivalent
# to another provider's verified EMPTY.
_YFINANCE_NEWS_EMPTY_LIMITATION = (
    "yfinance's Ticker.get_news() can return an empty list either because "
    "there genuinely is no news, or because the upstream request to "
    "Yahoo's internal news endpoint failed in a way yfinance does not "
    "raise for (it only raises for an HTTP 429 rate limit). This project "
    "does not reimplement yfinance's internal HTTP client to recover the "
    "real upstream outcome, so this capture's EMPTY status could not be "
    "verified against it."
)


@dataclass
class _RawCapture:
    status: str
    items: list[CaptureItem]
    error: CaptureError | None
    provider_metadata: dict
    requested_query: dict


def _safe_message(exc: Exception) -> str:
    """``str(exc)`` -- Alpha Vantage's own errors are already scrubbed of the
    API key by ``get_scrubbed`` before they reach here; other providers need
    no credential, so nothing further to redact in the free text itself."""
    return str(exc)


def _epoch_to_iso(epoch: float | None) -> str | None:
    if epoch is None:
        return None
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _yfinance_item_id(article: dict) -> str | None:
    content = article.get("content")
    if isinstance(content, dict) and content.get("id"):
        return str(content["id"])
    if article.get("id"):
        return str(article["id"])
    return None


def capture_news_yfinance(ticker: str, config: dict) -> _RawCapture:
    canonical = normalize_symbol(ticker)
    limit = config["news_article_limit"]
    requested_query = {"ticker": ticker, "canonical_symbol": canonical, "limit": limit}
    try:
        raw_items = _fetch_raw_news_items(canonical, limit)
    except Exception as exc:  # the raw fetch is not wrapped in VendorError translation here
        return _RawCapture(ERROR, [], CaptureError(type(exc).__name__, _safe_message(exc)), {}, requested_query)

    if not raw_items:
        return _RawCapture(
            EMPTY, [], None,
            unverifiable_empty_metadata(_YFINANCE_NEWS_EMPTY_LIMITATION),
            requested_query,
        )

    items = []
    for article in raw_items:
        data = _extract_article_data(article)
        items.append(CaptureItem(
            normalized=normalized_item(
                provider_item_id=_yfinance_item_id(article),
                published_at=data["pub_date"].isoformat() if data["pub_date"] else None,
                title=data["title"],
                text=data["summary"] or None,
                author_or_source=data["publisher"],
                url=data["link"] or None,
            ),
            provider_payload=article,
        ))
    return _RawCapture(SUCCESS, dedupe_items(items), None, {}, requested_query)


def capture_news_alpha_vantage(ticker: str, config: dict) -> _RawCapture:
    limit = config["news_article_limit"]
    requested_query = {"ticker": ticker, "limit": limit}
    try:
        response_text = _make_api_request("NEWS_SENTIMENT", {"tickers": ticker, "limit": str(limit)})
    except AlphaVantageNotConfiguredError as exc:
        return _RawCapture(ERROR, [], CaptureError("NOT_CONFIGURED", _safe_message(exc)), {}, requested_query)
    except Exception as exc:
        return _RawCapture(ERROR, [], CaptureError(type(exc).__name__, _safe_message(exc)), {}, requested_query)

    feed = _feed(response_text)
    if not feed:
        return _RawCapture(EMPTY, [], None, {}, requested_query)

    items = []
    for article in feed:
        published = _parse_time_published(article.get("time_published"))
        items.append(CaptureItem(
            normalized=normalized_item(
                provider_item_id=article.get("url") or None,
                published_at=published.isoformat() if published else None,
                title=article.get("title"),
                text=article.get("summary") or None,
                author_or_source=article.get("source"),
                url=article.get("url") or None,
            ),
            provider_payload=article,
        ))
    return _RawCapture(SUCCESS, dedupe_items(items), None, {}, requested_query)


def capture_social_stocktwits(ticker: str, config: dict) -> _RawCapture:
    limit = 30
    requested_query = {"ticker": ticker, "limit": limit}
    symbol = _stocktwits_symbol(ticker)
    if any(char in symbol for char in ("^", "=")):
        # StockTwits has no message stream for index/future-style symbols --
        # a structural gap, not an operational failure, so this is never
        # attempted over the network.
        return _RawCapture(
            UNSUPPORTED, [], None,
            {"reason": f"StockTwits has no symbol stream for {symbol!r} (index/derivative-shaped)"},
            requested_query,
        )
    try:
        raw_messages = _fetch_raw_messages(ticker, timeout=10.0)
    except (OSError, http.client.HTTPException, json.JSONDecodeError) as exc:
        return _RawCapture(ERROR, [], CaptureError(type(exc).__name__, _safe_message(exc)), {}, requested_query)

    if not raw_messages:
        return _RawCapture(EMPTY, [], None, {}, requested_query)

    items = []
    for message in raw_messages[:limit]:
        entities = message.get("entities") or {}
        sentiment_obj = entities.get("sentiment") or {}
        sentiment = sentiment_obj.get("basic") if isinstance(sentiment_obj, dict) else None
        items.append(CaptureItem(
            normalized=normalized_item(
                provider_item_id=str(message["id"]) if message.get("id") is not None else None,
                published_at=message.get("created_at") or None,
                text=message.get("body") or None,
                author_or_source=(message.get("user") or {}).get("username"),
                engagement={"sentiment": sentiment} if sentiment else None,
            ),
            provider_payload=message,
        ))
    return _RawCapture(SUCCESS, dedupe_items(items), None, {}, requested_query)


def capture_social_reddit(ticker: str, config: dict) -> _RawCapture:
    subreddits = DEFAULT_SUBREDDITS
    requested_query = {
        "ticker": ticker, "subreddits": list(subreddits), "limit": _FEED_PAGE,
    }
    search_ticker = crypto_base(ticker) or ticker
    fetched = _fetch_subreddit_rss(search_ticker, "+".join(subreddits), _FEED_PAGE, 10.0)
    if fetched is None:
        return _RawCapture(
            ERROR, [], CaptureError("FetchFailed", "Reddit RSS fetch failed (network/HTTP error)"),
            {}, requested_query,
        )

    if not fetched:
        return _RawCapture(EMPTY, [], None, {}, requested_query)

    items = [
        CaptureItem(
            normalized=normalized_item(
                # The RSS feed carries no post id, URL, author, or engagement
                # counts at all (see tradingagents/dataflows/vendors/reddit.py's
                # own module docstring) -- never fabricated here.
                provider_item_id=None,
                published_at=_epoch_to_iso(post.get("created_utc")),
                title=post.get("title") or None,
                text=post.get("selftext") or None,
                author_or_source=post.get("subreddit"),
            ),
            provider_payload=post,
        )
        for post in fetched
    ]
    # No dedupe: every item's provider_item_id is None (no unambiguous
    # identity to compare), so dedupe_items would be a no-op here by design.
    return _RawCapture(SUCCESS, items, None, {}, requested_query)


NEWS_PROVIDERS = {
    YFINANCE: capture_news_yfinance,
    ALPHA_VANTAGE: capture_news_alpha_vantage,
}

SOCIAL_PROVIDERS = {
    STOCKTWITS: capture_social_stocktwits,
    REDDIT: capture_social_reddit,
}
