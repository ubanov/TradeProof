import json
from datetime import datetime, timezone

from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.date_window import coverage_gap, in_window
from tradingagents.dataflows.vendors.alpha_vantage.common import (
    _make_api_request,
    format_datetime_for_api,
)


def _parse_time_published(raw) -> datetime | None:
    """Alpha Vantage's ``time_published`` is ``YYYYMMDDTHHMMSS``, UTC."""
    if not raw or not isinstance(raw, str):
        return None
    try:
        return datetime.strptime(raw, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _feed(response_text) -> list[dict]:
    """The response's ``feed`` list, or ``[]`` for anything else (a non-JSON
    body, an error shape, a JSON object with no ``feed`` key) -- never
    raises, mirroring every other vendor's "degrade, never crash" contract."""
    try:
        payload = json.loads(response_text)
    except (TypeError, ValueError):
        return []
    feed = payload.get("feed") if isinstance(payload, dict) else None
    return feed if isinstance(feed, list) else []


def _relevance(item: dict, ticker: str) -> float | None:
    """This item's ``ticker_sentiment`` relevance score for ``ticker``, when
    Alpha Vantage supplied one -- a deterministic, vendor-provided
    company-relevance signal, not a heuristic this project invents."""
    for entry in item.get("ticker_sentiment") or []:
        if str(entry.get("ticker", "")).upper() == ticker.upper():
            try:
                return float(entry.get("relevance_score"))
            except (TypeError, ValueError):
                return None
    return None


def _render_item(item: dict) -> str:
    title = item.get("title") or "No title"
    source = item.get("source") or "Unknown"
    summary = item.get("summary") or ""
    url = item.get("url") or ""
    lines = [f"### {title} (source: {source})"]
    if summary:
        lines.append(summary)
    if url:
        lines.append(f"Link: {url}")
    lines.append("")
    return "\n".join(lines)


def _in_window_items(feed: list[dict], start_date: str, end_date: str) -> list[dict]:
    """Point-in-time post-check: Alpha Vantage receives ``time_from``/
    ``time_to`` on the request, but that is a request to the vendor, not a
    local guarantee -- this project never trusts a vendor's own date
    filtering blindly (see fork README: "News/Sentiment acquisition").
    Every item's own ``time_published`` is independently re-verified against
    the requested window with the SAME ``in_window`` check every other
    dated source already uses, so a vendor that ignored or mishandled the
    request can never leak a future-published item into a historical run.
    """
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    kept: list[dict] = []
    seen_titles: set[str] = set()
    for item in feed:
        if not in_window(_parse_time_published(item.get("time_published")), start_dt, end_dt):
            continue
        title = item.get("title")
        if title:
            if title in seen_titles:
                continue
            seen_titles.add(title)
        kept.append(item)
    return kept


def get_news(ticker, start_date, end_date) -> str:
    """Ticker news from Alpha Vantage's NEWS_SENTIMENT feed, trimmed to the
    requested window.

    Args:
        ticker: Stock symbol for news articles.
        start_date: Start date for news search.
        end_date: End date for news search.

    Returns:
        A markdown block in the same shape ``get_news_yfinance`` already
        produces (so point-in-time filtering/classification stays uniform
        across vendors), or the project's standard "unavailable, not an
        absence" sentinel when the vendor's own feed never reached this
        window (see ``coverage_gap``).
    """
    # Without a limit the endpoint returns 50 articles, each with per-ticker
    # sentiment arrays, and all of it reaches the prompt.
    params = {
        "tickers": ticker,
        "time_from": format_datetime_for_api(start_date),
        "time_to": format_datetime_for_api(end_date, end_of_day=True),
        "limit": str(get_config()["news_article_limit"]),
    }
    feed = _feed(_make_api_request("NEWS_SENTIMENT", params))
    kept = _in_window_items(feed, start_date, end_date)

    if not kept:
        gap = coverage_gap(
            (_parse_time_published(a.get("time_published")) for a in feed),
            start_date, end_date, "Alpha Vantage news", f"news for {ticker}",
        )
        return gap or f"No news found for {ticker} between {start_date} and {end_date}"

    return f"## {ticker} News, from {start_date} to {end_date}:\n\n" + "\n".join(_render_item(a) for a in kept)


def get_global_news(curr_date, look_back_days: int | None = None, limit: int | None = None) -> str:
    """Global/macro news from Alpha Vantage's NEWS_SENTIMENT feed (topic-
    filtered, not ticker-filtered), trimmed to the requested window.

    Args:
        curr_date: Current date in yyyy-mm-dd format.
        look_back_days: Number of days to look back; ``None`` uses
            ``global_news_lookback_days`` from the active config.
        limit: Maximum number of articles; ``None`` uses
            ``global_news_article_limit`` from the active config.

    Returns:
        A markdown block in the same shape ``get_global_news_yfinance``
        already produces, or the standard "unavailable" sentinel.
    """
    from datetime import timedelta

    config = get_config()
    if look_back_days is None:
        look_back_days = config["global_news_lookback_days"]
    if limit is None:
        limit = config["global_news_article_limit"]

    curr_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    start_date = (curr_dt - timedelta(days=look_back_days)).strftime("%Y-%m-%d")

    params = {
        "topics": "financial_markets,economy_macro,economy_monetary",
        "time_from": format_datetime_for_api(start_date),
        "time_to": format_datetime_for_api(curr_date, end_of_day=True),
        "limit": str(limit),
    }
    feed = _feed(_make_api_request("NEWS_SENTIMENT", params))
    kept = _in_window_items(feed, start_date, curr_date)[:limit]

    if not kept:
        gap = coverage_gap(
            (_parse_time_published(a.get("time_published")) for a in feed),
            start_date, curr_date, "Alpha Vantage global news", "market news",
        )
        return gap or f"No global news found between {start_date} and {curr_date}"

    return (
        f"## Global Market News, from {start_date} to {curr_date}:\n\n"
        + "\n".join(_render_item(a) for a in kept)
    )


def get_insider_transactions(symbol: str, curr_date: str | None = None) -> dict[str, str] | str:
    """Returns latest and historical insider transactions by key stakeholders.

    Covers transactions by founders, executives, board members, etc.

    Args:
        symbol: Ticker symbol. Example: "IBM".
        curr_date: When given, only transactions on or before it (yyyy-mm-dd).

    Returns:
        Dictionary containing insider transaction data or JSON string.
    """

    params = {
        "symbol": symbol,
    }

    response = _make_api_request("INSIDER_TRANSACTIONS", params)
    if not curr_date:
        return response
    payload = json.loads(response)
    payload["data"] = [t for t in payload["data"] if t["transaction_date"] <= curr_date]
    return json.dumps(payload)
