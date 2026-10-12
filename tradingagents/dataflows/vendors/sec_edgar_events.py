"""SEC EDGAR company events (Form 8-K material events) for the News Analyst.

Reuses ``sec_edgar.py``'s CIK resolution, HTTP/cache plumbing, and
User-Agent policy -- this module adds exactly ONE new endpoint (the
submissions filing index, never fetched by the Fundamentals vendor, which
only ever reads XBRL company-facts) plus a deterministic Form 8-K Item-code
categorization, with NO LLM involved anywhere in extraction.

First scope (see fork README: "SEC EDGAR company events"): Form 8-K only.
Form 4, 10-Q/10-K-as-news, 6-K, and real exhibit text (beyond a short
best-effort excerpt of the primary document itself) are explicitly out of
scope for this version.

SEC company events are regulatory filings, not press/news coverage: a
filer can go long stretches with no 8-K that reads as news, and most 8-Ks
that DO exist are thin "furnished" wrappers pointing at an exhibit this
module does not fetch -- see ``content_status`` on ``CompanyEvent``. This
is a complement to News, never a substitute for a real news archive.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta

import requests

from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.errors import VendorRateLimitError
from tradingagents.dataflows.vendors.sec_edgar import _cached_json, _user_agent, cik_for

_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
_ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_nodash}/{primary_document}"

# Deterministic, SEC-native Item code -> category (Form 8-K only). Never an
# opinion of materiality beyond what SEC's own Item taxonomy already
# encodes; a filing can carry more than one Item/category.
ITEM_CATEGORIES: dict[str, str] = {
    "1.01": "material agreement",
    "1.02": "material agreement",
    "1.03": "bankruptcy",
    "2.01": "acquisition/disposition",
    "2.02": "earnings/results",
    "2.03": "financing/debt",
    "2.04": "default/acceleration",
    "2.06": "impairment",
    "5.02": "management/director change",
    "7.01": "regulation FD disclosure",
    "8.01": "other material event",
    "9.01": "financial statements/exhibits",
}

# A short, deterministic excerpt is only worth fetching a second document
# for; capped so a ticker with an unusually filing-heavy window still makes
# a bounded number of requests (see fork README: "minimize requests").
_EXCERPT_MAX_CHARS = 400
_MAX_EXCERPT_FETCHES = 5


@dataclass(frozen=True)
class CompanyEvent:
    """One Form 8-K filing, deterministically classified -- never the
    product of an LLM. Every instance this module returns already passed
    the point-in-time check (``filed_at`` within the requested window and
    on or before the as-of date), so ``point_in_time_valid`` is always True
    here; it exists as an explicit field so a downstream reader never has
    to re-derive that guarantee itself."""

    form: str
    filed_at: str
    accepted_at: str | None
    accession_number: str
    primary_document: str
    item_numbers: tuple[str, ...]
    categories: tuple[str, ...]
    url: str
    short_excerpt: str | None
    content_status: str  # "EXCERPT_AVAILABLE" | "WRAPPER_ONLY"
    point_in_time_valid: bool = True
    source: str = "SEC EDGAR"


def _fetch_text(url: str) -> str:
    """Same identification/error policy as ``sec_edgar._fetch_json``, for a
    non-JSON (HTML filing document) response."""
    try:
        response = requests.get(url, headers={"User-Agent": _user_agent()}, timeout=30)
        response.raise_for_status()
        return response.text
    except requests.RequestException as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        raise VendorRateLimitError(f"SEC EDGAR request failed ({status or type(exc).__name__})") from exc


def _categories_for(item_numbers: tuple[str, ...]) -> tuple[str, ...]:
    categories: list[str] = []
    for item in item_numbers:
        category = ITEM_CATEGORIES.get(item)
        if category and category not in categories:
            categories.append(category)
    return tuple(categories)


def _extract_excerpt(url: str, item_numbers: tuple[str, ...]) -> tuple[str | None, str]:
    """Best-effort, deterministic excerpt of the filing's own text right
    after its first recognized Item heading -- a plain-text strip, not an
    HTML parser (see module docstring: no complex parser in this first
    version). ``WRAPPER_ONLY`` covers both "the document has no usable text
    after its Item heading" (a thin furnished wrapper, e.g. an Item 2.02
    earnings 8-K that only points at an unfetched press-release exhibit)
    and "the document could not be fetched at all" -- from a News Analyst's
    perspective both mean the same thing: no excerpt, metadata only.
    """
    try:
        html = _fetch_text(url)
    except VendorRateLimitError:
        return None, "WRAPPER_ONLY"
    plain = re.sub(r"<[^>]+>", " ", html)
    plain = re.sub(r"&#\d+;|&[a-zA-Z]+;", " ", plain)
    plain = re.sub(r"\s+", " ", plain).strip()
    for item in item_numbers:
        pattern = re.compile(rf"Item\s*{re.escape(item)}\.?\s*", re.IGNORECASE)
        match = pattern.search(plain)
        if match:
            excerpt = plain[match.end():match.end() + _EXCERPT_MAX_CHARS].strip()
            if excerpt:
                return excerpt, "EXCERPT_AVAILABLE"
    return None, "WRAPPER_ONLY"


def _parse_items(raw: str) -> tuple[str, ...]:
    if not raw:
        return ()
    return tuple(i.strip() for i in raw.split(",") if i.strip())


def fetch_company_events(
    ticker: str, curr_date: str, look_back_days: int | None = None,
) -> tuple[list[CompanyEvent], int]:
    """Form 8-K company events for ``ticker``, filed within
    ``[curr_date - look_back_days, curr_date]``.

    Point-in-time: a filing is gated SOLELY on ``filingDate`` (never
    ``reportDate``/``periodOfReport``, which describe the period the filing
    covers, not when the market could have known about it) -- one filed
    after ``curr_date`` is never returned, and one filed before the window
    start is excluded too, regardless of what period it reports on.

    Returns ``(events, raw_filing_count)`` where ``raw_filing_count`` is
    every filing of ANY form found within the window (for source_coverage's
    "how much filing activity happened" signal), while ``events`` holds
    only the Form 8-K ones -- the only form this first version returns.
    """
    if look_back_days is None:
        look_back_days = get_config().get("sec_company_events_lookback_days", 30)
    start_date = (datetime.strptime(curr_date, "%Y-%m-%d") - timedelta(days=look_back_days)).strftime("%Y-%m-%d")

    cik = cik_for(ticker)
    if cik is None:
        return [], 0

    data = _cached_json(_SUBMISSIONS_URL.format(cik=cik), f"submissions_CIK{cik}.json")
    recent = (data.get("filings") or {}).get("recent") or {}
    forms = recent.get("form", [])
    filing_dates = recent.get("filingDate", [])
    accession_numbers = recent.get("accessionNumber", [])
    primary_documents = recent.get("primaryDocument", [])
    items_raw = recent.get("items", [])
    acceptance = recent.get("acceptanceDateTime", [])

    raw_filing_count = 0
    events: list[CompanyEvent] = []
    excerpt_fetches = 0
    for i, form in enumerate(forms):
        filed_at = filing_dates[i] if i < len(filing_dates) else ""
        if not filed_at or filed_at > curr_date or filed_at < start_date:
            continue
        raw_filing_count += 1
        if form != "8-K":
            continue

        item_numbers = _parse_items(items_raw[i] if i < len(items_raw) else "")
        accession_number = accession_numbers[i] if i < len(accession_numbers) else ""
        primary_document = primary_documents[i] if i < len(primary_documents) else ""
        accepted_at = acceptance[i] if i < len(acceptance) else None
        url = _ARCHIVE_URL.format(
            cik_int=int(cik), accession_nodash=accession_number.replace("-", ""), primary_document=primary_document,
        )

        short_excerpt: str | None = None
        content_status = "WRAPPER_ONLY"
        if primary_document and excerpt_fetches < _MAX_EXCERPT_FETCHES:
            excerpt_fetches += 1
            short_excerpt, content_status = _extract_excerpt(url, item_numbers)

        events.append(CompanyEvent(
            form=form, filed_at=filed_at, accepted_at=accepted_at,
            accession_number=accession_number, primary_document=primary_document,
            item_numbers=item_numbers, categories=_categories_for(item_numbers),
            url=url, short_excerpt=short_excerpt, content_status=content_status,
        ))

    events.sort(key=lambda e: e.filed_at)
    return events, raw_filing_count


def render_company_events(events: list[CompanyEvent], ticker: str, start_date: str, end_date: str) -> str:
    """Compact, ready-for-prompt rendering -- never a full filing's text."""
    if not events:
        return (
            f"<No SEC Form 8-K company events found for {ticker.upper()} between "
            f"{start_date} and {end_date}. SEC company events are regulatory filings, "
            f"not general news coverage -- this reflects regulatory filing activity "
            f"only, not an absence of news about {ticker.upper()}.>"
        )
    lines = [f"## SEC COMPANY EVENTS ({ticker.upper()}, {start_date} to {end_date})", ""]
    for event in events:
        lines.append(f"{event.filed_at} | {event.form} | Items {', '.join(event.item_numbers) or 'none listed'}")
        if event.categories:
            lines.append(f"Categories: {'; '.join(event.categories)}")
        lines.append(f"Accession: {event.accession_number}")
        if event.content_status == "EXCERPT_AVAILABLE" and event.short_excerpt:
            lines.append(f"Summary/excerpt: {event.short_excerpt}")
        else:
            lines.append("Summary/excerpt: not available (WRAPPER_ONLY -- metadata only, see primary document URL)")
        lines.append(f"Source: {event.source}")
        lines.append(f"URL: {event.url}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
