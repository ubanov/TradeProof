"""Structured, run-scoped coverage tracking for News/Sentiment data sources
(see fork README: "Source coverage" and audit "News/Sentiment acquisition").

Every News/Sentiment vendor call already degrades gracefully into one of a
handful of well-known placeholder strings instead of raising -- that is what
lets the LLM-facing behavior stay completely unchanged here. This module
reuses that EXACT existing semantics (never invents a new signal) and just
classifies the string a caller already has into one of ten deterministic
states, recording it into a small, run-scoped, serializable snapshot that
never holds a payload or a secret -- only source/status/counts/a short
reason.

Collection follows the exact pattern already established by
``llm_trace.py``'s ``_active_tracer``/``_active_seed_context``: a single
contextvar holds the active run's collector (``None`` when nothing is
tracking, so every recording call is a cheap no-op outside a real run), set
once per ticker by ``execute_analysis`` and read back at the end of that same
run. No threading/async concerns beyond what that existing mechanism already
handles, since this reuses it unchanged.
"""

from __future__ import annotations

import contextvars
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any


class SourceStatus(str, Enum):
    """The ten states a single source-fetch attempt can end in. Deliberately
    the exact set from the News/Sentiment acquisition audit -- no additions,
    no renaming of an already-reviewed taxonomy."""

    SUCCESS_WITH_DATA = "SUCCESS_WITH_DATA"
    SUCCESS_EMPTY = "SUCCESS_EMPTY"
    UNSUPPORTED_HISTORICAL = "UNSUPPORTED_HISTORICAL"
    AUTH_MISSING = "AUTH_MISSING"
    RATE_LIMITED = "RATE_LIMITED"
    PROVIDER_ERROR = "PROVIDER_ERROR"
    TIMEOUT = "TIMEOUT"
    FILTERED_TO_ZERO = "FILTERED_TO_ZERO"
    NOT_CONFIGURED = "NOT_CONFIGURED"
    SCREENED_TO_ZERO = "SCREENED_TO_ZERO"


# A status in this set means "a source legitimately has nothing to show" --
# the vendor was reached (or correctly withheld on point-in-time grounds) and
# the absence of usable content is itself the real answer.
LEGITIMATE_ABSENCE_STATUSES = frozenset({
    SourceStatus.UNSUPPORTED_HISTORICAL,
    SourceStatus.SUCCESS_EMPTY,
    SourceStatus.FILTERED_TO_ZERO,
    SourceStatus.SCREENED_TO_ZERO,
    SourceStatus.NOT_CONFIGURED,
})

# A status in this set means the attempt simply failed technically -- it says
# nothing about whether there was anything to find, so it must never be
# treated as equivalent to a legitimate absence (see ``decide_data_status``).
TECHNICAL_FAILURE_STATUSES = frozenset({
    SourceStatus.AUTH_MISSING,
    SourceStatus.RATE_LIMITED,
    SourceStatus.PROVIDER_ERROR,
    SourceStatus.TIMEOUT,
})


@dataclass(frozen=True)
class SourceCoverageEntry:
    """One source's outcome for one call. Compact and secret-free by
    construction: no payload field exists to accidentally fill with one."""

    source: str
    status: SourceStatus
    raw_count: int | None = None
    usable_count: int | None = None
    reason: str | None = None
    requested_start: str | None = None
    requested_end: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"status": self.status.value}
        for field_name in ("raw_count", "usable_count", "reason", "requested_start", "requested_end"):
            value = getattr(self, field_name)
            if value is not None:
                data[field_name] = value
        return data


class SourceCoverageCollector:
    """Accumulates entries for one ticker run, grouped as
    ``{top_level: {bucket: {source: entry}}}`` -- e.g.
    ``news.company.yfinance`` / ``sentiment.stocktwits``."""

    def __init__(self) -> None:
        self._data: dict[str, dict[str, dict[str, SourceCoverageEntry]]] = {}

    def record(self, top_level: str, bucket: str, source: str, entry: SourceCoverageEntry) -> None:
        self._data.setdefault(top_level, {}).setdefault(bucket, {})[source] = entry

    def snapshot(self) -> dict[str, Any]:
        """Compact, JSON-serializable view -- a flat bucket (no ``source``
        sub-level) when it holds exactly one source, matching the task's own
        example shape (``sentiment.stocktwits: {status: ...}``, not
        ``sentiment.stocktwits.stocktwits: {...}``)."""
        out: dict[str, Any] = {}
        for top_level, buckets in self._data.items():
            out[top_level] = {}
            for bucket, sources in buckets.items():
                if len(sources) == 1:
                    (only_entry,) = sources.values()
                    out[top_level][bucket] = only_entry.to_dict()
                else:
                    out[top_level][bucket] = {src: e.to_dict() for src, e in sources.items()}
        return out


_active_collector: contextvars.ContextVar[SourceCoverageCollector | None] = contextvars.ContextVar(
    "source_coverage_active_collector", default=None
)
# Disambiguates a shared call site (get_news is called by both the News
# Analyst's tool loop and the Sentiment Analyst's pre-fetch) -- set briefly
# around the one call that needs the non-default top-level bucket. Default
# "news" covers every other router-routed call without any caller needing
# to touch this at all.
_category_prefix: contextvars.ContextVar[str] = contextvars.ContextVar(
    "source_coverage_category_prefix", default="news"
)


def set_active_source_coverage_collector(collector: SourceCoverageCollector | None) -> None:
    """Called once per ticker run by ``execute_analysis``, exactly like
    ``llm_trace.set_active_tracer`` -- always called (even with ``None``), so
    a batch run never leaks one ticker's collector into the next."""
    _active_collector.set(collector)


def get_active_source_coverage_collector() -> SourceCoverageCollector | None:
    return _active_collector.get()


def source_coverage_category(name: str):
    """Context manager: tag every coverage recording made inside the `with`
    block as belonging to ``name`` (e.g. "sentiment") instead of the default
    "news". A thin wrapper around the contextvar so call sites read clearly."""

    class _Scope:
        def __enter__(self):
            self._token = _category_prefix.set(name)
            return self

        def __exit__(self, *exc_info):
            _category_prefix.reset(self._token)
            return False

    return _Scope()


def record_source_coverage(
    bucket: str, source: str, entry: SourceCoverageEntry, *, top_level: str | None = None,
) -> None:
    """Record one entry into the active collector; a complete no-op with no
    active collector (tests, a direct vendor call outside a graph run, trace
    mode off) -- mirrors ``llm_trace.record_structured_parse_result``."""
    collector = _active_collector.get()
    if collector is None:
        return
    collector.record(top_level or _category_prefix.get(), bucket, source, entry)


def get_source_coverage_snapshot() -> dict[str, Any] | None:
    """The active run's accumulated coverage, or ``None`` with nothing active."""
    collector = _active_collector.get()
    return collector.snapshot() if collector is not None else None


# ---------------------------------------------------------------------------
# Classifiers -- pure functions over text/exceptions this project's vendor
# code ALREADY produces; no vendor function's return type or behavior
# changes. Reused identically by the router (News) and by the Sentiment
# Analyst's own direct calls.
# ---------------------------------------------------------------------------

_UNSUPPORTED_HISTORICAL_MARKERS = ("it only serves recent items", "the window extends past today")
_WITHHELD_MARKERS = ("no historical vintage",)


def classify_router_text(method: str, text: str) -> tuple[SourceStatus, int | None]:
    """Classify a SUCCESSFUL ``route_to_vendor`` return value for one of the
    four News-analyst methods this task covers. Never called for any other
    method (callers gate on that explicitly), so this never affects
    OHLCV/fundamentals/insider-transactions routing."""
    if not isinstance(text, str):
        return SourceStatus.SUCCESS_WITH_DATA, None
    lowered = text.lower()
    if any(marker in lowered for marker in _UNSUPPORTED_HISTORICAL_MARKERS):
        return SourceStatus.UNSUPPORTED_HISTORICAL, 0
    if "withheld" in lowered and any(marker in lowered for marker in _WITHHELD_MARKERS):
        return SourceStatus.UNSUPPORTED_HISTORICAL, 0
    if "currently unavailable (network error" in lowered:
        return SourceStatus.PROVIDER_ERROR, 0
    if text.startswith(("NO_DATA_AVAILABLE:", "DATA_UNAVAILABLE:")):
        # route_to_vendor's own wrapper sentinel for an exhausted vendor
        # chain -- a technical failure, not a legitimate empty result (see
        # ``record_news_router_outcome``, which already classifies this
        # correctly from the exception site; this prefix check makes any
        # OTHER caller re-classifying the final returned text, e.g. the
        # Sentiment Analyst's own short-circuit decision, agree with it).
        return SourceStatus.PROVIDER_ERROR, 0
    if text.startswith(("<no ", "No news found", "No global news found")):
        return SourceStatus.SUCCESS_EMPTY, 0
    if method == "get_macro_indicators":
        if "no observations for" in lowered:
            return SourceStatus.SUCCESS_EMPTY, 0
        if "not found" in lowered or text.startswith("FRED:"):
            return SourceStatus.PROVIDER_ERROR, 0
        count = len(re.findall(r"^\| \d{4}-\d{2}-\d{2} \|", text, flags=re.MULTILINE))
        return SourceStatus.SUCCESS_WITH_DATA, count or None
    return SourceStatus.SUCCESS_WITH_DATA, None


# Source-coverage recording for the router (News Analyst's tool-calling
# path) is scoped to exactly these four methods -- every other routed
# method (OHLCV, fundamentals, insider transactions, ...) is completely
# untouched. "bucket" is the sub-group within the active top-level category
# (default "news"; the Sentiment Analyst's own direct get_news call
# overrides the top level to "sentiment" via ``source_coverage_category``,
# which also flattens the bucket to "news" instead of "company" there).
_NEWS_COVERAGE_BUCKETS = {
    "get_news": "company",
    "get_global_news": "macro",
    "get_macro_indicators": "macro",
    "get_prediction_markets": "prediction_markets",
}


def _coverage_window(method: str, args: tuple) -> tuple[str | None, str | None]:
    """Best-effort (requested_start, requested_end) from this call's own
    positional args -- purely descriptive, never used for any filtering."""
    if method == "get_news" and len(args) >= 3:
        return args[1], args[2]
    if method == "get_global_news" and args:
        return None, args[0]
    if method == "get_macro_indicators" and len(args) >= 2:
        return None, args[1]
    if method == "get_prediction_markets" and len(args) >= 3:
        return None, args[2]
    return None, None


def record_news_router_outcome(
    method: str, vendor: str, args: tuple, *, text: str | None = None, exc: Exception | None = None,
) -> SourceStatus | None:
    """Classify and record one ``route_to_vendor`` vendor attempt for one of
    the four News-analyst methods; a no-op for every other method (so this
    is safe to call unconditionally from ``route_to_vendor``'s loop without
    it needing to special-case anything about OHLCV/fundamentals/etc.) and a
    no-op with no active collector (see ``record_source_coverage``).

    Returns the classified ``SourceStatus`` (``None`` for an unrelated
    method) so the caller can decide whether to advance a multi-vendor chain
    on it (see ``route_to_vendor``) without classifying the same text twice.
    """
    if method not in _NEWS_COVERAGE_BUCKETS:
        return None
    top_level = _category_prefix.get()
    bucket = "news" if top_level == "sentiment" else _NEWS_COVERAGE_BUCKETS[method]
    requested_start, requested_end = _coverage_window(method, args)
    if exc is not None:
        status = classify_router_exception(exc)
        entry = SourceCoverageEntry(
            source=vendor, status=status, usable_count=0, reason=str(exc)[:200],
            requested_start=requested_start, requested_end=requested_end,
        )
    else:
        status, count = classify_router_text(method, text)
        entry = SourceCoverageEntry(
            source=vendor, status=status, usable_count=count,
            requested_start=requested_start, requested_end=requested_end,
        )
    record_source_coverage(bucket, vendor, entry, top_level=top_level)
    return status


def classify_router_exception(exc: Exception) -> SourceStatus:
    """Classify one of the three typed ``VendorError`` exceptions (or any
    other exception) already caught by ``route_to_vendor``. ``NoMarketDataError``
    for ``get_news``/``get_global_news`` specifically always means the vendor
    call itself raised (see ``yahoo/news.py``'s blanket ``except Exception``),
    never "no items in window" -- that path returns a string, not this
    exception -- so it is a technical failure here, not a legitimate empty."""
    from tradingagents.dataflows.errors import VendorNotConfiguredError, VendorRateLimitError

    if isinstance(exc, VendorRateLimitError):
        return SourceStatus.RATE_LIMITED
    if isinstance(exc, VendorNotConfiguredError):
        return SourceStatus.AUTH_MISSING
    # Checked by TEXT, not just isinstance(exc, TimeoutError): requests'
    # own Timeout/ConnectTimeout/ReadTimeout subclass OSError, not the
    # builtin TimeoutError, and NoMarketDataError (yahoo/news.py's blanket
    # wrapper) carries the original exception's text in its own message --
    # either shape must still classify as TIMEOUT, not a generic error.
    detail = str(exc).lower()
    if "timeout" in detail or "timed out" in detail:
        return SourceStatus.TIMEOUT
    return SourceStatus.PROVIDER_ERROR


def classify_stocktwits_text(text: str) -> SourceCoverageEntry:
    if not isinstance(text, str):
        return SourceCoverageEntry(source="stocktwits", status=SourceStatus.PROVIDER_ERROR, usable_count=0)
    if text.startswith("<stocktwits unavailable:"):
        return SourceCoverageEntry(
            source="stocktwits", status=SourceStatus.PROVIDER_ERROR, usable_count=0, reason=text[:200],
        )
    lowered = text.lower()
    if any(marker in lowered for marker in _UNSUPPORTED_HISTORICAL_MARKERS):
        return SourceCoverageEntry(source="stocktwits", status=SourceStatus.UNSUPPORTED_HISTORICAL, usable_count=0)
    if "<none of the" in text and "StockTwits messages is about" in text:
        return SourceCoverageEntry(source="stocktwits", status=SourceStatus.SCREENED_TO_ZERO, usable_count=0)
    if text.startswith("<no StockTwits messages"):
        return SourceCoverageEntry(source="stocktwits", status=SourceStatus.SUCCESS_EMPTY, usable_count=0)
    match = re.search(r"Total: (\d+) most-recent messages", text)
    count = int(match.group(1)) if match else None
    return SourceCoverageEntry(source="stocktwits", status=SourceStatus.SUCCESS_WITH_DATA, usable_count=count)


def classify_reddit_text(text: str) -> SourceCoverageEntry:
    if not isinstance(text, str):
        return SourceCoverageEntry(source="reddit", status=SourceStatus.PROVIDER_ERROR, usable_count=0)
    if text.startswith("<Reddit unavailable: fetch failed"):
        return SourceCoverageEntry(
            source="reddit", status=SourceStatus.PROVIDER_ERROR, usable_count=0, reason=text[:200],
        )
    lowered = text.lower()
    if any(marker in lowered for marker in _UNSUPPORTED_HISTORICAL_MARKERS):
        return SourceCoverageEntry(source="reddit", status=SourceStatus.UNSUPPORTED_HISTORICAL, usable_count=0)
    if text.startswith("<no Reddit posts found"):
        return SourceCoverageEntry(source="reddit", status=SourceStatus.SUCCESS_EMPTY, usable_count=0)
    matches = re.findall(r"— (\d+) recent posts mentioning", text)
    if matches:
        return SourceCoverageEntry(
            source="reddit", status=SourceStatus.SUCCESS_WITH_DATA, usable_count=sum(int(n) for n in matches),
        )
    if "after screening>" in text:
        return SourceCoverageEntry(source="reddit", status=SourceStatus.SCREENED_TO_ZERO, usable_count=0)
    return SourceCoverageEntry(source="reddit", status=SourceStatus.SUCCESS_EMPTY, usable_count=0)


def classify_jev_note(text: str) -> SourceCoverageEntry | None:
    """``None`` means no Jev note is present in ``text`` at all (screening was
    never invoked for this source -- e.g. ``jev_screen`` returned ``None``
    because ``TYPESAFE_API_KEY`` is unset; the caller records NOT_CONFIGURED
    itself in that case, since that fact isn't present in any returned text)."""
    if not isinstance(text, str):
        return None
    if "<Jev screening unavailable" in text:
        return SourceCoverageEntry(
            source="jev_screening", status=SourceStatus.PROVIDER_ERROR, reason="jev screening failed; posts unscreened",
        )
    match = re.search(r"Screened by Jev: (\d+) of the (\d+) posts fetched are about", text)
    if match:
        kept, total = int(match.group(1)), int(match.group(2))
        status = SourceStatus.SCREENED_TO_ZERO if kept == 0 else SourceStatus.SUCCESS_WITH_DATA
        return SourceCoverageEntry(source="jev_screening", status=status, raw_count=total, usable_count=kept)
    return None


def decide_data_status(entries: list[SourceCoverageEntry]) -> str:
    """The short-circuit verdict for a list of sentiment-source entries
    (news/stocktwits/reddit -- never ``jev_screening``, which is a meta-filter
    on top of those, not a source of sentiment itself).

    - any entry with real usable content -> "HAS_DATA" (call the LLM)
    - no real content, but every empty entry is a LEGITIMATE absence
      (point-in-time unsupported, genuinely empty window, screened out, or
      not configured) -> "NO_DATA" (a real verdict: no sentiment observable)
    - no real content, and at least one entry failed technically -> never
      claim "NO_DATA" over a failure that might be masking real sentiment;
      "SOURCE_UNAVAILABLE" instead (conservative, see task: "mezcla de
      fuentes -> regla conservadora").
    """
    if any(e.status == SourceStatus.SUCCESS_WITH_DATA and (e.usable_count or 0) > 0 for e in entries):
        return "HAS_DATA"
    if any(e.status == SourceStatus.SUCCESS_WITH_DATA for e in entries):
        # SUCCESS_WITH_DATA with an unknown (None) count is still real content,
        # not a zero -- never downgraded to NO_DATA for lack of a parsed count.
        return "HAS_DATA"
    if any(e.status in TECHNICAL_FAILURE_STATUSES for e in entries):
        return "SOURCE_UNAVAILABLE"
    return "NO_DATA"
