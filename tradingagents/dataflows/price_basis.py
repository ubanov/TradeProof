"""Structured, run-scoped capture of the price basis a run's market data
actually came from (see fork README: "RecommendationRecord" -- price_basis).

Captured at the ONE shared, cached load point every yfinance-backed
market-data path already funnels through
(``tradingagents.dataflows.vendors.yahoo.ohlcv.load_ohlcv``) -- never a
second fetch, never parsed from the rendered ``market_report`` Markdown.
Only the ``fill_gaps=False`` call path records a reference: that is the "as
the vendor reported it" read (see ``load_ohlcv``'s own docstring) used by
``get_verified_market_snapshot`` -- the deterministic, source-of-truth tool
this snapshot is semantically closest to. The ``fill_gaps=True`` path (used
for indicator continuity) can carry a PRIOR session's close forward under
today's date (see ``_fill_price_gaps``), which would misrepresent a filled
cell as a genuine reference bar -- so it is deliberately never recorded here.

Collection follows the exact pattern already established by
``llm_trace.py``'s ``_active_tracer``/``_active_seed_context`` and
``source_coverage.py``'s collector: a single contextvar holds the active
run's collector (``None`` when nothing is tracking, so every recording call
is a cheap no-op outside a real run), set once per run by
``execute_analysis`` and read back at the end of that same run.
"""

from __future__ import annotations

import contextvars
import hashlib
import json
from dataclasses import dataclass
from typing import Any

import pandas as pd

# The ONE price convention this project's yfinance-backed OHLCV path uses
# today (see ``load_ohlcv``'s explicit ``auto_adjust=True``) -- named
# explicitly rather than left as an implicit vendor default, so a future
# second convention (e.g. a raw/nominal fetch) has an unambiguous sibling
# constant to add, not a guess about what "today's mode" meant.
PRICE_MODE_YAHOO_AUTO_ADJUSTED = "YAHOO_AUTO_ADJUSTED"


def _canonical_number(value: Any) -> float | None:
    """``None`` for a value the vendor never reported (NaN); otherwise the
    exact float, with NO rounding.

    Deliberately not rounded: this fingerprint exists to let a future
    RecommendationOutcome detect when Yahoo has retroactively reprocessed
    (e.g. re-adjusted for a new split/dividend) historical bars this record
    already froze -- rounding away small differences would hide exactly the
    change this field exists to catch. Determinism across two occurrences of
    the SAME value is still guaranteed without rounding: Python's float
    formatting (what both ``json.dumps`` and ``repr`` use) is the shortest
    string that round-trips to the identical bit pattern, so the same float
    value always serializes to the same text, and two genuinely different
    values are never collapsed into one.
    """
    if value is None or pd.isna(value):
        return None
    return float(value)


def _canonical_bar(row: pd.Series) -> dict[str, Any]:
    return {
        "date": pd.Timestamp(row["Date"]).strftime("%Y-%m-%d"),
        "open": _canonical_number(row.get("Open")),
        "high": _canonical_number(row.get("High")),
        "low": _canonical_number(row.get("Low")),
        "close": _canonical_number(row.get("Close")),
        "volume": _canonical_number(row.get("Volume")),
    }


def _canonical_json(data: Any) -> bytes:
    """Same convention as ``tradingagents.recommendation_record.canonical_json``
    (sorted keys, fixed separators, UTF-8) -- duplicated as one local,
    trivial stdlib call rather than importing the ledger module from this
    lower-level dataflows module, to avoid a cross-layer dependency for a
    one-line function."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def compute_dataset_fingerprint(
    *, provider: str, symbol: str, price_mode: str, data: pd.DataFrame
) -> str | None:
    """Deterministic hash of the bars actually used.

    ``data`` must be exactly what ``load_ohlcv(symbol, curr_date,
    fill_gaps=False)`` returns: every row from the cached ~5-year-to-today
    download, filtered to dates on or before the run's own analysis_date,
    with any row that never got a reported Close dropped entirely (never
    filled/invented -- see ``load_ohlcv``'s own ``fill_gaps=False`` branch).
    That is bit-for-bit the same frame ``_verified_rows``
    (``vendors/yahoo/snapshot.py``) hands to ``build_verified_market_snapshot``
    -- the source for ALL of that snapshot's numbers (the latest row, the
    stockstats indicators, and the recent-closes table), not just the
    single latest bar. So this fingerprint can honestly be described as:
    "the exact structured dataset that fed the Market Analyst's verified
    snapshot" -- not a subset or a re-derivation of it.

    Enters the hash: ``provider``, ``symbol``, ``price_mode``, and every
    bar's date + OHLCV values (dates as ISO strings, numbers at full float
    precision -- see ``_canonical_number``'s own docstring for why rounding
    is deliberately NOT applied -- missing values as JSON ``null``, rows
    sorted by date regardless of input order). Never enters it: the
    dataframe's own row index, any file path, any timestamp, or a
    pandas-specific serialization. Two datasets with the same
    symbol/provider/price_mode and the same bar values always produce the
    same fingerprint, whatever order the rows arrived in or how NaNs were
    spelled -- and any genuinely different bar value, however small,
    changes it.
    """
    if data is None or data.empty:
        return None
    bars = [_canonical_bar(row) for _, row in data.iterrows()]
    bars.sort(key=lambda bar: bar["date"])
    payload = {"provider": provider, "symbol": symbol, "price_mode": price_mode, "bars": bars}
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


@dataclass(frozen=True)
class PriceBasisEntry:
    provider: str
    price_mode: str
    symbol: str
    requested_date: str
    reference_bar_date: str | None
    reference_bar: dict[str, Any] | None
    dataset_fingerprint: str
    # The real instant the vendor was last contacted for this exact cached
    # dataset (the cache file's own mtime -- see ``load_ohlcv``) -- NOT "the
    # instant this run looked at it". ``None`` only in the pathological case
    # where that instant could not be read back honestly (see
    # ``load_ohlcv``); never a stand-in ``datetime.now()``.
    fetched_at: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "price_mode": self.price_mode,
            "symbol": self.symbol,
            "requested_analysis_date": self.requested_date,
            "reference_bar_date": self.reference_bar_date,
            # Explicit, never silent: a reader can tell "the exact session"
            # from "the nearest prior session" without comparing two date
            # strings themselves.
            "reference_bar_matches_requested_date": (
                self.reference_bar_date == self.requested_date
                if self.reference_bar_date is not None else None
            ),
            "reference_bar": self.reference_bar,
            "dataset_fingerprint": self.dataset_fingerprint,
            "fetched_at": self.fetched_at,
        }


class PriceBasisCollector:
    """Accumulates at most one entry per distinct (symbol, requested_date)
    queried this run -- a normal run produces exactly one, since there is
    one ticker and ``get_verified_market_snapshot`` is normally called with
    the run's own analysis_date."""

    def __init__(self) -> None:
        self._entries: dict[tuple[str, str], PriceBasisEntry] = {}

    def record(self, entry: PriceBasisEntry) -> None:
        key = (entry.symbol, entry.requested_date)
        if key not in self._entries:
            # First call wins. load_ohlcv's cache makes every call for the
            # same (symbol, requested_date, fill_gaps=False) byte-identical
            # within one run, so there is nothing to reconcile between calls.
            self._entries[key] = entry

    def entry_for(self, symbol: str, requested_date: str) -> PriceBasisEntry | None:
        return self._entries.get((symbol, requested_date))


_active_collector: contextvars.ContextVar[PriceBasisCollector | None] = contextvars.ContextVar(
    "price_basis_active_collector", default=None
)


def set_active_price_basis_collector(collector: PriceBasisCollector | None) -> None:
    """Always called (even with ``None``), mirroring
    ``set_active_source_coverage_collector`` -- a batch run must not leak one
    ticker's price-basis entries into the next."""
    _active_collector.set(collector)


def record_price_basis_reference(
    *, provider: str, price_mode: str, symbol: str, requested_date: str,
    data: pd.DataFrame, fetched_at: str | None,
) -> None:
    """Called from ``load_ohlcv`` with the EXACT dataframe it is about to
    return to its own caller -- never re-fetched, never re-derived. A no-op
    outside an active run (no collector set) or when there is nothing usable
    to record."""
    collector = _active_collector.get()
    if collector is None:
        return
    fingerprint = compute_dataset_fingerprint(
        provider=provider, symbol=symbol, price_mode=price_mode, data=data
    )
    if fingerprint is None:
        return
    latest = data.iloc[-1]
    reference_bar_date = pd.Timestamp(latest["Date"]).strftime("%Y-%m-%d")
    collector.record(PriceBasisEntry(
        provider=provider, price_mode=price_mode, symbol=symbol,
        requested_date=requested_date, reference_bar_date=reference_bar_date,
        reference_bar=_canonical_bar(latest), dataset_fingerprint=fingerprint,
        fetched_at=fetched_at,
    ))


def get_price_basis_snapshot(symbol: str, analysis_date: str) -> dict[str, Any] | None:
    """The run's price_basis for EXACTLY ``analysis_date``, or ``None`` when
    no market-data tool fetched (via the ``fill_gaps=False`` path) that
    exact date during this run -- e.g. the market analyst was not selected,
    never called ``get_verified_market_snapshot``, or only queried other
    dates. Never reconstructed or guessed."""
    collector = _active_collector.get()
    if collector is None:
        return None
    entry = collector.entry_for(symbol, analysis_date)
    return entry.to_dict() if entry is not None else None
