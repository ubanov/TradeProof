"""Regression test for a real failed run: MSFT/2026-09-30, non-interactive
``tradingagents run``, local Ollama model ``qwen3.8-flash-next`` (a small,
quantized model, more prone than a top-tier hosted one to formatting
artifacts in tool-call argument text).

The Market Analyst's ``get_stock_data`` tool call carried a date argument
with a trailing newline (e.g. ``"2026-09-01\\n"``). ``get_YFin_data_online``
(tradingagents/dataflows/vendors/yahoo/market.py) does a bare
``datetime.strptime(date, "%Y-%m-%d")`` with zero tolerance for incidental
whitespace, raising exactly ``ValueError: unconverted data remains: \\n`` --
which propagated out of the whole market_analyst stage and aborted the run
(``status="error"``, ``failure_stage="market_analyst"``, no
RecommendationRecord written -- confirmed by reading, never modifying, the
real run's ``result.json``/``partial_report.md``).

Root cause: ``tradingagents/dataflows/date_window.py``'s ``as_of``/
``as_of_window`` -- the ONE centralized point every dated tool
(``get_stock_data``, ``get_indicators``, ``get_news``, ``get_fundamentals``,
...) already routes a model's raw date arguments through before any
vendor call -- returned the model's raw string completely unmodified
whenever no run-date clamping was needed, never validating or normalizing
it. The fix strips incidental surrounding whitespace there (and only
there, via ``_parse``/``_canonical``) before any date reaches a vendor --
``get_YFin_data_online`` itself is intentionally left exactly as strict as
before (see test below proving it still rejects a raw corrupted date when
called directly, bypassing normalization).

Pre-existing bug, not a regression: this exact code (bare ``strptime`` in
the vendor, permissive pass-through in ``as_of_window``) existed unchanged
across the take_profit/price_basis/classification/horizon_rules/
execution_rules commits (see ``git log -- tradingagents/dataflows/vendors/
yahoo/market.py tradingagents/dataflows/date_window.py`` -- untouched by
any of them), and the crash happens before any network call is made, so it
does not depend on a yfinance version or behavior change either.
"""

from __future__ import annotations

import pandas as pd
import pytest

import tradingagents.dataflows.vendors.yahoo.market as yfin
from tradingagents.dataflows.date_window import as_of, as_of_window
from tradingagents.dataflows.router import route_to_vendor


class _FakeTicker:
    """Stands in for ``yf.Ticker`` -- no network, captures the exact
    start/end the vendor function ends up requesting."""

    captured: dict | None = None

    def __init__(self, symbol):
        self.symbol = symbol

    def history(self, start, end):
        type(self).captured = {"symbol": self.symbol, "start": start, "end": end}
        idx = pd.to_datetime(["2026-09-29", "2026-09-30"])
        return pd.DataFrame(
            {"Open": [510.0, 511.0], "High": [515.0, 519.0], "Low": [505.0, 510.0],
             "Close": [512.0, 512.9], "Volume": [1_000_000, 1_100_000]},
            index=idx,
        )


# --- The real failure, reproduced exactly, no network/LLM -------------------


@pytest.mark.unit
def test_get_yfin_data_online_itself_still_rejects_a_raw_corrupted_date():
    """BEFORE this fix's layer is involved at all: calling the vendor
    function directly with the model's exact malformed argument shape
    still raises the exact real-world error -- proving the vendor was
    never made permissive, and this is exactly what crashed the real run
    when the (unfixed) date_window.py handed it this same raw string."""
    with pytest.raises(ValueError, match=r"unconverted data remains"):
        yfin.get_YFin_data_online("MSFT", "2026-09-01\n", "2026-09-30")


@pytest.mark.unit
def test_as_of_window_no_longer_passes_through_a_trailing_newline():
    """AFTER the fix: the centralized point every dated tool already
    shares strips the incidental newline before anything downstream ever
    sees it."""
    start, end = as_of_window("2026-09-01\n", "2026-09-30", "2026-09-30")
    assert start == "2026-09-01"
    assert end == "2026-09-30"
    assert "\n" not in start and "\n" not in end


@pytest.mark.unit
def test_as_of_window_strips_a_corrupted_end_date_too():
    start, end = as_of_window("2026-09-01", "2026-09-30\n", "2026-09-30")
    assert start == "2026-09-01"
    assert end == "2026-09-30"


@pytest.mark.unit
def test_full_market_analyst_call_shape_now_succeeds_end_to_end(monkeypatch):
    """The exact call shape the real Market Analyst made (get_stock_data's
    own start_date argument carrying a trailing newline), driven through
    the REAL as_of_window -> route_to_vendor -> get_YFin_data_online chain,
    with only the network call mocked out. Before the fix this raised
    ValueError inside route_to_vendor; after it, it succeeds and the
    vendor receives the clean, canonical dates."""
    _FakeTicker.captured = None
    monkeypatch.setattr(yfin.yf, "Ticker", _FakeTicker)

    start_date, end_date = as_of_window("2026-09-01\n", "2026-09-30", "2026-09-30")
    result = route_to_vendor("get_stock_data", "MSFT", start_date, end_date)

    assert "Stock data for MSFT" in result
    assert _FakeTicker.captured["symbol"] == "MSFT"
    assert _FakeTicker.captured["start"] == "2026-09-01"  # clean, no newline reached yfinance


# --- Correct final date / no look-ahead --------------------------------------


@pytest.mark.unit
def test_normalization_does_not_change_the_resolved_date_value():
    """Stripping must never alter WHICH date is served -- only remove
    incidental whitespace around the same value."""
    assert as_of("2026-09-30\n", "2026-09-30") == as_of("2026-09-30", "2026-09-30") == "2026-09-30"


@pytest.mark.unit
def test_a_corrupted_future_end_date_is_still_clamped_not_just_stripped():
    """No look-ahead: a model asking for a window past the run's own date
    must still be clamped to trade_date, whether or not its text happened
    to carry a stray newline -- normalization must not accidentally
    bypass the point-in-time guard."""
    clean = as_of_window("2026-10-01", "2026-10-10", "2026-09-30")
    corrupted = as_of_window("2026-10-01\n", "2026-10-10\n", "2026-09-30")
    assert clean == corrupted == ("2026-09-21", "2026-09-30")


@pytest.mark.unit
def test_as_of_still_refuses_to_serve_a_date_after_trade_date():
    assert as_of("2026-12-25", "2026-09-30") == "2026-09-30"


# --- Existing formats/behavior are not broken --------------------------------


@pytest.mark.unit
def test_clean_dates_are_completely_unaffected():
    assert as_of_window("2026-09-01", "2026-09-30", "2026-09-30") == ("2026-09-01", "2026-09-30")
    assert as_of("2026-09-15", "2026-09-30") == "2026-09-15"


@pytest.mark.unit
def test_as_of_with_no_trade_date_still_passes_a_clean_date_through():
    assert as_of("2026-09-30", "") == "2026-09-30"


@pytest.mark.unit
def test_genuinely_malformed_dates_are_still_not_silently_accepted():
    """Not permissive parsing: a date that is not YYYY-MM-DD even once
    stripped (e.g. a totally different format) is handled exactly as
    before -- as_of still falls back to trade_date, it is never guessed
    at or reformatted."""
    assert as_of("09/30/2026", "2026-09-30") == "2026-09-30"  # falls back, as before
    assert as_of("not-a-date", "2026-09-30") == "2026-09-30"


@pytest.mark.unit
def test_none_and_non_string_dates_are_handled_without_raising():
    assert as_of(None, "2026-09-30") == "2026-09-30"
    assert as_of(None, "") is None
