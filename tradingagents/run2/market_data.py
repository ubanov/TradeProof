"""Market-only, point-in-time data for run2 (fork spec: "run2 V1 -- MARKET
ONLY"). No news, no fundamentals, no social, no Decision Memory.

Deliberately skips ``tradingagents/agents/tools.py`` and
``market_analyst.py``: those exist for the main pipeline's LLM-driven
tool-calling loop (the model decides which indicators to request, one round
at a time). run2 wants the opposite shape -- a single, deterministic,
Python-callable "give me everything as of this date" bundle the LLM never
drives -- so it calls the underlying dataflow functions directly.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
from stockstats import wrap

from tradingagents.dataflows.vendors.yahoo.ohlcv import load_ohlcv
from tradingagents.dataflows.vendors.yahoo.snapshot import build_verified_market_snapshot
from tradingagents.run2.schemas import MarketFeatures


def fetch_market_context(ticker: str, analysis_date: str, *, look_back_days: int = 60) -> str:
    """Deterministic Markdown market snapshot -- no LLM involved. Already
    point-in-time safe: ``build_verified_market_snapshot`` filters to rows
    on or before ``analysis_date`` (fork spec: "no information from after
    analysis_date may be used")."""
    return build_verified_market_snapshot(ticker, analysis_date, look_back_days=look_back_days)


def fetch_price_reference(ticker: str, analysis_date: str) -> dict[str, Any]:
    """The latest verified close on or before ``analysis_date`` -- stored in
    ``result.json`` so a LATER, separate evaluation step can compute
    ``return_5d``/``return_10d``/``return_20d``/``return_40d`` against it.
    run2 itself never computes a future return (fork spec: "el pipeline de
    analisis debe tener acceso cero al futuro").

    Re-applies the ``<= analysis_date`` cutoff defensively on top of
    ``load_ohlcv``'s own filtering, the same belt-and-suspenders pattern
    ``build_verified_market_snapshot._verified_rows`` uses, since this value
    is later trusted as the run's own look-ahead-safety evidence.
    """
    data = load_ohlcv(ticker, analysis_date, fill_gaps=False)
    if data is None or data.empty:
        raise ValueError(f"No OHLCV data available for {ticker}.")
    df = data.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df[df["Date"] <= pd.to_datetime(analysis_date)].sort_values("Date")
    if df.empty:
        raise ValueError(f"No OHLCV rows on or before {analysis_date} for {ticker}.")
    latest = df.iloc[-1]
    return {
        "symbol": ticker,
        "reference_date": latest["Date"].strftime("%Y-%m-%d"),
        "close": float(latest["Close"]) if pd.notna(latest.get("Close")) else None,
    }


# ---------------------------------------------------------------------------
# Deterministic market feature pack (fork spec: "run2" hardening --
# "MARKET-ONLY cuantitativo, auditable y barato de evaluar").
#
# Every window below is a session count (rows of the point-in-time-filtered
# OHLCV frame), not a calendar-day count. All percentage fields are
# percentage POINTS (e.g. 2.5 means 2.5%), rounded to 4 decimals for a
# stable, readable result.json. A field is `None` whenever there is not
# enough history for its window -- never a fabricated or zero-filled value.
# ---------------------------------------------------------------------------

RETURN_WINDOWS: tuple[int, ...] = (5, 10, 20, 40, 60, 120)
SMA_WINDOWS: tuple[int, ...] = (20, 50, 100, 200)
# The fork spec asks for slope_sma20/50/200 specifically (no slope_sma100).
SLOPE_SMA_WINDOWS: tuple[int, ...] = (20, 50, 200)
# slope_smaN = % change of the SMA's own value over this many sessions -- a
# deliberately simple, documented proxy for "is the average itself rising or
# falling", not a linear regression (no new dependency).
SLOPE_LOOKBACK_SESSIONS = 10
STRUCTURE_WINDOWS: tuple[int, ...] = (20, 60, 120)
REALIZED_VOL_WINDOWS: tuple[int, ...] = (20, 60)
VOLUME_WINDOW = 20
# recent_structure compares the rolling high/low of the most recent half of
# this window against the prior half -- a simplified, deterministic proxy
# for "higher highs/higher lows" vs. "lower highs/lower lows", not true
# swing-point detection.
RECENT_STRUCTURE_WINDOW = 40


def _round(value: float | None, ndigits: int = 4) -> float | None:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return round(value, ndigits)


def _safe_indicator(stock_df: pd.DataFrame, column: str, n: int, *, min_sessions: int) -> float | None:
    """One stockstats indicator's latest value, or ``None`` when there is
    not enough history (``min_sessions``) or the computed value itself is
    NaN (stockstats' own insufficient-window behaviour) -- never raises."""
    if n < min_sessions:
        return None
    try:
        value = stock_df[column].iloc[-1]
    except Exception:  # noqa: BLE001 -- one bad indicator must not sink the whole feature pack
        return None
    return _round(float(value)) if pd.notna(value) else None


def _recent_structure(high: pd.Series, low: pd.Series, n: int) -> str | None:
    """"HH_HL" (higher highs + higher lows), "LH_LL" (lower highs + lower
    lows), or "MIXED" -- comparing the most recent half of
    ``RECENT_STRUCTURE_WINDOW`` sessions against the prior half. See the
    constant's own comment for why this is a simplified proxy, not true
    swing-point detection."""
    if n < RECENT_STRUCTURE_WINDOW:
        return None
    half = RECENT_STRUCTURE_WINDOW // 2
    first_high = high.iloc[-RECENT_STRUCTURE_WINDOW:-half].max()
    second_high = high.iloc[-half:].max()
    first_low = low.iloc[-RECENT_STRUCTURE_WINDOW:-half].min()
    second_low = low.iloc[-half:].min()
    if second_high > first_high and second_low > first_low:
        return "HH_HL"
    if second_high < first_high and second_low < first_low:
        return "LH_LL"
    return "MIXED"


def compute_market_features(ticker: str, analysis_date: str) -> MarketFeatures:
    """The deterministic feature pack Regime/Signal/Challenger are given as
    already-computed evidence (fork spec: "No quiero que el LLM tenga que
    calcular medias, retornos, ATR, etc. mentalmente a partir de series
    crudas"). Methodology, field by field:

    - ``return_Nd`` = ``(close[t] / close[t-N] - 1) * 100`` -- simple
      percentage return over the trailing N sessions ending at
      ``analysis_date``.
    - ``smaN`` = simple moving average of Close over the trailing N
      sessions; ``distance_pct_to_smaN`` = ``(close - smaN) / smaN * 100``;
      ``price_above_smaN`` = ``close > smaN``.
    - ``slope_smaN`` = ``% change of the smaN series itself`` over
      ``SLOPE_LOOKBACK_SESSIONS`` (10) sessions -- positive means the
      average itself is rising.
    - ``rolling_high_K``/``rolling_low_K`` = max(High)/min(Low) over the
      trailing K sessions; ``distance_pct_to_high_K``/``..._low_K`` =
      percentage distance of the current close from that extreme.
      ``drawdown_from_Kd_high`` is the same formula as
      ``distance_pct_to_high_K`` under a second, spec-requested name (see
      ``MarketFeatures.drawdown_from_20d_high``'s own docstring note).
    - ``recent_structure``: see ``_recent_structure``.
    - ``rsi14``/``atr14`` = stockstats' ``rsi_14``/``atr_14`` columns
      (confirmed equal to stockstats' own unsuffixed ``rsi``/``atr``
      defaults); ``adx14`` = stockstats' ``adx`` column (stockstats does not
      expose an ``adx_14``-suffixed variant; its default window is the
      standard 14) -- avoids adding a new TA-library dependency.
      ``atr14_pct`` = ``atr14 / close * 100``.
    - ``realized_vol_N`` = standard deviation of daily simple returns over
      the trailing N sessions, in percentage points -- deliberately NOT
      annualized (kept simple, see fork spec: "metodologia simple y
      documentada").
    - ``volume_ratio_20`` = ``last_volume / mean(volume, 20)``;
      ``volume_zscore_20`` = ``(last_volume - mean) / std(volume, 20)``.
      Both ``None`` when the symbol has no usable Volume column.

    Uses ``load_ohlcv(..., fill_gaps=True)`` (a continuous series, needed
    for stable rolling-window math -- the same choice the rest of this
    project's indicator computation already makes), then re-applies the
    ``<= analysis_date`` cutoff defensively before computing anything, so no
    row after ``analysis_date`` can influence any field even if a caller's
    mocked/cached data included one (see the leakage tests in
    ``tests/test_run2_market_data.py``).
    """
    data = load_ohlcv(ticker, analysis_date, fill_gaps=True)
    if data is None or data.empty:
        raise ValueError(f"No OHLCV data available for {ticker}.")
    df = data.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df[df["Date"] <= pd.to_datetime(analysis_date)].sort_values("Date").reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No OHLCV rows on or before {analysis_date} for {ticker}.")

    close, high, low = df["Close"], df["High"], df["Low"]
    volume = df["Volume"] if "Volume" in df.columns else None
    n = len(df)
    last_close = float(close.iloc[-1])

    fields: dict[str, Any] = {"as_of": analysis_date, "sessions_available": n}

    for w in RETURN_WINDOWS:
        fields[f"return_{w}d"] = (
            _round((last_close / float(close.iloc[-1 - w]) - 1.0) * 100.0) if n > w else None
        )

    sma_series: dict[int, pd.Series] = {}
    for w in SMA_WINDOWS:
        if n >= w:
            series = close.rolling(w).mean()
            sma_series[w] = series
            sma_val = float(series.iloc[-1])
            fields[f"sma{w}"] = _round(sma_val)
            fields[f"distance_pct_to_sma{w}"] = _round((last_close - sma_val) / sma_val * 100.0)
            fields[f"price_above_sma{w}"] = bool(last_close > sma_val)
        else:
            fields[f"sma{w}"] = None
            fields[f"distance_pct_to_sma{w}"] = None
            fields[f"price_above_sma{w}"] = None

    for w in SLOPE_SMA_WINDOWS:
        series = sma_series.get(w)
        if series is not None and len(series.dropna()) > SLOPE_LOOKBACK_SESSIONS:
            now_val = series.iloc[-1]
            prior_val = series.iloc[-1 - SLOPE_LOOKBACK_SESSIONS]
            fields[f"slope_sma{w}"] = (
                _round((now_val - prior_val) / prior_val * 100.0) if prior_val else None
            )
        else:
            fields[f"slope_sma{w}"] = None

    for w in STRUCTURE_WINDOWS:
        if n >= w:
            rolling_high = float(high.rolling(w).max().iloc[-1])
            rolling_low = float(low.rolling(w).min().iloc[-1])
            distance_to_high = _round((last_close - rolling_high) / rolling_high * 100.0)
            distance_to_low = _round((last_close - rolling_low) / rolling_low * 100.0)
            fields[f"rolling_high_{w}"] = _round(rolling_high)
            fields[f"rolling_low_{w}"] = _round(rolling_low)
            fields[f"distance_pct_to_high_{w}"] = distance_to_high
            fields[f"distance_pct_to_low_{w}"] = distance_to_low
            fields[f"drawdown_from_{w}d_high"] = distance_to_high
        else:
            fields[f"rolling_high_{w}"] = None
            fields[f"rolling_low_{w}"] = None
            fields[f"distance_pct_to_high_{w}"] = None
            fields[f"distance_pct_to_low_{w}"] = None
            fields[f"drawdown_from_{w}d_high"] = None

    fields["recent_structure"] = _recent_structure(high, low, n)

    stock_df = wrap(df.copy())
    fields["rsi14"] = _safe_indicator(stock_df, "rsi_14", n, min_sessions=15)
    fields["adx14"] = _safe_indicator(stock_df, "adx", n, min_sessions=28)
    atr14 = _safe_indicator(stock_df, "atr_14", n, min_sessions=15)
    fields["atr14"] = atr14
    fields["atr14_pct"] = _round(atr14 / last_close * 100.0) if atr14 is not None else None

    returns = close.pct_change().dropna()
    for w in REALIZED_VOL_WINDOWS:
        fields[f"realized_vol_{w}"] = (
            _round(float(returns.tail(w).std()) * 100.0) if len(returns) >= w else None
        )

    if volume is not None and volume.notna().any() and n >= VOLUME_WINDOW:
        vol_mean = float(volume.rolling(VOLUME_WINDOW).mean().iloc[-1])
        vol_std = float(volume.rolling(VOLUME_WINDOW).std().iloc[-1])
        last_vol = float(volume.iloc[-1])
        fields["volume_ratio_20"] = _round(last_vol / vol_mean) if vol_mean else None
        fields["volume_zscore_20"] = _round((last_vol - vol_mean) / vol_std) if vol_std else None
    else:
        fields["volume_ratio_20"] = None
        fields["volume_zscore_20"] = None

    return MarketFeatures(**fields)
