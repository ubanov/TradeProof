"""run2's market-only, point-in-time data fetch (fork README: "run2" V1
restriction: no information from after analysis_date may be used).
``fetch_price_reference``/``fetch_market_context`` re-apply the
``<= analysis_date`` cutoff defensively on top of ``load_ohlcv``'s own
filtering -- these tests prove that defense actually holds even if
``load_ohlcv`` (mocked here) ever returned a future row.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import tradingagents.run2.market_data as market_data


def _synthetic_ohlcv(n_days: int, *, seed: int = 0, start: str = "2024-01-01") -> pd.DataFrame:
    """A deterministic (seeded), reproducible OHLCV frame -- no network call,
    no cached fixture file, just enough realism (drift + noise, a plausible
    High/Low spread, non-zero Volume) for SMA/RSI/ATR/ADX to compute
    non-degenerate values."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=n_days)
    close = 100.0 + np.cumsum(0.05 + rng.normal(0, 1.0, n_days))
    high = close + rng.uniform(0.1, 1.0, n_days)
    low = close - rng.uniform(0.1, 1.0, n_days)
    open_ = close + rng.normal(0, 0.3, n_days)
    volume = rng.integers(1_000_000, 2_000_000, n_days).astype(float)
    return pd.DataFrame({
        "Date": dates, "Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume,
    })


def _ohlcv_with_future_row():
    return pd.DataFrame({
        "Date": ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"],
        "Open": [100.0, 101.0, 102.0, 999.0],
        "High": [101.0, 102.0, 103.0, 1000.0],
        "Low": [99.0, 100.0, 101.0, 998.0],
        "Close": [100.5, 101.5, 102.5, 999.5],
        "Volume": [1000, 1100, 1200, 9999],
    })


@pytest.mark.unit
def test_1_price_reference_never_uses_a_row_after_analysis_date(monkeypatch):
    monkeypatch.setattr(market_data, "load_ohlcv", lambda *a, **k: _ohlcv_with_future_row())

    ref = market_data.fetch_price_reference("NVDA", "2026-01-07")

    assert ref["reference_date"] == "2026-01-07"
    assert ref["close"] == 102.5
    assert ref["close"] != 999.5


@pytest.mark.unit
def test_2_price_reference_raises_when_no_row_on_or_before_the_date(monkeypatch):
    future_only = pd.DataFrame({
        "Date": ["2026-02-01"], "Open": [1.0], "High": [1.0], "Low": [1.0],
        "Close": [1.0], "Volume": [1],
    })
    monkeypatch.setattr(market_data, "load_ohlcv", lambda *a, **k: future_only)

    with pytest.raises(ValueError):
        market_data.fetch_price_reference("NVDA", "2026-01-07")


@pytest.mark.unit
def test_3_price_reference_raises_on_empty_data(monkeypatch):
    monkeypatch.setattr(market_data, "load_ohlcv", lambda *a, **k: pd.DataFrame())

    with pytest.raises(ValueError):
        market_data.fetch_price_reference("NVDA", "2026-01-07")


@pytest.mark.unit
def test_4_fetch_market_context_delegates_to_the_verified_snapshot_builder(monkeypatch):
    calls = {}

    def _fake_snapshot(symbol, curr_date, look_back_days=30, indicators=None):
        calls["args"] = (symbol, curr_date, look_back_days)
        return "## snapshot markdown"

    monkeypatch.setattr(market_data, "build_verified_market_snapshot", _fake_snapshot)

    text = market_data.fetch_market_context("NVDA", "2026-01-07", look_back_days=45)

    assert text == "## snapshot markdown"
    assert calls["args"] == ("NVDA", "2026-01-07", 45)


# ---------------------------------------------------------------------------
# Deterministic market feature pack (fork spec: "run2" hardening).
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_5_return_and_sma_formulas_match_manual_computation(monkeypatch):
    df = _synthetic_ohlcv(260)
    monkeypatch.setattr(market_data, "load_ohlcv", lambda *a, **k: df.copy())
    analysis_date = df["Date"].iloc[-1].strftime("%Y-%m-%d")

    features = market_data.compute_market_features("NVDA", analysis_date)

    close = df["Close"]
    expected_return_20d = (close.iloc[-1] / close.iloc[-1 - 20] - 1.0) * 100.0
    assert features.return_20d == pytest.approx(expected_return_20d, abs=1e-3)

    expected_sma50 = close.rolling(50).mean().iloc[-1]
    assert features.sma50 == pytest.approx(expected_sma50, abs=1e-3)
    expected_distance = (close.iloc[-1] - expected_sma50) / expected_sma50 * 100.0
    assert features.distance_pct_to_sma50 == pytest.approx(expected_distance, abs=1e-3)
    assert features.price_above_sma50 == bool(close.iloc[-1] > expected_sma50)

    sma20_series = close.rolling(20).mean()
    expected_slope20 = (
        (sma20_series.iloc[-1] - sma20_series.iloc[-11]) / sma20_series.iloc[-11] * 100.0
    )
    assert features.slope_sma20 == pytest.approx(expected_slope20, abs=1e-3)


@pytest.mark.unit
def test_6_realized_vol_matches_manual_stdev_of_daily_returns(monkeypatch):
    df = _synthetic_ohlcv(260)
    monkeypatch.setattr(market_data, "load_ohlcv", lambda *a, **k: df.copy())
    analysis_date = df["Date"].iloc[-1].strftime("%Y-%m-%d")

    features = market_data.compute_market_features("NVDA", analysis_date)

    returns = df["Close"].pct_change().dropna()
    expected = float(returns.tail(20).std()) * 100.0
    assert features.realized_vol_20 == pytest.approx(expected, abs=1e-3)


@pytest.mark.unit
def test_7_structure_and_drawdown_formulas_are_self_consistent(monkeypatch):
    df = _synthetic_ohlcv(260)
    monkeypatch.setattr(market_data, "load_ohlcv", lambda *a, **k: df.copy())
    analysis_date = df["Date"].iloc[-1].strftime("%Y-%m-%d")

    features = market_data.compute_market_features("NVDA", analysis_date)

    high, low, close = df["High"], df["Low"], df["Close"]
    for w in (20, 60, 120):
        expected_high = float(high.rolling(w).max().iloc[-1])
        expected_low = float(low.rolling(w).min().iloc[-1])
        assert getattr(features, f"rolling_high_{w}") == pytest.approx(expected_high, abs=1e-3)
        assert getattr(features, f"rolling_low_{w}") == pytest.approx(expected_low, abs=1e-3)
        expected_distance_high = (close.iloc[-1] - expected_high) / expected_high * 100.0
        assert getattr(features, f"distance_pct_to_high_{w}") == pytest.approx(expected_distance_high, abs=1e-3)
        # drawdown_from_Nd_high is documented as the same formula as
        # distance_pct_to_high_N under this methodology -- see
        # MarketFeatures.drawdown_from_20d_high's docstring note.
        assert getattr(features, f"drawdown_from_{w}d_high") == getattr(features, f"distance_pct_to_high_{w}")


@pytest.mark.unit
def test_8_insufficient_history_yields_none_not_fabricated_values(monkeypatch):
    df = _synthetic_ohlcv(10)
    monkeypatch.setattr(market_data, "load_ohlcv", lambda *a, **k: df.copy())
    analysis_date = df["Date"].iloc[-1].strftime("%Y-%m-%d")

    features = market_data.compute_market_features("NVDA", analysis_date)

    assert features.sma20 is None
    assert features.sma200 is None
    assert features.return_20d is None
    assert features.rsi14 is None
    assert features.adx14 is None
    assert features.recent_structure is None
    # Still computes whatever a 10-session history actually supports.
    assert features.return_5d is not None
    assert features.sessions_available == 10


@pytest.mark.unit
def test_9_compute_market_features_raises_when_no_row_on_or_before_the_date(monkeypatch):
    future_only = pd.DataFrame({
        "Date": pd.bdate_range("2026-02-01", periods=5),
        "Open": [1.0] * 5, "High": [1.0] * 5, "Low": [1.0] * 5, "Close": [1.0] * 5, "Volume": [1] * 5,
    })
    monkeypatch.setattr(market_data, "load_ohlcv", lambda *a, **k: future_only)

    with pytest.raises(ValueError):
        market_data.compute_market_features("NVDA", "2026-01-07")


# ---------------------------------------------------------------------------
# Leakage tests (fork spec: "run2" hardening, section 6) -- the feature pack
# must be byte-for-byte identical whether or not the underlying data source
# includes rows after analysis_date, covering means, highs/lows, returns,
# ATR, volume, and drawdown all at once (they all live on one MarketFeatures
# object, so one full-object comparison proves all of them simultaneously).
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_10_market_features_unaffected_by_many_wildly_different_future_rows(monkeypatch):
    df = _synthetic_ohlcv(260)
    analysis_date = df["Date"].iloc[-1].strftime("%Y-%m-%d")

    monkeypatch.setattr(market_data, "load_ohlcv", lambda *a, **k: df.copy())
    without_future = market_data.compute_market_features("NVDA", analysis_date)

    future_dates = pd.bdate_range(start=df["Date"].iloc[-1] + pd.Timedelta(days=1), periods=30)
    future_rows = pd.DataFrame({
        "Date": future_dates,
        "Open": [100_000.0] * 30, "High": [200_000.0] * 30, "Low": [50_000.0] * 30,
        "Close": [150_000.0] * 30, "Volume": [999_999_999.0] * 30,
    })
    padded = pd.concat([df, future_rows], ignore_index=True)
    monkeypatch.setattr(market_data, "load_ohlcv", lambda *a, **k: padded.copy())
    with_future = market_data.compute_market_features("NVDA", analysis_date)

    assert with_future.model_dump() == without_future.model_dump()
    # Explicit, spec-named checks on top of the full-object comparison:
    assert with_future.sma50 == without_future.sma50  # medias
    assert with_future.sma200 == without_future.sma200
    assert with_future.rolling_high_60 == without_future.rolling_high_60  # highs/lows
    assert with_future.rolling_low_60 == without_future.rolling_low_60
    assert with_future.return_20d == without_future.return_20d  # retornos
    assert with_future.atr14 == without_future.atr14  # ATR
    assert with_future.volume_ratio_20 == without_future.volume_ratio_20  # volumen
    assert with_future.volume_zscore_20 == without_future.volume_zscore_20
    assert with_future.drawdown_from_60d_high == without_future.drawdown_from_60d_high  # drawdown


@pytest.mark.unit
def test_11_market_features_unaffected_by_a_single_future_row(monkeypatch):
    df = _synthetic_ohlcv(260)
    analysis_date = df["Date"].iloc[-1].strftime("%Y-%m-%d")

    monkeypatch.setattr(market_data, "load_ohlcv", lambda *a, **k: df.copy())
    without_future = market_data.compute_market_features("NVDA", analysis_date)

    one_future_row = pd.DataFrame({
        "Date": [df["Date"].iloc[-1] + pd.Timedelta(days=1)],
        "Open": [1_000_000.0], "High": [1_000_000.0], "Low": [1_000_000.0],
        "Close": [1_000_000.0], "Volume": [1.0],
    })
    padded = pd.concat([df, one_future_row], ignore_index=True)
    monkeypatch.setattr(market_data, "load_ohlcv", lambda *a, **k: padded.copy())
    with_future = market_data.compute_market_features("NVDA", analysis_date)

    assert with_future.model_dump() == without_future.model_dump()
