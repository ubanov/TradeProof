"""Offline First Barrier evaluator (fork README: "run2" First Barrier --
"Evaluador offline"). Separate module from the inference pipeline -- it MAY
look at data after ``analysis_date`` because it only ever runs after the
fact. Barrier detection must use High/Low, never Close alone; intraday order
within a single session is never guessed (``AMBIGUOUS``). See
``tests/test_run2_cli.py::test_4b`` for the import-boundary proof that the
inference side never reaches into this module.
"""

from __future__ import annotations

import pandas as pd
import pytest

import tradingagents.run2.evaluator as evaluator


def _sessions(start="2026-01-08", n=40):
    return pd.bdate_range(start=start, periods=n)


def _frame(dates, *, high, low, close, open_=None):
    n = len(dates)
    return pd.DataFrame({
        "Date": dates,
        "Open": open_ if open_ is not None else close,
        "High": high, "Low": low, "Close": close,
        "Volume": [1_000_000.0] * n,
    })


REFERENCE_PRICE = 100.0
BARRIER_PCT = 5.0
HORIZON = 40


@pytest.mark.unit
def test_1_upper_barrier_hit_first_evaluates_long(monkeypatch):
    dates = _sessions(n=HORIZON)
    # Flat for a few sessions, then the upper barrier (105) is pierced on
    # session 10 via High, well before any approach to the lower barrier (95).
    high = [101.0] * 9 + [106.0] + [101.0] * (HORIZON - 10)
    low = [99.0] * HORIZON
    close = [100.0] * HORIZON
    df = _frame(dates, high=high, low=low, close=close)
    monkeypatch.setattr(evaluator, "load_ohlcv", lambda *a, **k: df)

    result = evaluator.evaluate_first_barrier(
        ticker="NVDA", analysis_date="2026-01-07", reference_price=REFERENCE_PRICE,
        horizon_sessions=HORIZON, barrier_pct=BARRIER_PCT,
    )

    assert result["first_barrier"] == "LONG"
    assert result["sessions_to_first_barrier"] == 10
    assert result["upper_barrier_hit_date"] == dates[9].strftime("%Y-%m-%d")
    assert result["lower_barrier_hit_date"] is None
    assert result["upper_barrier"] == pytest.approx(105.0)
    assert result["lower_barrier"] == pytest.approx(95.0)


@pytest.mark.unit
def test_2_lower_barrier_hit_first_evaluates_short(monkeypatch):
    dates = _sessions(n=HORIZON)
    high = [101.0] * HORIZON
    low = [99.0] * 4 + [94.0] + [99.0] * (HORIZON - 5)
    close = [100.0] * HORIZON
    df = _frame(dates, high=high, low=low, close=close)
    monkeypatch.setattr(evaluator, "load_ohlcv", lambda *a, **k: df)

    result = evaluator.evaluate_first_barrier(
        ticker="NVDA", analysis_date="2026-01-07", reference_price=REFERENCE_PRICE,
        horizon_sessions=HORIZON, barrier_pct=BARRIER_PCT,
    )

    assert result["first_barrier"] == "SHORT"
    assert result["sessions_to_first_barrier"] == 5
    assert result["lower_barrier_hit_date"] == dates[4].strftime("%Y-%m-%d")
    assert result["upper_barrier_hit_date"] is None


@pytest.mark.unit
def test_3_neither_barrier_hit_within_horizon_is_no_position(monkeypatch):
    dates = _sessions(n=HORIZON)
    high = [102.0] * HORIZON  # never reaches 105
    low = [98.0] * HORIZON  # never reaches 95
    close = [100.0] * HORIZON
    df = _frame(dates, high=high, low=low, close=close)
    monkeypatch.setattr(evaluator, "load_ohlcv", lambda *a, **k: df)

    result = evaluator.evaluate_first_barrier(
        ticker="NVDA", analysis_date="2026-01-07", reference_price=REFERENCE_PRICE,
        horizon_sessions=HORIZON, barrier_pct=BARRIER_PCT,
    )

    assert result["first_barrier"] == "NO_POSITION"
    assert result["sessions_to_first_barrier"] is None
    assert result["upper_barrier_hit_date"] is None
    assert result["lower_barrier_hit_date"] is None
    assert result["available_future_sessions"] == HORIZON
    assert result["horizon_complete"] is True


@pytest.mark.unit
def test_4_both_barriers_hit_same_session_is_ambiguous(monkeypatch):
    dates = _sessions(n=HORIZON)
    high = [101.0] * 6 + [106.0] + [101.0] * (HORIZON - 7)
    low = [99.0] * 6 + [94.0] + [99.0] * (HORIZON - 7)  # same session (index 6) as the high breach
    close = [100.0] * HORIZON
    df = _frame(dates, high=high, low=low, close=close)
    monkeypatch.setattr(evaluator, "load_ohlcv", lambda *a, **k: df)

    result = evaluator.evaluate_first_barrier(
        ticker="NVDA", analysis_date="2026-01-07", reference_price=REFERENCE_PRICE,
        horizon_sessions=HORIZON, barrier_pct=BARRIER_PCT,
    )

    assert result["first_barrier"] == "AMBIGUOUS"
    assert result["sessions_to_first_barrier"] == 7
    assert result["upper_barrier_hit_date"] == dates[6].strftime("%Y-%m-%d")
    assert result["lower_barrier_hit_date"] == dates[6].strftime("%Y-%m-%d")


@pytest.mark.unit
def test_5_barriers_are_evaluated_with_high_low_not_close_alone(monkeypatch):
    """A session whose Close never approaches the barrier, but whose High
    pierces it intraday, must still count as a hit (fork spec: "No
    determinar las barreras usando unicamente Close")."""
    dates = _sessions(n=HORIZON)
    high = [101.0] * 4 + [106.0] + [101.0] * (HORIZON - 5)
    low = [99.0] * HORIZON
    close = [100.5] * HORIZON  # close stays well inside both barriers throughout
    df = _frame(dates, high=high, low=low, close=close)
    monkeypatch.setattr(evaluator, "load_ohlcv", lambda *a, **k: df)

    result = evaluator.evaluate_first_barrier(
        ticker="NVDA", analysis_date="2026-01-07", reference_price=REFERENCE_PRICE,
        horizon_sessions=HORIZON, barrier_pct=BARRIER_PCT,
    )

    assert result["first_barrier"] == "LONG"
    assert result["sessions_to_first_barrier"] == 5


@pytest.mark.unit
def test_6_return_and_excursion_metrics_use_the_full_horizon_window(monkeypatch):
    dates = _sessions(n=HORIZON)
    high = [100.0 + i * 0.1 for i in range(HORIZON)]
    low = [95.0] * HORIZON
    close = [100.0 + i * 0.05 for i in range(HORIZON)]
    df = _frame(dates, high=high, low=low, close=close)
    monkeypatch.setattr(evaluator, "load_ohlcv", lambda *a, **k: df)

    result = evaluator.evaluate_first_barrier(
        ticker="NVDA", analysis_date="2026-01-07", reference_price=REFERENCE_PRICE,
        horizon_sessions=HORIZON, barrier_pct=BARRIER_PCT,
    )

    last_close = close[-1]
    max_high = max(high)
    min_low = min(low)
    assert result["return_at_horizon_pct"] == pytest.approx((last_close - 100.0) / 100.0 * 100.0)
    assert result["mfe_long_pct"] == pytest.approx((max_high - 100.0) / 100.0 * 100.0)
    assert result["mae_long_pct"] == pytest.approx((min_low - 100.0) / 100.0 * 100.0)
    assert result["mfe_short_pct"] == pytest.approx(-result["mae_long_pct"])
    assert result["mae_short_pct"] == pytest.approx(-result["mfe_long_pct"])
    # Documented-duplicate fields (same methodology as
    # MarketFeatures.drawdown_from_20d_high -- see evaluator.py docstring).
    assert result["max_high_pct"] == result["mfe_long_pct"]
    assert result["max_low_pct"] == result["mae_long_pct"]


@pytest.mark.unit
def test_7_truncated_horizon_with_no_barrier_touched_yet_is_incomplete_not_no_position(monkeypatch):
    """Evaluating very soon after analysis_date (fewer than horizon_sessions
    real trading days have elapsed yet), with neither barrier touched so
    far, must NOT be reported as the definitive "NO_POSITION" -- the outcome
    is simply not known yet (fork spec: "run2" First Barrier evaluator,
    INCOMPLETE label)."""
    dates = _sessions(n=31)  # fewer than HORIZON (40)
    high = [101.0] * 31
    low = [99.0] * 31
    close = [100.0] * 31
    df = _frame(dates, high=high, low=low, close=close)
    monkeypatch.setattr(evaluator, "load_ohlcv", lambda *a, **k: df)

    result = evaluator.evaluate_first_barrier(
        ticker="NVDA", analysis_date="2026-01-07", reference_price=REFERENCE_PRICE,
        horizon_sessions=HORIZON, barrier_pct=BARRIER_PCT,
    )

    assert result["sessions_evaluated"] == 31
    assert result["available_future_sessions"] == 31
    assert result["horizon_complete"] is False
    assert result["first_barrier"] == "INCOMPLETE"
    assert result["sessions_to_first_barrier"] is None


@pytest.mark.unit
def test_7b_incomplete_horizon_but_upper_touched_first_is_still_definitive_long(monkeypatch):
    dates = _sessions(n=31)
    high = [101.0] * 11 + [106.0] + [101.0] * (31 - 12)
    low = [99.0] * 31
    close = [100.0] * 31
    df = _frame(dates, high=high, low=low, close=close)
    monkeypatch.setattr(evaluator, "load_ohlcv", lambda *a, **k: df)

    result = evaluator.evaluate_first_barrier(
        ticker="NVDA", analysis_date="2026-01-07", reference_price=REFERENCE_PRICE,
        horizon_sessions=HORIZON, barrier_pct=BARRIER_PCT,
    )

    assert result["horizon_complete"] is False
    assert result["available_future_sessions"] == 31
    assert result["first_barrier"] == "LONG"
    assert result["sessions_to_first_barrier"] == 12


@pytest.mark.unit
def test_7c_incomplete_horizon_but_lower_touched_first_is_still_definitive_short(monkeypatch):
    dates = _sessions(n=31)
    high = [101.0] * 31
    low = [99.0] * 11 + [94.0] + [99.0] * (31 - 12)
    close = [100.0] * 31
    df = _frame(dates, high=high, low=low, close=close)
    monkeypatch.setattr(evaluator, "load_ohlcv", lambda *a, **k: df)

    result = evaluator.evaluate_first_barrier(
        ticker="NVDA", analysis_date="2026-01-07", reference_price=REFERENCE_PRICE,
        horizon_sessions=HORIZON, barrier_pct=BARRIER_PCT,
    )

    assert result["horizon_complete"] is False
    assert result["first_barrier"] == "SHORT"
    assert result["sessions_to_first_barrier"] == 12


@pytest.mark.unit
def test_7d_incomplete_horizon_with_both_barriers_touched_same_session_is_ambiguous(monkeypatch):
    dates = _sessions(n=31)
    high = [101.0] * 11 + [106.0] + [101.0] * (31 - 12)
    low = [99.0] * 11 + [94.0] + [99.0] * (31 - 12)
    close = [100.0] * 31
    df = _frame(dates, high=high, low=low, close=close)
    monkeypatch.setattr(evaluator, "load_ohlcv", lambda *a, **k: df)

    result = evaluator.evaluate_first_barrier(
        ticker="NVDA", analysis_date="2026-01-07", reference_price=REFERENCE_PRICE,
        horizon_sessions=HORIZON, barrier_pct=BARRIER_PCT,
    )

    assert result["horizon_complete"] is False
    assert result["first_barrier"] == "AMBIGUOUS"
    assert result["sessions_to_first_barrier"] == 12


@pytest.mark.unit
def test_7e_available_future_sessions_reports_everything_beyond_the_horizon_too(monkeypatch):
    """available_future_sessions is the UNCAPPED count of real future
    sessions -- it can exceed horizon_sessions, distinct from
    sessions_evaluated (which is capped at horizon_sessions)."""
    dates = _sessions(n=HORIZON + 10)
    high = [101.0] * (HORIZON + 10)
    low = [99.0] * (HORIZON + 10)
    close = [100.0] * (HORIZON + 10)
    df = _frame(dates, high=high, low=low, close=close)
    monkeypatch.setattr(evaluator, "load_ohlcv", lambda *a, **k: df)

    result = evaluator.evaluate_first_barrier(
        ticker="NVDA", analysis_date="2026-01-07", reference_price=REFERENCE_PRICE,
        horizon_sessions=HORIZON, barrier_pct=BARRIER_PCT,
    )

    assert result["available_future_sessions"] == HORIZON + 10
    assert result["sessions_evaluated"] == HORIZON
    assert result["horizon_complete"] is True
    assert result["first_barrier"] == "NO_POSITION"


@pytest.mark.unit
def test_8_evaluate_result_reads_ticker_date_and_reference_price_from_a_result_dict(monkeypatch):
    dates = _sessions(n=HORIZON)
    high = [106.0] * HORIZON
    low = [99.0] * HORIZON
    close = [100.0] * HORIZON
    df = _frame(dates, high=high, low=low, close=close)
    monkeypatch.setattr(evaluator, "load_ohlcv", lambda *a, **k: df)

    fake_result = {
        "ticker": "NVDA", "analysis_date": "2026-01-07", "horizon_sessions": HORIZON,
        "barrier_pct": BARRIER_PCT, "price_reference": {"close": REFERENCE_PRICE},
    }
    evaluation = evaluator.evaluate_result(fake_result)
    assert evaluation["first_barrier"] == "LONG"
    assert evaluation["ticker"] == "NVDA"
