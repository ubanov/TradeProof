"""Deterministic, untuned baselines (fork spec: "run2" hardening,
"Baselines deterministas gratuitos") -- fixed rules, no parameters to tune,
never fed into any LLM prompt (see ``pipeline.py``: they are stored under
``result["baselines"]`` only).
"""

from __future__ import annotations

import pytest

from tradingagents.run2.baselines import momentum_baseline, moving_average_baseline
from tradingagents.run2.schemas import MarketFeatures


def _features(**overrides):
    base = {"as_of": "2026-01-07", "sessions_available": 300}
    base.update(overrides)
    return MarketFeatures(**base)


@pytest.mark.unit
def test_1_momentum_baseline_long_when_both_returns_positive_and_above_sma50():
    f = _features(return_20d=5.0, return_60d=10.0, price_above_sma50=True)
    result = momentum_baseline(f)
    assert result.action == "LONG"


@pytest.mark.unit
def test_2_momentum_baseline_short_when_both_returns_negative_and_below_sma50():
    f = _features(return_20d=-5.0, return_60d=-10.0, price_above_sma50=False)
    result = momentum_baseline(f)
    assert result.action == "SHORT"


@pytest.mark.unit
def test_3_momentum_baseline_no_position_when_mixed():
    f = _features(return_20d=5.0, return_60d=-10.0, price_above_sma50=True)
    result = momentum_baseline(f)
    assert result.action == "NO_POSITION"


@pytest.mark.unit
def test_4_momentum_baseline_no_position_when_trend_not_aligned():
    f = _features(return_20d=5.0, return_60d=10.0, price_above_sma50=False)
    result = momentum_baseline(f)
    assert result.action == "NO_POSITION"


@pytest.mark.unit
def test_5_momentum_baseline_no_position_on_insufficient_history():
    f = _features()  # return_20d/return_60d/price_above_sma50 all None
    result = momentum_baseline(f)
    assert result.action == "NO_POSITION"
    assert result.reason == "insufficient_history"


@pytest.mark.unit
def test_6_moving_average_baseline_long_on_bullish_stack():
    f = _features(sma50=110.0, sma200=100.0, price_above_sma50=True, price_above_sma200=True)
    result = moving_average_baseline(f)
    assert result.action == "LONG"


@pytest.mark.unit
def test_7_moving_average_baseline_short_on_bearish_stack():
    f = _features(sma50=90.0, sma200=100.0, price_above_sma50=False, price_above_sma200=False)
    result = moving_average_baseline(f)
    assert result.action == "SHORT"


@pytest.mark.unit
def test_8_moving_average_baseline_no_position_when_not_aligned():
    f = _features(sma50=110.0, sma200=100.0, price_above_sma50=True, price_above_sma200=False)
    result = moving_average_baseline(f)
    assert result.action == "NO_POSITION"


@pytest.mark.unit
def test_9_moving_average_baseline_no_position_on_insufficient_history():
    f = _features()
    result = moving_average_baseline(f)
    assert result.action == "NO_POSITION"
    assert result.reason == "insufficient_history"


@pytest.mark.unit
def test_10_baselines_never_return_an_action_outside_long_short_no_position():
    for f in (
        _features(return_20d=5.0, return_60d=10.0, price_above_sma50=True),
        _features(sma50=90.0, sma200=100.0, price_above_sma50=False, price_above_sma200=False),
        _features(),
    ):
        assert momentum_baseline(f).action in ("LONG", "SHORT", "NO_POSITION")
        assert moving_average_baseline(f).action in ("LONG", "SHORT", "NO_POSITION")
