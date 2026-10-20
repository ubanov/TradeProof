"""Deterministic, untuned baselines (fork spec: "run2" hardening,
"Baselines deterministas gratuitos") -- for later, separate comparison
against run2's LLM-driven Final Decision only.

Each function is a fixed, documented rule with no free parameters to tune
and no optimization against historical outcomes -- the fork spec is
explicit: "No quiero optimizacion ni tuning. Solo resultados." Both take an
already-computed ``MarketFeatures`` (see ``market_data.compute_market_features``)
and are pure, side-effect-free functions. Neither is ever passed into the
Regime Analyzer or First Barrier Signal Analyst's prompt -- ``pipeline.py``
stores their output directly under ``result["baselines"]`` and nothing else
reads it.
"""

from __future__ import annotations

from tradingagents.run2.schemas import BaselineResult, MarketFeatures


def momentum_baseline(features: MarketFeatures) -> BaselineResult:
    """LONG when the 20- and 60-session returns are both positive and the
    close is above its 50-session average; SHORT when both are negative and
    the close is below it; NO_POSITION otherwise (including when there is
    not enough history for return_20d/return_60d/price_above_sma50)."""
    r20, r60 = features.return_20d, features.return_60d
    above_sma50 = features.price_above_sma50

    if r20 is None or r60 is None or above_sma50 is None:
        return BaselineResult(action="NO_POSITION", reason="insufficient_history")
    if r20 > 0 and r60 > 0 and above_sma50:
        return BaselineResult(
            action="LONG",
            reason=f"return_20d={r20:.2f}% and return_60d={r60:.2f}% both positive, price above SMA50",
        )
    if r20 < 0 and r60 < 0 and not above_sma50:
        return BaselineResult(
            action="SHORT",
            reason=f"return_20d={r20:.2f}% and return_60d={r60:.2f}% both negative, price below SMA50",
        )
    return BaselineResult(action="NO_POSITION", reason="momentum and trend not aligned")


def moving_average_baseline(features: MarketFeatures) -> BaselineResult:
    """LONG when price > SMA50 > SMA200 (classic bullish moving-average
    stack); SHORT when price < SMA50 < SMA200 (classic bearish stack);
    NO_POSITION otherwise (including insufficient history for SMA50/SMA200)."""
    sma50, sma200 = features.sma50, features.sma200
    above_50, above_200 = features.price_above_sma50, features.price_above_sma200

    if sma50 is None or sma200 is None or above_50 is None or above_200 is None:
        return BaselineResult(action="NO_POSITION", reason="insufficient_history")
    if above_50 and above_200 and sma50 > sma200:
        return BaselineResult(
            action="LONG", reason=f"price above SMA50 ({sma50:.2f}) above SMA200 ({sma200:.2f})",
        )
    if not above_50 and not above_200 and sma50 < sma200:
        return BaselineResult(
            action="SHORT", reason=f"price below SMA50 ({sma50:.2f}) below SMA200 ({sma200:.2f})",
        )
    return BaselineResult(action="NO_POSITION", reason="moving averages not aligned")
