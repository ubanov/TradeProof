"""Simple, untuned, deterministic directional rules for the offline
quantitative benchmark (see ``quant_benchmark.py`` and fork README:
"Experimental research"). No LLM, no optimization, no parameter search --
every threshold here is a frozen, hand-picked constant, exactly as the fork
spec asks ("No queremos optimizacion", "paragrams iniciales estan
congelados").

Every function returns ``{"prediction": "LONG" | "SHORT" | "NO_PREDICTION",
"reason": str}`` -- a DELIBERATELY different vocabulary from
``tradingagents.run2.schemas.BaselineResult``/``BarrierSignal``:
``NO_PREDICTION`` means "this rule declined to pick a direction" (an
abstention by the RULE), which must never be confused with
``NO_POSITION`` (the REAL market outcome when neither barrier was touched --
see ``evaluator.py``). ``wrap_baseline_result`` performs exactly that
translation for the two baselines reused from ``baselines.py``.

``momentum``/``moving_average`` are reused byte-for-byte from
``tradingagents.run2.baselines`` -- this module never redefines their logic,
only translates their output vocabulary for this benchmark's scoring.
``contrarian_momentum`` inverts ONLY `momentum_baseline`'s own LONG/SHORT
output; it is included because the 12-date development set documented in
the fork README's "Experimental research" section suggested a possible
reversal pattern -- those 12 dates are NOT an independent validation set for
that hypothesis, and this benchmark is the first attempt at an out-of-sample
check of it.
"""

from __future__ import annotations

from collections.abc import Callable

from tradingagents.run2.baselines import momentum_baseline, moving_average_baseline
from tradingagents.run2.schemas import BaselineResult, MarketFeatures

NO_PREDICTION = "NO_PREDICTION"

# RSI contrarian: a fixed, textbook overbought/oversold split. Deliberately
# not tuned against this (or any) dataset (fork spec: "NO buscar thresholds
# mejores").
RSI_OVERBOUGHT = 70.0
RSI_OVERSOLD = 30.0


def wrap_baseline_result(result: BaselineResult) -> dict[str, str]:
    """Translate a ``BaselineResult`` (``momentum_baseline``/
    ``moving_average_baseline``, reused unmodified) into this module's
    ``{"prediction", "reason"}`` shape -- its own ``NO_POSITION`` becomes
    ``NO_PREDICTION`` here (an abstention, in this benchmark's vocabulary),
    never the real evaluator's ``NO_POSITION`` market outcome."""
    prediction = NO_PREDICTION if result.action == "NO_POSITION" else result.action
    return {"prediction": prediction, "reason": result.reason}


def momentum(features: MarketFeatures) -> dict[str, str]:
    """Reuses ``tradingagents.run2.baselines.momentum_baseline`` exactly,
    unmodified -- see that function's own docstring for its rule."""
    return wrap_baseline_result(momentum_baseline(features))


def moving_average(features: MarketFeatures) -> dict[str, str]:
    """Reuses ``tradingagents.run2.baselines.moving_average_baseline``
    exactly, unmodified -- see that function's own docstring for its rule."""
    return wrap_baseline_result(moving_average_baseline(features))


def return20(features: MarketFeatures) -> dict[str, str]:
    """``return_20d > 0 -> LONG``; ``< 0 -> SHORT``; ``0`` or missing ->
    ``NO_PREDICTION``."""
    r = features.return_20d
    if not r:  # None or exactly 0.0
        return {"prediction": NO_PREDICTION, "reason": "return_20d is null or zero"}
    return {"prediction": "LONG" if r > 0 else "SHORT", "reason": f"return_20d={r:.4f}%"}


def return60(features: MarketFeatures) -> dict[str, str]:
    """Same rule as ``return20``, using ``return_60d``."""
    r = features.return_60d
    if not r:
        return {"prediction": NO_PREDICTION, "reason": "return_60d is null or zero"}
    return {"prediction": "LONG" if r > 0 else "SHORT", "reason": f"return_60d={r:.4f}%"}


def contrarian_return20(features: MarketFeatures) -> dict[str, str]:
    """Exact sign-inverse of ``return20``: ``return_20d > 0 -> SHORT``;
    ``< 0 -> LONG``."""
    r = features.return_20d
    if not r:
        return {"prediction": NO_PREDICTION, "reason": "return_20d is null or zero"}
    return {"prediction": "SHORT" if r > 0 else "LONG", "reason": f"contrarian return_20d={r:.4f}%"}


def contrarian_return60(features: MarketFeatures) -> dict[str, str]:
    """Exact sign-inverse of ``return60``."""
    r = features.return_60d
    if not r:
        return {"prediction": NO_PREDICTION, "reason": "return_60d is null or zero"}
    return {"prediction": "SHORT" if r > 0 else "LONG", "reason": f"contrarian return_60d={r:.4f}%"}


def contrarian_momentum(features: MarketFeatures) -> dict[str, str]:
    """Inverts ONLY ``momentum_baseline``'s own LONG/SHORT output
    (``LONG -> SHORT``, ``SHORT -> LONG``); its ``NO_POSITION`` stays an
    abstention (``NO_PREDICTION``), never flipped into a directional call."""
    base = momentum_baseline(features)
    flipped = {"LONG": "SHORT", "SHORT": "LONG", "NO_POSITION": NO_PREDICTION}[base.action]
    return {"prediction": flipped, "reason": f"inverted momentum_baseline: {base.reason}"}


def rsi_contrarian(features: MarketFeatures) -> dict[str, str]:
    """Fixed overbought/oversold rule, no tuning: ``rsi14 >= 70 -> SHORT``;
    ``rsi14 <= 30 -> LONG``; otherwise (including missing) ->
    ``NO_PREDICTION``."""
    rsi = features.rsi14
    if rsi is None:
        return {"prediction": NO_PREDICTION, "reason": "rsi14 is null"}
    if rsi >= RSI_OVERBOUGHT:
        return {"prediction": "SHORT", "reason": f"rsi14={rsi:.2f} >= {RSI_OVERBOUGHT:.0f} (overbought)"}
    if rsi <= RSI_OVERSOLD:
        return {"prediction": "LONG", "reason": f"rsi14={rsi:.2f} <= {RSI_OVERSOLD:.0f} (oversold)"}
    return {"prediction": NO_PREDICTION, "reason": f"rsi14={rsi:.2f} in the neutral zone"}


# Stable, ordered registry -- the single source both the CSV/summary builder
# in quant_benchmark.py and this module's own tests iterate over, so adding a
# baseline here never requires a second list to stay in sync.
QUANT_BASELINES: dict[str, Callable[[MarketFeatures], dict[str, str]]] = {
    "momentum": momentum,
    "moving_average": moving_average,
    "return20": return20,
    "return60": return60,
    "contrarian20": contrarian_return20,
    "contrarian60": contrarian_return60,
    "contrarian_momentum": contrarian_momentum,
    "rsi_contrarian": rsi_contrarian,
}
