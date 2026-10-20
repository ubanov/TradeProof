"""Offline First Barrier evaluator -- answers, AFTER THE FACT, which barrier
(+``barrier_pct``%/-``barrier_pct``% from the reference price) was actually
reached first within ``horizon_sessions`` trading sessions of
``analysis_date``.

Deliberately a SEPARATE module from the inference pipeline
(``pipeline.py``/``stages.py``/``market_data.py``): this is the ONE place in
run2 allowed to look at data after ``analysis_date``, because it only ever
runs once that much time has genuinely passed. The inference pipeline must
NEVER import this module (enforced by
``tests/test_run2_cli.py``'s import-boundary test) and this module must
NEVER be called from inside a live analysis run -- it is a standalone
utility, invoked later, by hand or by a separate offline script, against an
already-written ``result.json``.

Barrier detection uses High/Low, never Close alone (fork spec: "No
determinar las barreras usando unicamente Close"): a session's Close can sit
comfortably inside both barriers while its High or Low already pierced one
of them intraday. Close is used only for ``return_at_horizon_pct``, which is
a distinct question ("where did the price end up") from "which barrier was
hit first".

Intraday order within a single session is NOT inferable from daily OHLC bars
-- a session whose High and Low both cross their respective barriers is
reported as ``"AMBIGUOUS"``, never guessed.

A fifth label, ``"INCOMPLETE"``, covers the case where this evaluator runs
before ``horizon_sessions`` have actually elapsed yet (e.g. evaluating 31
real trading sessions after ``analysis_date`` when ``horizon_sessions`` is
40): if neither barrier has been touched in the sessions available so far,
the outcome is genuinely not yet known and must not be reported as
``"NO_POSITION"`` (a NO_POSITION label means the full horizon ran its
course with neither barrier touched -- a materially different, and
definitive, finding). LONG/SHORT/AMBIGUOUS are resolved as soon as the
triggering session is reached, regardless of whether the full horizon has
elapsed -- once a barrier is reached, the order is settled for good.

These REAL-MARKET-OUTCOME labels (``LONG``/``SHORT``/``NO_POSITION``/
``AMBIGUOUS``/``INCOMPLETE``) are a separate vocabulary from the LLM
predictor's own output space (``tradingagents.run2.schemas.BarrierSignal``),
which is a forced binary choice (LONG/SHORT only, see that module's
docstring) -- the market can genuinely produce a NO_POSITION/AMBIGUOUS/
INCOMPLETE outcome even though the predictor itself is never allowed to
predict one. ``classify_for_accuracy``/``score_prediction`` below bridge the
two vocabularies for later accuracy analysis.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from tradingagents.dataflows.vendors.yahoo.ohlcv import load_ohlcv
from tradingagents.run2.storage import load_result


def _forward_window(ticker: str, analysis_date: str) -> pd.DataFrame:
    """Every available trading session strictly AFTER ``analysis_date`` --
    UNCAPPED, so the caller can tell how many real future sessions actually
    exist (``available_future_sessions``) before deciding whether the
    horizon has fully elapsed. Fetches "as of today" (the widest
    ``load_ohlcv`` cutoff that still returns only genuinely past/settled
    data) since this function only ever runs well after ``analysis_date``.
    """
    data = load_ohlcv(ticker, date.today().isoformat(), fill_gaps=False)
    if data is None or data.empty:
        raise ValueError(f"No OHLCV data available for {ticker}.")
    df = data.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date")
    cutoff = pd.to_datetime(analysis_date)
    forward = df[df["Date"] > cutoff]
    return forward.reset_index(drop=True)


def evaluate_first_barrier(
    *, ticker: str, analysis_date: str, reference_price: float, horizon_sessions: int, barrier_pct: float,
) -> dict[str, Any]:
    """Walk forward session by session from ``analysis_date`` and report
    which of +``barrier_pct``%/-``barrier_pct``% (from ``reference_price``)
    is reached first, using each session's High/Low (fork spec: "run2"
    First Barrier evaluator). Returns ``"INCOMPLETE"`` instead of
    ``"NO_POSITION"`` when fewer than ``horizon_sessions`` real sessions are
    available yet and neither barrier has been touched so far -- see module
    docstring.
    """
    upper_barrier = reference_price * (1 + barrier_pct / 100.0)
    lower_barrier = reference_price * (1 - barrier_pct / 100.0)

    full_window = _forward_window(ticker, analysis_date)
    available_future_sessions = len(full_window)
    horizon_complete = available_future_sessions >= horizon_sessions
    window = full_window.head(horizon_sessions)

    upper_hit_date: str | None = None
    lower_hit_date: str | None = None
    first_barrier: str | None = None
    sessions_to_first_barrier: int | None = None

    for i, row in enumerate(window.itertuples(index=False), start=1):
        upper_now = row.High >= upper_barrier
        lower_now = row.Low <= lower_barrier
        row_date = row.Date.strftime("%Y-%m-%d")
        if upper_now and upper_hit_date is None:
            upper_hit_date = row_date
        if lower_now and lower_hit_date is None:
            lower_hit_date = row_date
        if first_barrier is None and (upper_now or lower_now):
            if upper_now and lower_now:
                # Both barriers crossed within the same daily bar: daily OHLC
                # cannot tell which happened first intraday -- never guess.
                first_barrier = "AMBIGUOUS"
            elif upper_now:
                first_barrier = "LONG"
            else:
                first_barrier = "SHORT"
            sessions_to_first_barrier = i
            break

    if first_barrier is None:
        # Neither barrier was touched in the sessions walked so far: this is
        # only a genuine, definitive NO_POSITION once the full horizon has
        # actually elapsed. Otherwise the outcome simply is not known yet.
        first_barrier = "NO_POSITION" if horizon_complete else "INCOMPLETE"

    sessions_evaluated = len(window)
    if sessions_evaluated:
        last_close = float(window.iloc[-1]["Close"])
        max_high = float(window["High"].max())
        min_low = float(window["Low"].min())
        return_at_horizon_pct = (last_close - reference_price) / reference_price * 100.0
        mfe_long_pct = (max_high - reference_price) / reference_price * 100.0
        mae_long_pct = (min_low - reference_price) / reference_price * 100.0
    else:
        max_high = min_low = None
        return_at_horizon_pct = mfe_long_pct = mae_long_pct = None

    # Symmetric SHORT-side readings of the same two extremes (fork spec: "si
    # es sencillo, incluir metricas simetricas para SHORT"): a short
    # position's favorable move is the price falling (toward min_low), its
    # adverse move is the price rising (toward max_high).
    mfe_short_pct = -mae_long_pct if mae_long_pct is not None else None
    mae_short_pct = -mfe_long_pct if mfe_long_pct is not None else None

    return {
        "ticker": ticker,
        "analysis_date": analysis_date,
        "horizon_sessions": horizon_sessions,
        "barrier_pct": barrier_pct,
        "sessions_evaluated": sessions_evaluated,
        "available_future_sessions": available_future_sessions,
        "horizon_complete": horizon_complete,
        "reference_price": reference_price,
        "upper_barrier": upper_barrier,
        "lower_barrier": lower_barrier,
        "upper_barrier_hit_date": upper_hit_date,
        "lower_barrier_hit_date": lower_hit_date,
        "first_barrier": first_barrier,
        "sessions_to_first_barrier": sessions_to_first_barrier,
        "return_at_horizon_pct": return_at_horizon_pct,
        "mfe_long_pct": mfe_long_pct,
        "mae_long_pct": mae_long_pct,
        "mfe_short_pct": mfe_short_pct,
        "mae_short_pct": mae_short_pct,
        # Identical to mfe_long_pct/mae_long_pct under this methodology --
        # kept as separate, more generic names per the fork spec's own field
        # list (same documented-duplicate convention as
        # MarketFeatures.drawdown_from_20d_high).
        "max_high_pct": mfe_long_pct,
        "max_low_pct": mae_long_pct,
    }


def evaluate_result(result_or_path: dict[str, Any] | str | Path) -> dict[str, Any]:
    """Convenience wrapper: pulls ticker/analysis_date/reference_price/
    horizon_sessions/barrier_pct out of a run2 ``result.json`` (already
    written to disk, or passed in-memory) and evaluates it."""
    result = result_or_path if isinstance(result_or_path, dict) else load_result(Path(result_or_path))
    price_reference = result["price_reference"]
    return evaluate_first_barrier(
        ticker=result["ticker"],
        analysis_date=result["analysis_date"],
        reference_price=price_reference["close"],
        horizon_sessions=result["horizon_sessions"],
        barrier_pct=result["barrier_pct"],
    )


# ---------------------------------------------------------------------------
# Accuracy support (fork spec: "run2" binary simplification -- "Metricas
# futuras"). Deliberately small: a per-case classifier/scorer, not a full
# backtest runner (bucketing by confidence, aggregate accuracy across many
# runs, etc. are explicitly future work -- "No hace falta implementar
# todavia un backtest complejo").
# ---------------------------------------------------------------------------

#: What a real market outcome (evaluator.first_barrier) means for the
#: predictor's binary LONG/SHORT accuracy: "EVALUABLE" (real LONG/SHORT,
#: counts toward accuracy), "NO_BARRIER_HIT" (real NO_POSITION -- tracked
#: separately, never counted as a correct prediction, per fork spec: "no
#: contar como acierto"), "EXCLUDED" (real AMBIGUOUS/INCOMPLETE -- the
#: outcome cannot be determined at all, so it is dropped from analysis
#: entirely, per fork spec: "excluir").
_ACCURACY_CATEGORY_BY_OUTCOME = {
    "LONG": "EVALUABLE",
    "SHORT": "EVALUABLE",
    "NO_POSITION": "NO_BARRIER_HIT",
    "AMBIGUOUS": "EXCLUDED",
    "INCOMPLETE": "EXCLUDED",
}


def classify_for_accuracy(first_barrier: str) -> str:
    """Map a real market outcome (``evaluate_first_barrier``'s
    ``"first_barrier"``) to its accuracy-analysis category -- see this
    section's module-level comment for the exact meaning of each category.
    """
    try:
        return _ACCURACY_CATEGORY_BY_OUTCOME[first_barrier]
    except KeyError:
        raise ValueError(f"Unknown first_barrier outcome: {first_barrier!r}") from None


def score_prediction(
    *, prediction: str, confidence: float | None, evaluation: dict[str, Any],
) -> dict[str, Any]:
    """Score one run2 prediction (``"LONG"``/``"SHORT"``, or ``"UNRESOLVED"``
    for a technical failure -- never scoreable) against its real outcome
    (an ``evaluate_first_barrier`` result). ``correct`` is ``True``/``False``
    only when the real outcome is itself evaluable (category
    ``"EVALUABLE"``, i.e. a genuine real LONG/SHORT); it is ``None`` for a
    technical-failure prediction and for ``"NO_BARRIER_HIT"``/``"EXCLUDED"``
    outcomes, since "no accurate/inaccurate" judgement is meaningful there.

    ``accuracy = correct LONG/SHORT predictions / evaluable LONG/SHORT
    cases`` (fork spec) is computed by a caller aggregating many of these
    records and filtering to ``category == "EVALUABLE"`` -- aggregation
    itself is out of scope here (see module-level comment).
    """
    actual = evaluation["first_barrier"]
    category = classify_for_accuracy(actual)
    evaluable = prediction in ("LONG", "SHORT") and category == "EVALUABLE"
    correct = (prediction == actual) if evaluable else None
    return {
        "prediction": prediction,
        "confidence": confidence,
        "actual_first_barrier": actual,
        "category": category,
        "correct": correct,
        "sessions_to_first_barrier": evaluation.get("sessions_to_first_barrier"),
    }


def score_result(result_or_path: dict[str, Any] | str | Path) -> dict[str, Any]:
    """Convenience wrapper: evaluates a run2 ``result.json`` (see
    ``evaluate_result``) and scores its own ``final_decision``/
    ``barrier_signal`` against the real outcome in one call."""
    result = result_or_path if isinstance(result_or_path, dict) else load_result(Path(result_or_path))
    evaluation = evaluate_result(result)
    prediction = result["final_decision"]["action"]
    confidence = (result.get("barrier_signal") or {}).get("confidence")
    return {**evaluation, **score_prediction(prediction=prediction, confidence=confidence, evaluation=evaluation)}
