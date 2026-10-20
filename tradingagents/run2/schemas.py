"""Structured schemas for run2's two LLM stages (see ``pipeline.py``):
Market Regime Analyzer and Binary First Barrier Predictor. ``FinalDecision``
is the one exception -- it is built deterministically by ``pipeline.py``,
never parsed from an LLM, so it has no need for a repair path.

run2 answers exactly one question (fork spec: "run2" -- binary First Barrier
experiment): from the reference price on ``analysis_date``, within the next
``horizon_sessions`` trading sessions, which barrier (+``barrier_pct``% or
-``barrier_pct``%) will be reached first? The predictor MUST choose LONG or
SHORT -- NO_POSITION is deliberately not a valid prediction (a prior version
of run2 allowed it, and the LLM used it as a dominant "mixed signals" escape
hatch, defeating the point of measuring raw predictive power; see
``pipeline.py``'s module docstring). There is no final-price prediction, no
expected-return range, and no confidence threshold gating the decision.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

RUN2_SCHEMA_VERSION = "3.0"

TrendDirection = Literal["UP", "DOWN", "SIDEWAYS"]
Momentum = Literal["STRONG_POSITIVE", "POSITIVE", "NEUTRAL", "NEGATIVE", "STRONG_NEGATIVE"]
VolatilityRegime = Literal["LOW", "NORMAL", "HIGH"]
# BarrierSignal.direction: binary, forced choice -- the LLM predictor can
# never produce NO_POSITION (see module docstring).
PredictionDirection = Literal["LONG", "SHORT"]
# FinalDecision.action: the deterministic outcome -- LONG/SHORT mirror a
# valid predictor output outright; UNRESOLVED means a technical/structural
# pipeline failure (a stage invalid even after its one repair attempt),
# NEVER market uncertainty -- see pipeline.py's REASON_* constants.
FinalAction = Literal["LONG", "SHORT", "UNRESOLVED"]
# BaselineResult.action only: the deterministic baselines (momentum/
# moving_average, see baselines.py) are experimental controls, not subject
# to the predictor's forced binary choice, and may legitimately report
# NO_POSITION ("insufficient evidence/not aligned").
BaselineAction = Literal["LONG", "SHORT", "NO_POSITION"]
RecentStructure = Literal["HH_HL", "LH_LL", "MIXED"]


class KeyLevels(BaseModel):
    support: list[float] = Field(default_factory=list)
    resistance: list[float] = Field(default_factory=list)


class RegimeAnalysis(BaseModel):
    """Stage 1: Market Regime Analyzer output (fork spec: "run2" section 1)."""

    primary_trend: TrendDirection
    secondary_trend: TrendDirection
    momentum: Momentum
    volatility_regime: VolatilityRegime
    trend_strength: float = Field(ge=0.0, le=1.0)
    regime_confidence: float = Field(ge=0.0, le=1.0)
    key_levels: KeyLevels = Field(default_factory=KeyLevels)
    observations: list[str] = Field(default_factory=list)


class BarrierSignal(BaseModel):
    """Stage 2: Binary First Barrier Predictor output (fork spec: "run2" --
    binary First Barrier experiment).

    Forced binary choice: predicts which of +barrier_pct%/-barrier_pct% from
    the reference price is reached first within horizon_sessions. LONG means
    +barrier_pct% first, SHORT means -barrier_pct% first -- there is no
    third option; ``direction`` cannot be NO_POSITION (see module
    docstring). ``confidence`` is diagnostic only: it is recorded for later
    predictive-power analysis but NEVER gates or changes
    ``final_decision`` -- a valid LONG/SHORT signal is the final decision
    outright, regardless of how low its confidence is (fork spec: "LONG
    confidence 0.51 -> LONG"). There is deliberately no
    ``expected_return_range`` field -- run2 does not predict where the price
    ends up, only which barrier is crossed first.
    """

    direction: PredictionDirection
    confidence: float = Field(ge=0.0, le=1.0)
    thesis: str
    evidence_for: list[str] = Field(default_factory=list)
    evidence_against: list[str] = Field(default_factory=list)
    invalidation_conditions: list[str] = Field(default_factory=list)


class FinalDecision(BaseModel):
    """Deterministic final output. Mirrors ``BarrierSignal.direction``
    outright when the signal parsed successfully (fork spec: "no aplicar
    ningun threshold posterior de confidence"); becomes ``UNRESOLVED`` only
    when a stage failed to produce valid structured output even after its
    one repair attempt (see ``pipeline.py``'s ``REASON_*`` constants) --
    never fabricates LONG/SHORT from an invalid or missing signal.
    ``UNRESOLVED`` represents EXCLUSIVELY a technical/structural pipeline
    failure, never market uncertainty -- that distinction is the entire
    reason NO_POSITION was removed as a predictor output (fork spec: "run2"
    binary simplification)."""

    action: FinalAction
    reason: str


class MarketFeatures(BaseModel):
    """Deterministic, point-in-time market feature pack (fork spec: "run2"
    hardening -- "MARKET-ONLY cuantitativo, auditable y barato de evaluar").

    Computed entirely in Python by ``market_data.compute_market_features``,
    never by an LLM -- every field here is either a plain arithmetic
    transform of the OHLCV history on or before ``as_of``, or ``None`` when
    there was not enough history for that particular window (never a
    fabricated value). See that function's docstring for the exact formula
    behind each field. Fed to the Regime Analyzer and Binary First Barrier
    Predictor as pre-computed evidence so the model never has to compute a
    mean, a return, or ATR mentally from a raw price series.
    """

    as_of: str
    sessions_available: int

    return_5d: float | None = None
    return_10d: float | None = None
    return_20d: float | None = None
    return_40d: float | None = None
    return_60d: float | None = None
    return_120d: float | None = None

    sma20: float | None = None
    sma50: float | None = None
    sma100: float | None = None
    sma200: float | None = None
    distance_pct_to_sma20: float | None = None
    distance_pct_to_sma50: float | None = None
    distance_pct_to_sma100: float | None = None
    distance_pct_to_sma200: float | None = None
    slope_sma20: float | None = None
    slope_sma50: float | None = None
    slope_sma200: float | None = None
    price_above_sma20: bool | None = None
    price_above_sma50: bool | None = None
    price_above_sma100: bool | None = None
    price_above_sma200: bool | None = None

    rolling_high_20: float | None = None
    rolling_high_60: float | None = None
    rolling_high_120: float | None = None
    rolling_low_20: float | None = None
    rolling_low_60: float | None = None
    rolling_low_120: float | None = None
    distance_pct_to_high_20: float | None = None
    distance_pct_to_high_60: float | None = None
    distance_pct_to_high_120: float | None = None
    distance_pct_to_low_20: float | None = None
    distance_pct_to_low_60: float | None = None
    distance_pct_to_low_120: float | None = None
    recent_structure: RecentStructure | None = None

    rsi14: float | None = None
    adx14: float | None = None

    atr14: float | None = None
    atr14_pct: float | None = None
    realized_vol_20: float | None = None
    realized_vol_60: float | None = None

    volume_ratio_20: float | None = None
    volume_zscore_20: float | None = None

    # Numerically identical to distance_pct_to_high_{20,60,120} under this
    # module's methodology (both are "how far is the close from the rolling
    # N-session high, in percent") -- kept as separate named fields because
    # the fork spec lists them under distinct "Structure" and "Drawdown"
    # categories; see compute_market_features's docstring.
    drawdown_from_20d_high: float | None = None
    drawdown_from_60d_high: float | None = None
    drawdown_from_120d_high: float | None = None


class BaselineResult(BaseModel):
    """One deterministic, untuned baseline rule's output (fork spec: "run2"
    hardening, "Baselines deterministas gratuitos") -- for later comparison
    only. Never fed to any LLM prompt; see ``baselines.py``. Unlike
    ``BarrierSignal``, a baseline MAY report NO_POSITION -- it is not
    subject to the predictor's forced binary choice."""

    action: BaselineAction
    reason: str
