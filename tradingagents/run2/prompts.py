"""Prompt text for run2's two LLM stages (Market Regime Analyzer, Binary
First Barrier Predictor). Plain Python string-building -- run2 does not use
``tradingagents/prompts/`` (that tree is sized for the main pipeline's much
larger, templated prompt set); these short prompts are simpler kept inline,
matching run2's "simpler architecture" goal.

Every prompt repeats the market-only / no-look-ahead constraint explicitly,
even though ``market_data.py`` already enforces it structurally -- the
instruction is what stops the model from trying to reason about news or
events it might otherwise assume from training data.

Every prompt also leads with the deterministic ``market_features`` block
(see ``market_data.compute_market_features``) ahead of the raw OHLCV
window, with an explicit "already calculated, do not recompute" instruction
-- the fork spec's hardening goal is that the model interprets pre-computed,
auditable numbers rather than estimating means/returns/ATR itself from a
raw price series.
"""

from __future__ import annotations

import hashlib

from tradingagents.agents.structured import NO_EXTERNAL_TOOLS
from tradingagents.agents.structured_recovery import STRUCTURED_RECOVERY_PROMPT_MARKER

_MARKET_ONLY_CONSTRAINT = (
    "Use ONLY the verified market data provided below. Do not use news, "
    "fundamentals, insider data, social media, or any information dated "
    "after the analysis date. Do not call external tools or search the web."
)

_FEATURES_INSTRUCTION = (
    "The `market_features` JSON below is ALREADY CALCULATED and point-in-time "
    "safe as of the analysis date (returns, moving averages and their "
    "slopes, distance to highs/lows, RSI/ADX, ATR, realized volatility, "
    "volume, drawdown). Do not recompute any of it, do not estimate it from "
    "the raw OHLCV window, and do not state a number that is not present in "
    "`market_features` or the OHLCV window. When you state a conclusion, "
    "cite the specific feature name(s) that support it (e.g. 'return_20d', "
    "'sma50', 'rsi14'). A `null` feature means there was not enough history "
    "for it -- treat that as missing evidence, not as zero."
)


def render_regime_prompt(
    *, ticker: str, analysis_date: str, horizon_sessions: int, market_features_json: str,
    market_context: str, language: str,
) -> str:
    return (
        f"{NO_EXTERNAL_TOOLS}\n\n"
        f"You are the Market Regime Analyzer for {ticker} as of {analysis_date}. "
        f"{_MARKET_ONLY_CONSTRAINT}\n\n"
        f"{_FEATURES_INSTRUCTION}\n\n"
        f"Characterize the CURRENT market regime, with an eye toward the next "
        f"~{horizon_sessions} trading sessions: primary trend, secondary trend, "
        f"momentum, volatility regime, high/low structure, distance to relevant "
        f"moving averages, trend strength, any possible regime change, relevant "
        f"support/resistance levels, volume (if available), and extension/"
        f"overextension. If the evidence is contradictory or weak, say so "
        f"explicitly rather than forcing a confident read. Respond in {language}.\n\n"
        f"### Deterministic market features (already computed -- do not recompute)\n"
        f"{market_features_json}\n\n"
        f"### Recent OHLCV window (supporting context)\n{market_context}\n"
    )


def render_barrier_signal_prompt(
    *, ticker: str, analysis_date: str, horizon_sessions: int, barrier_pct: float,
    reference_price: float | None, reference_date: str | None, market_features_json: str,
    market_context: str, regime_json: str, language: str,
) -> str:
    return (
        f"{NO_EXTERNAL_TOOLS}\n\n"
        f"You are the Binary First Barrier Predictor for {ticker} as of {analysis_date}. "
        f"{_MARKET_ONLY_CONSTRAINT}\n\n"
        f"{_FEATURES_INSTRUCTION}\n\n"
        f"Reference price: {reference_price} (verified close on {reference_date}).\n\n"
        f"Using only information available up to {analysis_date}, predict which event "
        f"will occur FIRST within the next {horizon_sessions} trading sessions:\n\n"
        f"  A) price reaches +{barrier_pct}% from the reference price\n"
        f"  B) price reaches -{barrier_pct}% from the reference price\n\n"
        f"You MUST choose LONG or SHORT. Do not answer NO_POSITION, neutral, uncertain, "
        f"unclear, or any equivalent -- that option does not exist for this task. LONG "
        f"means A occurs first; SHORT means B occurs first.\n\n"
        f"This is NOT a prediction of the closing price at session {horizon_sessions} -- "
        f"it does not matter at all where the price finishes. Only which barrier is "
        f"crossed FIRST matters. The two barriers are symmetric BY DESIGN (both are "
        f"exactly {barrier_pct}% from the reference price) -- that symmetry is a fixed "
        f"property of the task, never evidence for either side and never a justification "
        f"to abstain. That the stock's volatility makes both barriers reachable also does "
        f"not resolve the direction -- you must still judge which one is RELATIVELY more "
        f"likely to be reached first. Do not reason \"both barriers are equidistant, so I "
        f"cannot decide\" -- the equidistance is a given constant of this task, never a "
        f"basis for uncertainty; express any uncertainty through `confidence` instead. "
        f"`confidence` is purely diagnostic (relative strength of the evidence); it will "
        f"never gate or change which direction is recorded, so answer honestly rather "
        f"than inflating or deflating it. Base your decision on the market regime and the "
        f"deterministic features below; cite specific feature names and values from the "
        f"data. Do not invent an indicator that is not present in it. Give a thesis, "
        f"evidence for and against, and the conditions that would invalidate this call. "
        f"Respond in {language}.\n\n"
        f"### Market Regime Analysis\n{regime_json}\n\n"
        f"### Deterministic market features (already computed -- do not recompute)\n"
        f"{market_features_json}\n\n"
        f"### Recent OHLCV window (supporting context)\n{market_context}\n"
    )


# ---------------------------------------------------------------------------
# Prompt fingerprints for run2's own provenance (fork spec: "run2" hardening
# -- "Quiero que la provenance de run2 sea especifica de run2"). Every
# placeholder below is a FIXED, non-run-specific token -- never the real
# ticker/date/market data -- so the resulting hash is a fingerprint of the
# prompt's WORDING (changes only if someone edits the instructional text in
# this file), never of any particular run's inputs. This is the run2
# equivalent of the main pipeline's ``llm_trace.hash_prompt_files()``, which
# hashes static template FILES; run2 has no template files (these are
# f-string functions), so it hashes each render function's output under
# fixed placeholders instead.
# ---------------------------------------------------------------------------

_FINGERPRINT_PLACEHOLDERS = {
    "ticker": "<TICKER>",
    "analysis_date": "<ANALYSIS_DATE>",
    "horizon_sessions": 0,
    "barrier_pct": 0.0,
    "reference_price": 0.0,
    "reference_date": "<REFERENCE_DATE>",
    "market_features_json": "<MARKET_FEATURES_JSON>",
    "market_context": "<MARKET_CONTEXT>",
    "language": "<LANGUAGE>",
}


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def prompt_fingerprints(*, repair_used: bool = False) -> dict[str, str]:
    """``{stage_name: sha256_hex}`` for the two prompts run2 can actually
    use, plus a ``"repair_recovery"`` entry (only when ``repair_used`` is
    true) fingerprinting the generic repair-prompt marker any stage's one
    repair attempt uses (see ``structured_recovery.py``). Fed into
    ``pipeline.py``'s ``build_run_provenance(..., prompt_hashes=...)`` call
    so run2's provenance never falls back to hashing the main pipeline's
    ``tradingagents/prompts/`` tree (news/fundamentals/Bull-Bear/Risk/PM),
    which run2 never reads."""
    p = _FINGERPRINT_PLACEHOLDERS
    regime_text = render_regime_prompt(
        ticker=p["ticker"], analysis_date=p["analysis_date"], horizon_sessions=p["horizon_sessions"],
        market_features_json=p["market_features_json"], market_context=p["market_context"],
        language=p["language"],
    )
    barrier_text = render_barrier_signal_prompt(
        ticker=p["ticker"], analysis_date=p["analysis_date"], horizon_sessions=p["horizon_sessions"],
        barrier_pct=p["barrier_pct"], reference_price=p["reference_price"], reference_date=p["reference_date"],
        market_features_json=p["market_features_json"], market_context=p["market_context"],
        regime_json="<REGIME_JSON>", language=p["language"],
    )
    hashes = {
        "market_regime_analyzer": _sha256(regime_text),
        "binary_first_barrier_predictor": _sha256(barrier_text),
    }
    if repair_used:
        hashes["repair_recovery"] = _sha256(STRUCTURED_RECOVERY_PROMPT_MARKER)
    return hashes
