"""Thin per-stage wrappers: render a prompt (``prompts.py``), run it through
``validation.run_structured_stage`` against the right schema. Each function
is independently testable with a stub LLM -- see ``tests/test_run2_stages.py``.
"""

from __future__ import annotations

from typing import Any

from tradingagents.run2.prompts import render_barrier_signal_prompt, render_regime_prompt
from tradingagents.run2.schemas import BarrierSignal, RegimeAnalysis
from tradingagents.run2.validation import run_structured_stage


def run_regime_analysis(
    llm: Any, *, ticker: str, analysis_date: str, horizon_sessions: int, market_features_json: str,
    market_context: str, language: str, repair_llm: Any | None = None,
) -> tuple[RegimeAnalysis | None, dict[str, Any]]:
    prompt = render_regime_prompt(
        ticker=ticker, analysis_date=analysis_date, horizon_sessions=horizon_sessions,
        market_features_json=market_features_json, market_context=market_context, language=language,
    )
    return run_structured_stage(
        llm, RegimeAnalysis, prompt, "Market Regime Analyzer",
        llm_role="generation", call_identity="run2_regime", repair_llm=repair_llm,
    )


def run_barrier_signal(
    llm: Any, *, ticker: str, analysis_date: str, horizon_sessions: int, barrier_pct: float,
    reference_price: float | None, reference_date: str | None, market_features_json: str,
    market_context: str, regime: RegimeAnalysis, language: str, repair_llm: Any | None = None,
) -> tuple[BarrierSignal | None, dict[str, Any]]:
    prompt = render_barrier_signal_prompt(
        ticker=ticker, analysis_date=analysis_date, horizon_sessions=horizon_sessions,
        barrier_pct=barrier_pct, reference_price=reference_price, reference_date=reference_date,
        market_features_json=market_features_json, market_context=market_context,
        regime_json=regime.model_dump_json(indent=2), language=language,
    )
    return run_structured_stage(
        llm, BarrierSignal, prompt, "Binary First Barrier Predictor",
        llm_role="generation", call_identity="run2_barrier_signal", repair_llm=repair_llm,
    )
