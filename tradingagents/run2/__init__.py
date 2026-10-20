"""run2: a second, independent, market-data-only experimental pipeline --
the binary First Barrier experiment.

See fork README: "run2 (binary First Barrier experiment)".

run2 answers one narrow question, as a FORCED BINARY CHOICE -- "from the
reference price on analysis_date, within the next horizon_sessions trading
sessions, which barrier (+barrier_pct% or -barrier_pct%) will be reached
first?" -- via two stages (Market Regime Analyzer -> Binary First Barrier
Predictor) plus deterministic validation, producing LONG or SHORT.
NO_POSITION is deliberately not a valid predictor output (an earlier version
allowed it; the LLM used it as a dominant "mixed signals" escape hatch,
defeating the point of measuring raw predictive power) -- the only other
possible outcome is UNRESOLVED, representing exclusively a technical/
structural pipeline failure, never market uncertainty. It is a
deliberately separate, simpler architecture: it does not import or
construct ``TradingAgentsGraph``, does not use news/fundamentals/social/
Decision Memory, and does not run the Bull/Bear debate, Research Manager,
Trader, Risk, or Portfolio Manager. See ``pipeline.py`` for the orchestrator,
``storage.py`` for its own, separate persistence root, and
``evaluator.py`` for the SEPARATE, offline utility that checks -- after the
fact, using data this package's inference side never sees -- which barrier
was actually reached first.
"""

from tradingagents.run2.config import RUN2_DEFAULT_CONFIG, resolve_run2_config
from tradingagents.run2.pipeline import run_run2_analysis
from tradingagents.run2.schemas import (
    RUN2_SCHEMA_VERSION,
    BarrierSignal,
    FinalDecision,
    RegimeAnalysis,
)
from tradingagents.run2.storage import default_run2_root

__all__ = [
    "RUN2_DEFAULT_CONFIG",
    "RUN2_SCHEMA_VERSION",
    "BarrierSignal",
    "FinalDecision",
    "RegimeAnalysis",
    "default_run2_root",
    "resolve_run2_config",
    "run_run2_analysis",
]
