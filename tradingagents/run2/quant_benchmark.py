"""Offline, deterministic, LLM-free quantitative benchmark for run2's market
features (fork README: "Experimental research" / "quant benchmark").

Answers a narrower question than run2's own LLM pipeline: do the
deterministic ``market_features`` (see ``market_data.compute_market_features``)
contain any simple predictive signal for the First Barrier target at all,
before trying to improve on an LLM? Compares a handful of frozen,
hand-written rules (``quant_baselines.py``) against the real market outcome
(``evaluator.evaluate_first_barrier``) over a historical date range -- no
LLM call anywhere in this module, no training, no parameter search.

Structurally separate from the LLM inference pipeline
(``pipeline.py``/``stages.py``/``prompts.py``/``validation.py``): this
module never imports any of them, nor ``tradingagents.llm_clients`` or
``tradingagents.agents.structured`` -- enforced by
``tests/test_run2_quant_benchmark.py``'s import-boundary test. It reuses
``market_data.compute_market_features``/``fetch_price_reference`` (features
-- past and present only) and ``evaluator.evaluate_first_barrier`` (the
label -- genuinely future) exactly as they are; the fact that these live in
two different modules is itself the leakage guard (see fork spec:
"features(t) = pasado y presente; label(t) = futuro").

CLI:

    python -m tradingagents.run2.quant_benchmark \\
        --ticker NVDA --start-date 2024-01-01 --end-date 2026-06-30 \\
        --step-sessions 5 --horizon-sessions 40 --barrier-pct 5 \\
        --output ~/Desktop/nvda-first-barrier-benchmark

Writes ``observations.csv`` (one row per analysis_date), ``summary.json``,
and ``summary.md`` under ``--output`` (a directory, created if needed).
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from tradingagents.dataflows.vendors.yahoo.ohlcv import load_ohlcv
from tradingagents.run2.evaluator import evaluate_first_barrier
from tradingagents.run2.market_data import compute_market_features, fetch_price_reference
from tradingagents.run2.quant_baselines import QUANT_BASELINES

# Real market outcomes (evaluator.py) that count as a genuine LONG/SHORT
# label for accuracy purposes -- AMBIGUOUS/INCOMPLETE/NO_POSITION never do
# (see evaluator.classify_for_accuracy, reused in spirit here).
_EVALUABLE_OUTCOMES = ("LONG", "SHORT")


# ---------------------------------------------------------------------------
# Observation-date selection (pure, no I/O)
# ---------------------------------------------------------------------------


def generate_observation_dates(
    all_dates: Sequence[str], start_date: str, end_date: str, step_sessions: int,
) -> list[str]:
    """Every ``step_sessions``-th REAL trading date (not calendar day) in
    ``all_dates`` that falls within ``[start_date, end_date]`` -- avoids
    evaluating every consecutive session, which would produce extremely
    correlated, near-duplicate samples (fork spec: "No quiero evaluar todas
    las sesiones consecutivas"). ``step_sessions=5`` is roughly one
    observation per trading week.

    ``all_dates`` and the bounds are ``YYYY-MM-DD`` strings, compared
    lexicographically (which sorts identically to chronologically for that
    format) -- no pandas/timestamp machinery needed for this pure function.
    """
    if step_sessions < 1:
        raise ValueError(f"step_sessions must be >= 1, got {step_sessions}")
    in_range = sorted(d for d in all_dates if start_date <= d <= end_date)
    return in_range[::step_sessions]


def _load_trading_dates(ticker: str, end_date: str) -> list[str]:
    """Every real trading date on or before ``end_date`` the vendor has for
    ``ticker``, as ``YYYY-MM-DD`` strings -- a thin wrapper around the
    already point-in-time-safe ``load_ohlcv`` (fork spec: "reutilizar...
    compute_market_features, evaluate_first_barrier... no duplicar
    formulas")."""
    data = load_ohlcv(ticker, end_date, fill_gaps=False)
    if data is None or data.empty:
        raise ValueError(f"No OHLCV data available for {ticker}.")
    dates = pd.to_datetime(data["Date"], errors="coerce").dropna().sort_values()
    return [d.strftime("%Y-%m-%d") for d in dates]


# ---------------------------------------------------------------------------
# Per-observation scoring
# ---------------------------------------------------------------------------


def score_prediction(prediction: str, actual: str) -> bool | None:
    """``True``/``False`` only when ``prediction`` is directional
    (LONG/SHORT, never NO_PREDICTION) AND ``actual`` is a genuine market
    LONG/SHORT (never NO_POSITION/AMBIGUOUS/INCOMPLETE) -- ``None``
    otherwise (not scoreable either way)."""
    if prediction not in ("LONG", "SHORT"):
        return None
    if actual not in _EVALUABLE_OUTCOMES:
        return None
    return prediction == actual


def build_observation(
    *, ticker: str, analysis_date: str, horizon_sessions: int, barrier_pct: float,
) -> dict[str, Any] | None:
    """One row: deterministic features/baseline predictions (point-in-time,
    ``<= analysis_date`` only) plus the real, separately-computed label (the
    ONE place this function looks at the future -- via
    ``evaluator.evaluate_first_barrier``). Returns ``None`` (skip) only when
    there is no usable data for ``ticker`` at all -- an incomplete horizon is
    kept (tracked as ``"INCOMPLETE"``), never silently dropped.
    """
    try:
        features = compute_market_features(ticker, analysis_date)
        price_reference = fetch_price_reference(ticker, analysis_date)
        evaluation = evaluate_first_barrier(
            ticker=ticker, analysis_date=analysis_date, reference_price=price_reference["close"],
            horizon_sessions=horizon_sessions, barrier_pct=barrier_pct,
        )
    except ValueError:
        return None

    actual = evaluation["first_barrier"]
    row: dict[str, Any] = {
        "analysis_date": analysis_date,
        "reference_price": price_reference["close"],
        "actual_first_barrier": actual,
        "sessions_to_first_barrier": evaluation["sessions_to_first_barrier"],
        "return_at_horizon_pct": evaluation["return_at_horizon_pct"],
        "return_20d": features.return_20d,
        "return_60d": features.return_60d,
        "rsi14": features.rsi14,
        "adx14": features.adx14,
    }
    for name, rule in QUANT_BASELINES.items():
        prediction = rule(features)["prediction"]
        row[f"{name}_prediction"] = prediction
        row[f"{name}_correct"] = score_prediction(prediction, actual)
    return row


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _wilson_ci95(successes: int, n: int) -> tuple[float, float] | None:
    """95% Wilson score interval for a binomial proportion -- simple,
    closed-form, no scipy/statsmodels dependency (fork spec: "No anadir
    dependencias pesadas solo para esto")."""
    if n == 0:
        return None
    z = 1.96
    p_hat = successes / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def _actual_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    actuals = [r["actual_first_barrier"] for r in rows]
    return {
        "actual_long_count": actuals.count("LONG"),
        "actual_short_count": actuals.count("SHORT"),
        "actual_no_barrier_hit_count": actuals.count("NO_POSITION"),
        "actual_ambiguous_count": actuals.count("AMBIGUOUS"),
        "actual_incomplete_count": actuals.count("INCOMPLETE"),
    }


def _majority_class_accuracy_pct(actual_counts: dict[str, int]) -> float | None:
    """Accuracy of "always predict the more common real class" -- the
    comparison baseline any directional rule must beat to show it is doing
    better than exploiting class imbalance (fork spec: "Comparacion contra
    azar")."""
    evaluable = actual_counts["actual_long_count"] + actual_counts["actual_short_count"]
    if evaluable == 0:
        return None
    return max(actual_counts["actual_long_count"], actual_counts["actual_short_count"]) / evaluable * 100.0


def _metrics_for_baseline(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    total = len(rows)
    predictions = [r[f"{name}_prediction"] for r in rows]
    corrects = [r[f"{name}_correct"] for r in rows]
    actuals = [r["actual_first_barrier"] for r in rows]

    scored = [c for c in corrects if c is not None]
    correct = sum(1 for c in scored if c)
    incorrect = len(scored) - correct
    ci = _wilson_ci95(correct, len(scored))

    long_evaluable = [c for p, c, a in zip(predictions, corrects, actuals, strict=True) if p == "LONG" and a in _EVALUABLE_OUTCOMES]
    short_evaluable = [c for p, c, a in zip(predictions, corrects, actuals, strict=True) if p == "SHORT" and a in _EVALUABLE_OUTCOMES]

    return {
        "total_observations": total,
        "evaluable_market_labels": sum(1 for a in actuals if a in _EVALUABLE_OUTCOMES),
        "predictions_made": sum(1 for p in predictions if p in ("LONG", "SHORT")),
        "coverage_pct": (sum(1 for p in predictions if p in ("LONG", "SHORT")) / total * 100.0) if total else None,
        "correct": correct,
        "incorrect": incorrect,
        "accuracy_pct": (correct / len(scored) * 100.0) if scored else None,
        "accuracy_ci95_low_pct": ci[0] * 100.0 if ci else None,
        "accuracy_ci95_high_pct": ci[1] * 100.0 if ci else None,
        "long_predictions": sum(1 for p in predictions if p == "LONG"),
        "long_correct": sum(1 for c in long_evaluable if c),
        "long_accuracy_pct": (sum(1 for c in long_evaluable if c) / len(long_evaluable) * 100.0) if long_evaluable else None,
        "short_predictions": sum(1 for p in predictions if p == "SHORT"),
        "short_correct": sum(1 for c in short_evaluable if c),
        "short_accuracy_pct": (sum(1 for c in short_evaluable if c) / len(short_evaluable) * 100.0) if short_evaluable else None,
        "no_prediction_count": sum(1 for p in predictions if p == "NO_PREDICTION"),
    }


def _period_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    actual_counts = _actual_counts(rows)
    return {
        "total_observations": len(rows),
        **actual_counts,
        "majority_class_accuracy_pct": _majority_class_accuracy_pct(actual_counts),
        "baselines": {name: _metrics_for_baseline(rows, name) for name in QUANT_BASELINES},
    }


def summarize(
    rows: list[dict[str, Any]], *, ticker: str, start_date: str, end_date: str,
    step_sessions: int, horizon_sessions: int, barrier_pct: float,
) -> dict[str, Any]:
    by_year: dict[str, list[dict[str, Any]]] = {}
    by_quarter: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        year = row["analysis_date"][:4]
        month = int(row["analysis_date"][5:7])
        quarter = f"{year}-Q{(month - 1) // 3 + 1}"
        by_year.setdefault(year, []).append(row)
        by_quarter.setdefault(quarter, []).append(row)

    return {
        "meta": {
            "ticker": ticker, "start_date": start_date, "end_date": end_date,
            "step_sessions": step_sessions, "horizon_sessions": horizon_sessions,
            "barrier_pct": barrier_pct, "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "overall": _period_summary(rows),
        "by_year": {year: _period_summary(year_rows) for year, year_rows in sorted(by_year.items())},
        "by_quarter": {q: _period_summary(q_rows) for q, q_rows in sorted(by_quarter.items())},
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def write_observations_csv(rows: list[dict[str, Any]], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def write_summary_json(summary: dict[str, Any], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _fmt_pct(value: float | None) -> str:
    return f"{value:.1f}%" if value is not None else "n/a"


def _markdown_table(period: dict[str, Any]) -> list[str]:
    lines = ["| Baseline | Coverage | Accuracy | 95% CI | LONG acc. | SHORT acc. | N |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name, m in period["baselines"].items():
        lines.append(
            f"| {name} | {_fmt_pct(m['coverage_pct'])} | {_fmt_pct(m['accuracy_pct'])} | "
            f"[{_fmt_pct(m['accuracy_ci95_low_pct'])}, {_fmt_pct(m['accuracy_ci95_high_pct'])}] | "
            f"{_fmt_pct(m['long_accuracy_pct'])} | {_fmt_pct(m['short_accuracy_pct'])} | "
            f"{m['correct'] + m['incorrect']} |"
        )
    return lines


def write_summary_markdown(summary: dict[str, Any], path: Path) -> Path:
    meta = summary["meta"]
    overall = summary["overall"]
    lines = [
        f"# run2 quantitative benchmark: {meta['ticker']}",
        "",
        f"- Range: {meta['start_date']} to {meta['end_date']} (step {meta['step_sessions']} sessions)",
        f"- Horizon: {meta['horizon_sessions']} sessions; barrier: +/-{meta['barrier_pct']}%",
        f"- Generated: {meta['generated_at']}",
        "",
        "## Overall",
        "",
        f"- Observations: {overall['total_observations']}",
        f"- Actual LONG: {overall['actual_long_count']}; SHORT: {overall['actual_short_count']}; "
        f"NO_BARRIER_HIT: {overall['actual_no_barrier_hit_count']}; "
        f"AMBIGUOUS: {overall['actual_ambiguous_count']}; INCOMPLETE: {overall['actual_incomplete_count']}",
        f"- Majority-class baseline accuracy: {_fmt_pct(overall['majority_class_accuracy_pct'])}",
        "",
        *_markdown_table(overall),
        "",
        "## By year",
        "",
    ]
    for year, year_summary in sorted(summary["by_year"].items()):
        lines.append(f"### {year} (n={year_summary['total_observations']})")
        lines.append("")
        lines += _markdown_table(year_summary)
        lines.append("")
    lines += [
        "## Methodology notes",
        "",
        "- No LLM call anywhere in this benchmark; every rule is a frozen, hand-written formula.",
        "- `momentum`/`moving_average` reuse `tradingagents.run2.baselines` exactly, unmodified.",
        "- `contrarian_momentum` inverts only `momentum`'s own LONG/SHORT output; its NO_POSITION "
        "stays an abstention (`NO_PREDICTION`), never flipped into a directional call.",
        "- `contrarian_momentum` is included because a 12-observation NVDA development set "
        "(see fork README: \"Experimental research\") suggested a possible reversal pattern -- "
        "those 12 dates are NOT an independent validation set for that hypothesis; this is the "
        "first out-of-sample-style check of it.",
        "- Parameters (RSI 70/30, barrier %, horizon sessions) are frozen inputs, never tuned "
        "against this or any other dataset.",
        "- `NO_PREDICTION` (a rule declining to pick a direction) is tracked separately from "
        "`NO_POSITION` (the real market outcome when neither barrier was touched within the horizon).",
        "- `majority_class_accuracy_pct` is the accuracy of always predicting the more common real "
        "class -- the bar any rule must clear to show it beats class imbalance, not a coin flip.",
    ]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Orchestration + CLI
# ---------------------------------------------------------------------------


def run_benchmark(
    *, ticker: str, start_date: str, end_date: str, output_dir: Path,
    step_sessions: int = 5, horizon_sessions: int = 40, barrier_pct: float = 5.0,
) -> dict[str, Any]:
    all_dates = _load_trading_dates(ticker, end_date)
    observation_dates = generate_observation_dates(all_dates, start_date, end_date, step_sessions)
    rows = [
        row for row in (
            build_observation(ticker=ticker, analysis_date=d, horizon_sessions=horizon_sessions, barrier_pct=barrier_pct)
            for d in observation_dates
        )
        if row is not None
    ]
    summary = summarize(
        rows, ticker=ticker, start_date=start_date, end_date=end_date,
        step_sessions=step_sessions, horizon_sessions=horizon_sessions, barrier_pct=barrier_pct,
    )
    output_dir = Path(output_dir)
    write_observations_csv(rows, output_dir / "observations.csv")
    write_summary_json(summary, output_dir / "summary.json")
    write_summary_markdown(summary, output_dir / "summary.md")
    return summary


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline, LLM-free quantitative benchmark for run2's market features "
        "against the First Barrier target.",
    )
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--step-sessions", type=int, default=5, help="Default: 5 (~one observation per trading week).")
    parser.add_argument("--horizon-sessions", type=int, default=40)
    parser.add_argument("--barrier-pct", type=float, default=5.0)
    parser.add_argument("--output", required=True, help="Output directory for observations.csv/summary.json/summary.md.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    summary = run_benchmark(
        ticker=args.ticker, start_date=args.start_date, end_date=args.end_date,
        output_dir=Path(args.output), step_sessions=args.step_sessions,
        horizon_sessions=args.horizon_sessions, barrier_pct=args.barrier_pct,
    )
    print(json.dumps(summary["overall"], indent=2, sort_keys=True))
    print(f"\nWritten to {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
