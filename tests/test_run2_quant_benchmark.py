"""Offline, deterministic, LLM-free quantitative benchmark (fork README:
"Experimental research" / quant benchmark). Covers date selection, each
baseline rule, the NO_PREDICTION/NO_POSITION distinction, accuracy/coverage/
majority-class math, yearly grouping, output file generation, and that this
benchmark never leaks future data into its own inputs.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pandas as pd
import pytest

import tradingagents.dataflows.vendors.yahoo.snapshot as snapshot_module
import tradingagents.run2.evaluator as evaluator
import tradingagents.run2.market_data as market_data
import tradingagents.run2.quant_baselines as qbl
import tradingagents.run2.quant_benchmark as qb
from tradingagents.run2.schemas import MarketFeatures

RUN2_PACKAGE_DIR = Path(__file__).resolve().parent.parent / "tradingagents" / "run2"


def _features(**overrides):
    base = {"as_of": "2026-01-07", "sessions_available": 300}
    base.update(overrides)
    return MarketFeatures(**base)


# ---------------------------------------------------------------------------
# 1. Deterministic observation-date generation with step_sessions
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_1_generate_observation_dates_steps_through_real_trading_dates():
    dates = [f"2024-01-{d:02d}" for d in range(1, 21)]  # 20 consecutive "trading days"
    result = qb.generate_observation_dates(dates, "2024-01-01", "2024-01-20", step_sessions=5)
    assert result == ["2024-01-01", "2024-01-06", "2024-01-11", "2024-01-16"]


@pytest.mark.unit
def test_1b_generate_observation_dates_respects_the_date_bounds():
    dates = [f"2024-01-{d:02d}" for d in range(1, 21)]
    result = qb.generate_observation_dates(dates, "2024-01-05", "2024-01-15", step_sessions=5)
    assert result[0] >= "2024-01-05"
    assert result[-1] <= "2024-01-15"
    assert result == ["2024-01-05", "2024-01-10", "2024-01-15"]


@pytest.mark.unit
def test_1c_step_sessions_below_one_is_rejected():
    with pytest.raises(ValueError):
        qb.generate_observation_dates(["2024-01-01"], "2024-01-01", "2024-01-01", step_sessions=0)


# ---------------------------------------------------------------------------
# 2-9. Each baseline rule
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_2_momentum_baseline_is_reused_unmodified(monkeypatch):
    """quant_baselines.momentum must delegate to baselines.momentum_baseline
    -- never redefine its logic (fork spec: "Reutilizar EXACTAMENTE...
    No cambiar su definicion")."""
    from tradingagents.run2 import baselines as real_baselines

    calls = []
    original = real_baselines.momentum_baseline

    def _spy(features):
        calls.append(features)
        return original(features)

    monkeypatch.setattr(qbl, "momentum_baseline", _spy)
    f = _features(return_20d=5.0, return_60d=10.0, price_above_sma50=True)
    result = qbl.momentum(f)
    assert calls == [f]
    assert result["prediction"] == "LONG"


@pytest.mark.unit
def test_3_moving_average_baseline_is_reused_unmodified(monkeypatch):
    from tradingagents.run2 import baselines as real_baselines

    calls = []
    original = real_baselines.moving_average_baseline

    def _spy(features):
        calls.append(features)
        return original(features)

    monkeypatch.setattr(qbl, "moving_average_baseline", _spy)
    f = _features(sma50=110.0, sma200=100.0, price_above_sma50=True, price_above_sma200=True)
    result = qbl.moving_average(f)
    assert calls == [f]
    assert result["prediction"] == "LONG"


@pytest.mark.unit
def test_4_return20_baseline():
    assert qbl.return20(_features(return_20d=1.5))["prediction"] == "LONG"
    assert qbl.return20(_features(return_20d=-1.5))["prediction"] == "SHORT"
    assert qbl.return20(_features(return_20d=0.0))["prediction"] == "NO_PREDICTION"
    assert qbl.return20(_features(return_20d=None))["prediction"] == "NO_PREDICTION"


@pytest.mark.unit
def test_5_return60_baseline():
    assert qbl.return60(_features(return_60d=2.0))["prediction"] == "LONG"
    assert qbl.return60(_features(return_60d=-2.0))["prediction"] == "SHORT"
    assert qbl.return60(_features(return_60d=None))["prediction"] == "NO_PREDICTION"


@pytest.mark.unit
def test_6_contrarian_return20_is_the_exact_sign_inverse_of_return20():
    assert qbl.contrarian_return20(_features(return_20d=1.5))["prediction"] == "SHORT"
    assert qbl.contrarian_return20(_features(return_20d=-1.5))["prediction"] == "LONG"
    assert qbl.contrarian_return20(_features(return_20d=None))["prediction"] == "NO_PREDICTION"


@pytest.mark.unit
def test_7_contrarian_return60_is_the_exact_sign_inverse_of_return60():
    assert qbl.contrarian_return60(_features(return_60d=2.0))["prediction"] == "SHORT"
    assert qbl.contrarian_return60(_features(return_60d=-2.0))["prediction"] == "LONG"


@pytest.mark.unit
def test_8_contrarian_momentum_inverts_only_the_directional_output():
    long_features = _features(return_20d=5.0, return_60d=10.0, price_above_sma50=True)
    short_features = _features(return_20d=-5.0, return_60d=-10.0, price_above_sma50=False)
    no_position_features = _features()  # insufficient history -> momentum says NO_POSITION

    assert qbl.momentum(long_features)["prediction"] == "LONG"
    assert qbl.contrarian_momentum(long_features)["prediction"] == "SHORT"

    assert qbl.momentum(short_features)["prediction"] == "SHORT"
    assert qbl.contrarian_momentum(short_features)["prediction"] == "LONG"

    assert qbl.momentum(no_position_features)["prediction"] == "NO_PREDICTION"
    assert qbl.contrarian_momentum(no_position_features)["prediction"] == "NO_PREDICTION"


@pytest.mark.unit
def test_9_rsi_contrarian_uses_fixed_70_30_thresholds_no_tuning():
    assert qbl.rsi_contrarian(_features(rsi14=75.0))["prediction"] == "SHORT"
    assert qbl.rsi_contrarian(_features(rsi14=70.0))["prediction"] == "SHORT"
    assert qbl.rsi_contrarian(_features(rsi14=25.0))["prediction"] == "LONG"
    assert qbl.rsi_contrarian(_features(rsi14=30.0))["prediction"] == "LONG"
    assert qbl.rsi_contrarian(_features(rsi14=50.0))["prediction"] == "NO_PREDICTION"
    assert qbl.rsi_contrarian(_features(rsi14=None))["prediction"] == "NO_PREDICTION"


# ---------------------------------------------------------------------------
# 10. NO_PREDICTION (rule abstention) vs NO_POSITION (real market outcome)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_10_no_prediction_is_distinct_from_the_real_no_position_outcome():
    # A baseline that abstains reports "NO_PREDICTION", never "NO_POSITION".
    assert qbl.momentum(_features())["prediction"] == "NO_PREDICTION"
    assert qbl.return20(_features())["prediction"] == "NO_PREDICTION"
    # score_prediction never scores an abstention either way.
    assert qb.score_prediction("NO_PREDICTION", "LONG") is None
    assert qb.score_prediction("NO_PREDICTION", "NO_POSITION") is None
    # The real market outcome vocabulary (evaluator.py) is untouched --
    # NO_POSITION only ever comes from evaluate_first_barrier, never a baseline.
    assert "NO_POSITION" not in {rule(_features())["prediction"] for rule in qbl.QUANT_BASELINES.values()}


# ---------------------------------------------------------------------------
# 11-14. Accuracy / coverage / LONG-SHORT accuracy / majority-class math
# ---------------------------------------------------------------------------


def _row(actual, prediction, **extra):
    """A synthetic observation row with EVERY baseline set to the same
    ``prediction`` -- fine for testing the aggregation math in
    ``summarize``/``_metrics_for_baseline`` generically (via the
    ``"return20"`` column), since these tests exercise the arithmetic, not
    any particular rule's own logic (see the dedicated per-baseline tests
    above for that)."""
    row = {"analysis_date": "2024-01-01", "actual_first_barrier": actual}
    for name in qbl.QUANT_BASELINES:
        row[f"{name}_prediction"] = prediction
        row[f"{name}_correct"] = qb.score_prediction(prediction, actual)
    row.update(extra)
    return row


@pytest.mark.unit
def test_11_accuracy_pct_counts_only_scoreable_predictions():
    rows = [
        _row("LONG", "LONG"), _row("LONG", "SHORT"), _row("SHORT", "SHORT"),
        _row("NO_POSITION", "LONG"),  # not scoreable: real outcome not evaluable
        _row("LONG", "NO_PREDICTION"),  # not scoreable: no prediction made
    ]
    metrics = qb._metrics_for_baseline(rows, "return20")
    assert metrics["correct"] == 2
    assert metrics["incorrect"] == 1
    assert metrics["accuracy_pct"] == pytest.approx(2 / 3 * 100.0)


@pytest.mark.unit
def test_12_coverage_pct_is_predictions_made_over_total_observations():
    rows = [_row("LONG", "LONG"), _row("SHORT", "NO_PREDICTION"), _row("LONG", "SHORT"), _row("SHORT", "NO_PREDICTION")]
    metrics = qb._metrics_for_baseline(rows, "return20")
    assert metrics["predictions_made"] == 2
    assert metrics["total_observations"] == 4
    assert metrics["coverage_pct"] == pytest.approx(50.0)
    assert metrics["no_prediction_count"] == 2


@pytest.mark.unit
def test_13_long_and_short_accuracy_are_computed_independently():
    rows = [
        _row("LONG", "LONG"), _row("LONG", "LONG"), _row("SHORT", "LONG"),  # LONG: 2/3 correct
        _row("SHORT", "SHORT"), _row("LONG", "SHORT"),  # SHORT: 1/2 correct
    ]
    metrics = qb._metrics_for_baseline(rows, "return20")
    assert metrics["long_predictions"] == 3
    assert metrics["long_correct"] == 2
    assert metrics["long_accuracy_pct"] == pytest.approx(2 / 3 * 100.0)
    assert metrics["short_predictions"] == 2
    assert metrics["short_correct"] == 1
    assert metrics["short_accuracy_pct"] == pytest.approx(50.0)


@pytest.mark.unit
def test_14_majority_class_accuracy_reflects_class_imbalance():
    actual_counts = {"actual_long_count": 55, "actual_short_count": 45}
    assert qb._majority_class_accuracy_pct(actual_counts) == pytest.approx(55.0)

    balanced = {"actual_long_count": 10, "actual_short_count": 10}
    assert qb._majority_class_accuracy_pct(balanced) == pytest.approx(50.0)

    none_evaluable = {"actual_long_count": 0, "actual_short_count": 0}
    assert qb._majority_class_accuracy_pct(none_evaluable) is None


# ---------------------------------------------------------------------------
# 15. Yearly (and quarterly) grouping
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_15_summarize_groups_observations_by_year_and_quarter():
    rows = [
        _row("LONG", "LONG", analysis_date="2024-02-01"),
        _row("SHORT", "SHORT", analysis_date="2024-08-01"),
        _row("LONG", "SHORT", analysis_date="2025-03-01"),
    ]
    summary = qb.summarize(
        rows, ticker="NVDA", start_date="2024-01-01", end_date="2025-12-31",
        step_sessions=5, horizon_sessions=40, barrier_pct=5.0,
    )
    assert set(summary["by_year"]) == {"2024", "2025"}
    assert summary["by_year"]["2024"]["total_observations"] == 2
    assert summary["by_year"]["2025"]["total_observations"] == 1
    assert set(summary["by_quarter"]) == {"2024-Q1", "2024-Q3", "2025-Q1"}
    assert summary["overall"]["total_observations"] == 3


# ---------------------------------------------------------------------------
# 16. No future leakage into features or baseline predictions
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_16_build_observation_features_unaffected_by_future_but_label_can_change(monkeypatch):
    base_df = pd.DataFrame({
        "Date": pd.bdate_range("2024-01-01", periods=260),
        "Open": [100.0] * 260, "High": [101.0] * 260, "Low": [99.0] * 260,
        "Close": [100.0] * 260, "Volume": [1_000_000.0] * 260,
    })
    analysis_date = base_df["Date"].iloc[-1].strftime("%Y-%m-%d")

    # Variant A: no real future data exists yet at all.
    monkeypatch.setattr(market_data, "load_ohlcv", lambda *a, **k: base_df.copy())
    monkeypatch.setattr(snapshot_module, "load_ohlcv", lambda *a, **k: base_df.copy())
    monkeypatch.setattr(evaluator, "load_ohlcv", lambda *a, **k: base_df.copy())
    without_future = qb.build_observation(ticker="NVDA", analysis_date=analysis_date, horizon_sessions=40, barrier_pct=5.0)

    # Variant B: append 5 future sessions, the 5th of which pierces +5%.
    future_rows = pd.DataFrame({
        "Date": pd.bdate_range(base_df["Date"].iloc[-1] + pd.Timedelta(days=1), periods=5),
        "Open": [100.0] * 5, "High": [100.0, 100.0, 100.0, 100.0, 106.0],
        "Low": [99.0] * 5, "Close": [100.0] * 5, "Volume": [1_000_000.0] * 5,
    })
    padded = pd.concat([base_df, future_rows], ignore_index=True)
    monkeypatch.setattr(market_data, "load_ohlcv", lambda *a, **k: padded.copy())
    monkeypatch.setattr(snapshot_module, "load_ohlcv", lambda *a, **k: padded.copy())
    monkeypatch.setattr(evaluator, "load_ohlcv", lambda *a, **k: padded.copy())
    with_future = qb.build_observation(ticker="NVDA", analysis_date=analysis_date, horizon_sessions=40, barrier_pct=5.0)

    # Every feature-derived field (point-in-time, <= analysis_date) is identical.
    for key in ("reference_price", "return_20d", "return_60d", "rsi14", "adx14"):
        assert without_future[key] == with_future[key], key
    for name in qbl.QUANT_BASELINES:
        assert without_future[f"{name}_prediction"] == with_future[f"{name}_prediction"], name

    # The real, future-dependent label legitimately differs.
    assert without_future["actual_first_barrier"] == "INCOMPLETE"
    assert with_future["actual_first_barrier"] == "LONG"
    assert with_future["sessions_to_first_barrier"] == 5


# ---------------------------------------------------------------------------
# 17-19. Output files
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_17_observations_csv_is_written_with_one_row_per_date(tmp_path):
    rows = [_row("LONG", "LONG", analysis_date="2024-01-01"), _row("SHORT", "SHORT", analysis_date="2024-01-08")]
    path = qb.write_observations_csv(rows, tmp_path / "observations.csv")
    assert path.exists()
    df = pd.read_csv(path)
    assert len(df) == 2
    assert list(df["analysis_date"]) == ["2024-01-01", "2024-01-08"]


@pytest.mark.unit
def test_18_summary_json_is_written_and_valid(tmp_path):
    rows = [_row("LONG", "LONG", analysis_date="2024-01-01")]
    summary = qb.summarize(
        rows, ticker="NVDA", start_date="2024-01-01", end_date="2024-12-31",
        step_sessions=5, horizon_sessions=40, barrier_pct=5.0,
    )
    path = qb.write_summary_json(summary, tmp_path / "summary.json")
    assert path.exists()
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["meta"]["ticker"] == "NVDA"
    assert loaded["overall"]["total_observations"] == 1


@pytest.mark.unit
def test_19_summary_markdown_is_written_and_readable(tmp_path):
    rows = [_row("LONG", "LONG", analysis_date="2024-01-01")]
    summary = qb.summarize(
        rows, ticker="NVDA", start_date="2024-01-01", end_date="2024-12-31",
        step_sessions=5, horizon_sessions=40, barrier_pct=5.0,
    )
    path = qb.write_summary_markdown(summary, tmp_path / "summary.md")
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "NVDA" in text
    assert "## Overall" in text
    assert "return20" in text


# ---------------------------------------------------------------------------
# 20. No LLM call anywhere in this benchmark
# ---------------------------------------------------------------------------


_FORBIDDEN_LLM_MODULE_PREFIXES = (
    "tradingagents.llm_clients",
    "tradingagents.agents.structured",
    "tradingagents.run2.pipeline",
    "tradingagents.run2.stages",
    "tradingagents.run2.prompts",
    "tradingagents.run2.validation",
    "langchain",
)


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    names |= {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
    return names


@pytest.mark.unit
def test_20_quant_benchmark_modules_never_import_any_llm_machinery():
    for filename in ("quant_benchmark.py", "quant_baselines.py"):
        imported = _imported_modules(RUN2_PACKAGE_DIR / filename)
        violations = {
            name for name in imported
            if any(name == prefix or name.startswith(prefix + ".") for prefix in _FORBIDDEN_LLM_MODULE_PREFIXES)
        }
        assert not violations, f"{filename} imports LLM machinery: {violations}"
