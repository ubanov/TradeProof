"""End-to-end run2 pipeline -- binary First Barrier experiment (fork README:
"run2"), with every external dependency (LLM clients, market data) replaced
by deterministic fakes so the test suite never makes a network or provider
call. Covers the fork spec's required-test items: no confidence threshold
(a valid LONG/SHORT signal IS the final decision outright), persistent stage
failure degrading to UNRESOLVED (fail-closed, never NO_POSITION -- that
label no longer represents a predictor output), exactly two LLM calls with
no repair (Regime + Binary First Barrier Predictor, no Challenger), a repair
incrementing the count correctly, and that run2's own inference never leaks
future data into its inputs.
"""

from __future__ import annotations

import pandas as pd
import pytest

import tradingagents.run2.pipeline as run2_pipeline
from tradingagents.run2.schemas import BarrierSignal, MarketFeatures, RegimeAnalysis


class _Resp:
    def __init__(self, content: str):
        self.content = content


class _FakeChatModel:
    def __init__(self, *, structured_by_schema=None, plain_responses=None):
        self.structured_by_schema = structured_by_schema or {}
        self.plain_responses = list(plain_responses or [])
        self.plain_calls = 0

    def with_structured_output(self, schema):
        outer = self

        class _Bound:
            def invoke(self, prompt, **kwargs):
                spec = outer.structured_by_schema.get(schema)
                if spec is None:
                    raise AssertionError(f"no structured result/error configured for {schema}")
                if isinstance(spec, Exception):
                    raise spec
                return spec

        return _Bound()

    def invoke(self, prompt, **kwargs):
        self.plain_calls += 1
        content = self.plain_responses.pop(0) if self.plain_responses else ""
        return _Resp(content)


class _FakeClient:
    def __init__(self, llm):
        self._llm = llm

    def get_llm(self):
        return self._llm


def _regime(**overrides):
    base = {
        "primary_trend": "UP", "secondary_trend": "UP", "momentum": "POSITIVE",
        "volatility_regime": "NORMAL", "trend_strength": 0.6, "regime_confidence": 0.6,
    }
    base.update(overrides)
    return RegimeAnalysis(**base)


def _signal(**overrides):
    base = {
        "direction": "LONG", "confidence": 0.55, "thesis": "t",
        "evidence_for": ["a"], "evidence_against": [], "invalidation_conditions": ["x"],
    }
    base.update(overrides)
    return BarrierSignal(**base)


def _features(**overrides):
    base = {
        "as_of": "2026-01-07", "sessions_available": 300,
        "return_20d": 4.0, "return_60d": 8.0, "sma50": 100.0, "sma200": 90.0,
        "price_above_sma50": True, "price_above_sma200": True,
    }
    base.update(overrides)
    return MarketFeatures(**base)


def _patch_market_data(monkeypatch, *, features=None):
    monkeypatch.setattr(run2_pipeline, "fetch_market_context", lambda *a, **k: "## fake market context")
    monkeypatch.setattr(
        run2_pipeline, "fetch_price_reference",
        lambda ticker, date: {"symbol": ticker, "reference_date": date, "close": 150.0},
    )
    monkeypatch.setattr(
        run2_pipeline, "compute_market_features",
        lambda ticker, date: features if features is not None else _features(as_of=date),
    )


def _patch_llm_clients(monkeypatch, *, generation_llm, validator_llm=None):
    by_model = {"gen-model": generation_llm}
    if validator_llm is not None:
        by_model["val-model"] = validator_llm

    def _fake_create_llm_client(provider, model, base_url=None, **kwargs):
        return _FakeClient(by_model[model])

    monkeypatch.setattr(run2_pipeline, "create_llm_client", _fake_create_llm_client)


def _run(monkeypatch, tmp_path, *, regime=None, signal=None, validator_llm=None, features=None, **kw):
    _patch_market_data(monkeypatch, features=features)
    generation = _FakeChatModel(structured_by_schema={
        RegimeAnalysis: regime if regime is not None else _regime(),
        BarrierSignal: signal if signal is not None else _signal(),
    })
    _patch_llm_clients(monkeypatch, generation_llm=generation, validator_llm=validator_llm)

    kw.setdefault("ticker", "NVDA")
    kw.setdefault("analysis_date", "2026-01-07")
    kw.setdefault("generation_llm", "gen-model")
    kw.setdefault("llm_provider", "ollama")
    kw.setdefault("root", tmp_path)
    if validator_llm is not None:
        kw.setdefault("validator_llm", "val-model")
    return run2_pipeline.run_run2_analysis(**kw)


@pytest.mark.unit
def test_1_valid_long_signal_is_final_decision_regardless_of_low_confidence(monkeypatch, tmp_path):
    result, path = _run(monkeypatch, tmp_path, signal=_signal(direction="LONG", confidence=0.01))
    assert result["final_decision"]["action"] == "LONG"
    assert result["final_decision"]["reason"] == "valid_binary_first_barrier_signal"
    assert path.exists()


@pytest.mark.unit
def test_2_valid_short_signal_is_final_decision_regardless_of_low_confidence(monkeypatch, tmp_path):
    result, _ = _run(monkeypatch, tmp_path, signal=_signal(direction="SHORT", confidence=0.01))
    assert result["final_decision"]["action"] == "SHORT"
    assert result["final_decision"]["reason"] == "valid_binary_first_barrier_signal"


@pytest.mark.unit
def test_3_high_confidence_does_not_change_anything_either(monkeypatch, tmp_path):
    """No threshold anywhere -- confidence is purely diagnostic (fork spec:
    "NO aplicar ningun threshold posterior de confidence"). Symmetric check
    to test_1/test_2: neither a very low nor a very high confidence changes
    which direction is recorded."""
    result, _ = _run(monkeypatch, tmp_path, signal=_signal(direction="SHORT", confidence=0.99))
    assert result["final_decision"]["action"] == "SHORT"


@pytest.mark.unit
def test_4_persistent_regime_failure_ends_in_unresolved_never_no_position(monkeypatch, tmp_path):
    result, _ = _run(monkeypatch, tmp_path, regime=RuntimeError("boom"))
    assert result["final_decision"]["action"] == "UNRESOLVED"
    assert result["final_decision"]["reason"] == "regime_stage_failed"
    assert result["regime_analysis"] is None
    assert result["barrier_signal"] is None


@pytest.mark.unit
def test_5_persistent_signal_failure_ends_in_unresolved_never_no_position(monkeypatch, tmp_path):
    result, _ = _run(monkeypatch, tmp_path, signal=RuntimeError("boom"))
    assert result["final_decision"]["action"] == "UNRESOLVED"
    assert result["final_decision"]["reason"] == "signal_stage_failed"
    assert result["regime_analysis"] is not None
    assert result["barrier_signal"] is None


@pytest.mark.unit
@pytest.mark.parametrize("failure_kwargs", [
    {"regime": RuntimeError("boom")},
    {"signal": RuntimeError("boom")},
])
def test_6_every_persistent_stage_failure_is_unresolved_never_long_short_or_no_position(monkeypatch, tmp_path, failure_kwargs):
    result, _ = _run(monkeypatch, tmp_path, **failure_kwargs)
    assert result["final_decision"]["action"] == "UNRESOLVED"
    assert result["final_decision"]["action"] not in ("LONG", "SHORT", "NO_POSITION")


@pytest.mark.unit
def test_7_repair_is_attempted_at_most_once_and_a_failed_repair_still_resolves_to_unresolved(monkeypatch, tmp_path):
    _patch_market_data(monkeypatch)
    generation = _FakeChatModel(
        structured_by_schema={RegimeAnalysis: RuntimeError("structured output failed")},
        plain_responses=["free text regime description with no usable structured numbers"],
    )
    validator = _FakeChatModel(plain_responses=["not valid json at all"])
    _patch_llm_clients(monkeypatch, generation_llm=generation, validator_llm=validator)

    result, _ = run2_pipeline.run_run2_analysis(
        ticker="NVDA", analysis_date="2026-01-07", generation_llm="gen-model",
        validator_llm="val-model", llm_provider="ollama", root=tmp_path,
    )

    assert result["final_decision"]["action"] == "UNRESOLVED"
    assert result["final_decision"]["reason"] == "regime_stage_failed"
    assert result["validation"]["regime"]["status"] == "failed"
    assert result["validation"]["regime"]["repair_attempts"] == 1
    # 1 structured attempt (raised) + 1 freetext fallback + 1 repair attempt
    # = 3 real calls for this one stage; the signal stage never ran.
    assert result["validation"]["regime"]["llm_call_count"] == 3
    assert result["token_usage"]["llm_calls"] == 3


@pytest.mark.unit
def test_8_normal_execution_without_repair_makes_exactly_two_llm_calls(monkeypatch, tmp_path):
    """Regime + Binary First Barrier Predictor only -- there is no
    Challenger (fork spec: "run2" binary simplification, "Llamadas LLM")."""
    result, _ = _run(monkeypatch, tmp_path)
    assert result["token_usage"]["llm_calls"] == 2


@pytest.mark.unit
def test_9_no_challenger_pm_risk_or_bull_bear_stage_exists_in_the_result(monkeypatch, tmp_path):
    result, _ = _run(monkeypatch, tmp_path)
    for forbidden_key in ("challenger", "portfolio_manager", "risk", "bull", "bear", "news", "fundamentals"):
        assert forbidden_key not in result
        assert forbidden_key not in result["validation"]


@pytest.mark.unit
def test_10_a_successful_repair_still_increments_llm_calls_correctly(monkeypatch, tmp_path):
    _patch_market_data(monkeypatch)
    repaired_regime_json = _regime().model_dump_json()
    generation = _FakeChatModel(
        structured_by_schema={
            RegimeAnalysis: RuntimeError("structured output failed"),
            BarrierSignal: _signal(direction="LONG"),
        },
        plain_responses=["free text regime description, trend_strength 0.6, regime_confidence 0.6"],
    )
    validator = _FakeChatModel(plain_responses=[repaired_regime_json])
    _patch_llm_clients(monkeypatch, generation_llm=generation, validator_llm=validator)

    result, _ = run2_pipeline.run_run2_analysis(
        ticker="NVDA", analysis_date="2026-01-07", generation_llm="gen-model",
        validator_llm="val-model", llm_provider="ollama", root=tmp_path,
    )

    assert result["regime_analysis"] is not None
    assert result["validation"]["regime"]["status"] == "repaired"
    assert result["validation"]["regime"]["llm_call_count"] == 3  # structured + fallback + repair
    assert result["validation"]["barrier_signal"]["llm_call_count"] == 1  # clean structured success
    # 3 (regime, repaired) + 1 (signal) = 4.
    assert result["token_usage"]["llm_calls"] == 4


@pytest.mark.unit
def test_11_run2_writes_under_its_own_separate_root(monkeypatch, tmp_path):
    result, path = _run(monkeypatch, tmp_path)
    assert tmp_path in path.parents
    assert path.parts[-5:-2] == ("NVDA", "2026-01-07", "runs")
    assert (path.parent / "complete_report.md").exists()


@pytest.mark.unit
def test_12_trace_llm_writes_debug_calls(monkeypatch, tmp_path):
    _, path = _run(monkeypatch, tmp_path, trace_llm=True)
    debug_dir = path.parent / "debug" / "calls"
    assert debug_dir.exists()
    assert any(p.name.endswith("_regime") for p in debug_dir.iterdir())
    assert any(p.name.endswith("_barrier_signal") for p in debug_dir.iterdir())


@pytest.mark.unit
def test_13_future_analysis_date_is_rejected(monkeypatch, tmp_path):
    _patch_market_data(monkeypatch)
    _patch_llm_clients(monkeypatch, generation_llm=_FakeChatModel())
    with pytest.raises(ValueError, match="future"):
        run2_pipeline.run_run2_analysis(
            ticker="NVDA", analysis_date="2999-01-01", generation_llm="gen-model",
            llm_provider="ollama", root=tmp_path,
        )


@pytest.mark.unit
def test_14_barrier_pct_and_horizon_sessions_are_recorded_in_the_result(monkeypatch, tmp_path):
    result, _ = _run(monkeypatch, tmp_path, horizon_sessions=20, barrier_pct=3.0)
    assert result["horizon_sessions"] == 20
    assert result["barrier_pct"] == 3.0


@pytest.mark.unit
def test_15_baselines_are_recorded_but_never_reach_a_prompt(monkeypatch, tmp_path):
    captured_prompts = []

    class _CapturingChatModel(_FakeChatModel):
        def with_structured_output(self, schema):
            bound = super().with_structured_output(schema)
            original_invoke = bound.invoke

            def _invoke(prompt, **kwargs):
                captured_prompts.append(prompt)
                return original_invoke(prompt, **kwargs)

            bound.invoke = _invoke
            return bound

    _patch_market_data(monkeypatch)
    generation = _CapturingChatModel(structured_by_schema={
        RegimeAnalysis: _regime(), BarrierSignal: _signal(),
    })
    _patch_llm_clients(monkeypatch, generation_llm=generation)

    result, _ = run2_pipeline.run_run2_analysis(
        ticker="NVDA", analysis_date="2026-01-07", generation_llm="gen-model",
        llm_provider="ollama", root=tmp_path,
    )

    assert "momentum" in result["baselines"]
    assert "moving_average" in result["baselines"]
    assert captured_prompts, "expected at least one LLM prompt to have been captured"
    for prompt in captured_prompts:
        assert "baseline" not in prompt.lower()


@pytest.mark.unit
def test_16_barrier_symmetry_is_not_a_reason_to_abstain_in_the_prompt(monkeypatch, tmp_path):
    """Fork spec: "run2" binary simplification -- the prompt must forbid the
    equidistance-of-barriers non-argument and must never offer NO_POSITION
    as an option."""
    captured_prompts = []

    class _CapturingChatModel(_FakeChatModel):
        def with_structured_output(self, schema):
            bound = super().with_structured_output(schema)
            original_invoke = bound.invoke

            def _invoke(prompt, **kwargs):
                captured_prompts.append(prompt)
                return original_invoke(prompt, **kwargs)

            bound.invoke = _invoke
            return bound

    _patch_market_data(monkeypatch)
    generation = _CapturingChatModel(structured_by_schema={RegimeAnalysis: _regime(), BarrierSignal: _signal()})
    _patch_llm_clients(monkeypatch, generation_llm=generation)

    run2_pipeline.run_run2_analysis(
        ticker="NVDA", analysis_date="2026-01-07", generation_llm="gen-model",
        llm_provider="ollama", root=tmp_path,
    )

    barrier_prompt = next(p for p in captured_prompts if "Binary First Barrier Predictor" in p)
    assert "MUST choose LONG or SHORT" in barrier_prompt
    assert "symmetric" in barrier_prompt.lower()
    assert "NO_POSITION" in barrier_prompt  # named explicitly as forbidden
    assert "cannot decide" in barrier_prompt.lower()


@pytest.mark.unit
def test_17_inference_result_is_unaffected_by_wildly_different_future_rows(monkeypatch, tmp_path):
    """run2's own inference (market context, price reference, market
    features) must never change because of rows dated after analysis_date
    -- the ENTIRE result.json (minus run_id/timestamps, which legitimately
    differ between calls) must be identical whether or not the underlying
    data source includes extreme future rows (fork spec: "run2" First
    Barrier, leakage tests)."""
    import tradingagents.dataflows.vendors.yahoo.snapshot as snapshot_module
    import tradingagents.run2.market_data as market_data

    base_df = pd.DataFrame({
        "Date": pd.bdate_range("2024-01-01", periods=260),
        "Open": [100.0] * 260, "High": [101.0] * 260, "Low": [99.0] * 260,
        "Close": [100.0 + i * 0.01 for i in range(260)], "Volume": [1_000_000.0] * 260,
    })
    future_rows = pd.DataFrame({
        "Date": pd.bdate_range(base_df["Date"].iloc[-1] + pd.Timedelta(days=1), periods=30),
        "Open": [999_999.0] * 30, "High": [999_999.0] * 30, "Low": [999_999.0] * 30,
        "Close": [999_999.0] * 30, "Volume": [1.0] * 30,
    })
    analysis_date = base_df["Date"].iloc[-1].strftime("%Y-%m-%d")

    def _run_with(df):
        monkeypatch.setattr(market_data, "load_ohlcv", lambda *a, **k: df.copy())
        monkeypatch.setattr(snapshot_module, "load_ohlcv", lambda *a, **k: df.copy())
        generation = _FakeChatModel(structured_by_schema={RegimeAnalysis: _regime(), BarrierSignal: _signal()})
        _patch_llm_clients(monkeypatch, generation_llm=generation)
        result, _ = run2_pipeline.run_run2_analysis(
            ticker="NVDA", analysis_date=analysis_date, generation_llm="gen-model",
            llm_provider="ollama", root=tmp_path, seed=42,
        )
        for key in ("run_id", "created_at", "timing"):
            result.pop(key, None)
        return result

    without_future = _run_with(base_df)
    with_future = _run_with(pd.concat([base_df, future_rows], ignore_index=True))

    assert with_future == without_future
