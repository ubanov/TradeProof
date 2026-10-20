"""run2's own, self-contained provenance (fork spec: "run2" hardening --
"Quiero que la provenance de run2 sea especifica de run2"). ``pipeline.py``
always passes ``prompt_hashes`` explicitly to ``build_run_provenance``, so it
never falls back to ``llm_trace.hash_prompt_files()`` -- hashing the ENTIRE
main-pipeline ``tradingagents/prompts/`` tree (news, fundamentals, Bull/Bear,
Research Manager, Trader, Risk, Portfolio Manager), none of which run2
reads. These tests prove that never happens and that run2's own two prompts
(Market Regime Analyzer, Binary First Barrier Predictor -- plus the generic
repair prompt, when used) are the only things fingerprinted. There is no
Challenger anymore, so no third prompt and no "review" model.
"""

from __future__ import annotations

import pytest

import tradingagents.run2.pipeline as run2_pipeline
from tradingagents.llm_trace import hash_prompt_files
from tradingagents.run2.schemas import BarrierSignal, MarketFeatures, RegimeAnalysis

RUN2_PROMPT_KEYS = {"market_regime_analyzer", "binary_first_barrier_predictor"}


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


def _regime():
    return RegimeAnalysis(
        primary_trend="UP", secondary_trend="UP", momentum="POSITIVE",
        volatility_regime="NORMAL", trend_strength=0.6, regime_confidence=0.6,
    )


def _signal(direction="LONG", confidence=0.6):
    return BarrierSignal(
        direction=direction, confidence=confidence, thesis="t",
        evidence_for=["a"], evidence_against=[], invalidation_conditions=["x"],
    )


def _features(**overrides):
    base = {"as_of": "2026-01-07", "sessions_available": 300}
    base.update(overrides)
    return MarketFeatures(**base)


def _run(monkeypatch, tmp_path, *, signal=None, generation=None, validator_llm=None):
    monkeypatch.setattr(run2_pipeline, "fetch_market_context", lambda *a, **k: "## fake market context")
    monkeypatch.setattr(
        run2_pipeline, "fetch_price_reference",
        lambda ticker, date: {"symbol": ticker, "reference_date": date, "close": 150.0},
    )
    monkeypatch.setattr(run2_pipeline, "compute_market_features", lambda ticker, date: _features(as_of=date))

    generation = generation or _FakeChatModel(structured_by_schema={
        RegimeAnalysis: _regime(),
        BarrierSignal: signal if signal is not None else _signal(),
    })
    by_model = {"gen-model": generation}
    if validator_llm is not None:
        by_model["val-model"] = validator_llm

    monkeypatch.setattr(
        run2_pipeline, "create_llm_client",
        lambda provider, model, base_url=None, **kwargs: _FakeClient(by_model[model]),
    )

    kwargs = {
        "ticker": "NVDA", "analysis_date": "2026-01-07", "generation_llm": "gen-model",
        "llm_provider": "ollama", "root": tmp_path,
    }
    if validator_llm is not None:
        kwargs["validator_llm"] = "val-model"
    return run2_pipeline.run_run2_analysis(**kwargs)


@pytest.mark.unit
def test_1_run2_provenance_never_contains_the_main_pipelines_prompt_hashes(monkeypatch, tmp_path):
    result, _ = _run(monkeypatch, tmp_path)
    run2_hashes = set(result["provenance"]["prompts"]["hashes"].values())
    main_pipeline_hashes = set(hash_prompt_files().values())
    assert not (run2_hashes & main_pipeline_hashes)


@pytest.mark.unit
def test_2_run2_provenance_contains_exactly_its_own_two_prompts_when_no_repair_happened(monkeypatch, tmp_path):
    result, _ = _run(monkeypatch, tmp_path)
    assert set(result["provenance"]["prompts"]["hashes"]) == RUN2_PROMPT_KEYS


@pytest.mark.unit
def test_3_run2_provenance_adds_the_repair_recovery_fingerprint_only_when_a_repair_happened(monkeypatch, tmp_path):
    generation = _FakeChatModel(
        structured_by_schema={RegimeAnalysis: RuntimeError("boom")},
        plain_responses=["free text regime description, trend_strength 0.6, regime_confidence 0.6"],
    )
    validator = _FakeChatModel(plain_responses=['{"primary_trend": "UP", "secondary_trend": "UP", '
                                                 '"momentum": "POSITIVE", "volatility_regime": "NORMAL", '
                                                 '"trend_strength": 0.6, "regime_confidence": 0.6}'])
    result, _ = _run(monkeypatch, tmp_path, generation=generation, validator_llm=validator)
    assert set(result["provenance"]["prompts"]["hashes"]) == RUN2_PROMPT_KEYS | {"repair_recovery"}


@pytest.mark.unit
def test_4_run2_provenance_nested_block_has_ticker_date_horizon_and_barrier_pct(monkeypatch, tmp_path):
    result, _ = _run(monkeypatch, tmp_path)
    run2_block = result["provenance"]["run2"]
    assert run2_block["ticker"] == "NVDA"
    assert run2_block["analysis_date"] == "2026-01-07"
    assert run2_block["horizon_sessions"] == result["horizon_sessions"]
    assert run2_block["barrier_pct"] == result["barrier_pct"]
    assert run2_block["schema_version"] == result["schema_version"]
    # No confidence thresholds exist anymore (fork spec: "First Barrier" --
    # NO_POSITION/LONG/SHORT are never gated by confidence).
    assert "min_confidence" not in run2_block
    assert "thresholds" not in run2_block


@pytest.mark.unit
def test_5_run2_provenance_still_has_git_commit_dirty_and_decision_memory_false(monkeypatch, tmp_path):
    result, _ = _run(monkeypatch, tmp_path)
    provenance = result["provenance"]
    assert "commit" in provenance["git"]
    assert "dirty" in provenance["git"]
    assert provenance["execution"]["decision_memory_enabled"] is False
    assert provenance["execution"]["decision_memory_root"] is None


@pytest.mark.unit
def test_6_models_have_no_review_role_anymore(monkeypatch, tmp_path):
    """Challenger (the only stage that ever used a "review" model) is gone
    -- run2's own ``models``/``reasoning_effort`` dicts should not carry a
    "review" entry at all."""
    result, _ = _run(monkeypatch, tmp_path)
    assert "review" not in result["models"]
    assert "review" not in result["reasoning_effort"]
    assert set(result["models"]) == {"generation", "validator"}
