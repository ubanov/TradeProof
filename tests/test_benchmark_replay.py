"""Benchmark replay: a controlled, read-only model comparison that reuses
the frozen inputs of an already-completed --trace-llm run (see fork README:
"Benchmark replay"). Covers trace loading, final-call identification,
message reconstruction, model-spec parsing, seed derivation, structured/
free-text preservation, and the read-only/no-tool-execution guarantees.
Numbered per the task's own "TESTS" list (1-15)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, ToolMessage

import tradingagents.benchmark_replay as br
from tradingagents.benchmark_replay import (
    ReplayError,
    _normalize_tool_history,
    _safe_path_component,
    find_source_call,
    parse_model_spec,
    reconstruct_messages,
    resolve_debug_dir,
    run_benchmark,
)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _write_call(
    calls_dir: Path,
    number: int,
    stage: str,
    *,
    attempt: str = "normal",
    suffix: str | None = None,
    llm_role: str | None = "generation",
    call_identity: str | None = None,
    model: str = "source-model",
    provider: str = "openai-chat",
    messages: list[dict] | None = None,
    tools: list[dict] | None = None,
    generation_params: dict | None = None,
    tool_call_count: int = 0,
    finish_reason: str = "stop",
    structured_requested: bool = False,
) -> Path:
    dir_suffix = f"_{suffix}" if suffix else ""
    call_dir = calls_dir / f"{number:03d}_{stage.replace(' ', '_')}{dir_suffix}"
    call_dir.mkdir(parents=True, exist_ok=True)
    _write_json(call_dir / "request.json", {
        "call_number": number, "stage": stage, "attempt": attempt,
        "llm_role": llm_role, "call_identity": call_identity,
        "model": model, "provider": provider,
        "messages": messages if messages is not None else [
            {"role": "system", "content": "sys"}, {"role": "human", "content": "hi"},
        ],
        "tools": tools, "response_format": None,
        "generation_params": generation_params or {"seed": 42},
    })
    _write_json(call_dir / "metadata.json", {
        "call_number": number, "stage": stage, "attempt": attempt,
        "success": True, "structured_requested": structured_requested,
        "fallback_triggered": False, "tool_call_count": tool_call_count,
        "finish_reason": finish_reason,
    })
    return call_dir


def _build_trace(tmp_path: Path, ticker: str = "LLY") -> Path:
    """A minimal but realistic trace: one tool round + one final call for
    Market Analyst, a single structured call for Sentiment, and a
    structured-then-fallback pair plus a Validator call for Research
    Manager (to exercise call_identity-based filtering)."""
    run_dir = tmp_path / "run"
    debug_dir = run_dir / ticker / "debug"
    calls_dir = debug_dir / "calls"
    _write_json(debug_dir / "run_manifest.json", {
        "ticker": ticker, "provider": "ollama", "base_seed": 12345,
    })

    # Market Analyst: one tool round, then the final free-text synthesis.
    _write_call(
        calls_dir, 1, "Market Analyst", call_identity=None, llm_role="generation",
        messages=[
            {"role": "system", "content": "sys"}, {"role": "human", "content": ticker},
            {"role": "ai", "content": "", "tool_calls": [{"name": "get_stock_data", "args": {}, "id": "call_1", "type": "tool_call"}]},
        ],
        tools=[{"type": "function", "function": {"name": "get_stock_data"}}],
        tool_call_count=1, finish_reason="tool_calls",
    )
    _write_call(
        calls_dir, 2, "Market Analyst", call_identity=None, llm_role="generation",
        messages=[
            {"role": "system", "content": "sys"}, {"role": "human", "content": ticker},
            {"role": "ai", "content": "", "tool_calls": [{"name": "get_stock_data", "args": {}, "id": "call_1", "type": "tool_call"}]},
            {"role": "tool", "content": "price=100", "name": "get_stock_data"},
        ],
        tool_call_count=0, finish_reason="stop",
    )

    # Sentiment Analyst: a single structured call.
    _write_call(
        calls_dir, 3, "Sentiment Analyst", suffix="structured", attempt="structured",
        call_identity="sentiment_analyst", llm_role="generation",
        messages=[{"role": "human", "content": "sentiment prompt"}],
        tools=[{"type": "function", "function": {"name": "SentimentReport"}}],
        tool_call_count=1, finish_reason="tool_calls", structured_requested=True,
    )

    # Research Manager: structured attempt (fails to parse) then fallback
    # (the one actually used), plus a Validator call sharing the same stage
    # tag that must be excluded from the "final call" search.
    _write_call(
        calls_dir, 4, "Research Manager", suffix="structured", attempt="structured",
        call_identity="research_manager", llm_role="generation",
        messages=[{"role": "human", "content": "research manager prompt"}],
        tools=[{"type": "function", "function": {"name": "ResearchPlan"}}],
        tool_call_count=1, finish_reason="tool_calls", structured_requested=True,
        generation_params={"seed": 99},
    )
    _write_call(
        calls_dir, 5, "Research Manager", suffix="fallback", attempt="fallback",
        call_identity="research_manager", llm_role="generation",
        messages=[{"role": "human", "content": "research manager prompt"}],
        tool_call_count=0, finish_reason="stop", structured_requested=False,
        generation_params={"seed": 100},
    )
    _write_call(
        calls_dir, 6, "Research Manager", suffix="structured", attempt="structured",
        call_identity="research_manager_validation", llm_role="validator",
        messages=[{"role": "human", "content": "validator prompt"}],
        tool_call_count=0, finish_reason="stop", structured_requested=True,
        generation_params={"seed": 7},
    )
    return run_dir


class _FakeLLM:
    """No structured support unless given a schema; records every call.

    ``reenter_tool_mode`` only fires when the invocation actually carries a
    ``tools`` kwarg -- a model cannot attempt a tool call when none is
    declared, so this mirrors the real failure mode (a provider ignoring
    ``tool_choice="none"`` while ``tools`` is still present) rather than an
    unconditional one. ``retry_response_text``, when set, is the text
    returned specifically on a call made WITHOUT ``tools`` (i.e. the
    tool-history-normalized retry) -- letting a test simulate a model that
    fails with tools bound but can synthesize fine once they're stripped.
    """

    def __init__(
        self, response_text="free-text response", reenter_tool_mode=False,
        structured_fails=False, retry_response_text=None,
    ):
        self.response_text = response_text
        self.reenter_tool_mode = reenter_tool_mode
        self.structured_fails = structured_fails
        self.retry_response_text = retry_response_text
        self.invoke_calls: list[dict] = []
        self.structured_calls: list[dict] = []

    def invoke(self, messages, **kwargs):
        self.invoke_calls.append({"messages": messages, "kwargs": kwargs})
        has_tools = "tools" in kwargs
        if self.reenter_tool_mode and has_tools:
            return SimpleNamespace(
                content="",
                tool_calls=[{"name": "get_news", "args": {}, "id": "call_x", "type": "tool_call"}],
                usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                response_metadata={"finish_reason": "tool_calls"},
            )
        text = self.response_text if (has_tools or self.retry_response_text is None) else self.retry_response_text
        return SimpleNamespace(
            content=text,
            tool_calls=None,
            usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            response_metadata={"finish_reason": "stop"},
        )

    def with_structured_output(self, schema, include_raw=False):
        assert include_raw is True
        parent = self

        class _Wrapper:
            def invoke(self, messages, **kwargs):
                parent.structured_calls.append({"messages": messages, "kwargs": kwargs})
                raw = SimpleNamespace(
                    content="{}", usage_metadata={"input_tokens": 20, "output_tokens": 8, "total_tokens": 28},
                    response_metadata={"finish_reason": "stop"},
                )
                if parent.structured_fails:
                    return {"raw": raw, "parsed": None, "parsing_error": ValueError("schema mismatch")}
                return {"raw": raw, "parsed": _minimal_valid_instance(schema), "parsing_error": None}

        return _Wrapper()


def _minimal_valid_instance(schema):
    """A fully schema-valid instance for the two structured schemas this
    test fixture's stages use -- a real with_structured_output() call
    always returns a complete, validated instance (never a partial
    model_construct()), so the fake must too."""
    if schema.__name__ == "SentimentReport":
        return schema.model_validate({
            "overall_band": "Neutral", "overall_score": 0.0, "confidence": "low", "narrative": "test",
        })
    if schema.__name__ == "ResearchPlan":
        return schema.model_validate({
            "recommendation": "Hold", "rationale": "test", "strategic_actions": "test",
        })
    return schema.model_construct()


class _FakeClient:
    def __init__(self, llm, provider, model):
        self._llm = llm
        self.provider = provider
        self.model = model

    def get_llm(self):
        return self._llm


@pytest.fixture
def fake_llm_factory(monkeypatch):
    """Patches create_llm_client to return a fresh _FakeLLM per call,
    recording every (provider, model, kwargs) it was constructed with.
    ``result.per_model[model_name] = {"response_text": ..., ...}`` lets a
    test give a specific model distinct fake behavior (e.g. a tool-re-entry
    or structured-failure simulation) before calling run_benchmark."""
    created: list[dict] = []
    llms: list[_FakeLLM] = []
    per_model: dict[str, dict] = {}

    def _create(provider, model, base_url=None, **kwargs):
        llm = _FakeLLM(**per_model.get(model, {}))
        llms.append(llm)
        created.append({"provider": provider, "model": model, "base_url": base_url, "kwargs": kwargs})
        return _FakeClient(llm, provider, model)

    monkeypatch.setattr(br, "create_llm_client", _create)
    return SimpleNamespace(created=created, llms=llms, per_model=per_model)


# --- 1: correct trace loading ------------------------------------------


@pytest.mark.unit
def test_1_loads_the_correct_source_call_for_each_supported_stage(tmp_path):
    run_dir = _build_trace(tmp_path)
    debug_dir, ticker = resolve_debug_dir(run_dir)
    assert ticker == "LLY"
    assert find_source_call(debug_dir, "market").call_number == 2
    assert find_source_call(debug_dir, "sentiment").call_number == 3
    # Research Manager must pick the FALLBACK (5), not the failed
    # structured attempt (4), and never the Validator's own call (6).
    rm_call = find_source_call(debug_dir, "research_manager")
    assert rm_call.call_number == 5
    assert rm_call.call_identity == "research_manager"


# --- 2: source run is never modified -------------------------------------


@pytest.mark.unit
def test_2_source_run_is_never_modified(tmp_path, fake_llm_factory):
    run_dir = _build_trace(tmp_path)
    before = {
        p: p.read_bytes()
        for p in run_dir.rglob("*") if p.is_file()
    }
    run_benchmark(
        from_run=run_dir, stage_keys=["market"], model_specs=[parse_model_spec("fake-model")],
        output_dir=tmp_path / "out",
    )
    after = {p: p.read_bytes() for p in run_dir.rglob("*") if p.is_file()}
    assert before == after
    assert set(before) == set(after)


# --- 3: replay never executes tools --------------------------------------


@pytest.mark.unit
def test_3_replay_never_binds_or_executes_the_original_tools(tmp_path, fake_llm_factory):
    """The Market Analyst source call had real tools (get_stock_data) bound
    -- replay must never pass them to the new model at all."""
    run_dir = _build_trace(tmp_path)
    run_benchmark(
        from_run=run_dir, stage_keys=["market"], model_specs=[parse_model_spec("fake-model")],
        output_dir=tmp_path / "out",
    )
    llm = fake_llm_factory.llms[0]
    assert len(llm.invoke_calls) == 1
    # No tools= kwarg was ever passed to invoke, and bind_tools was never called
    # (the fake doesn't even define it -- a real bind_tools call would AttributeError).
    assert "tools" not in llm.invoke_calls[0]["kwargs"]


# --- 4: same source input sent to two distinct models --------------------


@pytest.mark.unit
def test_4_same_source_input_sent_to_two_distinct_models(tmp_path, fake_llm_factory):
    run_dir = _build_trace(tmp_path)
    run_benchmark(
        from_run=run_dir, stage_keys=["market"],
        model_specs=[parse_model_spec("model-a"), parse_model_spec("model-b")],
        output_dir=tmp_path / "out",
    )
    assert [c["model"] for c in fake_llm_factory.created] == ["model-a", "model-b"]
    msgs_a = fake_llm_factory.llms[0].invoke_calls[0]["messages"]
    msgs_b = fake_llm_factory.llms[1].invoke_calls[0]["messages"]
    assert [m.content for m in msgs_a] == [m.content for m in msgs_b]


# --- 5: messages are semantically equivalent to the original -------------


@pytest.mark.unit
def test_5_reconstructed_messages_match_the_original_content_and_roles(tmp_path, fake_llm_factory):
    run_dir = _build_trace(tmp_path)
    debug_dir, _ = resolve_debug_dir(run_dir)
    call = find_source_call(debug_dir, "market")
    recon = reconstruct_messages(call.messages)
    assert [type(m).__name__ for m in recon.messages] == ["SystemMessage", "HumanMessage", "AIMessage", "ToolMessage"]
    assert recon.messages[0].content == "sys"
    assert recon.messages[1].content == "LLY"
    assert recon.messages[3].content == "price=100"
    # Missing tool_call_id (old trace format) -> positional inference, not "exact".
    assert recon.tool_call_id_source == "positional_inference"
    assert recon.messages[3].tool_call_id == "call_1"


# --- 6: model names containing ':' parse correctly ------------------------


@pytest.mark.unit
@pytest.mark.parametrize("spec,expected_model,expected_effort", [
    ("qwen3.8-flash-next:125b-a6b-q4_K_M", "qwen3.8-flash-next:125b-a6b-q4_K_M", None),
    ("qwen3.8-flash-next:125b-a6b-q4_K_M@medium", "qwen3.8-flash-next:125b-a6b-q4_K_M", "medium"),
    ("deepseek-v4.1-flash:cloud@high", "deepseek-v4.1-flash:cloud", "high"),
    ("gemma4:26b", "gemma4:26b", None),
])
def test_6_model_names_with_colons_parse_correctly(spec, expected_model, expected_effort):
    parsed = parse_model_spec(spec)
    assert parsed.model == expected_model
    assert parsed.reasoning_effort == expected_effort


@pytest.mark.unit
def test_6_empty_or_malformed_spec_raises_clear_error():
    with pytest.raises(ReplayError):
        parse_model_spec("")
    with pytest.raises(ReplayError):
        parse_model_spec("@medium")
    with pytest.raises(ReplayError):
        parse_model_spec("model@")


# --- 7: reasoning effort applies to the correct model ---------------------


@pytest.mark.unit
def test_7_reasoning_effort_is_applied_per_model(tmp_path, fake_llm_factory, monkeypatch):
    captured_kwargs = []
    orig = br.build_llm_kwargs

    def _spy(config, role=None):
        kwargs = orig(config, role=role)
        captured_kwargs.append((config.get("replay_reasoning_effort"), role))
        return kwargs

    monkeypatch.setattr(br, "build_llm_kwargs", _spy)
    run_dir = _build_trace(tmp_path)
    run_benchmark(
        from_run=run_dir, stage_keys=["market"],
        model_specs=[parse_model_spec("model-a@low"), parse_model_spec("model-b@high")],
        output_dir=tmp_path / "out",
    )
    assert captured_kwargs == [("low", "replay"), ("high", "replay")]


# --- 8: structured request is preserved ------------------------------------


@pytest.mark.unit
def test_8_structured_request_is_preserved_for_sentiment_and_research_manager(tmp_path, fake_llm_factory):
    run_dir = _build_trace(tmp_path)
    run_benchmark(
        from_run=run_dir, stage_keys=["sentiment", "research_manager"],
        model_specs=[parse_model_spec("fake-model")], output_dir=tmp_path / "out",
    )
    for llm in fake_llm_factory.llms:
        assert len(llm.structured_calls) == 1
        assert len(llm.invoke_calls) == 0  # structured succeeded -- no free-text fallback needed


# --- 9: free-text request is preserved -------------------------------------


@pytest.mark.unit
def test_9_free_text_request_is_preserved_for_market_news_fundamentals(tmp_path, fake_llm_factory):
    run_dir = _build_trace(tmp_path)
    run_benchmark(
        from_run=run_dir, stage_keys=["market"], model_specs=[parse_model_spec("fake-model")],
        output_dir=tmp_path / "out",
    )
    llm = fake_llm_factory.llms[0]
    assert len(llm.invoke_calls) == 1
    assert len(llm.structured_calls) == 0


# --- 10: unsupported stage fails with a clear message ---------------------


@pytest.mark.unit
def test_10_unsupported_stage_fails_with_clear_message(tmp_path):
    run_dir = _build_trace(tmp_path)
    debug_dir, _ = resolve_debug_dir(run_dir)
    with pytest.raises(ReplayError, match="Unsupported stage"):
        find_source_call(debug_dir, "trader")


# --- 11: missing trace artifacts fail cleanly ------------------------------


@pytest.mark.unit
def test_11_missing_trace_artifacts_fail_cleanly(tmp_path):
    empty_run = tmp_path / "empty_run"
    empty_run.mkdir()
    with pytest.raises(ReplayError, match="No --trace-llm artifacts"):
        resolve_debug_dir(empty_run)


@pytest.mark.unit
def test_11b_stage_never_run_fails_cleanly(tmp_path):
    run_dir = _build_trace(tmp_path)
    debug_dir, _ = resolve_debug_dir(run_dir)
    with pytest.raises(ReplayError, match="No trace calls found for stage"):
        find_source_call(debug_dir, "news")


@pytest.mark.unit
def test_11c_non_final_call_refuses_rather_than_guessing(tmp_path):
    """A stage whose only recorded call is still mid tool-loop (not final)
    must refuse, per the task's explicit instruction, never guess."""
    run_dir = tmp_path / "run2"
    debug_dir = run_dir / "NVDA" / "debug"
    calls_dir = debug_dir / "calls"
    _write_json(debug_dir / "run_manifest.json", {"ticker": "NVDA", "provider": "ollama"})
    _write_call(
        calls_dir, 1, "News Analyst", tool_call_count=3, finish_reason="tool_calls",
        structured_requested=False,
    )
    with pytest.raises(ReplayError, match="no call in the trace that looks like a final synthesis"):
        find_source_call(debug_dir, "news")


# --- 12: output tree is correct ---------------------------------------


@pytest.mark.unit
def test_12_output_tree_is_correct(tmp_path, fake_llm_factory):
    run_dir = _build_trace(tmp_path)
    output_dir = tmp_path / "out"
    run_benchmark(
        from_run=run_dir, stage_keys=["market", "sentiment"],
        model_specs=[parse_model_spec("fake-model@medium")], output_dir=output_dir,
    )
    assert (output_dir / "manifest.json").is_file()
    assert (output_dir / "summary.json").is_file()
    model_dir_name = _safe_path_component("fake-model@medium")
    for stage_dir in ("Market_Analyst", "Sentiment_Analyst"):
        call_dir = output_dir / stage_dir / model_dir_name
        assert (call_dir / "request.json").is_file()
        assert (call_dir / "metadata.json").is_file()
        assert (call_dir / "response.txt").is_file() or (call_dir / "response.json").is_file()


# --- 13: metadata is correct -----------------------------------------------


@pytest.mark.unit
def test_13_metadata_contains_all_required_summary_fields(tmp_path, fake_llm_factory):
    run_dir = _build_trace(tmp_path)
    summary = run_benchmark(
        from_run=run_dir, stage_keys=["market"], model_specs=[parse_model_spec("fake-model@medium")],
        output_dir=tmp_path / "out",
    )
    entry = summary["results"][0]
    for key in (
        "source_run", "ticker", "source_call_identity", "stage", "model", "reasoning_effort",
        "seed", "elapsed_seconds", "input_tokens", "output_tokens", "total_tokens",
        "structured_success", "fallback_triggered", "parse_success", "error",
    ):
        assert key in entry, f"missing summary field: {key}"
    assert entry["model"] == "fake-model"
    assert entry["reasoning_effort"] == "medium"


# --- 14: stable seed --------------------------------------------------------


@pytest.mark.unit
def test_14_seed_is_stable_across_repeated_runs_and_distinct_per_model(tmp_path, fake_llm_factory):
    run_dir = _build_trace(tmp_path)
    summary1 = run_benchmark(
        from_run=run_dir, stage_keys=["market"],
        model_specs=[parse_model_spec("model-a"), parse_model_spec("model-b")],
        output_dir=tmp_path / "out1", base_seed=555,
    )
    summary2 = run_benchmark(
        from_run=run_dir, stage_keys=["market"],
        model_specs=[parse_model_spec("model-a"), parse_model_spec("model-b")],
        output_dir=tmp_path / "out2", base_seed=555,
    )
    seeds1 = [r["seed"] for r in summary1["results"]]
    seeds2 = [r["seed"] for r in summary2["results"]]
    assert seeds1 == seeds2  # stable across repeated runs with the same base seed
    assert seeds1[0] != seeds1[1]  # distinct per model


@pytest.mark.unit
def test_14b_without_seed_reuses_the_source_calls_own_recorded_seed(tmp_path, fake_llm_factory):
    run_dir = _build_trace(tmp_path)
    summary = run_benchmark(
        from_run=run_dir, stage_keys=["market"], model_specs=[parse_model_spec("fake-model")],
        output_dir=tmp_path / "out",
    )
    assert summary["results"][0]["seed"] == 42  # the source call's own generation_params.seed


# --- 15: does not interfere with the normal pipeline -----------------------


@pytest.mark.unit
def test_15_does_not_touch_default_config_or_global_pipeline_state(tmp_path, fake_llm_factory):
    from tradingagents.default_config import DEFAULT_CONFIG

    before = dict(DEFAULT_CONFIG)
    run_dir = _build_trace(tmp_path)
    run_benchmark(
        from_run=run_dir, stage_keys=["market"], model_specs=[parse_model_spec("fake-model")],
        output_dir=tmp_path / "out", provider="anthropic",
    )
    assert before == DEFAULT_CONFIG  # never mutated globally, even with a provider override


@pytest.mark.unit
def test_15b_provider_override_affects_only_this_benchmark_not_default_config(tmp_path, fake_llm_factory):
    run_dir = _build_trace(tmp_path)
    run_benchmark(
        from_run=run_dir, stage_keys=["market"], model_specs=[parse_model_spec("fake-model")],
        output_dir=tmp_path / "out", provider="anthropic",
    )
    assert fake_llm_factory.created[0]["provider"] == "anthropic"


# --- Additional: provider resolution must use the run manifest's own
# llm_provider, never LangChain's internal client-type string -------------


@pytest.mark.unit
def test_provider_defaults_to_run_manifest_llm_provider_not_langchain_internal_type(tmp_path, fake_llm_factory):
    run_dir = _build_trace(tmp_path)  # run_manifest.json records provider="ollama";
    # the source call's own request.json records provider="openai-chat" (LangChain's
    # internal client type) -- the latter must NEVER be used as create_llm_client's
    # provider argument.
    run_benchmark(
        from_run=run_dir, stage_keys=["market"], model_specs=[parse_model_spec("fake-model")],
        output_dir=tmp_path / "out",
    )
    assert fake_llm_factory.created[0]["provider"] == "ollama"


@pytest.mark.unit
def test_safe_path_component_handles_colons_and_at_signs():
    assert ":" not in _safe_path_component("qwen3.8-flash-next:125b-a6b-q4_K_M@medium")
    assert "@" not in _safe_path_component("qwen3.8-flash-next:125b-a6b-q4_K_M@medium")


@pytest.mark.unit
def test_disambiguates_multiple_tickers_with_explicit_ticker_flag(tmp_path):
    run_dir = tmp_path / "multi_run"
    for ticker in ("AAPL", "MSFT"):
        debug_dir = run_dir / ticker / "debug"
        _write_json(debug_dir / "run_manifest.json", {"ticker": ticker, "provider": "ollama"})
    with pytest.raises(ReplayError, match="more than one ticker"):
        resolve_debug_dir(run_dir)
    debug_dir, ticker = resolve_debug_dir(run_dir, ticker="AAPL")
    assert ticker == "AAPL"


# =============================================================================
# Fix-check: (1) News/Market/Fundamentals analysts replayed WITHOUT their
# original tools bound could cause a tool-trained model to re-enter "tool
# mode" (empty content, tool_calls populated) against a system prompt that
# still describes tools in plain text -- replay now binds the SAME tools
# with tool_choice="none" and treats any re-entry (or empty content) as an
# explicit error, never a silent empty "success". (2) Research Manager's
# structured-fail -> free-text-fallback-success path was wrongly surfaced
# as an error; it's now REPLAY SUCCESS with output_mode="fallback", exactly
# matching production's own acceptance of a non-empty fallback. Numbered
# per this task's own "TESTS" lists (1-16).
# =============================================================================


def _build_tool_loop_trace(tmp_path, ticker="LLY", stage="News Analyst"):
    """call1 asks for a tool, call2 gets that result and asks for another,
    call3 gets the second result and stops -- the exact 3-call shape the
    task's test #1 describes, with the SAME tools bound throughout
    (matching the real trace: every round in a tool loop carries the same
    tool definitions, even the one that ends up not using them)."""
    run_dir = tmp_path / "news_run"
    debug_dir = run_dir / ticker / "debug"
    calls_dir = debug_dir / "calls"
    _write_json(debug_dir / "run_manifest.json", {"ticker": ticker, "provider": "ollama"})
    tools = [{"type": "function", "function": {"name": "get_news", "description": "..."}}]
    sys_msg = {"role": "system", "content": "You have access to the following tools: get_news."}

    _write_call(
        calls_dir, 1, stage, call_identity=None,
        messages=[sys_msg, {"role": "human", "content": ticker}],
        tools=tools, tool_call_count=1, finish_reason="tool_calls",
    )
    _write_call(
        calls_dir, 2, stage, call_identity=None,
        messages=[
            sys_msg, {"role": "human", "content": ticker},
            {"role": "ai", "content": "", "tool_calls": [{"name": "get_news", "args": {}, "id": "call_1", "type": "tool_call"}]},
            {"role": "tool", "content": "headline batch 1", "name": "get_news"},
        ],
        tools=tools, tool_call_count=1, finish_reason="tool_calls",
    )
    _write_call(
        calls_dir, 3, stage, call_identity=None,
        messages=[
            sys_msg, {"role": "human", "content": ticker},
            {"role": "ai", "content": "", "tool_calls": [{"name": "get_news", "args": {}, "id": "call_1", "type": "tool_call"}]},
            {"role": "tool", "content": "headline batch 1", "name": "get_news"},
            {"role": "ai", "content": "", "tool_calls": [{"name": "get_news", "args": {}, "id": "call_2", "type": "tool_call"}]},
            {"role": "tool", "content": "headline batch 2", "name": "get_news"},
        ],
        tools=tools, tool_call_count=0, finish_reason="stop",
    )
    return run_dir, calls_dir


# --- News 1-7 ----------------------------------------------------------


@pytest.mark.unit
def test_news_1_selects_the_third_final_synthesis_call(tmp_path):
    run_dir, _ = _build_tool_loop_trace(tmp_path)
    debug_dir, _ = resolve_debug_dir(run_dir)
    call = find_source_call(debug_dir, "news")
    assert call.call_number == 3


@pytest.mark.unit
def test_news_2_a_tool_calls_finish_reason_is_never_final_comparable():
    call = SimpleNamespace(metadata={"structured_requested": False, "tool_call_count": 2, "finish_reason": "tool_calls"})
    assert br._is_final_synthesis_call(call) is False


@pytest.mark.unit
def test_news_3_replay_never_binds_the_real_get_news_tool_as_executable(tmp_path, fake_llm_factory):
    """tools ARE passed through (to keep the model's environment honest),
    but ONLY alongside tool_choice="none" -- never bound via bind_tools in
    a way that could be executed, and this module never calls any tool
    function regardless of what the model returns."""
    run_dir, _ = _build_tool_loop_trace(tmp_path)
    run_benchmark(
        from_run=run_dir, stage_keys=["news"], model_specs=[parse_model_spec("fake-model")],
        output_dir=tmp_path / "out",
    )
    llm = fake_llm_factory.llms[0]
    assert len(llm.invoke_calls) == 1
    kwargs = llm.invoke_calls[0]["kwargs"]
    assert kwargs.get("tool_choice") == "none"
    assert kwargs.get("tools") == [{"type": "function", "function": {"name": "get_news", "description": "..."}}]


@pytest.mark.unit
def test_news_4_prior_tool_result_messages_remain_in_frozen_messages(tmp_path):
    run_dir, _ = _build_tool_loop_trace(tmp_path)
    debug_dir, _ = resolve_debug_dir(run_dir)
    call = find_source_call(debug_dir, "news")
    recon = reconstruct_messages(call.messages)
    tool_messages = [m for m in recon.messages if type(m).__name__ == "ToolMessage"]
    assert [m.content for m in tool_messages] == ["headline batch 1", "headline batch 2"]


@pytest.mark.unit
def test_news_5_model_cannot_trigger_a_real_tool_during_replay(tmp_path, fake_llm_factory):
    """Even when the fake model "wants" to call a tool (simulating a
    tool-trained model ignoring tool_choice="none"), no tool is executed.
    Here the tools-free retry (see _normalize_tool_history) ALSO fails --
    the retry's own response is empty -- so the overall result is still an
    explicit error, reached via the normalized retry path, never by
    accepting the tool-call attempt as text."""
    run_dir, _ = _build_tool_loop_trace(tmp_path)
    fake_llm_factory.per_model["fake-model"] = {"reenter_tool_mode": True, "response_text": ""}
    summary = run_benchmark(
        from_run=run_dir, stage_keys=["news"], model_specs=[parse_model_spec("fake-model")],
        output_dir=tmp_path / "out",
    )
    entry = summary["results"][0]
    assert entry["status"] == "error"
    assert entry["replay_input_mode"] == "tool_history_normalized"
    assert "tools-free retry" in entry["error"]
    assert not (tmp_path / "out" / "News_Analyst" / "fake-model" / "response.txt").exists()


@pytest.mark.unit
def test_news_6_stage_with_no_separable_final_call_is_unsupported_for_replay(tmp_path):
    run_dir = tmp_path / "stuck_run"
    debug_dir = run_dir / "NVDA" / "debug"
    calls_dir = debug_dir / "calls"
    _write_json(debug_dir / "run_manifest.json", {"ticker": "NVDA", "provider": "ollama"})
    _write_call(calls_dir, 1, "News Analyst", tool_call_count=3, finish_reason="tool_calls")
    summary = run_benchmark(
        from_run=run_dir, stage_keys=["news"], model_specs=[parse_model_spec("fake-model")],
        output_dir=tmp_path / "out",
    )
    entry = summary["results"][0]
    assert entry["status"] == "unsupported_for_replay"
    assert entry["reason"]


@pytest.mark.unit
def test_news_7_empty_response_is_never_counted_as_success(tmp_path, fake_llm_factory):
    run_dir, _ = _build_tool_loop_trace(tmp_path)
    fake_llm_factory.per_model["fake-model"] = {"response_text": ""}
    summary = run_benchmark(
        from_run=run_dir, stage_keys=["news"], model_specs=[parse_model_spec("fake-model")],
        output_dir=tmp_path / "out",
    )
    entry = summary["results"][0]
    assert entry["status"] == "error"
    assert not (tmp_path / "out" / "News_Analyst" / "fake-model" / "response.txt").exists()


# --- Research Manager 8-13 ----------------------------------------------


@pytest.mark.unit
def test_rm_8_structured_success_reports_output_mode_structured(tmp_path, fake_llm_factory):
    run_dir = _build_trace(tmp_path)
    summary = run_benchmark(
        from_run=run_dir, stage_keys=["research_manager"], model_specs=[parse_model_spec("fake-model")],
        output_dir=tmp_path / "out",
    )
    entry = summary["results"][0]
    assert entry["status"] == "success"
    assert entry["output_mode"] == "structured"
    assert entry["structured_success"] is True
    assert entry["fallback_triggered"] is False


@pytest.mark.unit
def test_rm_9_structured_fail_valid_fallback_is_success_with_fallback_mode(tmp_path, fake_llm_factory):
    run_dir = _build_trace(tmp_path)
    fake_llm_factory.per_model["fake-model"] = {
        "structured_fails": True,
        "response_text": "**Recommendation**: Hold\n**Rationale**: x\n**Strategic Actions**: y",
    }
    summary = run_benchmark(
        from_run=run_dir, stage_keys=["research_manager"], model_specs=[parse_model_spec("fake-model")],
        output_dir=tmp_path / "out",
    )
    entry = summary["results"][0]
    assert entry["status"] == "success"
    assert entry["output_mode"] == "fallback"
    assert entry["structured_success"] is False
    assert entry["fallback_triggered"] is True
    assert entry["fallback_success"] is True
    assert entry.get("error") is None


@pytest.mark.unit
def test_rm_10_non_empty_fallback_text_is_not_an_error_merely_because_structured_failed(tmp_path, fake_llm_factory):
    run_dir = _build_trace(tmp_path)
    fake_llm_factory.per_model["fake-model"] = {"structured_fails": True, "response_text": "some usable text"}
    summary = run_benchmark(
        from_run=run_dir, stage_keys=["research_manager"], model_specs=[parse_model_spec("fake-model")],
        output_dir=tmp_path / "out",
    )
    entry = summary["results"][0]
    assert entry["status"] == "success"
    assert entry.get("error") is None
    model_dir = tmp_path / "out" / "Research_Manager" / "fake-model"
    assert (model_dir / "response.txt").read_text(encoding="utf-8") == "some usable text"


@pytest.mark.unit
def test_rm_11_structured_fail_plus_empty_fallback_is_an_error(tmp_path, fake_llm_factory):
    run_dir = _build_trace(tmp_path)
    fake_llm_factory.per_model["fake-model"] = {"structured_fails": True, "response_text": "   "}
    summary = run_benchmark(
        from_run=run_dir, stage_keys=["research_manager"], model_specs=[parse_model_spec("fake-model")],
        output_dir=tmp_path / "out",
    )
    entry = summary["results"][0]
    assert entry["status"] == "error"
    assert entry["fallback_success"] is False


@pytest.mark.unit
def test_rm_12_reproduces_the_real_qwen_gemma_structured_failure_bug(tmp_path, fake_llm_factory):
    """The exact real-run shape: structured parse fails ("schema mismatch"),
    free-text fallback succeeds -- must NOT surface as
    "fallback, error: structured output returned no parsed result"."""
    run_dir = _build_trace(tmp_path)
    fake_llm_factory.per_model["qwen-like"] = {
        "structured_fails": True,
        "response_text": "**Recommendation**: Buy\n**Rationale**: x\n**Strategic Actions**: y",
    }
    summary = run_benchmark(
        from_run=run_dir, stage_keys=["research_manager"], model_specs=[parse_model_spec("qwen-like")],
        output_dir=tmp_path / "out",
    )
    entry = summary["results"][0]
    assert entry["status"] == "success"
    assert entry.get("error") is None


@pytest.mark.unit
def test_rm_13_deepseek_like_structured_success_still_works(tmp_path, fake_llm_factory):
    run_dir = _build_trace(tmp_path)
    summary = run_benchmark(
        from_run=run_dir, stage_keys=["research_manager"], model_specs=[parse_model_spec("deepseek-like")],
        output_dir=tmp_path / "out",
    )
    entry = summary["results"][0]
    assert entry["status"] == "success"
    assert entry["output_mode"] == "structured"


# --- Timing 14-16 ---------------------------------------------------------


@pytest.mark.unit
def test_timing_14_summary_has_elapsed_total_per_model(tmp_path, fake_llm_factory):
    run_dir = _build_trace(tmp_path)
    summary = run_benchmark(
        from_run=run_dir, stage_keys=["market", "sentiment"], model_specs=[parse_model_spec("fake-model")],
        output_dir=tmp_path / "out",
    )
    assert "fake-model" in summary["elapsed_seconds_total_by_model"]
    assert summary["elapsed_seconds_total_by_model"]["fake-model"] >= 0


@pytest.mark.unit
def test_timing_15_summary_has_benchmark_wall_elapsed(tmp_path, fake_llm_factory):
    run_dir = _build_trace(tmp_path)
    summary = run_benchmark(
        from_run=run_dir, stage_keys=["market"], model_specs=[parse_model_spec("fake-model")],
        output_dir=tmp_path / "out",
    )
    assert "benchmark_wall_elapsed_seconds" in summary
    assert "model_call_elapsed_seconds" in summary
    assert summary["benchmark_wall_elapsed_seconds"] >= 0


@pytest.mark.unit
def test_timing_16_cli_prints_final_duration(tmp_path, fake_llm_factory):
    from typer.testing import CliRunner

    from cli.main import app

    run_dir = _build_trace(tmp_path)
    runner = CliRunner()
    result = runner.invoke(app, [
        "benchmark-models", "--from-run", str(run_dir), "--stages", "market",
        "--model", "fake-model", "--output-dir", str(tmp_path / "out"),
    ])
    assert result.exit_code == 0, result.output
    assert "Benchmark completed in" in result.output


@pytest.mark.unit
def test_cli_model_names_with_colon_never_trigger_rich_emoji_markup_crash(tmp_path, fake_llm_factory):
    """Real-run regression: "<model>:cloud" immediately followed by this
    command's own ": " separator forms the literal substring ":cloud:",
    which Rich's console markup parser reads as an emoji shortcode --
    rendered as an actual unicode character that crashed a real benchmark
    on a legacy Windows console's codepage, on an otherwise fully
    successful run. Every model/stage/error value the CLI prints must be
    escaped before reaching console.print."""
    from typer.testing import CliRunner

    from cli.main import app

    run_dir = _build_trace(tmp_path)
    runner = CliRunner()
    result = runner.invoke(app, [
        "benchmark-models", "--from-run", str(run_dir), "--stages", "market",
        "--model", "some-model:cloud", "--output-dir", str(tmp_path / "out"),
    ])
    assert result.exit_code == 0, result.output
    assert "some-model:cloud" in result.output


# --- output-dir reuse guard: abort before any LLM call if --output-dir ---
# --- already holds artifacts from a previous benchmark run --------------


@pytest.mark.unit
def test_output_dir_1_new_output_dir_runs_normally(tmp_path, fake_llm_factory):
    run_dir = _build_trace(tmp_path)
    out_dir = tmp_path / "out" / "does-not-exist-yet"
    summary = run_benchmark(
        from_run=run_dir, stage_keys=["market"], model_specs=[parse_model_spec("fake-model")],
        output_dir=out_dir,
    )
    assert summary["results"][0]["status"] == "success"
    assert (out_dir / "summary.json").exists()


@pytest.mark.unit
def test_output_dir_2_existing_empty_output_dir_runs_normally(tmp_path, fake_llm_factory):
    run_dir = _build_trace(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True)
    summary = run_benchmark(
        from_run=run_dir, stage_keys=["market"], model_specs=[parse_model_spec("fake-model")],
        output_dir=out_dir,
    )
    assert summary["results"][0]["status"] == "success"


@pytest.mark.unit
def test_output_dir_3_output_dir_with_prior_benchmark_fails(tmp_path, fake_llm_factory):
    run_dir = _build_trace(tmp_path)
    out_dir = tmp_path / "out"
    run_benchmark(
        from_run=run_dir, stage_keys=["market"], model_specs=[parse_model_spec("fake-model")],
        output_dir=out_dir,
    )
    with pytest.raises(ReplayError, match="already contains benchmark artifacts"):
        run_benchmark(
            from_run=run_dir, stage_keys=["market"], model_specs=[parse_model_spec("fake-model")],
            output_dir=out_dir,
        )


@pytest.mark.unit
def test_output_dir_4_failure_happens_before_any_llm_call(tmp_path, fake_llm_factory):
    run_dir = _build_trace(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True)
    (out_dir / "leftover.txt").write_text("stale artifact from a prior run", encoding="utf-8")
    with pytest.raises(ReplayError, match="already contains benchmark artifacts"):
        run_benchmark(
            from_run=run_dir, stage_keys=["market"], model_specs=[parse_model_spec("fake-model")],
            output_dir=out_dir,
        )
    assert fake_llm_factory.created == []
    assert fake_llm_factory.llms == []


@pytest.mark.unit
def test_output_dir_5_existing_files_are_not_modified(tmp_path, fake_llm_factory):
    run_dir = _build_trace(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True)
    leftover = out_dir / "summary.json"
    original_bytes = b'{"from": "a previous benchmark run"}'
    leftover.write_bytes(original_bytes)
    with pytest.raises(ReplayError, match="already contains benchmark artifacts"):
        run_benchmark(
            from_run=run_dir, stage_keys=["market"], model_specs=[parse_model_spec("fake-model")],
            output_dir=out_dir,
        )
    assert leftover.read_bytes() == original_bytes
    assert list(out_dir.iterdir()) == [leftover]


# --- Progress reporting 1-6: visible progress during a long benchmark ---


@pytest.mark.unit
def test_progress_1_start_is_reported_before_the_llm_is_called(tmp_path, fake_llm_factory):
    run_dir = _build_trace(tmp_path)
    events: list[dict] = []
    created_at_start: list[int] = []

    def _on_progress(event: dict) -> None:
        if event.get("phase") == "start":
            # Snapshot taken INSIDE the callback, before replay_one runs --
            # a post-hoc check after run_benchmark returns would always see
            # the client already created, regardless of ordering.
            created_at_start.append(len(fake_llm_factory.created))
        events.append(event)

    run_benchmark(
        from_run=run_dir, stage_keys=["market"], model_specs=[parse_model_spec("fake-model")],
        output_dir=tmp_path / "out", on_progress=_on_progress,
    )
    start_events = [e for e in events if e.get("phase") == "start"]
    assert len(start_events) == 1
    assert created_at_start == [0]


@pytest.mark.unit
def test_progress_2_finish_is_reported_after_the_llm_call(tmp_path, fake_llm_factory):
    run_dir = _build_trace(tmp_path)
    events: list[dict] = []
    run_benchmark(
        from_run=run_dir, stage_keys=["market"], model_specs=[parse_model_spec("fake-model")],
        output_dir=tmp_path / "out", on_progress=events.append,
    )
    finish_events = [e for e in events if e.get("phase") == "finish"]
    assert len(finish_events) == 1
    assert finish_events[0]["output_mode"] == "free_text"
    assert finish_events[0]["total_tokens"] == 15
    # The client was already created and used by the time "finish" fired.
    assert len(fake_llm_factory.created) == 1


@pytest.mark.unit
def test_progress_3_error_is_reported_on_failure(tmp_path, fake_llm_factory):
    run_dir = _build_trace(tmp_path)
    fake_llm_factory.per_model["fake-model"] = {"response_text": ""}
    events: list[dict] = []
    run_benchmark(
        from_run=run_dir, stage_keys=["market"], model_specs=[parse_model_spec("fake-model")],
        output_dir=tmp_path / "out", on_progress=events.append,
    )
    error_events = [e for e in events if e.get("phase") == "error"]
    assert len(error_events) == 1
    assert error_events[0]["error"]
    assert not any(e.get("phase") == "finish" for e in events)


@pytest.mark.unit
def test_progress_4_every_cli_progress_line_has_a_timestamp(tmp_path, fake_llm_factory):
    import re

    from typer.testing import CliRunner

    from cli.main import app

    run_dir = _build_trace(tmp_path)
    runner = CliRunner()
    result = runner.invoke(app, [
        "benchmark-models", "--from-run", str(run_dir), "--stages", "market",
        "--model", "fake-model", "--output-dir", str(tmp_path / "out"),
    ])
    assert result.exit_code == 0, result.output
    stamp_re = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} ")
    progress_lines = [
        line for line in result.output.splitlines()
        if "Benchmark source:" in line or "Starting" in line or "Finished" in line or "ERROR" in line
    ]
    assert progress_lines
    assert all(stamp_re.match(line) for line in progress_lines)


@pytest.mark.unit
def test_progress_5_counter_n_of_total_is_correct(tmp_path, fake_llm_factory):
    run_dir = _build_trace(tmp_path)
    events: list[dict] = []
    run_benchmark(
        from_run=run_dir, stage_keys=["market", "sentiment"],
        model_specs=[parse_model_spec("model-a"), parse_model_spec("model-b")],
        output_dir=tmp_path / "out", on_progress=events.append,
    )
    start_events = [e for e in events if e.get("phase") == "start"]
    assert [e["total"] for e in start_events] == [4, 4, 4, 4]
    assert [e["index"] for e in start_events] == [1, 2, 3, 4]


@pytest.mark.unit
def test_progress_6_never_prints_the_full_prompt_or_response(tmp_path, fake_llm_factory):
    from typer.testing import CliRunner

    from cli.main import app

    run_dir = _build_trace(tmp_path)
    distinctive_response = "VERY-LONG-DISTINCTIVE-RESPONSE-BODY-" + ("x" * 500)
    fake_llm_factory.per_model["fake-model"] = {"response_text": distinctive_response}
    runner = CliRunner()
    result = runner.invoke(app, [
        "benchmark-models", "--from-run", str(run_dir), "--stages", "market",
        "--model", "fake-model", "--output-dir", str(tmp_path / "out"),
    ])
    assert result.exit_code == 0, result.output
    assert distinctive_response not in result.output
    for line in result.output.splitlines():
        assert len(line) < 300


# --- Tool-history replay 7-15: safe synthesis when a provider ignores ---
# --- tool_choice="none" and still attempts a real tool call -------------


@pytest.mark.unit
def test_toolhistory_7_provider_honoring_tool_choice_none_stays_exact(tmp_path, fake_llm_factory):
    run_dir, _ = _build_tool_loop_trace(tmp_path)
    summary = run_benchmark(
        from_run=run_dir, stage_keys=["news"], model_specs=[parse_model_spec("fake-model")],
        output_dir=tmp_path / "out",
    )
    entry = summary["results"][0]
    assert entry["status"] == "success"
    assert entry["replay_input_mode"] == "exact"
    assert entry["tool_history_detail"] is None
    llm = fake_llm_factory.llms[0]
    assert len(llm.invoke_calls) == 1


@pytest.mark.unit
def test_toolhistory_8_provider_ignoring_tool_choice_none_never_executes_a_tool(tmp_path, fake_llm_factory):
    """The model tries to call a tool on the first attempt despite
    tool_choice="none" -- this module has no code path that dispatches a
    real tool function, on either attempt, so there is structurally
    nothing to execute; the retry is made with NO ``tools`` parameter."""
    run_dir, _ = _build_tool_loop_trace(tmp_path)
    fake_llm_factory.per_model["fake-model"] = {
        "reenter_tool_mode": True, "response_text": "", "retry_response_text": "Synthesized without tools.",
    }
    run_benchmark(
        from_run=run_dir, stage_keys=["news"], model_specs=[parse_model_spec("fake-model")],
        output_dir=tmp_path / "out",
    )
    llm = fake_llm_factory.llms[0]
    assert len(llm.invoke_calls) == 2
    assert "tools" not in llm.invoke_calls[1]["kwargs"]


@pytest.mark.unit
def test_toolhistory_9_a_safe_tools_free_synthesis_route_exists(tmp_path, fake_llm_factory):
    run_dir, _ = _build_tool_loop_trace(tmp_path)
    fake_llm_factory.per_model["fake-model"] = {
        "reenter_tool_mode": True, "response_text": "", "retry_response_text": "Synthesized without tools.",
    }
    summary = run_benchmark(
        from_run=run_dir, stage_keys=["news"], model_specs=[parse_model_spec("fake-model")],
        output_dir=tmp_path / "out",
    )
    entry = summary["results"][0]
    assert entry["status"] == "success"
    assert entry["replay_input_mode"] == "tool_history_normalized"
    response_path = tmp_path / "out" / "News_Analyst" / "fake-model" / "response.txt"
    assert response_path.read_text(encoding="utf-8") == "Synthesized without tools."


@pytest.mark.unit
def test_toolhistory_10_tool_result_text_is_preserved_verbatim(tmp_path):
    run_dir, _ = _build_tool_loop_trace(tmp_path)
    debug_dir, _ = resolve_debug_dir(run_dir)
    call = find_source_call(debug_dir, "news")
    recon = reconstruct_messages(call.messages)
    normalized = _normalize_tool_history(recon.messages)
    combined = "\n".join(m.content for m in normalized.messages)
    assert "headline batch 1" in combined
    assert "headline batch 2" in combined


@pytest.mark.unit
def test_toolhistory_11_no_evidence_content_is_lost(tmp_path):
    run_dir, _ = _build_tool_loop_trace(tmp_path)
    debug_dir, _ = resolve_debug_dir(run_dir)
    call = find_source_call(debug_dir, "news")
    recon = reconstruct_messages(call.messages)
    original_tool_contents = [m.content for m in recon.messages if isinstance(m, ToolMessage)]
    normalized = _normalize_tool_history(recon.messages)
    combined = "\n".join(m.content for m in normalized.messages)
    for original in original_tool_contents:
        assert original in combined


@pytest.mark.unit
def test_toolhistory_12_metadata_reports_exact_vs_normalized(tmp_path, fake_llm_factory):
    run_dir, _ = _build_tool_loop_trace(tmp_path)
    summary_exact = run_benchmark(
        from_run=run_dir, stage_keys=["news"], model_specs=[parse_model_spec("exact-model")],
        output_dir=tmp_path / "out-exact",
    )
    assert summary_exact["results"][0]["replay_input_mode"] == "exact"
    assert summary_exact["results"][0]["tool_history_detail"] is None

    fake_llm_factory.per_model["normalized-model"] = {
        "reenter_tool_mode": True, "response_text": "", "retry_response_text": "Synthesized.",
    }
    summary_normalized = run_benchmark(
        from_run=run_dir, stage_keys=["news"], model_specs=[parse_model_spec("normalized-model")],
        output_dir=tmp_path / "out-normalized",
    )
    entry = summary_normalized["results"][0]
    assert entry["replay_input_mode"] == "tool_history_normalized"
    assert entry["tool_history_detail"]
    assert "tool-call round" in entry["tool_history_detail"]


@pytest.mark.unit
def test_toolhistory_13_normalized_history_has_no_tool_call_shaped_messages_left(tmp_path):
    run_dir, _ = _build_tool_loop_trace(tmp_path)
    debug_dir, _ = resolve_debug_dir(run_dir)
    call = find_source_call(debug_dir, "news")
    recon = reconstruct_messages(call.messages)
    normalized = _normalize_tool_history(recon.messages)
    assert not any(isinstance(m, ToolMessage) for m in normalized.messages)
    assert not any(isinstance(m, AIMessage) and m.tool_calls for m in normalized.messages)


@pytest.mark.unit
def test_toolhistory_14_empty_normalized_retry_response_is_still_an_error(tmp_path, fake_llm_factory):
    run_dir, _ = _build_tool_loop_trace(tmp_path)
    fake_llm_factory.per_model["fake-model"] = {
        "reenter_tool_mode": True, "response_text": "", "retry_response_text": "",
    }
    summary = run_benchmark(
        from_run=run_dir, stage_keys=["news"], model_specs=[parse_model_spec("fake-model")],
        output_dir=tmp_path / "out",
    )
    entry = summary["results"][0]
    assert entry["status"] == "error"
    assert entry["replay_input_mode"] == "tool_history_normalized"
    assert not (tmp_path / "out" / "News_Analyst" / "fake-model" / "response.txt").exists()


@pytest.mark.unit
def test_toolhistory_15_deepseek_msft_equivalent_succeeds_without_executable_tools(tmp_path, fake_llm_factory):
    """Reproduces the real DeepSeek/MSFT News regression: the model ignores
    tool_choice="none" and re-enters tool mode on the exact attempt, but can
    synthesize a real answer once no ``tools`` are offered at all."""
    run_dir, _ = _build_tool_loop_trace(tmp_path)
    fake_llm_factory.per_model["fake-model"] = {
        "reenter_tool_mode": True, "response_text": "",
        "retry_response_text": "Final MSFT news synthesis, no tools needed.",
    }
    summary = run_benchmark(
        from_run=run_dir, stage_keys=["news"], model_specs=[parse_model_spec("fake-model")],
        output_dir=tmp_path / "out",
    )
    entry = summary["results"][0]
    assert entry["status"] == "success"
    assert entry["output_mode"] == "free_text"
    assert entry["replay_input_mode"] == "tool_history_normalized"
    response_path = tmp_path / "out" / "News_Analyst" / "fake-model" / "response.txt"
    assert response_path.read_text(encoding="utf-8") == "Final MSFT news synthesis, no tools needed."
