"""Opt-in LLM trace/debug mode: a complete, auditable record of every LLM
call (see fork README: "LLM trace mode"). Off by default; when enabled,
request/response/metadata are captured per call under
<output_dir>/debug/, with credential-shaped keys redacted and a trace-write
failure degrading to a WARN rather than aborting the analysis."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from tradingagents.llm_trace import (
    LLMTraceCallbackHandler,
    record_structured_parse_result,
    redact,
    set_active_tracer,
    snapshot_prompts,
    traced_invoke,
    traced_node,
    write_run_manifest,
)


@pytest.fixture(autouse=True)
def _reset_tracer():
    set_active_tracer(None)
    yield
    set_active_tracer(None)


def _sys(content="sys prompt"):
    return SimpleNamespace(type="system", content=content, tool_calls=None, name=None)


def _human(content="user prompt"):
    return SimpleNamespace(type="human", content=content, tool_calls=None, name=None)


def _fake_llm_call(tracer, *, run_id="run-1", attempt="normal", invocation_params=None,
                    content="the answer", usage=None, finish_reason="stop", tool_calls=None):
    """Drive a handler through one request/response cycle without a real LLM."""
    from tradingagents.llm_trace import _current_attempt

    token = _current_attempt.set(attempt)
    try:
        tracer.on_chat_model_start(
            {"name": "fake-model"}, [[_sys(), _human()]], run_id=run_id,
            invocation_params=invocation_params or {"model": "fake-model"},
        )
    finally:
        _current_attempt.reset(token)
    message = SimpleNamespace(
        content=content, tool_calls=tool_calls,
        usage_metadata=usage, response_metadata={"finish_reason": finish_reason},
    )
    response = SimpleNamespace(generations=[[SimpleNamespace(message=message)]])
    tracer.on_llm_end(response, run_id=run_id)


def _call_dir(tmp_path, number, stage, attempt=None):
    name = f"{number:03d}_{stage}" + (f"_{attempt}" if attempt and attempt != "normal" else "")
    return tmp_path / "debug" / "calls" / name


# --- 10, 11, 15, 16: a normal call saves request/response/metadata ----------


@pytest.mark.unit
def test_normal_call_saves_rendered_request_and_raw_response(tmp_path):
    tracer = LLMTraceCallbackHandler(tmp_path / "debug")
    traced_node("Market Analyst", lambda state: _fake_llm_call(tracer))(None)

    call_dir = _call_dir(tmp_path, 1, "Market_Analyst")
    request = json.loads((call_dir / "request.json").read_text())
    assert request["stage"] == "Market Analyst"
    assert request["messages"] == [
        {"role": "system", "content": "sys prompt"},
        {"role": "human", "content": "user prompt"},
    ]
    raw = json.loads((call_dir / "response_raw.txt").read_text())
    assert raw["content"] == "the answer"


@pytest.mark.unit
def test_metadata_contains_model_tokens_duration_when_supplied(tmp_path):
    tracer = LLMTraceCallbackHandler(tmp_path / "debug")
    traced_node(
        "Trader",
        lambda state: _fake_llm_call(
            tracer, usage={"input_tokens": 12, "output_tokens": 7, "total_tokens": 19}
        ),
    )(None)
    metadata = json.loads((_call_dir(tmp_path, 1, "Trader") / "metadata.json").read_text())
    assert metadata["input_tokens"] == 12
    assert metadata["output_tokens"] == 7
    assert metadata["total_tokens"] == 19
    assert metadata["success"] is True
    assert "elapsed_seconds" in metadata
    assert metadata["finish_reason"] == "stop"


@pytest.mark.unit
def test_missing_token_counts_are_null_not_estimated(tmp_path):
    tracer = LLMTraceCallbackHandler(tmp_path / "debug")
    traced_node("Trader", lambda state: _fake_llm_call(tracer, usage=None))(None)
    metadata = json.loads((_call_dir(tmp_path, 1, "Trader") / "metadata.json").read_text())
    assert metadata["input_tokens"] is None
    assert metadata["output_tokens"] is None


# --- 15: call numbering is deterministic/increasing -------------------------


@pytest.mark.unit
def test_call_numbering_increases_deterministically(tmp_path):
    tracer = LLMTraceCallbackHandler(tmp_path / "debug")
    traced_node("Market Analyst", lambda state: _fake_llm_call(tracer, run_id="a"))(None)
    traced_node("Bull Researcher", lambda state: _fake_llm_call(tracer, run_id="b"))(None)
    traced_node("Bear Researcher", lambda state: _fake_llm_call(tracer, run_id="c"))(None)
    assert _call_dir(tmp_path, 1, "Market_Analyst").exists()
    assert _call_dir(tmp_path, 2, "Bull_Researcher").exists()
    assert _call_dir(tmp_path, 3, "Bear_Researcher").exists()


# --- 12, 13, 14: structured success / failure / fallback as a separate call -


@pytest.mark.unit
def test_structured_success_saves_parsed_response(tmp_path):
    tracer = LLMTraceCallbackHandler(tmp_path / "debug")
    set_active_tracer(tracer)

    def node(state):
        _fake_llm_call(tracer, attempt="structured")
        record_structured_parse_result(success=True, parsed={"risk_level": "LOW"})

    traced_node("Aggressive Analyst", node)(None)
    call_dir = _call_dir(tmp_path, 1, "Aggressive_Analyst", "structured")
    parsed = json.loads((call_dir / "response_parsed.json").read_text())
    assert parsed == {"risk_level": "LOW"}


@pytest.mark.unit
def test_structured_failure_records_validation_failure_without_fabricating_parsed_json(tmp_path):
    tracer = LLMTraceCallbackHandler(tmp_path / "debug")
    set_active_tracer(tracer)

    def node(state):
        _fake_llm_call(tracer, attempt="structured")
        record_structured_parse_result(success=False, error="2 validation errors for Schema")

    traced_node("Bull Researcher", node)(None)
    call_dir = _call_dir(tmp_path, 1, "Bull_Researcher", "structured")
    assert not (call_dir / "response_parsed.json").exists()
    metadata = json.loads((call_dir / "metadata.json").read_text())
    assert metadata["structured_parse_status"] == "failed"
    assert "validation errors" in metadata["structured_parse_error"]


@pytest.mark.unit
def test_fallback_is_stored_as_a_separate_traced_call_directory(tmp_path):
    tracer = LLMTraceCallbackHandler(tmp_path / "debug")
    set_active_tracer(tracer)

    def node(state):
        _fake_llm_call(tracer, run_id="s", attempt="structured", content="raw json")
        record_structured_parse_result(success=False, error="bad json")
        _fake_llm_call(tracer, run_id="f", attempt="fallback", content="free text answer")

    traced_node("Bull Researcher", node)(None)
    structured_dir = _call_dir(tmp_path, 1, "Bull_Researcher", "structured")
    fallback_dir = _call_dir(tmp_path, 2, "Bull_Researcher", "fallback")
    assert structured_dir.exists() and fallback_dir.exists()
    assert structured_dir != fallback_dir
    fallback_raw = json.loads((fallback_dir / "response_raw.txt").read_text())
    assert fallback_raw["content"] == "free text answer"


@pytest.mark.unit
def test_traced_invoke_tags_attempt_for_the_duration_of_one_call():
    from tradingagents.llm_trace import _current_attempt

    class _LLM:
        def invoke(self, prompt):
            assert _current_attempt.get() == "structured"
            return "result"

    assert traced_invoke(_LLM(), "prompt", attempt="structured") == "result"
    assert _current_attempt.get() == "normal"  # reset after the call


# --- 17-20: secret redaction --------------------------------------------


@pytest.mark.unit
def test_api_key_is_redacted():
    assert redact({"api_key": "sk-secret"})["api_key"] == "***REDACTED***"


@pytest.mark.unit
def test_authorization_is_redacted():
    assert redact({"Authorization": "Bearer xyz"})["Authorization"] == "***REDACTED***"


@pytest.mark.unit
@pytest.mark.parametrize("key", ["token", "api_token", "secret", "password", "client_secret"])
def test_credential_shaped_keys_are_redacted(key):
    assert redact({key: "xyz"})[key] == "***REDACTED***"


@pytest.mark.unit
def test_ordinary_token_count_keys_are_not_redacted():
    """The word 'token' inside a legitimate plural metric key (input_tokens,
    output_tokens, total_tokens) must survive -- these are explicitly
    requested trace fields, not credentials."""
    result = redact({"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})
    assert result == {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}


@pytest.mark.unit
def test_ordinary_natural_language_mentioning_token_is_not_redacted():
    """Redaction applies to config/metadata structures, never to prompt text
    -- a prompt that happens to contain the word 'token' must be untouched."""
    text = "The access token for the API should not be shared with third parties."
    assert redact({"content": text})["content"] == text


@pytest.mark.unit
def test_request_json_redacts_a_leaked_api_key_in_generation_params(tmp_path):
    tracer = LLMTraceCallbackHandler(tmp_path / "debug")
    traced_node(
        "Trader",
        lambda state: _fake_llm_call(
            tracer, invocation_params={"model": "m", "api_key": "sk-should-not-appear"}
        ),
    )(None)
    request_text = (_call_dir(tmp_path, 1, "Trader") / "request.json").read_text()
    assert "sk-should-not-appear" not in request_text
    assert "***REDACTED***" in request_text


# --- 21, 22: trace-write failure resilience ---------------------------------


@pytest.mark.unit
def test_trace_write_failure_emits_a_warn_and_does_not_raise(tmp_path, capsys):
    tracer = LLMTraceCallbackHandler(tmp_path / "debug")
    # Make the calls directory itself unwritable by pre-creating a FILE where
    # the handler expects to create a directory, forcing a real OSError.
    bad_dir = tmp_path / "debug" / "calls" / "001_Trader"
    bad_dir.parent.mkdir(parents=True)
    bad_dir.write_text("not a directory")

    traced_node("Trader", lambda state: _fake_llm_call(tracer))(None)  # must not raise
    captured = capsys.readouterr()
    assert "[WARN] LLM trace" in captured.out


@pytest.mark.unit
def test_analysis_continues_after_a_trace_write_failure(tmp_path):
    """The analysis result itself (not just absence of a crash) must still be
    produced when trace persistence fails."""
    tracer = LLMTraceCallbackHandler(tmp_path / "debug")
    bad_dir = tmp_path / "debug" / "calls" / "001_Trader"
    bad_dir.parent.mkdir(parents=True)
    bad_dir.write_text("not a directory")

    results = []

    def node(state):
        _fake_llm_call(tracer)
        results.append("trader completed")
        return state

    traced_node("Trader", node)(None)
    assert results == ["trader completed"]


# --- manifest / prompt snapshot ---------------------------------------------


@pytest.mark.unit
def test_prompt_files_are_snapshotted_and_hash_matches_content(tmp_path):
    hashes = snapshot_prompts(tmp_path / "debug", on_warn=lambda p, e: None)
    assert "global_policy.txt" in hashes
    snapshotted = (tmp_path / "debug" / "prompts" / "global_policy.txt").read_text(encoding="utf-8")
    import hashlib
    assert hashes["global_policy.txt"] == hashlib.sha256(snapshotted.encode("utf-8")).hexdigest()


@pytest.mark.unit
def test_manifest_contains_required_fields_and_no_secrets(tmp_path):
    config = {
        "temperature": 0.1, "max_tokens": 4096,
        "llm_provider": "ollama", "api_key": "sk-should-not-appear",
    }
    write_run_manifest(
        tmp_path / "debug",
        ticker="NVDA", analysis_date="2026-09-30", depth=3,
        analysts=["market", "social", "news", "fundamentals"], language="Spanish",
        provider="ollama", effective_deep_model="qwen3.8-flash-next:125b-a6b-q4_K_M",
        effective_quick_model="gpt-oss:20b",
        config=config, prompt_hashes={"global_policy.txt": "abc123"},
        on_warn=lambda p, e: None,
    )
    manifest = json.loads((tmp_path / "debug" / "run_manifest.json").read_text())
    assert manifest["ticker"] == "NVDA"
    assert manifest["analysis_date"] == "2026-09-30"
    assert manifest["depth"] == 3
    assert manifest["analysts"] == ["market", "social", "news", "fundamentals"]
    assert manifest["language"] == "Spanish"
    assert manifest["effective_deep_model"] == "qwen3.8-flash-next:125b-a6b-q4_K_M"
    assert manifest["effective_quick_model"] == "gpt-oss:20b"
    assert manifest["prompt_file_hashes"] == {"global_policy.txt": "abc123"}
    assert manifest["trace_format_version"] >= 1
    manifest_text = (tmp_path / "debug" / "run_manifest.json").read_text()
    assert "sk-should-not-appear" not in manifest_text
