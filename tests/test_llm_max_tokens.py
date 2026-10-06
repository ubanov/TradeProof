"""Configurable output-token cap (#1204).

Some model/gateway combinations (e.g. deepseek-v4-flash deployments) emit
unbounded reasoning/output and hang or trip an idle timeout. An opt-in
``max_tokens`` config knob is forwarded to every provider so a run can bound it;
Gemini names the parameter ``max_output_tokens``, so it is forwarded under the
right key per provider.
"""
from __future__ import annotations

import importlib

import pytest

import tradingagents.default_config as default_config_module
from tradingagents.llm_clients.factory import _coerce_max_tokens, build_llm_kwargs

# --- coercion / validation -------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("value,expected", [(1, 1), (8192, 8192), ("4096", 4096)])
def test_coerce_accepts_positive_ints_and_numeric_strings(value, expected):
    assert _coerce_max_tokens(value) == expected


@pytest.mark.unit
@pytest.mark.parametrize("bad", [0, -1, "0", "-5"])
def test_coerce_rejects_non_positive(bad):
    with pytest.raises(ValueError, match="> 0"):
        _coerce_max_tokens(bad)


@pytest.mark.unit
@pytest.mark.parametrize("bad", [True, False])
def test_coerce_rejects_booleans(bad):
    with pytest.raises(ValueError, match="boolean"):
        _coerce_max_tokens(bad)


@pytest.mark.unit
@pytest.mark.parametrize("bad", ["abc", "1.5", None])
def test_coerce_rejects_non_integers(bad):
    with pytest.raises(ValueError, match="integer"):
        _coerce_max_tokens(bad)


# --- forwarding into provider kwargs (right key per provider) --------------


@pytest.mark.unit
def test_not_forwarded_when_unset():
    kwargs = build_llm_kwargs({"llm_provider": "openai", "max_tokens": None})
    assert "max_tokens" not in kwargs
    assert "max_output_tokens" not in kwargs


@pytest.mark.unit
@pytest.mark.parametrize("provider", ["openai", "anthropic", "deepseek", "openai_compatible"])
def test_forwarded_as_max_tokens_for_non_google(provider):
    kwargs = build_llm_kwargs({"llm_provider": provider, "max_tokens": 8192})
    assert kwargs["max_tokens"] == 8192
    assert "max_output_tokens" not in kwargs


@pytest.mark.unit
def test_forwarded_as_max_output_tokens_for_google():
    # Gemini's kwarg name differs; forwarding plain max_tokens would be rejected.
    kwargs = build_llm_kwargs({"llm_provider": "google", "max_tokens": 8192})
    assert kwargs["max_output_tokens"] == 8192
    assert "max_tokens" not in kwargs


@pytest.mark.unit
def test_env_string_is_coerced():
    kwargs = build_llm_kwargs({"llm_provider": "openai", "max_tokens": "4096"})
    assert kwargs["max_tokens"] == 4096


@pytest.mark.unit
def test_invalid_value_fails_loudly():
    with pytest.raises(ValueError):
        build_llm_kwargs({"llm_provider": "openai", "max_tokens": 0})


# --- client-side allowlists carry the kwarg --------------------------------

@pytest.mark.unit
def test_openai_and_google_clients_accept_the_kwarg():
    from tradingagents.llm_clients import openai_client
    from tradingagents.llm_clients.google_client import GoogleClient  # noqa: F401
    assert "max_tokens" in openai_client._PASSTHROUGH_KWARGS
    # Google client forwards max_output_tokens through construction.
    llm = GoogleClient("gemini-3.5-flash", api_key="x", max_output_tokens=8192).get_llm()
    assert getattr(llm, "max_output_tokens", None) == 8192


# --- env overlay -----------------------------------------------------------

def _reload_with_env(monkeypatch, **overrides):
    for key in list(default_config_module._ENV_OVERRIDES):
        monkeypatch.delenv(key, raising=False)
    for key, val in overrides.items():
        monkeypatch.setenv(key, val)
    return importlib.reload(default_config_module)


@pytest.mark.unit
def test_default_is_none(monkeypatch):
    dc = _reload_with_env(monkeypatch)
    assert dc.DEFAULT_CONFIG["max_tokens"] is None


@pytest.mark.unit
def test_env_override_sets_config(monkeypatch):
    dc = _reload_with_env(monkeypatch, TRADINGAGENTS_MAX_TOKENS="8192")
    assert dc.DEFAULT_CONFIG["max_tokens"] == "8192"
    assert _coerce_max_tokens(dc.DEFAULT_CONFIG["max_tokens"]) == 8192


# --- Ollama default max_tokens: runaway-generation safety guardrail --------
#
# A local Ollama model has no provider-side output cap; a real run observed a
# single call consuming the entire context window (~131K tokens, 30+ minutes,
# empty content / no tool call, finish_reason="length") instead of a normal
# few-thousand-token response. Ollama alone gets a default max_tokens; every
# other provider's behavior must stay exactly as it was.


@pytest.mark.unit
def test_ollama_gets_the_default_cap_when_unset():
    from tradingagents.llm_clients.openai_client import OLLAMA_DEFAULT_MAX_TOKENS, OpenAIClient

    llm = OpenAIClient(model="gemma4:26b", provider="ollama").get_llm()
    assert OLLAMA_DEFAULT_MAX_TOKENS == 8192
    assert llm.max_tokens == 8192


@pytest.mark.unit
def test_ollama_explicit_max_tokens_overrides_the_default():
    from tradingagents.llm_clients.openai_client import OpenAIClient

    llm = OpenAIClient(model="gemma4:26b", provider="ollama", max_tokens=16384).get_llm()
    assert llm.max_tokens == 16384


@pytest.mark.unit
def test_ollama_default_survives_the_full_config_to_client_pipeline():
    """End-to-end: no TRADINGAGENTS_MAX_TOKENS set -> build_llm_kwargs forwards
    nothing (unchanged) -> OpenAIClient still applies the Ollama default."""
    from tradingagents.llm_clients.openai_client import OLLAMA_DEFAULT_MAX_TOKENS, OpenAIClient

    kwargs = build_llm_kwargs({"llm_provider": "ollama", "max_tokens": None})
    assert "max_tokens" not in kwargs  # build_llm_kwargs itself is unchanged

    llm = OpenAIClient(model="gemma4:26b", provider="ollama", **kwargs).get_llm()
    assert llm.max_tokens == OLLAMA_DEFAULT_MAX_TOKENS


@pytest.mark.unit
def test_ollama_env_override_wins_over_the_default_end_to_end():
    """TRADINGAGENTS_MAX_TOKENS=16384 flows through build_llm_kwargs and still
    overrides the Ollama default once it reaches the client."""
    from tradingagents.llm_clients.openai_client import OpenAIClient

    kwargs = build_llm_kwargs({"llm_provider": "ollama", "max_tokens": "16384"})
    llm = OpenAIClient(model="gemma4:26b", provider="ollama", **kwargs).get_llm()
    assert llm.max_tokens == 16384


@pytest.mark.unit
def test_openai_does_not_get_a_default_max_tokens():
    from tradingagents.llm_clients.openai_client import OpenAIClient

    llm = OpenAIClient(model="gpt-4.1", provider="openai", api_key="x").get_llm()
    assert llm.max_tokens is None


@pytest.mark.unit
def test_generic_openai_compatible_local_server_does_not_inherit_the_ollama_default():
    """The generic 'openai_compatible' provider (LM Studio, vLLM, llama.cpp)
    must not silently receive Ollama's guardrail -- only "ollama" declares it."""
    from tradingagents.llm_clients.openai_client import OpenAIClient

    llm = OpenAIClient(
        model="local-model", provider="openai_compatible", base_url="http://localhost:1234/v1"
    ).get_llm()
    assert llm.max_tokens is None


@pytest.mark.unit
def test_azure_is_a_structurally_separate_client_unaffected_by_the_ollama_default(monkeypatch):
    """Azure has its own client/kwargs allowlist entirely outside
    OpenAIClient/OPENAI_COMPATIBLE_PROVIDERS, so the Ollama default cannot
    reach it even in principle."""
    from tradingagents.llm_clients.azure_client import AzureOpenAIClient

    monkeypatch.delenv("AZURE_OPENAI_DEPLOYMENT_NAME", raising=False)
    monkeypatch.setenv("OPENAI_API_VERSION", "2025-03-01-preview")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com/")
    llm = AzureOpenAIClient("gpt-4.1", api_key="test-key").get_llm()
    assert getattr(llm, "max_tokens", None) is None


@pytest.mark.unit
def test_invalid_explicit_max_tokens_for_ollama_still_fails_loudly():
    """An invalid user-set value must still raise before ever reaching the
    Ollama-default logic -- the default must never paper over bad input."""
    with pytest.raises(ValueError):
        build_llm_kwargs({"llm_provider": "ollama", "max_tokens": 0})


@pytest.mark.unit
def test_ollama_default_does_not_change_base_url_or_context_length(monkeypatch):
    """The guardrail is only an output-token cap -- base_url resolution (and
    therefore num_ctx, which lives server-side in Ollama, untouched by this
    client) must be completely unaffected."""
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    from tradingagents.llm_clients.openai_client import OpenAIClient

    llm = OpenAIClient(model="gemma4:26b", provider="ollama").get_llm()
    assert str(llm.openai_api_base) == "http://localhost:11434/v1"
    assert llm.max_tokens == 8192
