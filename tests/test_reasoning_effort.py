"""Quick vs Deep reasoning effort (see fork README: "Quick vs Deep reasoning
effort"). ``quick_reasoning_effort`` / ``deep_reasoning_effort`` let each role
carry an independently configured reasoning level, forwarded as
``reasoning_effort`` to providers that accept it (OpenAI, and now Ollama's
OpenAI-compatible endpoint). Valid levels are model/backend-specific --
TradingAgents validates format only (non-empty string), never a fixed level
list. Unset means no override is sent, so the backend/model's own default
applies (e.g. Ollama's own per-model default via ``/api/show``)."""
from __future__ import annotations

import importlib

import pytest

import tradingagents.default_config as default_config_module
from tradingagents.llm_clients.factory import _coerce_reasoning_effort, build_llm_kwargs
from tradingagents.llm_clients.openai_client import OpenAIClient

# --- 16: validation ----------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("value", ["low", "medium", "xhigh", "max", " high "])
def test_coerce_accepts_and_strips_any_nonempty_string(value):
    assert _coerce_reasoning_effort(value) == value.strip()


@pytest.mark.unit
@pytest.mark.parametrize("bad", ["", "   "])
def test_coerce_rejects_empty_or_whitespace(bad):
    with pytest.raises(ValueError, match="empty"):
        _coerce_reasoning_effort(bad)


@pytest.mark.unit
@pytest.mark.parametrize("bad", [True, False])
def test_coerce_rejects_booleans(bad):
    with pytest.raises(ValueError, match="boolean"):
        _coerce_reasoning_effort(bad)


@pytest.mark.unit
@pytest.mark.parametrize("bad", [1, 2.5, ["low"]])
def test_coerce_rejects_non_strings(bad):
    with pytest.raises(ValueError, match="string"):
        _coerce_reasoning_effort(bad)


@pytest.mark.unit
def test_coerce_does_not_impose_a_fixed_level_list():
    """No hardcoded low/medium/high/max/xhigh allowlist -- the backend/model
    is the final authority on whether a level is valid."""
    assert _coerce_reasoning_effort("some-future-level-nobody-has-heard-of") == (
        "some-future-level-nobody-has-heard-of"
    )


# --- 5, 6, 7: build_llm_kwargs resolves quick/deep independently ------------


@pytest.mark.unit
def test_quick_role_uses_quick_reasoning_effort_for_openai():
    kwargs = build_llm_kwargs(
        {"llm_provider": "openai", "quick_reasoning_effort": "low", "deep_reasoning_effort": "max"},
        role="quick",
    )
    assert kwargs["reasoning_effort"] == "low"


@pytest.mark.unit
def test_deep_role_uses_deep_reasoning_effort_for_openai():
    kwargs = build_llm_kwargs(
        {"llm_provider": "openai", "quick_reasoning_effort": "low", "deep_reasoning_effort": "max"},
        role="deep",
    )
    assert kwargs["reasoning_effort"] == "max"


@pytest.mark.unit
def test_quick_and_deep_can_differ_simultaneously_for_ollama():
    config = {"llm_provider": "ollama", "quick_reasoning_effort": "low", "deep_reasoning_effort": "max"}
    quick_kwargs = build_llm_kwargs(config, role="quick")
    deep_kwargs = build_llm_kwargs(config, role="deep")
    assert quick_kwargs["reasoning_effort"] == "low"
    assert deep_kwargs["reasoning_effort"] == "max"
    assert quick_kwargs["reasoning_effort"] != deep_kwargs["reasoning_effort"]


@pytest.mark.unit
def test_openai_per_role_key_takes_precedence_over_legacy_global_key():
    kwargs = build_llm_kwargs(
        {"llm_provider": "openai", "openai_reasoning_effort": "medium", "quick_reasoning_effort": "low"},
        role="quick",
    )
    assert kwargs["reasoning_effort"] == "low"


@pytest.mark.unit
def test_openai_legacy_global_key_still_works_unchanged_when_role_omitted():
    """A caller that never migrated to per-role keys (role=None) keeps
    getting the old global behavior -- this is the backward-compat path."""
    kwargs = build_llm_kwargs({"llm_provider": "openai", "openai_reasoning_effort": "medium"})
    assert kwargs["reasoning_effort"] == "medium"


@pytest.mark.unit
def test_openai_legacy_global_key_is_fallback_when_role_given_but_unset():
    kwargs = build_llm_kwargs(
        {"llm_provider": "openai", "openai_reasoning_effort": "medium"}, role="quick"
    )
    assert kwargs["reasoning_effort"] == "medium"


# --- 8, 9: Ollama routing ----------------------------------------------------


@pytest.mark.unit
def test_ollama_receives_reasoning_effort_when_configured():
    kwargs = build_llm_kwargs(
        {"llm_provider": "ollama", "deep_reasoning_effort": "max"}, role="deep"
    )
    assert kwargs["reasoning_effort"] == "max"


@pytest.mark.unit
def test_ollama_without_configuration_sends_nothing():
    """No artificial default -- Ollama's own per-model default (e.g. "low"
    for DeepSeek V4 Pro, "medium" for Qwen3.8 Flash Next, per /api/show)
    applies when TradingAgents sends no override."""
    kwargs = build_llm_kwargs({"llm_provider": "ollama"}, role="deep")
    assert "reasoning_effort" not in kwargs


@pytest.mark.unit
def test_ollama_without_role_sends_nothing():
    """role=None (no caller opted into the per-role mechanism) must not
    accidentally leak a reasoning_effort for Ollama -- there is no legacy
    global Ollama knob to fall back to."""
    kwargs = build_llm_kwargs({"llm_provider": "ollama", "deep_reasoning_effort": "max"})
    assert "reasoning_effort" not in kwargs


@pytest.mark.unit
def test_other_openai_compatible_providers_are_not_generalized_to():
    """Explicitly NOT generalized to every OpenAI-compatible provider --
    xai/deepseek/qwen/groq/... support is unverified."""
    for provider in ("xai", "deepseek", "qwen", "groq", "openai_compatible"):
        kwargs = build_llm_kwargs(
            {"llm_provider": provider, "deep_reasoning_effort": "max"}, role="deep"
        )
        assert "reasoning_effort" not in kwargs, provider


# --- 10, 11: other providers unaffected / no unknown kwarg ------------------


@pytest.mark.unit
def test_google_and_anthropic_knobs_are_untouched():
    google_kwargs = build_llm_kwargs(
        {"llm_provider": "google", "google_thinking_level": "high",
         "deep_reasoning_effort": "max"}, role="deep",
    )
    assert google_kwargs == {"thinking_level": "high"}

    anthropic_kwargs = build_llm_kwargs(
        {"llm_provider": "anthropic", "anthropic_effort": "high",
         "deep_reasoning_effort": "max"}, role="deep",
    )
    assert anthropic_kwargs == {"effort": "high"}


@pytest.mark.unit
def test_openai_previous_behavior_unchanged_for_non_reasoning_model(monkeypatch):
    """Pre-existing regression guard (test_openai_reasoning_effort.py) still
    holds: a non-reasoning OpenAI model must not receive reasoning_effort,
    this time via the new per-role path."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    kwargs = build_llm_kwargs(
        {"llm_provider": "openai", "deep_reasoning_effort": "low"}, role="deep"
    )
    llm = OpenAIClient("gpt-4.1", provider="openai", **kwargs).get_llm()
    assert getattr(llm, "reasoning_effort", None) is None


@pytest.mark.unit
def test_openai_reasoning_model_still_receives_effort_via_per_role_path(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    kwargs = build_llm_kwargs(
        {"llm_provider": "openai", "deep_reasoning_effort": "high"}, role="deep"
    )
    llm = OpenAIClient("gpt-5.4-mini", provider="openai", **kwargs).get_llm()
    assert llm.reasoning_effort == "high"


@pytest.mark.unit
def test_ollama_model_receives_reasoning_effort_regardless_of_model_name_shape():
    """Unlike native OpenAI, Ollama's model-name shape (e.g.
    'deepseek-v4-pro:cloud', not a gpt-5/o-series name) must not gate the
    kwarg -- that gate is an OpenAI-only 400-avoidance quirk."""
    llm = OpenAIClient(
        "deepseek-v4-pro:cloud", provider="ollama", reasoning_effort="max"
    ).get_llm()
    assert llm.reasoning_effort == "max"


@pytest.mark.unit
def test_unsupported_provider_never_receives_the_kwarg():
    """A provider build_llm_kwargs never populates reasoning_effort for
    (e.g. xai) must not receive it even if somehow present in raw kwargs --
    the client only forwards kwargs declared in _PASSTHROUGH_KWARGS that are
    actually present in self.kwargs, which build_llm_kwargs controls."""
    kwargs = build_llm_kwargs({"llm_provider": "xai", "deep_reasoning_effort": "max"}, role="deep")
    assert kwargs == {}


# --- 12: coexistence with seed + max_tokens ---------------------------------


@pytest.mark.unit
def test_ollama_seed_max_tokens_and_reasoning_effort_coexist():
    """A single Ollama call can carry seed (best-effort deterministic
    replay), the default max_tokens=8192 guardrail, and reasoning_effort
    together -- none of the three mechanisms interferes with another.

    Checked at both layers: the client instance carries max_tokens and
    reasoning_effort as bound defaults, and the actual outgoing request
    payload (``_get_request_payload``, the same merge
    ``payload = {**self._default_params, **kwargs}`` that puts the seed into
    the real HTTP request) contains all three together.
    """
    from langchain_core.messages import HumanMessage

    kwargs = build_llm_kwargs(
        {"llm_provider": "ollama", "deep_reasoning_effort": "max"}, role="deep"
    )
    llm = OpenAIClient("deepseek-v4-pro:cloud", provider="ollama", **kwargs).get_llm()
    assert llm.reasoning_effort == "max"
    assert llm.max_tokens == 8192  # Ollama default guardrail, untouched

    payload = llm._get_request_payload([HumanMessage(content="hi")], seed=543350193)
    assert payload["seed"] == 543350193
    assert payload["reasoning_effort"] == "max"
    assert payload["max_completion_tokens"] == 8192


# --- env/config precedence ---------------------------------------------------


def _reload_with_env(monkeypatch, **overrides):
    for key in list(default_config_module._ENV_OVERRIDES):
        monkeypatch.delenv(key, raising=False)
    for key, val in overrides.items():
        monkeypatch.setenv(key, val)
    return importlib.reload(default_config_module)


@pytest.mark.unit
def test_default_is_none(monkeypatch):
    dc = _reload_with_env(monkeypatch)
    assert dc.DEFAULT_CONFIG["quick_reasoning_effort"] is None
    assert dc.DEFAULT_CONFIG["deep_reasoning_effort"] is None


@pytest.mark.unit
def test_env_overrides_set_config_independently(monkeypatch):
    dc = _reload_with_env(
        monkeypatch,
        TRADINGAGENTS_QUICK_REASONING_EFFORT="low",
        TRADINGAGENTS_DEEP_REASONING_EFFORT="max",
    )
    assert dc.DEFAULT_CONFIG["quick_reasoning_effort"] == "low"
    assert dc.DEFAULT_CONFIG["deep_reasoning_effort"] == "max"


@pytest.mark.unit
def test_empty_env_value_is_passthrough(monkeypatch):
    dc = _reload_with_env(
        monkeypatch,
        TRADINGAGENTS_QUICK_REASONING_EFFORT="",
        TRADINGAGENTS_DEEP_REASONING_EFFORT="",
    )
    assert dc.DEFAULT_CONFIG["quick_reasoning_effort"] is None
    assert dc.DEFAULT_CONFIG["deep_reasoning_effort"] is None


# --- canonical generation/review role naming (see fork README: "Model
# roles: Generation vs Review") -- build_llm_kwargs's role lookup is
# generic (f"{role}_reasoning_effort"), so these exercise the actual names
# TradingAgentsGraph calls it with today. --------------------------------


@pytest.mark.unit
def test_generation_reasoning_effort_only_reaches_generation_role():
    config = {
        "llm_provider": "ollama",
        "generation_reasoning_effort": "low", "review_reasoning_effort": "max",
    }
    generation_kwargs = build_llm_kwargs(config, role="generation")
    assert generation_kwargs["reasoning_effort"] == "low"


@pytest.mark.unit
def test_review_reasoning_effort_only_reaches_review_role():
    config = {
        "llm_provider": "ollama",
        "generation_reasoning_effort": "low", "review_reasoning_effort": "max",
    }
    review_kwargs = build_llm_kwargs(config, role="review")
    assert review_kwargs["reasoning_effort"] == "max"


@pytest.mark.unit
def test_generation_and_review_reasoning_effort_differ_simultaneously():
    config = {
        "llm_provider": "ollama",
        "generation_reasoning_effort": "low", "review_reasoning_effort": "max",
    }
    assert build_llm_kwargs(config, role="generation")["reasoning_effort"] == "low"
    assert build_llm_kwargs(config, role="review")["reasoning_effort"] == "max"


@pytest.mark.unit
def test_generation_and_review_reasoning_effort_can_be_equal():
    config = {
        "llm_provider": "ollama",
        "generation_reasoning_effort": "medium", "review_reasoning_effort": "medium",
    }
    assert build_llm_kwargs(config, role="generation")["reasoning_effort"] == "medium"
    assert build_llm_kwargs(config, role="review")["reasoning_effort"] == "medium"


@pytest.mark.unit
def test_openai_generation_role_key_takes_precedence_over_legacy_global_key():
    kwargs = build_llm_kwargs(
        {"llm_provider": "openai", "openai_reasoning_effort": "medium",
         "generation_reasoning_effort": "low"},
        role="generation",
    )
    assert kwargs["reasoning_effort"] == "low"
