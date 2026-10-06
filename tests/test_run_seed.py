"""Best-effort deterministic replay (see tradingagents/run_seed.py and fork
README: "Reproducible runs / Best-effort deterministic replay").

A configured seed of 0 means "auto-generate a fresh base seed for this run
and print it"; a non-zero value is reused exactly. Every LLM call then
derives its own stable seed from (base_seed, ticker, stage, call identity)
via SHA-256 -- never Python's randomized hash() -- so the same four inputs
always produce the same derived seed regardless of call order, batch
composition, or how many other calls happened first.
"""
from __future__ import annotations

import random

import pytest

from tradingagents.run_seed import (
    SEED_CAPABLE_PROVIDERS,
    derive_call_seed,
    generate_base_seed,
    resolve_base_seed,
    validate_configured_seed,
)

# --- 1: default config seed is 0 --------------------------------------------


@pytest.mark.unit
def test_default_config_seed_is_zero():
    from tradingagents.default_config import DEFAULT_CONFIG
    assert DEFAULT_CONFIG["seed"] == 0


# --- validation --------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("value,expected", [(0, 0), (12345, 12345), ("42", 42)])
def test_validate_accepts_zero_and_positive_ints(value, expected):
    assert validate_configured_seed(value) == expected


@pytest.mark.unit
@pytest.mark.parametrize("bad", [-1, "-5"])
def test_validate_rejects_negative(bad):
    with pytest.raises(ValueError, match=">= 0"):
        validate_configured_seed(bad)


@pytest.mark.unit
@pytest.mark.parametrize("bad", [True, False])
def test_validate_rejects_booleans(bad):
    with pytest.raises(ValueError, match="boolean"):
        validate_configured_seed(bad)


@pytest.mark.unit
@pytest.mark.parametrize("bad", ["abc", "1.5", None])
def test_validate_rejects_non_integers(bad):
    with pytest.raises(ValueError, match="integer"):
        validate_configured_seed(bad)


# --- 2, 3: resolve_base_seed ---------------------------------------------


@pytest.mark.unit
def test_seed_zero_generates_a_nonzero_base_seed(monkeypatch):
    monkeypatch.setattr(random, "SystemRandom", lambda: random.Random(1))
    base_seed = resolve_base_seed(0)
    assert base_seed != 0
    assert isinstance(base_seed, int)


@pytest.mark.unit
def test_seed_zero_generates_exactly_once_per_call(monkeypatch):
    """resolve_base_seed must call the RNG exactly once -- a caller invoking
    it twice for the same run (instead of once and threading the result)
    would be the actual bug, not something this function hides."""
    calls = []

    class _CountingRandom:
        def randint(self, a, b):
            calls.append((a, b))
            return 999

    monkeypatch.setattr(random, "SystemRandom", lambda: _CountingRandom())
    base_seed = resolve_base_seed(0)
    assert base_seed == 999
    assert len(calls) == 1


@pytest.mark.unit
def test_explicit_seed_is_reused_exactly():
    assert resolve_base_seed(12345) == 12345


@pytest.mark.unit
def test_generate_base_seed_is_nonzero_and_in_range():
    for _ in range(20):
        value = generate_base_seed()
        assert 1 <= value <= 2**31 - 1


# --- 4, 5, 6: derive_call_seed determinism / sensitivity --------------------


@pytest.mark.unit
def test_same_inputs_always_produce_the_same_derived_seed():
    a = derive_call_seed(12345, "NVDA", "Bull Researcher", "review_1")
    b = derive_call_seed(12345, "NVDA", "Bull Researcher", "review_1")
    assert a == b


@pytest.mark.unit
def test_changing_ticker_changes_the_derived_seed():
    a = derive_call_seed(12345, "NVDA", "Bull Researcher", "initial")
    b = derive_call_seed(12345, "AMD", "Bull Researcher", "initial")
    assert a != b


@pytest.mark.unit
def test_changing_stage_changes_the_derived_seed():
    a = derive_call_seed(12345, "NVDA", "Bull Researcher", "initial")
    b = derive_call_seed(12345, "NVDA", "Bear Researcher", "initial")
    assert a != b


@pytest.mark.unit
def test_changing_call_identity_changes_the_derived_seed():
    a = derive_call_seed(12345, "NVDA", "Bull Researcher", "review_1")
    b = derive_call_seed(12345, "NVDA", "Bull Researcher", "review_2")
    assert a != b


@pytest.mark.unit
def test_changing_base_seed_changes_the_derived_seed():
    a = derive_call_seed(12345, "NVDA", "Bull Researcher", "initial")
    b = derive_call_seed(54321, "NVDA", "Bull Researcher", "initial")
    assert a != b


@pytest.mark.unit
def test_derived_seed_is_within_the_shared_seed_space():
    value = derive_call_seed(12345, "NVDA", "Bull Researcher", "initial")
    assert 0 <= value < 2**31 - 1


# --- 7: batch order independence --------------------------------------------


@pytest.mark.unit
def test_derivation_does_not_depend_on_batch_order_or_other_tickers():
    """`batch --tickers NVDA,AMD --seed 12345` and `batch --tickers AMD
    --seed 12345` must derive the same seeds for AMD's own calls -- the
    function only ever sees AMD's own identity, never a global counter or
    knowledge of what ran before it."""
    identities = [
        ("Market Analyst", "tool_round_1"),
        ("Bull Researcher", "initial"),
        ("Bull Researcher", "review_1"),
        ("Research Verifier", "pass_1"),
    ]
    amd_alone = [derive_call_seed(12345, "AMD", stage, call) for stage, call in identities]
    amd_after_nvda = [derive_call_seed(12345, "AMD", stage, call) for stage, call in identities]
    assert amd_alone == amd_after_nvda


# --- 8: no use of Python's hash() / stable across a fresh interpreter -------


@pytest.mark.unit
def test_derive_call_seed_does_not_use_builtin_hash():
    import inspect

    from tradingagents import run_seed
    source = inspect.getsource(run_seed.derive_call_seed)
    assert "hash(" not in source


@pytest.mark.unit
def test_derive_call_seed_matches_a_hand_computed_sha256_value():
    """Pins the exact algorithm (canonical "|"-joined string, SHA-256, first
    8 bytes big-endian, modulo the shared seed space) so a future refactor
    that silently changes it is caught -- "stable across processes and
    Python versions" is a documented guarantee, not an implementation detail."""
    import hashlib
    canonical = "12345|NVDA|Bull Researcher|initial"
    digest = hashlib.sha256(canonical.encode("utf-8")).digest()
    expected = int.from_bytes(digest[:8], "big") % (2**31 - 1)
    assert derive_call_seed(12345, "NVDA", "Bull Researcher", "initial") == expected


# --- provider capability allowlist -------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("provider", ["openai", "azure", "ollama"])
def test_seed_capable_providers_include_openai_azure_ollama(provider):
    assert provider in SEED_CAPABLE_PROVIDERS


@pytest.mark.unit
@pytest.mark.parametrize(
    "provider",
    ["anthropic", "google", "xai", "deepseek", "qwen", "openai_compatible", "groq"],
)
def test_seed_capable_providers_exclude_unverified_providers(provider):
    assert provider not in SEED_CAPABLE_PROVIDERS
