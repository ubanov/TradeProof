"""Best-effort deterministic replay: a per-run base seed, deterministically
derived into a stable per-call seed from (base_seed, ticker, stage, logical
call identity) -- never from a global call counter or batch execution order.

This is explicitly BEST-EFFORT, not a guarantee of bit-for-bit identical
output -- see the fork README section "Reproducible runs / Best-effort
deterministic replay" for the full list of reasons it can't be (backend/model
version, hardware, external tool/API data, provider seed support, non-
deterministic GPU kernels, and any code/prompt/config change).

Design:

- ``resolve_base_seed`` turns the user's configured seed (0 = auto-generate)
  into one concrete integer, once per run (once per batch, not once per
  ticker -- callers must call this exactly once and thread the result down).
- ``derive_call_seed`` turns ``(base_seed, ticker, stage, call_identity)``
  into one stable per-call seed via SHA-256 over a canonical string --
  deliberately NOT Python's built-in ``hash()``, which is randomized per
  process (``PYTHONHASHSEED``) specifically to prevent this kind of reuse.
  The same four inputs always produce the same digest, and therefore the
  same derived seed, regardless of call order, which other tickers ran
  first, or how many other LLM calls happened earlier in the run.
"""

from __future__ import annotations

import hashlib
import random

# Providers whose current LangChain client actually forwards a `seed` request
# parameter to a Chat-Completions-shaped API documented to honor it: OpenAI's
# "reproducible outputs" feature, which Azure OpenAI mirrors exactly, and
# Ollama's OpenAI-compatible layer, which forwards it to its own native seed
# option. Every other provider in this project's registry either has no
# documented seed support in its current client (Anthropic, Google, Bedrock)
# or speaks the same OpenAI-compatible wire format without verified
# server-side seed support (xai, deepseek, qwen, glm, minimax, openrouter,
# mistral, kimi, groq, nvidia, the generic openai_compatible endpoint) --
# silently ignoring an unsupported seed would be harmless, but a provider
# that 400s on an unrecognised parameter would not be, so this stays a
# deliberate, explicit allowlist rather than "every OpenAI-compatible
# provider".
SEED_CAPABLE_PROVIDERS = frozenset({"openai", "azure", "ollama"})

# The derived per-call seed is reduced into this range so it is valid for
# every seed-capable provider's accepted integer range, regardless of exactly
# how narrow any one of them is -- comfortably within a signed 32-bit int and
# well within JSON-number precision.
_SEED_SPACE = 2**31 - 1


def validate_configured_seed(value) -> int:
    """Validate a user-configured seed (CLI flag, config, or env var).

    0 is a valid, special value meaning "auto-generate a base seed for this
    run". Negative values are rejected outright -- there is no supported
    provider or algorithmic reason to accept one.
    """
    if isinstance(value, bool):
        raise ValueError(f"seed must be an integer, not a boolean: {value!r}")
    try:
        n = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"seed must be an integer, got {value!r}") from exc
    if n < 0:
        raise ValueError(f"seed must be >= 0 (0 means auto-generate a base seed), got {n}")
    return n


def generate_base_seed() -> int:
    """A fresh, non-zero random base seed for a ``seed=0`` (auto) run."""
    return random.SystemRandom().randint(1, _SEED_SPACE)


def resolve_base_seed(configured_seed: int) -> int:
    """The effective base seed for this run.

    Call exactly once per run (once per batch, not once per ticker -- see
    module docstring): a fresh random seed when ``configured_seed`` is the
    auto-generate sentinel (0), otherwise exactly the configured value.
    """
    if configured_seed == 0:
        return generate_base_seed()
    return configured_seed


def derive_call_seed(base_seed: int, ticker: str, stage: str, call_identity: str) -> int:
    """Stable per-call seed from ``(base_seed, ticker, stage, call_identity)``.

    Deterministic across processes and Python versions. The four inputs are
    joined into one canonical "|"-separated string and hashed with SHA-256;
    the first 8 bytes of the digest, reduced modulo the shared seed space,
    are the derived seed. The same four inputs always produce the same
    result -- call order, other tickers in the same batch run, and how many
    other LLM calls happened first never factor in.
    """
    canonical = f"{base_seed}|{ticker}|{stage}|{call_identity}"
    digest = hashlib.sha256(canonical.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % _SEED_SPACE
