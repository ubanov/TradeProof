import os

_TRADINGAGENTS_HOME = os.path.join(os.path.expanduser("~"), ".tradingagents")

# Single source of truth for env-var → config-key overrides. To expose
# a new config key for environment-based override, add a row here — no
# entry-point script changes required. Coercion is driven by the type
# of the existing default, so users can keep writing plain strings in
# their .env file.
_ENV_OVERRIDES = {
    "TRADINGAGENTS_LLM_PROVIDER":         "llm_provider",
    # Legacy model-role names (deprecated, see fork README: "Legacy naming").
    # Still accepted as input; resolve_role_config() below is the single
    # compatibility layer that maps them onto the canonical generation_llm /
    # review_llm keys every internal reader uses from here on.
    "TRADINGAGENTS_DEEP_THINK_LLM":       "deep_think_llm",
    "TRADINGAGENTS_QUICK_THINK_LLM":      "quick_think_llm",
    "TRADINGAGENTS_QUICK_REASONING_EFFORT": "quick_reasoning_effort",
    "TRADINGAGENTS_DEEP_REASONING_EFFORT":  "deep_reasoning_effort",
    # Canonical model-role config (see fork README: "Model roles: Generation
    # vs Review"). generation = produce/structure/synthesize/propose/decide;
    # review = challenge/verify/audit/correct/repair -- a function in the
    # pipeline, not a statement about model size/speed/cost.
    "TRADINGAGENTS_GENERATION_LLM":              "generation_llm",
    "TRADINGAGENTS_REVIEW_LLM":                  "review_llm",
    "TRADINGAGENTS_GENERATION_REASONING_EFFORT": "generation_reasoning_effort",
    "TRADINGAGENTS_REVIEW_REASONING_EFFORT":     "review_reasoning_effort",
    "TRADINGAGENTS_LLM_BACKEND_URL":      "backend_url",
    "TRADINGAGENTS_OUTPUT_LANGUAGE":      "output_language",
    "TRADINGAGENTS_MAX_DEBATE_ROUNDS":    "max_debate_rounds",
    "TRADINGAGENTS_MAX_RISK_ROUNDS":      "max_risk_discuss_rounds",
    "TRADINGAGENTS_CHECKPOINT_ENABLED":   "checkpoint_enabled",
    "TRADINGAGENTS_BENCHMARK_TICKER":     "benchmark_ticker",
    "TRADINGAGENTS_TEMPERATURE":          "temperature",
    "TRADINGAGENTS_LLM_MAX_RETRIES":      "llm_max_retries",
    "TRADINGAGENTS_MAX_TOKENS":           "max_tokens",
    "TRADINGAGENTS_SEED":                 "seed",
    # Provider-specific reasoning/thinking knobs (None = each provider's own
    # default). Settable here for non-interactive runs; the CLI also offers an
    # interactive choice, which is skipped when the matching var is set.
    "TRADINGAGENTS_GOOGLE_THINKING_LEVEL":   "google_thinking_level",
    "TRADINGAGENTS_OPENAI_REASONING_EFFORT": "openai_reasoning_effort",
    "TRADINGAGENTS_ANTHROPIC_EFFORT":        "anthropic_effort",
}


_BOOL_TRUE = ("true", "1", "yes", "on")
_BOOL_FALSE = ("false", "0", "no", "off")


def _coerce(value: str, reference):
    """Coerce env-var string to the type of the existing default value.

    Invalid values raise ``ValueError`` rather than silently falling back to a
    default — a misspelled boolean (e.g. ``treu``) or non-numeric int should fail
    loudly at startup, not quietly misconfigure an unattended run.
    """
    if isinstance(reference, bool):
        normalized = value.strip().lower()
        if normalized in _BOOL_TRUE:
            return True
        if normalized in _BOOL_FALSE:
            return False
        raise ValueError(
            f"expected a boolean ({'/'.join(_BOOL_TRUE + _BOOL_FALSE)}), got {value!r}"
        )
    if isinstance(reference, int) and not isinstance(reference, bool):
        return int(value)
    if isinstance(reference, float):
        return float(value)
    return value


def _apply_env_overrides(config: dict) -> dict:
    """Apply TRADINGAGENTS_* env vars to the config dict in-place."""
    for env_var, key in _ENV_OVERRIDES.items():
        raw = os.environ.get(env_var)
        if raw is None or raw == "":
            continue
        try:
            config[key] = _coerce(raw, config.get(key))
        except ValueError as exc:
            raise ValueError(f"Invalid value for {env_var}: {exc}") from exc
    return config


def _resolve_pair(
    configured_primary: str | None, configured_secondary: str | None,
    cli_primary: str | None, cli_secondary: str | None,
) -> tuple[str, str]:
    """Shared precedence algorithm behind ``resolve_effective_models`` and
    ``resolve_effective_roles``: an explicit, non-empty CLI secondary override
    always wins; otherwise an explicit CLI primary override makes secondary
    follow it too (one flag runs the whole analysis on that model); otherwise
    secondary falls back to its own configured value if non-empty; otherwise
    secondary inherits whatever primary resolved to.

    An explicitly given but empty/whitespace CLI secondary value is treated
    the same as "no secondary override", not as "configured secondary".
    """
    configured_primary = (configured_primary or "").strip()
    configured_secondary = (configured_secondary or "").strip()

    cli_primary_clean = (cli_primary or "").strip()
    cli_secondary_given = cli_secondary is not None
    cli_secondary_clean = (cli_secondary or "").strip()

    effective_primary = cli_primary_clean or configured_primary

    if cli_secondary_given and cli_secondary_clean:
        effective_secondary = cli_secondary_clean
    elif cli_secondary_given or cli_primary_clean:
        effective_secondary = effective_primary
    elif configured_secondary:
        effective_secondary = configured_secondary
    else:
        effective_secondary = effective_primary

    return effective_primary, effective_secondary


def resolve_effective_models(
    config: dict, cli_deep: str | None = None, cli_quick: str | None = None
) -> tuple[str, str]:
    """Single source of truth for which deep/quick model a run actually uses.

    Deprecated naming (see fork README: "Legacy naming") -- kept for backward
    compatibility. ``resolve_effective_roles`` is the canonical
    generation/review equivalent; both share the same precedence algorithm
    (``_resolve_pair``).

    ``cli_deep``/``cli_quick`` are CLI-flag values; ``None`` means the flag was
    not given at all.
    """
    return _resolve_pair(
        config.get("deep_think_llm"), config.get("quick_think_llm"), cli_deep, cli_quick
    )


def resolve_effective_roles(
    config: dict, cli_review: str | None = None, cli_generation: str | None = None
) -> tuple[str, str]:
    """Single source of truth for which generation/review model a run
    actually uses (see fork README: "Model roles: Generation vs Review").

    ``cli_review``/``cli_generation`` are CLI-flag values; ``None`` means the
    flag was not given at all. ``cli_review`` plays the same "primary" role
    ``cli_deep`` does in ``resolve_effective_models`` -- an explicit
    ``--review-llm`` alone runs the whole analysis on that model unless
    ``--generation-llm`` is also given, mirroring the existing
    ``--deep-think-llm`` behavior users already know.
    """
    return _resolve_pair(
        config.get("review_llm"), config.get("generation_llm"), cli_review, cli_generation
    )


# Hardcoded fallback when neither the canonical nor the legacy config/env
# value is set for a role -- applied only inside resolve_role_config, never
# baked directly into the generation_llm/review_llm dict entries below, so
# "was this explicitly configured" can still be distinguished from "this is
# just the default" during resolution.
_GENERATION_LLM_DEFAULT = "gpt-6-luna"
_REVIEW_LLM_DEFAULT = "gpt-6-sol"


def resolve_role_config(config: dict) -> dict:
    """The single compatibility layer between canonical generation/review
    naming and legacy quick/deep naming (see fork README: "Legacy naming" /
    "Model roles: Generation vs Review"). Mutates and returns ``config`` in
    place, so it resolves the same way regardless of how ``config`` was
    built -- DEFAULT_CONFIG at import time, a CLI-assembled dict, or a bare
    dict a caller passes straight into ``TradingAgentsGraph``.

    Precedence per role: explicit canonical name (``generation_llm`` /
    ``review_llm``) > legacy equivalent (``quick_think_llm`` -> generation,
    ``deep_think_llm`` -> review) > hardcoded default. The resolved value is
    mirrored back onto the legacy key too, so code that still reads it (the
    interactive CLI flow's selections/remembered-preferences machinery) sees
    the exact same, single, correctly-resolved value -- never two diverging
    sources of truth. Reasoning effort follows the identical rule, with no
    hardcoded default (unset means "send no override, the backend/model's
    own default applies").
    """
    config["generation_llm"] = (
        config.get("generation_llm") or config.get("quick_think_llm") or _GENERATION_LLM_DEFAULT
    )
    config["review_llm"] = (
        config.get("review_llm") or config.get("deep_think_llm") or _REVIEW_LLM_DEFAULT
    )
    config["generation_reasoning_effort"] = (
        config.get("generation_reasoning_effort") or config.get("quick_reasoning_effort")
    )
    config["review_reasoning_effort"] = (
        config.get("review_reasoning_effort") or config.get("deep_reasoning_effort")
    )
    config["quick_think_llm"] = config["generation_llm"]
    config["deep_think_llm"] = config["review_llm"]
    config["quick_reasoning_effort"] = config["generation_reasoning_effort"]
    config["deep_reasoning_effort"] = config["review_reasoning_effort"]
    return config


DEFAULT_CONFIG = resolve_role_config(_apply_env_overrides({
    "results_dir": os.getenv("TRADINGAGENTS_RESULTS_DIR") or os.path.join(_TRADINGAGENTS_HOME, "logs"),
    "data_cache_dir": os.getenv("TRADINGAGENTS_CACHE_DIR") or os.path.join(_TRADINGAGENTS_HOME, "cache"),
    "memory_log_path": os.getenv("TRADINGAGENTS_MEMORY_LOG_PATH") or os.path.join(_TRADINGAGENTS_HOME, "memory", "trading_memory.md"),
    # Optional cap on the number of resolved memory log entries. When set,
    # the oldest resolved entries are pruned once this limit is exceeded.
    # Pending entries are never pruned. None disables rotation entirely.
    "memory_log_max_entries": None,
    # LLM settings
    "llm_provider": "openai",
    # Canonical model-role keys (see fork README: "Model roles: Generation vs
    # Review"). Left unset (None) here -- resolve_role_config (called below,
    # wrapping this whole literal) is what actually fills these in, honoring
    # explicit canonical > legacy quick/deep > hardcoded default, so "this
    # was never set" stays distinguishable from "this is just the default"
    # until that single resolution step runs.
    "generation_llm": None,
    "review_llm": None,
    "generation_reasoning_effort": None,
    "review_reasoning_effort": None,
    # Legacy aliases (deprecated, see fork README "Legacy naming": quick ->
    # generation, deep -> review). Still accepted as input; resolve_role_config
    # resolves and mirrors them, so no other code should read these directly.
    "quick_think_llm": None,
    "deep_think_llm": None,
    "quick_reasoning_effort": None,
    "deep_reasoning_effort": None,
    # When None, each provider's client falls back to its own default endpoint
    # (api.openai.com for OpenAI, generativelanguage.googleapis.com for Gemini, ...).
    # The CLI overrides this per provider when the user picks one. Keeping a
    # provider-specific URL here would leak (e.g. OpenAI's /v1 was previously
    # being forwarded to Gemini, producing malformed request URLs).
    "backend_url": None,
    # Provider-specific thinking configuration
    "google_thinking_level": None,      # "high", "minimal", etc.
    "openai_reasoning_effort": None,    # "medium", "high", "low"
    "anthropic_effort": None,           # "high", "medium", "low"
    # Sampling temperature, forwarded to every provider when set. None leaves
    # each provider at its own default. Lower values reduce run-to-run
    # variation on models that honor it; reasoning models largely ignore it
    # and no setting makes LLM output bit-identical across runs (see README).
    "temperature": None,
    # SDK retry budget forwarded to every provider chat client. None leaves each
    # provider/SDK at its own default (usually 2). Raise it to ride out bursty
    # 429 throttling on rate-limited deployments instead of aborting a run (#1091).
    "llm_max_retries": None,
    # Cap on output tokens forwarded to every provider chat client. None leaves
    # each provider at its own default. Set it to bound a model that emits
    # unbounded reasoning/output and hangs or trips a gateway idle timeout
    # (e.g. some deepseek-v4-flash deployments, #1204).
    "max_tokens": None,
    # Best-effort deterministic replay base seed. 0 means "auto-generate a
    # fresh base seed for this run and print it"; a non-zero value reuses it.
    # See fork README: "Reproducible runs / Best-effort deterministic replay".
    "seed": 0,
    # Checkpoint/resume: when True, LangGraph saves state after each node
    # so a crashed run can resume from the last successful step.
    "checkpoint_enabled": False,
    # Output language for analyst reports and final decision
    # Internal agent debate stays in English for reasoning quality
    "output_language": "English",
    # Debate and discussion settings
    "max_debate_rounds": 1,
    "max_risk_discuss_rounds": 1,
    "max_recur_limit": 100,
    # Shared analysis horizon for both Bull and Bear initial theses, so one
    # side cannot independently pick a materially different holding window
    # (e.g. Bull at 3 days, Bear at 3 months). A single fixed string for now;
    # see fork README "Research horizon".
    "research_horizon": "5-10 trading sessions",
    # News / data fetching parameters
    # Increase for longer lookback strategies or to broaden macro coverage;
    # decrease to reduce token usage in agent prompts.
    "news_article_limit": 20,             # max articles per ticker (ticker-news)
    "global_news_article_limit": 10,      # max articles for global/macro news
    "global_news_lookback_days": 7,       # macro news lookback window
    # Search queries used by get_global_news for macro headlines. Extend or
    # replace to broaden geographic / sector coverage.
    "global_news_queries": [
        "Federal Reserve interest rates inflation",
        "S&P 500 earnings GDP economic outlook",
        "geopolitical risk trade war sanctions",
        "ECB Bank of England BOJ central bank policy",
        "oil commodities supply chain energy",
    ],
    # Data vendor configuration
    # Category-level configuration (default for all tools in category).
    # The configured value is the exact vendor chain — requests are NOT silently
    # routed to vendors you didn't choose. For ordered fallback, list several,
    # e.g. "yfinance,alpha_vantage". "default" uses all available vendors.
    "data_vendors": {
        "core_stock_apis": "yfinance",       # Options: alpha_vantage, yfinance
        "technical_indicators": "yfinance",  # Options: alpha_vantage, yfinance
        "fundamental_data": "yfinance",      # Options: alpha_vantage, yfinance
        "news_data": "yfinance",             # Options: alpha_vantage, yfinance
        "macro_data": "fred",                # Options: fred (needs FRED_API_KEY)
        "prediction_markets": "polymarket",  # Options: polymarket (keyless)
    },
    # Tool-level configuration (takes precedence over category-level)
    "tool_vendors": {
        # Example: "get_stock_data": "alpha_vantage",  # Override category default
    },
    # Benchmark for alpha calculation in the reflection layer.
    # ``benchmark_ticker`` (when set) overrides the suffix map for all
    # tickers; leave it None to use ``benchmark_map`` for auto-detection
    # based on the ticker's exchange suffix. SPY remains the US default
    # so the reflection label keeps reading "Alpha vs SPY" for US tickers
    # while non-US tickers get their regional index automatically.
    # Trading days after the analysis date over which a decision's outcome is
    # measured, for reflection and for the backtest figures.
    "holding_period_days": 5,
    "benchmark_ticker": None,
    "benchmark_map": {
        ".NS":  "^NSEI",       # NSE India (Nifty 50)
        ".BO":  "^BSESN",      # BSE India (Sensex)
        ".T":   "^N225",       # Tokyo (Nikkei 225)
        ".HK":  "^HSI",        # Hong Kong (Hang Seng)
        ".L":   "^FTSE",       # London (FTSE 100)
        ".TO":  "^GSPTSE",     # Toronto (TSX Composite)
        ".AX":  "^AXJO",       # Australia (ASX 200)
        ".SS":  "000001.SS",   # Shanghai (SSE Composite)
        ".SZ":  "399001.SZ",   # Shenzhen (SZSE Component)
        ".SA":  "^BVSP",       # B3 Brazil (Ibovespa)
        "":     "SPY",         # default for US-listed tickers (no suffix)
    },
}))
