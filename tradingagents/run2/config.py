"""run2's own, fully independent config: default values, env-var overrides,
and a resolver with the same CLI-flag > env-var > default precedence as the
main pipeline (see ``tradingagents/default_config.py``) -- but a completely
separate dict and a separate env-var namespace (``RUN2_*``, not
``TRADINGAGENTS_*``), so run2 can never read or mutate
``tradingagents.default_config.DEFAULT_CONFIG``/``_ENV_OVERRIDES`` and
`tradingagents run` can never be affected by a ``RUN2_*`` env var.

The coercion helpers below are a deliberate, literal copy of
``default_config.py``'s ``_coerce``/``_apply_env_overrides`` pattern (project
convention: duplicate small, generic helpers locally rather than import
across an intentional module boundary -- see also
``tradingagents/dataflows/price_basis.py``'s ``_canonical_json`` and
``tradingagents/pm_replay.py``'s ``_atomic_write_json``).
"""

from __future__ import annotations

import os
from typing import Any

# Frozen initial values for the binary First Barrier experiment (fork spec:
# "run2" -- First Barrier): the horizon is ~2 months of trading sessions,
# and the barrier is +/-5% from the reference price on analysis_date. There
# is deliberately no confidence threshold here -- a valid LONG/SHORT signal
# is the final decision outright (see pipeline.py).
RUN2_DEFAULT_CONFIG: dict[str, Any] = {
    "horizon_sessions": 40,
    "barrier_pct": 5.0,
}

_RUN2_ENV_OVERRIDES: dict[str, str] = {
    "RUN2_HORIZON_SESSIONS": "horizon_sessions",
    "RUN2_BARRIER_PCT": "barrier_pct",
}

_BOOL_TRUE = ("true", "1", "yes", "on")
_BOOL_FALSE = ("false", "0", "no", "off")


def _coerce(value: str, reference: Any) -> Any:
    """Coerce an env-var string to the type of ``reference`` (the existing
    default value) -- bool checked before int, since ``bool`` is an ``int``
    subclass in Python."""
    if isinstance(reference, bool):
        normalized = value.strip().lower()
        if normalized in _BOOL_TRUE:
            return True
        if normalized in _BOOL_FALSE:
            return False
        raise ValueError(f"expected a boolean ({'/'.join(_BOOL_TRUE + _BOOL_FALSE)}), got {value!r}")
    if isinstance(reference, int) and not isinstance(reference, bool):
        return int(value)
    if isinstance(reference, float):
        return float(value)
    return value


def _apply_run2_env_overrides(config: dict[str, Any]) -> dict[str, Any]:
    """Apply ``RUN2_*`` env vars to ``config`` in place. An empty-string env
    var is treated as unset, matching ``default_config.py``'s behaviour."""
    for env_var, key in _RUN2_ENV_OVERRIDES.items():
        raw = os.environ.get(env_var)
        if raw is None or raw == "":
            continue
        try:
            config[key] = _coerce(raw, config.get(key))
        except ValueError as exc:
            raise ValueError(f"Invalid value for {env_var}: {exc}") from exc
    return config


def resolve_run2_config(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """``RUN2_DEFAULT_CONFIG``, with ``RUN2_*`` env vars applied, with
    explicit ``overrides`` (e.g. CLI flags) taking final precedence. A
    ``None`` value in ``overrides`` means "not provided" and is skipped, so a
    caller can pass every CLI option through unconditionally."""
    config = dict(RUN2_DEFAULT_CONFIG)
    _apply_run2_env_overrides(config)
    if overrides:
        for key, value in overrides.items():
            if value is not None:
                config[key] = value
    return config
