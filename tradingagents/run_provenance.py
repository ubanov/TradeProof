"""Run provenance: a structured, JSON-safe snapshot of the metadata needed
to reproduce or compare one analysis run -- package version, git identity,
configured models/reasoning-efforts/seeds, execution parameters, and
prompt-material hashes.

Built unconditionally, once per run (success or partial/failed), regardless
of ``--trace-llm`` -- this is the future metadata source for
RecommendationRecord. Trace mode's own ``run_manifest.json``
(``llm_trace.py``) is a separate, richer, opt-in artifact; it consumes the
SAME git/version/prompt-hash primitives this module uses (see
``execute_analysis``, which computes them once and feeds both), so the two
can never independently diverge on the fields they share.
"""

from __future__ import annotations

from typing import Any

from tradingagents.llm_trace import _git_info, _package_version, hash_prompt_files

PROVENANCE_SCHEMA_VERSION = 1


def _role_metadata(config: dict, role: str) -> dict[str, Any]:
    """Name/reasoning-effort/provider as CONFIGURED for ``role``
    ("generation"/"review"/"validator") -- never deduced retrospectively
    from rendered text. ``provider`` is this project's single run-wide
    ``llm_provider`` config value (there is no per-role provider override
    today); if that ever changes, this is the one place to update."""
    return {
        "name": config.get(f"{role}_llm"),
        "reasoning_effort": config.get(f"{role}_reasoning_effort"),
        "provider": config.get("llm_provider"),
    }


def build_run_provenance(
    *,
    selected_analysts: list[str],
    config: dict,
    configured_seed: int | None = None,
    base_seed: int | None = None,
    trace_llm: bool = False,
    trace_manifest_path: str | None = None,
    git_info: dict[str, str | bool | None] | None = None,
    package_version: str | None = None,
    prompt_hashes: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Unconditional provenance snapshot for one run.

    ``git_info``/``package_version``/``prompt_hashes`` may be passed in
    already-computed by the caller (so git/version/hashing run exactly once
    per run, shared with trace mode's manifest when it's also active);
    omitting them computes each here directly -- git/version are
    best-effort and never fail the run (see ``_git_info``/
    ``_package_version``), and a git failure yields explicit ``None``
    fields, never a fabricated commit.

    ``trace_manifest_path`` is the path to this run's ``run_manifest.json``
    when ``--trace-llm`` was on, else ``None`` -- the only two fields here
    that legitimately depend on tracing; everything else is always present.
    """
    resolved_git = git_info if git_info is not None else _git_info()
    resolved_version = package_version if package_version is not None else _package_version()
    resolved_prompt_hashes = prompt_hashes if prompt_hashes is not None else hash_prompt_files()

    return {
        "provenance_schema_version": PROVENANCE_SCHEMA_VERSION,
        "tradeproof_version": resolved_version,
        "git": {
            "commit": resolved_git.get("commit"),
            "branch": resolved_git.get("branch"),
            "dirty": resolved_git.get("dirty"),
        },
        "seed": {
            # Configured vs actually-used, the same distinction already
            # established for the trace manifest (see run_seed.py):
            # configured_seed is whatever the user passed (0 = auto);
            # base_used is what this run actually derived every call's
            # seed from.
            "configured": configured_seed,
            "base_used": base_seed,
        },
        "models": {
            "generation": _role_metadata(config, "generation"),
            "review": _role_metadata(config, "review"),
            "validator": {
                **_role_metadata(config, "validator"),
                # validator_llm may still be configured while validation is
                # toggled off -- both facts are kept, never collapsed into
                # one (see task: "distinguir configurado vs usado").
                "validation_enabled": bool(config.get("validation_enabled")),
            },
        },
        "execution": {
            "language": config.get("output_language", "English"),
            "depth": config.get("max_risk_discuss_rounds"),
            "selected_analysts": list(selected_analysts),
            # Decision Memory mode this run actually used (see fork README:
            # "Decision memory isolation") -- so a run's provenance always
            # says, on its own, whether it was stateful with respect to
            # past/future sessions.
            "decision_memory_enabled": bool(config.get("decision_memory_enabled", True)),
            "decision_memory_root": config.get("decision_memory_root"),
        },
        "prompts": {
            "hashes": resolved_prompt_hashes,
        },
        "trace": {
            "enabled": bool(trace_llm),
            "manifest_path": trace_manifest_path,
        },
    }
