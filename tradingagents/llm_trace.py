"""Opt-in LLM trace/debug mode: a complete, auditable record of every LLM
call a run makes -- the exact rendered request, the raw response, the parsed
structured result (or failure), and timing/token metadata -- written under
``<output_dir>/debug/``.

Off by default; when off, this module is not imported into the hot path at
all beyond a few no-op contextvar reads, and normal runs are unaffected.

Design (see fork README: "LLM trace mode"): every agent in this graph
already funnels its LLM calls through one of a handful of shared call sites
(``structured.py``, ``thesis_flow.py``, ``stance.py``,
``research_verifier.py``) or, for the tool-calling analysts, a plain
``chain.invoke(...)``. Rather than writing trace code into every agent file,
a single ``LLMTraceCallbackHandler`` is attached the same way
``StatsCallbackHandler`` already is (at LLM-client construction, via
``callbacks=[...]``), and two small context variables tell it which node is
currently running (set once per node, in ``graph/setup.py`` -- no change to
any individual agent file) and which "attempt" a call represents (structured
vs. fallback, set by the handful of shared helpers via ``traced_invoke``).
Tool-call requests/results are not captured via a separate hook: they are
already part of the message history of the *next* call on that node, which
``request.json`` records in full.
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import logging
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

logger = logging.getLogger(__name__)

TRACE_FORMAT_VERSION = 1

# Set once per graph node (see graph/setup.py's node-registration wrapper) --
# every LLM call made while that node's function is on the call stack is
# tagged with it automatically.
_current_stage: contextvars.ContextVar[str] = contextvars.ContextVar(
    "llm_trace_stage", default="unknown"
)
# Set by the shared invoke helpers immediately around one specific call, to
# distinguish a structured attempt from its free-text fallback within the
# same node.
_current_attempt: contextvars.ContextVar[str] = contextvars.ContextVar(
    "llm_trace_attempt", default="normal"
)
# Set by the shared invoke helpers for the duration of one call, purely for
# observability (see fork README: "Model roles: Generation vs Review") --
# neither affects routing or seed derivation, which already get their own
# values passed explicitly where needed.
_current_llm_role: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "llm_trace_llm_role", default=None
)
_current_call_identity: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "llm_trace_call_identity", default=None
)
# The directory of the most recently completed call, so the application layer
# (which knows whether structured parsing succeeded) can attach
# response_parsed.json / a parse-failure note without needing the LangChain
# run_id the callback machinery keeps internally.
_last_call_dir: contextvars.ContextVar[Path | None] = contextvars.ContextVar(
    "llm_trace_last_call_dir", default=None
)
# The single active tracer for this run (None when trace mode is off), so
# helpers can cheaply no-op without importing/constructing anything.
_active_tracer: contextvars.ContextVar[LLMTraceCallbackHandler | None] = contextvars.ContextVar(
    "llm_trace_active_tracer", default=None
)


@dataclass(frozen=True)
class SeedContext:
    """Best-effort deterministic replay context for the current run (see
    ``tradingagents/run_seed.py``). ``provider_supports_seed`` is resolved
    once, from the run's configured provider, not re-checked per call."""

    base_seed: int
    ticker: str
    provider_supports_seed: bool


# The active run's seed context (None when seeding is off / not applicable),
# set once per run exactly like ``_active_tracer`` -- and, like it, must be
# explicitly reset to None at the start of every run (including untraced,
# unseeded ones), since a batch processes tickers sequentially in the same
# thread/process and a context set for ticker 1 would otherwise leak into
# ticker 2 (see ``set_active_seed_context``).
_active_seed_context: contextvars.ContextVar[SeedContext | None] = contextvars.ContextVar(
    "llm_trace_active_seed_context", default=None
)


def set_active_seed_context(context: SeedContext | None) -> None:
    """Called once per run by ``execute_analysis`` (with ``None`` when no
    base seed is configured for this run), mirroring ``set_active_tracer``."""
    _active_seed_context.set(context)


def traced_node(name: str, fn):
    """Wrap one graph node so every LLM call made during its execution is
    tagged with ``name`` -- the only touch point needed in graph/setup.py;
    individual agent/node files are unchanged."""

    def wrapped(state):
        token = _current_stage.set(name)
        try:
            return fn(state)
        finally:
            _current_stage.reset(token)

    return wrapped


def traced_invoke(
    llm, prompt, *, attempt: str = "normal",
    call_identity: str | None = None, llm_role: str | None = None,
):
    """Call ``llm.invoke(prompt)``, tagging the active trace (if any) with
    ``attempt`` so the callback can tell a structured call apart from its
    free-text fallback. A thin, always-safe indirection: with no tracer
    active this is exactly ``llm.invoke(prompt)``.

    When a seed context is active (see ``SeedContext``) and the run's
    provider is seed-capable, also derives this call's best-effort
    reproducible seed from ``(base_seed, ticker, current stage, call
    identity)`` and passes it as the ``seed`` kwarg on this one invocation --
    never by reconstructing the client, so every other call on the same
    client keeps its own independently-derived seed. ``call_identity``
    distinguishes logically different calls that would otherwise share the
    same ``attempt`` (e.g. "review_1" vs "review_2"); callers that don't need
    that distinction can omit it and ``attempt`` alone is used.

    ``attempt`` is always folded into the derivation key alongside
    ``call_identity``, not replaced by it: a structured call and its
    free-text/JSON fallback are logically different attempts even when a
    caller passes the same ``call_identity`` for both (e.g. the Research
    Verifier's "pass_1" structured attempt and "pass_1" JSON fallback), and
    must never silently collide on the same derived seed.

    ``llm_role`` ("generation"/"review", see fork README: "Model roles:
    Generation vs Review") and ``call_identity`` are also recorded on the
    trace (when one is active) as this call's ``llm_role``/``call_identity``
    fields in ``request.json``, purely for observability -- neither affects
    routing (the caller already picked which LLM object to pass in) or seed
    derivation (unchanged, driven by ``attempt``/``call_identity`` as before).
    """
    attempt_token = _current_attempt.set(attempt)
    role_token = _current_llm_role.set(llm_role)
    identity_token = _current_call_identity.set(call_identity)
    try:
        seed_ctx = _active_seed_context.get()
        if seed_ctx is not None and seed_ctx.provider_supports_seed:
            from tradingagents.run_seed import derive_call_seed

            identity = f"{attempt}:{call_identity}" if call_identity else attempt
            call_seed = derive_call_seed(
                seed_ctx.base_seed,
                seed_ctx.ticker,
                _current_stage.get(),
                identity,
            )
            return llm.invoke(prompt, seed=call_seed)
        return llm.invoke(prompt)
    finally:
        _current_attempt.reset(attempt_token)
        _current_llm_role.reset(role_token)
        _current_call_identity.reset(identity_token)


def set_active_tracer(tracer: LLMTraceCallbackHandler | None) -> None:
    """Called once by ``execute_analysis`` right after constructing the
    handler (or with ``None`` when trace mode is off), so
    ``record_structured_parse_result`` can cheaply no-op without depending on
    the handler having already seen its first callback invocation."""
    _active_tracer.set(tracer)


def record_structured_parse_result(
    success: bool, parsed: dict | None = None, error: str | None = None
) -> None:
    """Called by the shared structured-output helpers immediately after a
    structured call returns (success) or raises (failure) -- the callback
    alone only sees the raw LLM output, not whether Pydantic validation of it
    succeeded. No-op when trace mode is off or there is no pending call.
    """
    tracer = _active_tracer.get()
    if tracer is None:
        return
    call_dir = _last_call_dir.get()
    if call_dir is None:
        return
    tracer.record_parse_result(call_dir, success=success, parsed=parsed, error=error)


def record_validator_debug(fields: dict) -> None:
    """Attach small, named debug fields to the last completed call (see
    ``LLMTraceCallbackHandler.record_debug_fields``) -- used by the Validator
    to record which parse path succeeded (structured/json_fallback/
    limited_recovery) and which LLM findings were dropped by the
    deterministic post-filter, and why. No-op when trace mode is off or
    there is no pending call, exactly like ``record_structured_parse_result``.
    """
    tracer = _active_tracer.get()
    if tracer is None:
        return
    call_dir = _last_call_dir.get()
    if call_dir is None:
        return
    tracer.record_debug_fields(call_dir, fields)


# ---------------------------------------------------------------------------
# Secret redaction
# ---------------------------------------------------------------------------

# "token(?!s)" deliberately excludes the plural "tokens" so legitimate,
# explicitly-requested metric keys (input_tokens/output_tokens/total_tokens)
# are never redacted -- only a credential-shaped singular "token"/"api_token"
# matches (see fork README: redaction must not catch ordinary token counts).
_REDACT_KEY_RE = re.compile(
    r"(api[_-]?key|apikey|authorization|bearer|secret|password|token(?!s))", re.IGNORECASE
)
_REDACTED = "***REDACTED***"


def redact(obj: Any) -> Any:
    """Recursively redact dict keys that look like credentials.

    Applied only to configuration/metadata structures that get serialized
    wholesale (e.g. generation parameters, the run manifest's config
    section) -- never to natural-language prompt/response text, so an
    ordinary sentence that happens to contain the word "token" is untouched.
    """
    if isinstance(obj, dict):
        return {
            k: (_REDACTED if isinstance(k, str) and _REDACT_KEY_RE.search(k) else redact(v))
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [redact(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# Message / response serialization
# ---------------------------------------------------------------------------


def _serialize_message(msg: Any) -> dict:
    role = getattr(msg, "type", None) or msg.__class__.__name__
    data: dict[str, Any] = {"role": role, "content": getattr(msg, "content", None)}
    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        data["tool_calls"] = [dict(tc) for tc in tool_calls]
    name = getattr(msg, "name", None)
    if name:
        data["name"] = name
    # Only ToolMessage carries this -- needed to correlate a tool result back
    # to the specific tool_call it answers (see fork README: "Benchmark
    # replay"). A trace captured before this field existed has no value
    # here; a replay consumer falls back to positional inference in that case.
    tool_call_id = getattr(msg, "tool_call_id", None)
    if tool_call_id:
        data["tool_call_id"] = tool_call_id
    return data


def _serialize_messages(messages) -> list[dict]:
    flat = []
    for batch in messages or []:
        flat.extend(batch)
    return [_serialize_message(m) for m in flat]


# ---------------------------------------------------------------------------
# Safe, resilient file writing -- a trace failure must never fail the analysis
# ---------------------------------------------------------------------------


def _safe_write_text(path: Path, text: str, on_warn) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    except OSError as exc:
        on_warn(path, exc)


def _safe_write_json(path: Path, value: Any, on_warn) -> None:
    try:
        text = json.dumps(redact(value), indent=2, ensure_ascii=False, default=str)
    except (TypeError, ValueError) as exc:
        on_warn(path, exc)
        return
    _safe_write_text(path, text, on_warn)


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------


def _git_info() -> dict[str, str | bool | None]:
    """Best-effort git commit/branch/dirty via a plain subprocess call -- no
    new dependency, and any failure (not a git repo, git not on PATH,
    timeout) silently yields nulls rather than breaking the manifest.
    ``dirty`` is True when ``git status --porcelain`` reports any output
    (uncommitted/untracked changes), None when it could not be determined --
    never guessed."""
    info: dict[str, str | bool | None] = {"commit": None, "branch": None, "dirty": None}
    for key, args in (
        ("commit", ["git", "rev-parse", "HEAD"]),
        ("branch", ["git", "rev-parse", "--abbrev-ref", "HEAD"]),
    ):
        try:
            result = subprocess.run(
                args, capture_output=True, text=True, timeout=2, check=True,
            )
            info[key] = result.stdout.strip() or None
        except Exception:
            pass
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, timeout=2, check=True,
        )
        info["dirty"] = bool(result.stdout.strip())
    except Exception:
        pass
    return info


def _package_version() -> str | None:
    """Best-effort TradeProof version for the code actually running in this
    checkout -- the single source both run_manifest and run_provenance read
    (see run_provenance.py), so they can never disagree.

    This checkout's own ``pyproject.toml`` wins over installed distribution
    metadata: ``importlib.metadata.version()`` reads whatever ``.egg-info``/
    ``.dist-info`` happens to be on ``sys.path``, which goes stale the moment
    ``pyproject.toml`` is bumped without a reinstall (an editable install's
    egg-info is only regenerated by ``pip install -e .``, not by editing the
    file) -- exactly the gap that let a real run_provenance validation read
    a stale installed version instead of this checkout's own. Installed
    metadata is still a reasonable fallback when this
    isn't a source checkout at all (e.g. a built/installed package with no
    pyproject.toml alongside it). Never fails the run and never fabricates a
    version: both sources failing yields ``None``.
    """
    try:
        import tomllib
        with open(Path(__file__).resolve().parent.parent / "pyproject.toml", "rb") as f:
            project_version = tomllib.load(f)["project"]["version"]
        if project_version:
            return project_version
    except Exception:
        pass
    try:
        from importlib.metadata import version
        return version("tradingagents")
    except Exception:
        return None


def list_prompt_files() -> list[str]:
    """Every built-in prompt file packaged under ``tradingagents/prompts/``,
    as relative POSIX paths -- the exact set "actually available to this
    run" (see fork README: "LLM trace mode -- prompt snapshot")."""
    root = resources.files("tradingagents.prompts")
    found: list[str] = []

    def _walk(node, prefix: str) -> None:
        for entry in node.iterdir():
            rel = f"{prefix}{entry.name}"
            if entry.is_dir():
                _walk(entry, rel + "/")
            elif entry.name.endswith(".txt"):
                found.append(rel)

    _walk(root, "")
    return sorted(found)


def hash_prompt_files() -> dict[str, str]:
    """``{relative_path: sha256_hex}`` for every built-in prompt file --
    pure, no filesystem writes. The hashing half of ``snapshot_prompts``,
    split out so run provenance (always on, see ``run_provenance.py``) and
    the trace-mode prompt snapshot below (opt-in, which also copies each
    file to ``debug_dir/prompts/``) share one implementation that can never
    diverge -- tracing is a consumer of this, never a second one.
    """
    hashes: dict[str, str] = {}
    root = resources.files("tradingagents.prompts")
    for rel in list_prompt_files():
        try:
            content = root.joinpath(rel).read_text(encoding="utf-8")
        except OSError:
            continue
        hashes[rel] = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return hashes


def snapshot_prompts(debug_dir: Path, on_warn) -> dict[str, str]:
    """Copy every built-in prompt file into ``debug_dir/prompts/`` and return
    ``{relative_path: sha256_hex}`` (from ``hash_prompt_files()``, never
    recomputed independently). The prompt-file *content* is what is
    hashed/snapshotted here; the actual rendered prompt sent to the model
    (placeholders filled in) is saved separately, per call, in
    ``calls/*/request.json`` -- the snapshot alone cannot answer "what did
    the model see," only "what template version was in play."
    """
    hashes = hash_prompt_files()
    root = resources.files("tradingagents.prompts")
    for rel in hashes:
        try:
            content = root.joinpath(rel).read_text(encoding="utf-8")
        except OSError as exc:
            on_warn(debug_dir / "prompts" / rel, exc)
            continue
        _safe_write_text(debug_dir / "prompts" / rel, content, on_warn)
    return hashes


def write_run_manifest(
    debug_dir: Path,
    *,
    ticker: str,
    analysis_date: str,
    depth: str | int,
    analysts: list[str],
    language: str,
    provider: str,
    effective_deep_model: str,
    effective_quick_model: str,
    config: dict,
    prompt_hashes: dict[str, str],
    on_warn,
    configured_seed: int | None = None,
    base_seed: int | None = None,
    seed_provider_supported: bool | None = None,
    git_info: dict[str, str | bool | None] | None = None,
    package_version: str | None = None,
) -> None:
    """``git_info``/``package_version`` may be passed in already-computed
    (see ``run_provenance.build_run_provenance``, called once per run
    regardless of trace mode) so git is never invoked twice for the same
    run; omitting either computes it here directly, unchanged from before."""
    manifest = {
        "trace_format_version": TRACE_FORMAT_VERSION,
        "ticker": ticker,
        "analysis_date": analysis_date,
        "run_started_at": datetime.now(timezone.utc).isoformat(),
        "depth": depth,
        "analysts": list(analysts),
        "language": language,
        "provider": provider,
        # Legacy field names (deprecated, see fork README "Legacy naming":
        # quick -> generation, deep -> review). generation_model/review_model
        # below are the canonical fields; these are kept for any existing
        # manifest consumer that already reads them.
        "effective_deep_model": effective_deep_model,
        "effective_quick_model": effective_quick_model,
        "generation_model": config.get("generation_llm"),
        "review_model": config.get("review_llm"),
        "model_parameters": {
            "temperature": config.get("temperature"),
            "max_tokens": config.get("max_tokens"),
            "top_p": config.get("top_p"),
            "llm_max_retries": config.get("llm_max_retries"),
        },
        # Best-effort deterministic replay (see tradingagents/run_seed.py):
        # configured_seed is whatever the user passed (0 = auto-generate);
        # base_seed is the actual value used to derive every call's seed this
        # run -- the one to reuse with --seed to attempt reproducing this run.
        # seed_provider_supported records whether this run's provider was
        # seed-capable at all; per-call effective seeds (when applied) are
        # already visible in each call's own request.json generation_params.
        "configured_seed": configured_seed,
        "base_seed": base_seed,
        "seed_provider_supported": seed_provider_supported,
        # Generation vs Review reasoning effort (see fork README): whatever
        # was *configured* for this run, None meaning "no override was sent,
        # the backend/model's own default applied" -- never a guessed
        # effective default. The actual per-call value, when one was sent, is
        # already visible in each call's own request.json generation_params.
        # quick_reasoning_effort/deep_reasoning_effort are the deprecated
        # legacy field names, kept for any existing manifest consumer.
        "generation_reasoning_effort": config.get("generation_reasoning_effort"),
        "review_reasoning_effort": config.get("review_reasoning_effort"),
        "quick_reasoning_effort": config.get("quick_reasoning_effort"),
        "deep_reasoning_effort": config.get("deep_reasoning_effort"),
        "package_version": package_version if package_version is not None else _package_version(),
        "git": git_info if git_info is not None else _git_info(),
        "prompt_file_hashes": prompt_hashes,
        "python_version": sys.version.split()[0],
    }
    _safe_write_json(debug_dir / "run_manifest.json", manifest, on_warn)


# ---------------------------------------------------------------------------
# The callback handler
# ---------------------------------------------------------------------------

_SAFE_NAME_RE = re.compile(r"[^a-zA-Z0-9_]+")


def _safe_component(text: str) -> str:
    return _SAFE_NAME_RE.sub("_", text).strip("_") or "unknown"


@dataclass
class _PendingCall:
    number: int
    stage: str
    attempt: str
    start_monotonic: float
    start_iso: str
    dir: Path


class LLMTraceCallbackHandler(BaseCallbackHandler):
    """Writes one ``calls/NNN_<stage>[_<attempt>]/`` directory per LLM call.

    Attached exactly like ``StatsCallbackHandler`` -- passed in the
    ``callbacks`` list at LLM-client construction -- so it sees every call
    any agent makes without any agent file needing to know it exists.
    """

    def __init__(self, debug_dir: Path):
        super().__init__()
        self.debug_dir = Path(debug_dir)
        self.calls_dir = self.debug_dir / "calls"
        self._lock = threading.Lock()
        self._counter = 0
        self._pending: dict[str, _PendingCall] = {}
        self._warned = False
        self.structured_calls = 0
        self.structured_failures = 0
        self.fallback_calls = 0

    # -- shared failure handling ------------------------------------------------

    def _warn(self, path: Path, exc: Exception) -> None:
        if not self._warned:
            print(
                f"[WARN] LLM trace: failed to write a trace artifact at {path} "
                f"({exc}); continuing analysis without it.",
                flush=True,
            )
            self._warned = True
        logger.debug("LLM trace write failure for %s", path, exc_info=True)

    # -- directory / numbering ----------------------------------------------

    def _next_call_dir(self, stage: str, attempt: str) -> tuple[int, Path]:
        with self._lock:
            self._counter += 1
            number = self._counter
        suffix = f"_{_safe_component(attempt)}" if attempt and attempt != "normal" else ""
        return number, self.calls_dir / f"{number:03d}_{_safe_component(stage)}{suffix}"

    # -- LangChain callback hooks --------------------------------------------

    def on_chat_model_start(
        self, serialized, messages, *, run_id, metadata=None, invocation_params=None, **kwargs
    ) -> None:
        stage = _current_stage.get()
        attempt = _current_attempt.get()
        number, call_dir = self._next_call_dir(stage, attempt)
        pending = _PendingCall(
            number=number, stage=stage, attempt=attempt,
            start_monotonic=time.monotonic(),
            start_iso=datetime.now(timezone.utc).isoformat(),
            dir=call_dir,
        )
        with self._lock:
            self._pending[str(run_id)] = pending
        _last_call_dir.set(call_dir)

        if attempt == "structured":
            self.structured_calls += 1
        elif attempt != "normal":
            self.fallback_calls += 1

        invocation_params = invocation_params or {}
        request = {
            "call_number": number,
            "stage": stage,
            "attempt": attempt,
            # "generation"/"review" (see fork README: "Model roles:
            # Generation vs Review"), or None for a call that didn't go
            # through a role-aware helper (e.g. Reflection's raw .invoke()).
            "llm_role": _current_llm_role.get(),
            "call_identity": _current_call_identity.get(),
            "model": invocation_params.get("model") or (serialized or {}).get("name"),
            "provider": invocation_params.get("_type") or (serialized or {}).get("id"),
            "messages": _serialize_messages(messages),
            "tools": invocation_params.get("tools"),
            "response_format": invocation_params.get("response_format"),
            "generation_params": {
                k: v for k, v in invocation_params.items()
                if k not in ("tools", "response_format", "model")
            },
        }
        _safe_write_json(call_dir / "request.json", request, self._warn)

    def on_llm_end(self, response, *, run_id, **kwargs) -> None:
        with self._lock:
            pending = self._pending.pop(str(run_id), None)
        if pending is None:
            return
        elapsed = time.monotonic() - pending.start_monotonic

        text, tool_calls, usage, finish_reason = None, None, None, None
        try:
            generation = response.generations[0][0]
            message = getattr(generation, "message", None)
            if message is not None:
                text = getattr(message, "content", None)
                tool_calls = getattr(message, "tool_calls", None) or None
                usage = getattr(message, "usage_metadata", None)
                finish_reason = (getattr(message, "response_metadata", None) or {}).get(
                    "finish_reason"
                )
            else:
                text = getattr(generation, "text", None)
        except (IndexError, AttributeError, TypeError):
            pass

        raw = {
            "content": text,
            "tool_calls": [dict(tc) for tc in tool_calls] if tool_calls else None,
            "finish_reason": finish_reason,
        }
        _safe_write_text(
            pending.dir / "response_raw.txt",
            json.dumps(redact(raw), indent=2, ensure_ascii=False, default=str),
            self._warn,
        )

        metadata_doc = {
            "call_number": pending.number,
            "stage": pending.stage,
            "attempt": pending.attempt,
            "start": pending.start_iso,
            "elapsed_seconds": round(elapsed, 3),
            "input_tokens": (usage or {}).get("input_tokens"),
            "output_tokens": (usage or {}).get("output_tokens"),
            "total_tokens": (usage or {}).get("total_tokens"),
            "success": True,
            "structured_requested": pending.attempt == "structured",
            "fallback_triggered": pending.attempt not in ("normal", "structured"),
            "tool_call_count": len(tool_calls) if tool_calls else 0,
            "finish_reason": finish_reason,
        }
        _safe_write_json(pending.dir / "metadata.json", metadata_doc, self._warn)

    def on_llm_error(self, error, *, run_id, **kwargs) -> None:
        with self._lock:
            pending = self._pending.pop(str(run_id), None)
        if pending is None:
            return
        elapsed = time.monotonic() - pending.start_monotonic
        metadata_doc = {
            "call_number": pending.number,
            "stage": pending.stage,
            "attempt": pending.attempt,
            "start": pending.start_iso,
            "elapsed_seconds": round(elapsed, 3),
            "success": False,
            "error": str(error),
            "structured_requested": pending.attempt == "structured",
        }
        _safe_write_json(pending.dir / "metadata.json", metadata_doc, self._warn)

    # -- application-layer hook (structured parse outcome) -------------------

    def record_parse_result(
        self, call_dir: Path, *, success: bool, parsed: dict | None, error: str | None
    ) -> None:
        if success and parsed is not None:
            _safe_write_json(call_dir / "response_parsed.json", parsed, self._warn)
        elif not success:
            self.structured_failures += 1
            # Never fabricate a parsed result: record the failure in metadata
            # instead of writing an invalid/partial response_parsed.json.
            meta_path = call_dir / "metadata.json"
            try:
                existing = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                existing = {}
            existing["structured_parse_status"] = "failed"
            existing["structured_parse_error"] = (error or "")[:500]
            _safe_write_json(meta_path, existing, self._warn)

    def record_debug_fields(self, call_dir: Path, fields: dict) -> None:
        """Merge small, named debug fields into a call's ``metadata.json`` --
        generic counterpart to ``record_parse_result`` for callers (today:
        the Validator, see ``agents/validator.py``) that need a few extra
        named booleans/lists on a specific call without a bespoke schema."""
        meta_path = call_dir / "metadata.json"
        try:
            existing = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = {}
        existing.update(fields)
        _safe_write_json(meta_path, existing, self._warn)

    # -- aggregate summary ----------------------------------------------------

    def write_summary(
        self, *, stats: dict, elapsed_seconds: float, generation_model: str, review_model: str
    ) -> None:
        summary = {
            "llm_calls": stats.get("llm_calls"),
            "structured_calls": self.structured_calls,
            "structured_failures": self.structured_failures,
            "fallback_calls": self.fallback_calls,
            "input_tokens": stats.get("tokens_in"),
            "output_tokens": stats.get("tokens_out"),
            "elapsed_seconds": round(elapsed_seconds, 3),
            "generation_model": generation_model,
            "review_model": review_model,
            # Legacy field names (deprecated, see fork README "Legacy
            # naming": quick -> generation, deep -> review), kept for any
            # existing summary consumer.
            "deep_model": review_model,
            "quick_model": generation_model,
        }
        _safe_write_json(self.debug_dir / "trace_summary.json", summary, self._warn)
