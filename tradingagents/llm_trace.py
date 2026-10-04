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


def traced_invoke(llm, prompt, *, attempt: str = "normal"):
    """Call ``llm.invoke(prompt)``, tagging the active trace (if any) with
    ``attempt`` so the callback can tell a structured call apart from its
    free-text fallback. A thin, always-safe indirection: with no tracer
    active this is exactly ``llm.invoke(prompt)``."""
    token = _current_attempt.set(attempt)
    try:
        return llm.invoke(prompt)
    finally:
        _current_attempt.reset(token)


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


def _git_info() -> dict[str, str | None]:
    """Best-effort git commit/branch via a plain subprocess call -- no new
    dependency, and any failure (not a git repo, git not on PATH, timeout)
    silently yields nulls rather than breaking the manifest."""
    info: dict[str, str | None] = {"commit": None, "branch": None}
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
    return info


def _package_version() -> str | None:
    try:
        from importlib.metadata import version
        return version("tradingagents")
    except Exception:
        try:
            import tomllib
            with open(Path(__file__).resolve().parent.parent / "pyproject.toml", "rb") as f:
                return tomllib.load(f)["project"]["version"]
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


def snapshot_prompts(debug_dir: Path, on_warn) -> dict[str, str]:
    """Copy every built-in prompt file into ``debug_dir/prompts/`` and return
    ``{relative_path: sha256_hex}``. The prompt-file *content* is what is
    hashed/snapshotted here; the actual rendered prompt sent to the model
    (placeholders filled in) is saved separately, per call, in
    ``calls/*/request.json`` -- the snapshot alone cannot answer "what did
    the model see," only "what template version was in play."
    """
    hashes: dict[str, str] = {}
    root = resources.files("tradingagents.prompts")
    for rel in list_prompt_files():
        try:
            content = root.joinpath(rel).read_text(encoding="utf-8")
        except OSError as exc:
            on_warn(debug_dir / "prompts" / rel, exc)
            continue
        hashes[rel] = hashlib.sha256(content.encode("utf-8")).hexdigest()
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
) -> None:
    manifest = {
        "trace_format_version": TRACE_FORMAT_VERSION,
        "ticker": ticker,
        "analysis_date": analysis_date,
        "run_started_at": datetime.now(timezone.utc).isoformat(),
        "depth": depth,
        "analysts": list(analysts),
        "language": language,
        "provider": provider,
        "effective_deep_model": effective_deep_model,
        "effective_quick_model": effective_quick_model,
        "model_parameters": {
            "temperature": config.get("temperature"),
            "max_tokens": config.get("max_tokens"),
            "top_p": config.get("top_p"),
            "llm_max_retries": config.get("llm_max_retries"),
        },
        "package_version": _package_version(),
        "git": _git_info(),
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

    # -- aggregate summary ----------------------------------------------------

    def write_summary(
        self, *, stats: dict, elapsed_seconds: float, deep_model: str, quick_model: str
    ) -> None:
        summary = {
            "llm_calls": stats.get("llm_calls"),
            "structured_calls": self.structured_calls,
            "structured_failures": self.structured_failures,
            "fallback_calls": self.fallback_calls,
            "input_tokens": stats.get("tokens_in"),
            "output_tokens": stats.get("tokens_out"),
            "elapsed_seconds": round(elapsed_seconds, 3),
            "deep_model": deep_model,
            "quick_model": quick_model,
        }
        _safe_write_json(self.debug_dir / "trace_summary.json", summary, self._warn)
