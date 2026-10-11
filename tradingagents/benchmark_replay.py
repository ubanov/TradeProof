"""Benchmark replay: a controlled, read-only model comparison that reuses
the frozen inputs already captured by a ``--trace-llm`` run.

This is NOT a second pipeline. It replays the EXACT serialized request of
one already-completed LLM call -- same messages, same structured-output
contract, same evidence -- against a different model/reasoning_effort,
swapping nothing else. It never executes tools, never writes to the source
run, never regenerates a prompt from ``AgentState``, and never runs any
downstream stage (see fork README: "Benchmark replay").

Supported stages (first scope, deliberately small -- see ``SUPPORTED_STAGES``):
Market/Sentiment/News/Fundamentals Analyst, Research Manager. Each stage's
"source call" is its FINAL synthesis call -- the one that already has every
tool result folded into the message history, identified from the trace's
own recorded metadata (never by re-running anything):

- a structured-output call (``structured_requested: true`` in the source
  call's metadata) is final by construction -- its one "tool call" IS the
  schema extraction, not a real tool-loop round;
- a free-text call is final when it made no tool calls and the model
  stopped on its own (``tool_call_count: 0`` and ``finish_reason: "stop"``).

If a stage's last recorded call matches neither shape, this refuses to
guess -- see ``ReplayError``.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from tradingagents.agents.schemas import (
    ResearchPlan,
    SentimentReport,
    render_research_plan,
    render_sentiment_report,
)
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.llm_clients import build_llm_kwargs, create_llm_client
from tradingagents.run_seed import SEED_CAPABLE_PROVIDERS, derive_call_seed

# Stage key (CLI-facing) -> the exact "stage" string the trace records.
_STAGE_DIR_NAMES: dict[str, str] = {
    "market": "Market Analyst",
    "sentiment": "Sentiment Analyst",
    "news": "News Analyst",
    "fundamentals": "Fundamentals Analyst",
    "research_manager": "Research Manager",
}
SUPPORTED_STAGES: tuple[str, ...] = tuple(_STAGE_DIR_NAMES)

# A stage's structured schema + render function, when it has one -- None
# means the stage is always free-text (Market/News/Fundamentals never bind
# a schema; confirmed by reading their node factories). Reused as-is, never
# a second definition of either schema.
_STAGE_SCHEMAS: dict[str, tuple[type, Any] | None] = {
    "market": None,
    "sentiment": (SentimentReport, render_sentiment_report),
    "news": None,
    "fundamentals": None,
    "research_manager": (ResearchPlan, render_research_plan),
}


class ReplayError(ValueError):
    """Raised for anything that would require guessing instead of replaying
    the exact recorded request -- an unsupported stage, missing trace
    artifacts, or a stage whose trace has no separable final-synthesis
    call. Never silently falls back to an approximation."""


# ---------------------------------------------------------------------------
# Model spec parsing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    """One ``--model`` argument. ``raw`` is kept verbatim for seed
    derivation and output-path naming, so two specs that differ only in
    reasoning effort never collide."""

    raw: str
    model: str
    reasoning_effort: str | None = None

    def safe_dir_name(self) -> str:
        return _safe_path_component(self.raw)


def parse_model_spec(spec: str) -> ModelSpec:
    """Parse ``"<model>"`` or ``"<model>@<reasoning_effort>"``.

    Model names in this project routinely contain ``:`` (e.g.
    ``qwen3.8-flash-next:125b-a6b-q4_K_M``), so ``:`` is never used as the
    reasoning-effort delimiter -- only ``@`` is, split from the RIGHT so a
    model name could even (implausibly) contain ``@`` itself without
    breaking this, as long as the actual reasoning effort doesn't.
    """
    spec = (spec or "").strip()
    if not spec:
        raise ReplayError("Empty --model spec.")
    if "@" in spec:
        model, _, effort = spec.rpartition("@")
        model = model.strip()
        effort = effort.strip()
        if not model:
            raise ReplayError(f"Model spec {spec!r} has no model name before '@'.")
        if not effort:
            raise ReplayError(f"Model spec {spec!r} has an empty reasoning effort after '@'.")
        return ModelSpec(raw=spec, model=model, reasoning_effort=effort)
    return ModelSpec(raw=spec, model=spec, reasoning_effort=None)


_UNSAFE_PATH_CHARS_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_path_component(value: str) -> str:
    """Filesystem-safe directory name for an arbitrary model spec (which may
    contain ``:``, ``@``, ``/``) -- collapses any run of disallowed
    characters into a single ``-``. Distinct from ``dataflows.symbols.
    safe_ticker_component``, which VALIDATES/REJECTS a ticker rather than
    sanitizing a model spec into something always safe to use."""
    collapsed = _UNSAFE_PATH_CHARS_RE.sub("-", value.strip())
    return collapsed.strip("-") or "model"


# ---------------------------------------------------------------------------
# Loading the source trace
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TraceCall:
    """One call's request, exactly as ``request.json``/``metadata.json``
    recorded it -- the single source of truth this module replays from."""

    call_number: int
    stage: str
    attempt: str
    llm_role: str | None
    call_identity: str | None
    model: str | None
    provider: str | None
    messages: list[dict]
    tools: list[dict] | None
    response_format: Any
    generation_params: dict
    metadata: dict
    call_dir: Path


def _load_call(call_dir: Path) -> TraceCall:
    request_path = call_dir / "request.json"
    if not request_path.is_file():
        raise ReplayError(f"Missing request.json under {call_dir}.")
    request = json.loads(request_path.read_text(encoding="utf-8"))
    metadata_path = call_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    return TraceCall(
        call_number=request.get("call_number", 0),
        stage=request.get("stage", ""),
        attempt=request.get("attempt", ""),
        llm_role=request.get("llm_role"),
        call_identity=request.get("call_identity"),
        model=request.get("model"),
        provider=request.get("provider"),
        messages=request.get("messages") or [],
        tools=request.get("tools"),
        response_format=request.get("response_format"),
        generation_params=request.get("generation_params") or {},
        metadata=metadata,
        call_dir=call_dir,
    )


def resolve_debug_dir(from_run: Path, ticker: str | None = None) -> tuple[Path, str]:
    """Find ``<from_run>/<ticker>/debug`` under a run directory, returning
    ``(debug_dir, ticker)``. ``from_run`` is the run's own folder (what
    ``--output-dir``/``--from-run`` pointed ``--trace-llm`` at for that run),
    not the ticker subdirectory itself."""
    from_run = Path(from_run)
    if not from_run.is_dir():
        raise ReplayError(f"--from-run directory does not exist: {from_run}")
    if ticker:
        debug_dir = from_run / ticker / "debug"
        if not debug_dir.is_dir():
            raise ReplayError(f"No debug/ trace directory found at {debug_dir}.")
        return debug_dir, ticker
    candidates = sorted(from_run.glob("*/debug/run_manifest.json"))
    if not candidates:
        raise ReplayError(
            f"No --trace-llm artifacts found under {from_run} "
            "(expected <ticker>/debug/run_manifest.json). Was this run executed with --trace-llm?"
        )
    if len(candidates) > 1:
        found = ", ".join(c.parent.parent.name for c in candidates)
        raise ReplayError(
            f"--from-run {from_run} contains more than one ticker's trace ({found}); "
            "pass --ticker to disambiguate."
        )
    debug_dir = candidates[0].parent
    return debug_dir, debug_dir.parent.name


def _load_run_manifest(debug_dir: Path) -> dict:
    manifest_path = debug_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise ReplayError(f"Missing run_manifest.json under {debug_dir}.")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _is_final_synthesis_call(call: TraceCall) -> bool:
    """See module docstring for the two shapes this recognizes."""
    if call.metadata.get("structured_requested"):
        return True
    return call.metadata.get("tool_call_count") == 0 and call.metadata.get("finish_reason") == "stop"


def find_source_call(debug_dir: Path, stage_key: str) -> TraceCall:
    """The one call to replay for ``stage_key`` -- see module docstring for
    the selection rule. Raises ``ReplayError`` (never guesses) when the
    stage is unsupported, the trace is missing, or no call in it qualifies."""
    if stage_key not in SUPPORTED_STAGES:
        raise ReplayError(
            f"Unsupported stage {stage_key!r}. Supported stages: {', '.join(SUPPORTED_STAGES)}."
        )
    calls_dir = debug_dir / "calls"
    if not calls_dir.is_dir():
        raise ReplayError(f"No trace calls found under {calls_dir}. Was this run executed with --trace-llm?")

    stage_name = _STAGE_DIR_NAMES[stage_key]
    call_dirs = sorted(d for d in calls_dir.iterdir() if d.is_dir())
    matching: list[TraceCall] = []
    for call_dir in call_dirs:
        try:
            call = _load_call(call_dir)
        except (json.JSONDecodeError, OSError):
            continue
        if call.stage != stage_name:
            continue
        # Exclude Validator's own calls (research_manager_validation /
        # _validation_repair / _validation_recheck), which share the node's
        # "Research Manager" stage tag but are not the Manager's own
        # generation output (see fork README: "Generation / Validation /
        # Review" -- Validator is explicitly out of scope for this first
        # replay pass).
        if stage_key == "research_manager" and (
            call.llm_role != "generation" or call.call_identity != "research_manager"
        ):
            continue
        matching.append(call)

    if not matching:
        raise ReplayError(
            f"No trace calls found for stage {stage_key!r} (\"{stage_name}\") under {calls_dir}. "
            "Was this analyst/stage selected for the source run?"
        )

    matching.sort(key=lambda c: c.call_number)
    qualifying = [c for c in matching if _is_final_synthesis_call(c)]
    if not qualifying:
        raise ReplayError(
            f"Stage {stage_key!r} has no call in the trace that looks like a final synthesis "
            "call (no call with structured_requested=true, or tool_call_count=0 and "
            "finish_reason='stop'). Refusing to guess which call to replay -- see fork README: "
            "\"Benchmark replay\"."
        )
    return qualifying[-1]


# ---------------------------------------------------------------------------
# Message reconstruction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReconstructedMessages:
    messages: list[BaseMessage]
    tool_call_id_source: str  # "exact" | "positional_inference" | "none"


def reconstruct_messages(raw_messages: list[dict]) -> ReconstructedMessages:
    """Rebuilds real LangChain message objects from a trace's serialized
    ``messages`` -- never a regenerated/approximated prompt. A trace
    recorded before this module's ``tool_call_id`` capture (see
    ``llm_trace.py::_serialize_message``) has no exact id on a ToolMessage;
    this falls back to matching the Nth tool message after an AI message to
    that AI message's Nth tool_call, in call order -- accurate whenever tool
    results were appended in request order, which is the only order this
    project's own tool loop ever produces them in. The result reports which
    source was used, never silently claiming "exact" when it wasn't.
    """
    messages: list[BaseMessage] = []
    pending_tool_call_ids: list[str] = []
    saw_tool_message = False
    used_positional_inference = False

    for raw in raw_messages:
        role = raw.get("role")
        content = raw.get("content") or ""
        if role == "system":
            messages.append(SystemMessage(content=content))
        elif role == "human":
            messages.append(HumanMessage(content=content))
        elif role == "ai":
            tool_calls = raw.get("tool_calls") or []
            messages.append(AIMessage(content=content, tool_calls=tool_calls))
            pending_tool_call_ids = [tc.get("id") for tc in tool_calls if tc.get("id")]
        elif role == "tool":
            saw_tool_message = True
            tool_call_id = raw.get("tool_call_id")
            if not tool_call_id:
                used_positional_inference = True
                tool_call_id = pending_tool_call_ids.pop(0) if pending_tool_call_ids else "unknown"
            messages.append(ToolMessage(content=content, tool_call_id=tool_call_id, name=raw.get("name")))
        else:
            raise ReplayError(f"Unrecognized serialized message role in trace: {role!r}")

    if not saw_tool_message:
        source = "none"
    elif used_positional_inference:
        source = "positional_inference"
    else:
        source = "exact"
    return ReconstructedMessages(messages=messages, tool_call_id_source=source)


# ---------------------------------------------------------------------------
# Tool-history normalization -- a deterministic, lossless-for-tool-results
# fallback used only when a provider does not honor tool_choice="none" and
# still attempts a real tool call during replay (real-run regression: a
# MSFT News Analyst replay against deepseek-v4.1-flash:cloud kept returning
# finish_reason="tool_calls" even with tool_choice="none" sent -- the
# provider/model simply ignored it). Collapses every AIMessage(tool_calls=
# ...) + ToolMessage(...) pair into plain System/Human/AI turns that never
# require a ``tools`` parameter to be valid, so there is nothing left for
# the model to call.
# ---------------------------------------------------------------------------


def _render_tool_calls(tool_calls: list[dict]) -> str:
    """Deterministic text restatement of an AIMessage's own recorded
    ``tool_calls`` -- never invents an argument, just restates the SAME
    structured data the trace already captured."""
    rendered = []
    for call in tool_calls:
        name = call.get("name", "?")
        args = call.get("args") or {}
        args_text = ", ".join(f"{k}={v!r}" for k, v in args.items())
        rendered.append(f"{name}({args_text})")
    return "[Tool call(s) recorded in the original trace: " + "; ".join(rendered) + "]"


@dataclass(frozen=True)
class NormalizedHistory:
    messages: list[BaseMessage]
    rounds_collapsed: int
    tool_messages_converted: int

    def detail(self) -> str:
        return (
            f"collapsed {self.rounds_collapsed} tool-call round(s) and "
            f"{self.tool_messages_converted} tool-result message(s) into plain "
            "System/Human/AI turns (no 'tools' parameter sent); every tool "
            "result's text content is preserved verbatim"
        )


def _normalize_tool_history(messages: list[BaseMessage]) -> NormalizedHistory:
    """Flattens a frozen AIMessage(tool_calls=...) + ToolMessage(...) history
    into plain turns with strict System/Human/AI alternation, so the request
    never needs a ``tools`` parameter at all -- nothing declared, nothing to
    call. Every ToolMessage's ``content`` is carried over byte-for-byte (just
    relabeled from a ``tool``-role message into a ``human``-role one with a
    clear prefix); an AIMessage with tool_calls keeps its own original text
    content (if any) and gains a deterministic restatement of what it called.
    Never drops or rewrites a tool result's actual text.
    """
    normalized: list[BaseMessage] = []
    rounds_collapsed = 0
    tool_messages_converted = 0
    pending_tool_lines: list[str] = []

    def _flush_pending() -> None:
        if pending_tool_lines:
            normalized.append(HumanMessage(content="\n\n".join(pending_tool_lines)))
            pending_tool_lines.clear()

    for message in messages:
        if isinstance(message, ToolMessage):
            label = message.name or message.tool_call_id or "tool"
            pending_tool_lines.append(f"[Tool result -- {label}]\n{message.content}")
            tool_messages_converted += 1
            continue
        _flush_pending()
        if isinstance(message, AIMessage) and message.tool_calls:
            original_text = (message.content or "").strip()
            rendering = _render_tool_calls(message.tool_calls)
            content = f"{original_text}\n{rendering}" if original_text else rendering
            normalized.append(AIMessage(content=content))
            rounds_collapsed += 1
            continue
        normalized.append(message)
    _flush_pending()

    return NormalizedHistory(
        messages=normalized, rounds_collapsed=rounds_collapsed, tool_messages_converted=tool_messages_converted,
    )


# ---------------------------------------------------------------------------
# Replay invocation -- preserves structured/free-text shape, never executes
# a real tool, always seed-aware when the provider supports it.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReplayOutcome:
    """``status`` is the ONE field that decides success/failure; the rest
    are diagnostic detail (see task: "structured fail + free-text fallback
    producing usable text is REPLAY SUCCESS with output_mode=fallback, not
    an error" -- a failed structured attempt that a successful fallback
    recovered from must never surface as ``status="error"``)."""

    structured_attempted: bool
    structured_success: bool
    fallback_triggered: bool
    fallback_success: bool
    output_mode: str  # "structured" | "fallback" | "free_text" | "error"
    status: str  # "success" | "error"
    parsed: Any
    text: str | None
    finish_reason: str | None
    usage: dict | None
    error: str | None
    replay_input_mode: str = "exact"  # "exact" | "tool_history_normalized"
    tool_history_detail: str | None = None


def _free_text_attempt(
    llm: Any, messages: list[BaseMessage], kwargs: dict[str, Any], tools: list[dict] | None,
) -> dict[str, Any]:
    """One free-text invocation attempt. Binds ``tools`` (unchanged, from
    the source call) with ``tool_choice="none"`` when present; passing
    ``tools=None`` sends no ``tools`` parameter at all. Never raises --
    any provider/model exception is captured and reported, not propagated."""
    invoke_kwargs = dict(kwargs)
    if tools:
        invoke_kwargs["tools"] = tools
        invoke_kwargs["tool_choice"] = "none"
    try:
        response = llm.invoke(messages, **invoke_kwargs)
    except Exception as exc:  # noqa: BLE001 - provider/model failure, reported not raised
        return {"error": str(exc)}
    usage = getattr(response, "usage_metadata", None)
    finish_reason = (getattr(response, "response_metadata", None) or {}).get("finish_reason")
    text = getattr(response, "content", None)
    # The model asked for a tool anyway (despite tool_choice="none", or
    # despite no tools being offered at all) -- never executed, but also
    # never silently accepted as a valid comparison output.
    reentered_tool_mode = bool(getattr(response, "tool_calls", None)) or finish_reason == "tool_calls"
    return {
        "error": None, "usage": usage, "finish_reason": finish_reason, "text": text,
        "reentered_tool_mode": reentered_tool_mode,
        "text_is_usable": bool((text or "").strip()) and not reentered_tool_mode,
    }


def _invoke_replay(
    llm: Any,
    messages: list[BaseMessage],
    schema: type | None,
    seed: int | None,
    provider: str,
    tools: list[dict] | None,
) -> ReplayOutcome:
    """Structured attempt (when the SOURCE call requested one) with a
    free-text fallback on failure -- the exact two-step shape every
    structured-capable agent in this project already uses (see
    ``agents/structured.py``), reimplemented narrowly here ONLY because this
    needs ``include_raw=True`` (for token usage / finish_reason on every
    path, success or failure) and an explicit, precomputed seed rather than
    the pipeline's own tracer-bound seed-context mechanism, which this
    read-only utility deliberately does not hook into.

    Never EXECUTES a tool, on any path -- see module docstring. For a
    stage with no structured schema (Market/News/Fundamentals Analyst)
    whose source call had real tools bound, those tools are still bound
    here too (unchanged, from ``tools``), but with ``tool_choice="none"``:
    the model's own system prompt usually still says "you have access to
    the following tools", and a tool-trained model given that text with NO
    ``tools`` parameter at all can get confused and emit an empty,
    tool-call-shaped response instead of answering (real-run regression).

    Some providers/models do not actually honor ``tool_choice="none"`` --
    a real MSFT News Analyst replay against deepseek-v4.1-flash:cloud kept
    returning ``finish_reason="tool_calls"`` even with it set (verified by
    inspecting the saved replay request: the frozen history ends on a tool
    result, ``tool_choice="none"`` was sent, and the model still re-entered
    tool mode -- a provider/model limitation, not a bug in how this module
    builds the request). When that happens -- ``tool_choice="none"`` sent
    but ignored -- this retries EXACTLY ONCE against the SAME frozen
    history with every AIMessage(tool_calls=...)/ToolMessage pair collapsed
    into plain System/Human/AI turns (see ``_normalize_tool_history``), so
    the retried request has no ``tools`` parameter at all -- nothing left
    to call. No tool is ever executed on either attempt; a tool call is
    always reported as an error, never accepted as text.
    """
    kwargs: dict[str, Any] = {}
    if seed is not None and provider.lower() in SEED_CAPABLE_PROVIDERS:
        kwargs["seed"] = seed

    structured_attempted = schema is not None
    structured_error: str | None = None

    if structured_attempted:
        try:
            structured_llm = llm.with_structured_output(schema, include_raw=True)
        except (NotImplementedError, AttributeError, TypeError) as exc:
            structured_llm = None
            structured_error = f"structured output not supported by this model/client: {exc}"
        if structured_llm is not None:
            try:
                raw_result = structured_llm.invoke(messages, **kwargs)
            except Exception as exc:  # noqa: BLE001 - provider/model failure, reported not raised
                structured_error = str(exc)
                raw_result = None
            if raw_result is not None:
                raw_message = raw_result.get("raw") if isinstance(raw_result, dict) else None
                parsing_error = raw_result.get("parsing_error") if isinstance(raw_result, dict) else None
                candidate = raw_result.get("parsed") if isinstance(raw_result, dict) else raw_result
                usage = getattr(raw_message, "usage_metadata", None) if raw_message is not None else None
                finish_reason = None
                if raw_message is not None:
                    finish_reason = (getattr(raw_message, "response_metadata", None) or {}).get("finish_reason")
                if parsing_error is None and candidate is not None:
                    return ReplayOutcome(
                        structured_attempted=True, structured_success=True, fallback_triggered=False,
                        fallback_success=False, output_mode="structured", status="success",
                        parsed=candidate, text=getattr(raw_message, "content", None),
                        finish_reason=finish_reason, usage=usage, error=None,
                    )
                structured_error = str(parsing_error) if parsing_error else "structured output returned no parsed result"

    attempt_tools = None if structured_attempted else tools
    attempt = _free_text_attempt(llm, messages, kwargs, attempt_tools)
    replay_input_mode = "exact"
    tool_history_detail: str | None = None

    needs_retry = bool(attempt_tools) and (
        attempt.get("error") is not None or attempt.get("reentered_tool_mode")
    )
    if needs_retry:
        normalized = _normalize_tool_history(messages)
        retry = _free_text_attempt(llm, normalized.messages, kwargs, None)
        replay_input_mode = "tool_history_normalized"
        tool_history_detail = normalized.detail()
        # The ORIGINAL (exact) attempt's own failure reason is preserved for
        # the error message below when the retry fails too -- "both the
        # exact and the tools-free retry failed" is more useful than only
        # the retry's own (possibly unrelated) failure.
        original_attempt = attempt
        attempt = retry
    else:
        original_attempt = None

    if attempt.get("error") is not None:
        error_message = attempt["error"]
        if original_attempt is not None:
            prior = (
                "model attempted to call a tool during replay despite tool_choice='none'"
                if original_attempt.get("reentered_tool_mode")
                else original_attempt.get("error")
            )
            error_message = f"{prior}; tools-free retry then failed: {error_message}"
        return ReplayOutcome(
            structured_attempted=structured_attempted, structured_success=False,
            fallback_triggered=structured_attempted, fallback_success=False,
            output_mode="error", status="error", parsed=None, text=None,
            finish_reason=None, usage=None, error=error_message,
            replay_input_mode=replay_input_mode, tool_history_detail=tool_history_detail,
        )

    usage = attempt.get("usage")
    finish_reason = attempt.get("finish_reason")
    text = attempt.get("text")

    if attempt["text_is_usable"]:
        output_mode = "fallback" if structured_attempted else "free_text"
        return ReplayOutcome(
            structured_attempted=structured_attempted, structured_success=False,
            fallback_triggered=structured_attempted, fallback_success=structured_attempted,
            output_mode=output_mode, status="success", parsed=None, text=text,
            finish_reason=finish_reason, usage=usage, error=None,
            replay_input_mode=replay_input_mode, tool_history_detail=tool_history_detail,
        )

    if attempt["reentered_tool_mode"]:
        error_message = "model attempted to call a tool during replay; no tool was executed"
        if original_attempt is not None:
            error_message = (
                "model attempted to call a tool during replay even after the tools-free "
                "retry; no tool was executed"
            )
    else:
        error_message = structured_error or "empty response"
        if original_attempt is not None and original_attempt.get("reentered_tool_mode"):
            error_message = (
                "model attempted to call a tool during replay despite tool_choice='none'; "
                "the tools-free retry then returned an empty response"
            )
    return ReplayOutcome(
        structured_attempted=structured_attempted, structured_success=False,
        fallback_triggered=structured_attempted, fallback_success=False,
        output_mode="error", status="error", parsed=None, text=text,
        finish_reason=finish_reason, usage=usage, error=error_message,
        replay_input_mode=replay_input_mode, tool_history_detail=tool_history_detail,
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def _ensure_fresh_output_dir(output_dir: Path) -> None:
    """Refuse to reuse an ``--output-dir`` that already holds artifacts from
    a previous benchmark run. Two invocations sharing a directory silently
    overwrite summary.json/manifest.json while leaving the earlier run's
    per-model directories behind, so the manifest no longer describes what's
    actually on disk. Merge/append is a deliberate non-goal for now -- the
    caller must pick a new, empty directory instead. Checked before any
    trace loading or LLM call, and never deletes or modifies what it finds.
    """
    if not output_dir.exists():
        return
    if any(output_dir.iterdir()):
        raise ReplayError(
            f"--output-dir already contains benchmark artifacts: {output_dir}. "
            "Reusing an existing benchmark directory is not supported yet "
            "(no merge/append) -- pass a new, empty --output-dir."
        )


def _resolve_seed(
    base_seed: int | None, source_call: TraceCall, ticker: str, stage_name: str, model_spec: ModelSpec,
) -> int | None:
    """Stable per-(stage, model) seed when ``--seed`` is given -- reuses
    ``derive_call_seed`` exactly as the pipeline itself does, never a second
    seed algorithm, with the model spec folded into the call-identity
    component so two models/efforts being compared never collide on the
    same derived seed. Without ``--seed``, reuses the SOURCE call's own
    recorded seed (same value for every model in this benchmark) when the
    trace captured one, else no seed is sent at all.
    """
    if base_seed is not None:
        identity = f"{source_call.call_identity or source_call.attempt}:{model_spec.raw}"
        return derive_call_seed(base_seed, ticker, stage_name, identity)
    return source_call.generation_params.get("seed")


def replay_one(
    *,
    source_call: TraceCall,
    model_spec: ModelSpec,
    stage_key: str,
    output_dir: Path,
    base_seed: int | None,
    provider_override: str | None,
    ticker: str,
    source_run: Path,
    source_run_provider: str | None = None,
) -> dict:
    """Replays exactly one (stage, model) pair; saves its own request/
    response/metadata; returns the compact summary entry for it. Never
    touches the source run -- everything written lives under ``output_dir``.

    ``source_run_provider`` is the SOURCE RUN's own ``llm_provider`` config
    value (``run_manifest.json["provider"]``, e.g. "ollama"/"openai") --
    deliberately NOT ``source_call.provider``, which is LangChain's own
    internal client-type string (e.g. "openai-chat") recorded for
    observability, not a value ``create_llm_client`` understands as a
    provider name.
    """
    stage_name = _STAGE_DIR_NAMES[stage_key]
    model_dir = output_dir / stage_name.replace(" ", "_") / model_spec.safe_dir_name()
    model_dir.mkdir(parents=True, exist_ok=True)

    provider = provider_override or source_run_provider or DEFAULT_CONFIG.get("llm_provider")
    seed = _resolve_seed(base_seed, source_call, ticker, stage_name, model_spec)
    schema_entry = _STAGE_SCHEMAS[stage_key]
    schema, render_fn = schema_entry if schema_entry else (None, None)

    reconstructed = reconstruct_messages(source_call.messages)

    entry: dict[str, Any] = {
        "source_run": str(source_run),
        "ticker": ticker,
        "stage": stage_key,
        "source_call_identity": source_call.call_identity,
        "source_call_number": source_call.call_number,
        "model": model_spec.model,
        "reasoning_effort": model_spec.reasoning_effort,
        "provider": provider,
        "seed": seed,
        "tool_call_id_source": reconstructed.tool_call_id_source,
    }

    _write_json(model_dir / "request.json", {
        **entry,
        "messages": source_call.messages,
        "structured_schema": schema.__name__ if schema is not None else None,
    })

    config = dict(DEFAULT_CONFIG)
    config["llm_provider"] = provider
    if model_spec.reasoning_effort:
        config["replay_reasoning_effort"] = model_spec.reasoning_effort
    llm_kwargs = build_llm_kwargs(config, role="replay")

    started = time.monotonic()
    error = None
    outcome = None
    try:
        client = create_llm_client(
            provider=provider, model=model_spec.model, base_url=DEFAULT_CONFIG.get("backend_url"), **llm_kwargs,
        )
        llm = client.get_llm()
        outcome = _invoke_replay(llm, reconstructed.messages, schema, seed, provider, source_call.tools)
    except Exception as exc:  # noqa: BLE001 - any provider/model/client failure, reported not raised
        error = str(exc)
    elapsed = time.monotonic() - started

    usage = (outcome.usage if outcome is not None else None) or {}
    response_text = outcome.text if outcome is not None else None

    # A failed structured attempt that a successful free-text fallback
    # recovered from is REPLAY SUCCESS (output_mode="fallback"), never an
    # error -- only write response.* when the overall status is "success"
    # (see task: "RESEARCH MANAGER FALLBACK VALIDO MARCADO COMO ERROR").
    status = outcome.status if outcome is not None else "error"
    if error is not None:
        status = "error"
    if status == "success":
        if outcome.parsed is not None:
            _write_json(model_dir / "response.json", outcome.parsed.model_dump(mode="json"))
            if render_fn is not None:
                (model_dir / "response.txt").write_text(render_fn(outcome.parsed), encoding="utf-8")
        elif response_text is not None:
            (model_dir / "response.txt").write_text(response_text, encoding="utf-8")

    metadata = {
        **entry,
        "elapsed_seconds": round(elapsed, 3),
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "structured_requested": schema is not None,
        "structured_attempted": outcome.structured_attempted if outcome is not None else (schema is not None),
        "structured_success": outcome.structured_success if outcome is not None else False,
        "fallback_triggered": outcome.fallback_triggered if outcome is not None else False,
        "fallback_success": outcome.fallback_success if outcome is not None else False,
        "parse_success": (outcome.structured_success or outcome.fallback_success) if outcome is not None else False,
        "output_mode": outcome.output_mode if outcome is not None else "error",
        "status": status,
        "finish_reason": outcome.finish_reason if outcome is not None else None,
        "error": error or (outcome.error if outcome is not None else None) if status == "error" else None,
        "replay_input_mode": outcome.replay_input_mode if outcome is not None else "exact",
        "tool_history_detail": outcome.tool_history_detail if outcome is not None else None,
    }
    _write_json(model_dir / "metadata.json", metadata)
    return metadata


def run_benchmark(
    *,
    from_run: Path,
    stage_keys: list[str],
    model_specs: list[ModelSpec],
    output_dir: Path,
    base_seed: int | None = None,
    provider: str | None = None,
    ticker: str | None = None,
    on_progress: Callable[[dict], None] | None = None,
) -> dict:
    """Entry point used by the CLI. Read-only with respect to ``from_run``:
    only ever opens files under it; every write goes under ``output_dir``.

    ``on_progress``, when given, is called synchronously -- BEFORE and
    AFTER each (stage, model) replay call, and once up front once the
    source run/ticker is resolved -- so a caller (the CLI) can print
    visible progress during a run that can otherwise take 30-60 minutes
    with no output at all. Never receives a prompt or a response body,
    only small structured fields (stage/model/counters/timing/tokens/
    mode/error). This function itself never prints anything.
    """
    _ensure_fresh_output_dir(output_dir)
    wall_started = time.monotonic()
    debug_dir, resolved_ticker = resolve_debug_dir(from_run, ticker)
    manifest = _load_run_manifest(debug_dir)  # fail fast if the trace is incomplete/missing
    source_run_provider = manifest.get("provider")

    if on_progress is not None:
        on_progress({"phase": "source_resolved", "source_run": str(from_run), "ticker": resolved_ticker})

    results: list[dict] = []
    # Resolved up front (no LLM call yet) so progress reporting can show an
    # exact N/total from its very first line, rather than a total that only
    # becomes accurate after some stages turn out unsupported.
    stage_calls: dict[str, TraceCall] = {}
    for stage_key in stage_keys:
        try:
            stage_calls[stage_key] = find_source_call(debug_dir, stage_key)
        except ReplayError as exc:
            # A trace-structural problem (no separable final-synthesis
            # call at all) -- distinct from a specific MODEL failing to
            # produce a usable reply against a valid frozen input, which is
            # a per-(stage, model) "error" result instead (see
            # ``replay_one``), never this stage-wide status.
            results.append({"stage": stage_key, "status": "unsupported_for_replay", "reason": str(exc)})
            if on_progress is not None:
                on_progress({"phase": "unsupported", "stage": stage_key, "reason": str(exc)})

    total = len(stage_calls) * len(model_specs)
    index = 0
    for stage_key, source_call in stage_calls.items():
        for model_spec in model_specs:
            index += 1
            if on_progress is not None:
                on_progress({
                    "phase": "start", "index": index, "total": total, "stage": stage_key,
                    "model": model_spec.model, "reasoning_effort": model_spec.reasoning_effort,
                })
            result = replay_one(
                source_call=source_call, model_spec=model_spec, stage_key=stage_key,
                output_dir=output_dir, base_seed=base_seed, provider_override=provider,
                ticker=resolved_ticker, source_run=from_run, source_run_provider=source_run_provider,
            )
            results.append(result)
            if on_progress is not None:
                on_progress({
                    "phase": "finish" if result.get("status") == "success" else "error",
                    "index": index, "total": total, "stage": stage_key,
                    "model": model_spec.model, "reasoning_effort": model_spec.reasoning_effort,
                    "elapsed_seconds": result.get("elapsed_seconds"),
                    "total_tokens": result.get("total_tokens"),
                    "output_mode": result.get("output_mode"),
                    "error": result.get("error"),
                })

    wall_elapsed = time.monotonic() - wall_started
    # Wall-clock time for the whole benchmark vs. the sum of each
    # individual call's own elapsed time -- never conflated (see task: "No
    # mezclar tiempo wall-clock y suma de llamadas sin etiquetarlo
    # claramente"): run_benchmark executes sequentially today, so the two
    # are expected to be close, but are reported as what they each actually
    # measure, not assumed interchangeable.
    model_call_elapsed = sum(r.get("elapsed_seconds", 0) or 0 for r in results if "elapsed_seconds" in r)
    per_model_elapsed: dict[str, float] = {}
    for r in results:
        if "model" in r and "elapsed_seconds" in r:
            per_model_elapsed[r["model"]] = per_model_elapsed.get(r["model"], 0.0) + (r["elapsed_seconds"] or 0)

    summary = {
        "source_run": str(from_run),
        "ticker": resolved_ticker,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "base_seed": base_seed,
        "stages": list(stage_keys),
        "models": [spec.raw for spec in model_specs],
        "benchmark_wall_elapsed_seconds": round(wall_elapsed, 3),
        "model_call_elapsed_seconds": round(model_call_elapsed, 3),
        "elapsed_seconds_total_by_model": {k: round(v, 3) for k, v in per_model_elapsed.items()},
        "results": results,
    }
    _write_json(output_dir / "summary.json", summary)
    _write_json(output_dir / "manifest.json", {
        "source_run": str(from_run),
        "source_debug_dir": str(debug_dir),
        "ticker": resolved_ticker,
        "stages": list(stage_keys),
        "models": [spec.raw for spec in model_specs],
        "base_seed": base_seed,
        "provider_override": provider,
        "generated_at": summary["generated_at"],
    })
    return summary
