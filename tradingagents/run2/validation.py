"""Generic structured-call-with-one-repair orchestration, shared by run2's
two LLM stages (Market Regime Analyzer, First Barrier Signal Analyst -- see
``stages.py``).

Deliberately NOT ``tradingagents.agents.validator.validate_and_repair``: that
orchestrator is built around the main pipeline's Generation/Validation/
Review architecture (a Validator LLM producing PASS/WARN/FAIL findings on a
candidate, repaired by re-invoking the SAME stage) and is tightly coupled to
that flow. run2 needs something much smaller: one structured call, and if it
does not parse into the expected schema (malformed JSON, or a semantic
violation caught by the schema's own ``model_validator`` -- see
``schemas.py``), exactly one repair attempt, then give up. That smaller
shape is ``run_structured_stage`` below, built entirely from already-generic
primitives (``tradingagents.agents.structured``,
``tradingagents.agents.structured_recovery``).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from pydantic import BaseModel

from tradingagents.agents.structured import bind_structured, invoke_structured_or_freetext_parsed
from tradingagents.agents.structured_recovery import recover_structured_from_text

T = TypeVar("T", bound=BaseModel)

STATUS_OK = "ok"
STATUS_REPAIRED = "repaired"
STATUS_FAILED = "failed"


def run_structured_stage(
    llm: Any,
    schema: type[T],
    prompt: str,
    stage_name: str,
    *,
    llm_role: str,
    call_identity: str | None = None,
    repair_llm: Any | None = None,
) -> tuple[T | None, dict[str, Any]]:
    """Run one structured LLM call for ``schema``; on any parse failure
    (malformed JSON, missing fields, or a ``model_validator`` rejection),
    make exactly ONE repair attempt via
    ``structured_recovery.recover_structured_from_text`` -- never a second
    one (fork spec: "si una salida estructurada falla: un unico intento de
    repair; si sigue fallando, degradar").

    ``repair_llm``, when given, performs the repair instead of ``llm`` --
    run2 passes its optional ``--validator-llm`` here, so a user who
    configures one gets a second, independent model attempting the fix
    (mirroring the main pipeline's Validator role) while a user who omits it
    still gets a working repair path (the stage's own ``llm`` repairs
    itself, exactly as ``recover_structured_from_text``'s established
    "same original LLM" convention already does elsewhere in this project).

    Returns ``(parsed_or_None, meta)``; ``meta`` always has a ``"status"``
    key in {"ok", "repaired", "failed"}, an accurate ``"llm_call_count"`` --
    the number of real ``.invoke()`` calls this stage actually made (1 for a
    clean structured success; 2 if the structured attempt itself was made
    and raised before falling back to free text, or 1 if structured output
    was not bound at all; +1 more whenever the one repair attempt runs) --
    plus the raw prompt/response text, used both for
    ``result.json["validation"]`` and for run2's own ``--trace-llm`` debug
    dump (see ``pipeline.py``). This count is derived purely from
    information already visible here (whether ``bind_structured`` returned
    something, and whether ``parsed`` came back non-``None``); it never
    estimates or invents a token count, since no provider/wrapper used here
    exposes one.
    """
    structured_llm = bind_structured(llm, schema, stage_name)
    attempted_structured = structured_llm is not None
    render: Callable[[T], str] = lambda parsed: parsed.model_dump_json(indent=2)  # noqa: E731
    text, parsed = invoke_structured_or_freetext_parsed(
        structured_llm, llm, prompt, render, stage_name,
        call_identity=call_identity, llm_role=llm_role,
    )
    if parsed is not None:
        # The structured call succeeded on the first attempt -- exactly one
        # real LLM call, no fallback and no repair.
        return parsed, {
            "stage": stage_name, "status": STATUS_OK, "repaired": False,
            "llm_call_count": 1, "prompt": prompt, "response_text": text,
        }

    # parsed is None only when invoke_structured_or_freetext_parsed took its
    # free-text fallback branch -- exactly one real fallback call, plus one
    # more if a structured attempt was actually made (and raised) before it.
    calls_so_far = 2 if attempted_structured else 1

    repair_model = repair_llm if repair_llm is not None else llm
    repair_role = "validator" if repair_llm is not None else llm_role
    repaired = recover_structured_from_text(text, schema, repair_model, stage_name, llm_role=repair_role)
    # The one repair attempt is always exactly one more real call, whether
    # or not it succeeds.
    calls_so_far += 1
    if repaired is not None:
        return repaired, {
            "stage": stage_name, "status": STATUS_REPAIRED, "repaired": True,
            "repair_attempts": 1, "llm_call_count": calls_so_far,
            "prompt": prompt, "response_text": text,
        }

    return None, {
        "stage": stage_name, "status": STATUS_FAILED, "repaired": False,
        "repair_attempts": 1, "llm_call_count": calls_so_far,
        "prompt": prompt, "response_text": text,
    }
