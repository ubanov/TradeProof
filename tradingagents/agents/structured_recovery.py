"""Generic canonical-structure recovery for a Validator-accepted free-text
response (see fork README: "Generation / Validation / Review").

When a stage's Validator repair call returns free text instead of its
expected schema's JSON shape, the structured object used to simply stay
``None`` for the rest of the run -- even once that free text was itself
accepted (PASS/WARN) on recheck. That silently destroyed
``structured.research_plan``/``structured.portfolio_decision`` for an
otherwise-valid final answer, and with ``portfolio_decision`` gone, the
RecommendationRecord ledger entry could never be written at all (real-run
regression: Research Manager and Portfolio Manager repairs both returned
Markdown that a subsequent validation pass accepted, yet
``research_plan_structured``/``portfolio_decision_structured`` ended up
``null``).

Same architecture as ``tradingagents.agents.researchers.canonical_recovery``
(schema-SPECIFIC there, for Bull/Bear's own patch shapes): ask the SAME
original LLM/role one narrow bookkeeping question ("what does the text
above already say, in this exact JSON shape?"), validate the answer with
the caller's own Pydantic schema, and reject the whole result unless every
number it introduces is already present in the free text (the same
membership check researchers/canonical_recovery.py uses, generalized to walk
an arbitrary model's fields rather than one hand-written list). Generic
across ResearchPlan/PortfolioDecision/TraderProposal (or any other
``validate_and_repair`` caller's schema) because this is called from the
one shared ``validate_and_repair`` entry point, not duplicated per stage.

A failed recovery (invalid JSON, failed validation, or a hallucinated
value) must never guess -- the caller keeps the free text as ``final_text``
but ``final_parsed`` stays ``None``, exactly as before this module existed.
"""

from __future__ import annotations

import json
import logging

from pydantic import BaseModel, ValidationError

from tradingagents.agents.integrity_shared import extract_numbers, numbers_match
from tradingagents.llm_trace import record_structured_parse_result, traced_invoke

logger = logging.getLogger(__name__)

# Same convention as researchers/canonical_recovery.py's own marker -- a
# recovery call is unambiguous in a trace or to a test double.
STRUCTURED_RECOVERY_PROMPT_MARKER = "STRUCTURED_RECOVERY_REQUEST"


def _strip_json_fence(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        return "\n".join(lines[1:-1]) if len(lines) >= 3 else text
    return text


def _parse_json_object(content: str) -> dict | None:
    try:
        data = json.loads(_strip_json_fence(content))
    except (json.JSONDecodeError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _numeric_leaves(value) -> list[float]:
    """Every int/float leaf reachable from ``value`` (a Pydantic model,
    dict, list, or scalar) -- walks an ARBITRARY schema's fields, so this
    module never needs a hand-written list of "the numeric fields" per
    caller. Bools are excluded (``bool`` is a ``int`` subclass in Python,
    and a True/False flag is never a price/number to verify)."""
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, BaseModel):
        return _numeric_leaves(value.model_dump(mode="json"))
    if isinstance(value, dict):
        return [n for v in value.values() for n in _numeric_leaves(v)]
    if isinstance(value, (list, tuple)):
        return [n for v in value for n in _numeric_leaves(v)]
    return []


def _values_present_in_text(values: list[float], text: str) -> bool:
    """Every number the recovered object introduces must already be
    literally present in the free text it was derived from -- a
    deterministic membership check (the same ``extract_numbers``/
    ``numbers_match`` primitives ``researchers/canonical_recovery.py``
    uses), never a new semantic price parser. Rejects a recovery that would
    let the bookkeeping call "improve" or invent content beyond what the
    stage's own accepted free text actually said."""
    if not values:
        return True
    pool = extract_numbers(text)
    return all(numbers_match(value, pool) for value in values)


def _recovery_prompt(stage: str, fallback_text: str, expected_schema: type[BaseModel]) -> str:
    schema_json = json.dumps(expected_schema.model_json_schema(), indent=2)
    return (
        f"{STRUCTURED_RECOVERY_PROMPT_MARKER}\n"
        "You are doing a narrow bookkeeping task, not giving a new opinion "
        f"or revising the {stage}'s decision. Below is the {stage}'s own "
        "free-text response, already accepted as final. Translate ONLY "
        "what it already says into a JSON object matching the schema "
        "below. Do not add, infer, or introduce any value, level, rating, "
        "or claim that is not already stated in the text.\n\n"
        "Respond with ONLY a single JSON object, no markdown fences, no "
        "commentary, matching this JSON schema:\n"
        f"{schema_json}\n\n"
        f"{stage}'s free-text response:\n"
        f"{fallback_text}\n"
    )


def recover_structured_from_text(
    fallback_text: str,
    expected_schema: type[BaseModel],
    llm,
    stage: str,
    *,
    llm_role: str | None = None,
) -> BaseModel | None:
    """Attempt to reconstruct ``expected_schema`` from ``fallback_text`` --
    the SAME original LLM/role that produced it, asked one narrow
    bookkeeping question, based exclusively on that text (no tools, no new
    evidence, no re-reading of upstream reports).

    ``llm_role`` is whatever role the call being recovered used (see fork
    README: "Model roles: Generation vs Review"), passed straight through
    by the caller -- never fixed to a constant role here.
    """
    try:
        response = traced_invoke(
            llm, _recovery_prompt(stage, fallback_text, expected_schema),
            attempt="canonical_recovery", llm_role=llm_role,
        )
        payload = _parse_json_object(response.content)
        if payload is None:
            record_structured_parse_result(
                success=False, error="structured recovery: invalid JSON"
            )
            return None

        try:
            candidate = expected_schema.model_validate(payload)
        except ValidationError:
            record_structured_parse_result(
                success=False, error="structured recovery: schema mismatch"
            )
            return None

        if not _values_present_in_text(_numeric_leaves(candidate), fallback_text):
            record_structured_parse_result(
                success=False,
                error="structured recovery: a recovered value is not present in the fallback text",
            )
            return None

        record_structured_parse_result(success=True, parsed=candidate.model_dump(mode="json"))
        return candidate
    except Exception as exc:
        record_structured_parse_result(success=False, error=str(exc))
        logger.debug("%s: structured recovery raised: %s", stage, exc)
        return None
