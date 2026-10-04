"""Shared structured-output flow for the Bull/Bear initial thesis and reviews.

Both researchers follow the same two-schema pattern (see fork README:
"Structured trade hypotheses", "Review data restrictions"):

- Initial phase: one ``InitialResearchThesis`` call, rendered to the markdown
  prose the rest of the system already treats the debate history as.
- Review phase: one ``ResearchReviewOutcome`` call per round, applied to the
  side's tracked trade-plan levels via KEEP/REVISE/WITHDRAW.

Structured output is optional and falls back to free text exactly like the
Trader/Portfolio Manager/Research Manager (``agents/structured.py``): a
provider or test double without ``with_structured_output`` degrades to plain
``llm.invoke(prompt).content`` with no thesis/review dict, so the trade-plan
tracking and deterministic metrics below simply do not populate for that
side rather than breaking the debate.
"""

from __future__ import annotations

import logging

from tradingagents.agents.fallback_log import warn_fallback
from tradingagents.agents.researchers.canonical_recovery import (
    CanonicalRecoveryStatus,
    recover_initial_thesis,
    recover_review_patch,
)
from tradingagents.agents.researchers.trade_metrics import (
    compute_trade_metrics,
    entry_reference as _entry_reference,
)
from tradingagents.agents.schemas import (
    InitialResearchThesis,
    ResearchReviewOutcome,
    render_initial_thesis,
    render_review_outcome,
)
from tradingagents.llm_trace import record_structured_parse_result, traced_invoke

logger = logging.getLogger(__name__)

# Appended to the rendered fallback text when its canonical recovery failed,
# so every later stage that reads this text directly (Research Manager,
# Verifier -- see fork README: "Canonical recovery for Research fallbacks")
# sees unambiguously that it is narrative only, never authoritative state,
# with no change required to those stages' own prompts/code.
NON_CANONICAL_NOTE = (
    "\n\n[NON-CANONICAL: this fallback text could not be converted into "
    "structured canonical state (canonical recovery failed); treat it as "
    "narrative only, not as an authoritative conviction/trade-plan update.]"
)


def invoke_initial_thesis(structured_llm, plain_llm, prompt, agent_name: str):
    """Return ``(rendered_markdown, thesis_dict_or_None, recovery_status)``.

    ``recovery_status`` is a ``CanonicalRecoveryStatus.value`` string:
    ``NOT_NEEDED`` when structured output succeeded directly, or
    ``APPLIED``/``UNAPPLIED`` when it fell back to free text and canonical
    recovery did/did not manage to reconstruct a minimal canonical thesis
    from that text (see ``canonical_recovery.py``).
    """
    if structured_llm is not None:
        try:
            result = traced_invoke(structured_llm, prompt, attempt="structured")
            if result is None:
                raise ValueError("structured output returned no parsed result")
            if not isinstance(result, InitialResearchThesis):
                result = InitialResearchThesis.model_validate(result)
            record_structured_parse_result(success=True, parsed=result.model_dump(mode="json"))
            return (
                render_initial_thesis(result),
                result.model_dump(mode="json"),
                CanonicalRecoveryStatus.NOT_NEEDED.value,
            )
        except Exception as exc:
            record_structured_parse_result(success=False, error=str(exc))
            warn_fallback(
                logger,
                f"{agent_name}: structured initial thesis unavailable; "
                "continuing with free-text fallback.",
                detail=str(exc),
            )
    fallback_text = traced_invoke(plain_llm, prompt, attempt="fallback").content
    thesis_dict, status = recover_initial_thesis(fallback_text, plain_llm, agent_name)
    if status is CanonicalRecoveryStatus.UNAPPLIED:
        warn_fallback(
            logger,
            f"{agent_name}: fallback canonical recovery failed; canonical state unchanged.",
        )
        return fallback_text + NON_CANONICAL_NOTE, thesis_dict, status.value
    logger.debug("%s: fallback canonical recovery applied.", agent_name)
    return fallback_text, thesis_dict, status.value


def invoke_review(
    structured_llm, plain_llm, prompt, agent_name: str, prior_thesis: dict | None = None
):
    """Return ``(rendered_markdown, review_dict_or_None, recovery_status)``.

    ``prior_thesis`` is the side's current canonical thesis dict (if any),
    passed through to canonical recovery so a fallback review can be
    represented as a patch over it -- never reconstructed from scratch. See
    ``invoke_initial_thesis`` for ``recovery_status``'s meaning.
    """
    if structured_llm is not None:
        try:
            result = traced_invoke(structured_llm, prompt, attempt="structured")
            if result is None:
                raise ValueError("structured output returned no parsed result")
            if not isinstance(result, ResearchReviewOutcome):
                result = ResearchReviewOutcome.model_validate(result)
            record_structured_parse_result(success=True, parsed=result.model_dump(mode="json"))
            return (
                render_review_outcome(result),
                result.model_dump(mode="json"),
                CanonicalRecoveryStatus.NOT_NEEDED.value,
            )
        except Exception as exc:
            record_structured_parse_result(success=False, error=str(exc))
            warn_fallback(
                logger,
                f"{agent_name}: structured review unavailable; continuing "
                "with free-text fallback.",
                detail=str(exc),
            )
    fallback_text = traced_invoke(plain_llm, prompt, attempt="fallback").content
    review_dict, status = recover_review_patch(fallback_text, prior_thesis, plain_llm, agent_name)
    if status is CanonicalRecoveryStatus.UNAPPLIED:
        warn_fallback(
            logger,
            f"{agent_name}: fallback canonical recovery failed; canonical state unchanged.",
        )
        return fallback_text + NON_CANONICAL_NOTE, review_dict, status.value
    logger.debug("%s: fallback canonical recovery applied.", agent_name)
    return fallback_text, review_dict, status.value


def _entry_field(action: str, old_entry: dict, new_value) -> dict:
    if action == "WITHDRAW":
        return {"type": old_entry.get("type", "point"), "price": None, "low": None, "high": None}
    if action == "REVISE" and new_value is not None:
        return {"type": "point", "price": new_value, "low": None, "high": None}
    return dict(old_entry)


def _scalar_field(action: str, old_value, new_value):
    if action == "WITHDRAW":
        return None
    if action == "REVISE" and new_value is not None:
        return new_value
    return old_value


def _retired_value_for(action: str, old_value, new_value) -> float | None:
    """The value that stops being active provenance after this change, or
    ``None`` if nothing was retired.

    WITHDRAW always retires the old value (if there was one). REVISE retires
    it only when the new value genuinely differs -- a REVISE that merely
    restates the same number isn't really replacing anything, and treating it
    as retired would flag the still-current number as reused the next time it
    appears. KEEP never retires. This is the one invariant both normal review
    rounds and repair rely on: a value that was REVISED or WITHDRAWN must
    never later be treated as active verified provenance (see
    ``managers/integrity_check.py``, WITHDRAWN_VALUE_REUSED).
    """
    if old_value is None:
        return None
    if action == "WITHDRAW":
        return old_value
    if action == "REVISE" and new_value is not None and new_value != old_value:
        return old_value
    return None


def apply_trade_plan_changes(thesis: dict, review: dict) -> tuple[dict, list[float]]:
    """Apply a review's KEEP/REVISE/WITHDRAW actions to a tracked thesis dict.

    Operates on the plain-dict state representation (not the pydantic model)
    so it works the same whether the thesis was set this round or an earlier
    one (a normal review or a repair -- see ``researchers/research_repair.py``,
    which applies this identically). From the first review onward, entry is
    tracked as a flattened point reference (its range midpoint, if the
    initial thesis was a range) since a review only ever revises a single
    number in practice.

    Returns ``(updated_thesis, newly_retired_values)``: the second element is
    the actual pre-change value of any field just WITHDRAWN or genuinely
    REVISED (not the review's self-reported ``old``, which may not match) --
    stored in debate state as ``withdrawn_values`` (the name predates REVISE
    also retiring a value; broadened rather than renamed to keep the diff
    small, see fork README).
    """
    updated = dict(thesis)
    retired: list[float] = []
    changes = review.get("trade_plan_changes") or {}

    entry_change = changes.get("entry") or {}
    old_entry = thesis.get("entry") or {}
    old_entry_ref = _entry_reference(old_entry.get("price"), old_entry.get("low"), old_entry.get("high"))
    retired_entry = _retired_value_for(
        entry_change.get("action", "KEEP"), old_entry_ref, entry_change.get("new")
    )
    if retired_entry is not None:
        retired.append(retired_entry)
    updated["entry"] = _entry_field(
        entry_change.get("action", "KEEP"), old_entry, entry_change.get("new")
    )

    tp_change = changes.get("take_profit") or {}
    retired_tp = _retired_value_for(
        tp_change.get("action", "KEEP"), thesis.get("take_profit"), tp_change.get("new")
    )
    if retired_tp is not None:
        retired.append(retired_tp)
    updated["take_profit"] = _scalar_field(
        tp_change.get("action", "KEEP"), thesis.get("take_profit"), tp_change.get("new")
    )

    sl_change = changes.get("stop_loss") or {}
    retired_sl = _retired_value_for(
        sl_change.get("action", "KEEP"), thesis.get("stop_loss"), sl_change.get("new")
    )
    if retired_sl is not None:
        retired.append(retired_sl)
    updated["stop_loss"] = _scalar_field(
        sl_change.get("action", "KEEP"), thesis.get("stop_loss"), sl_change.get("new")
    )

    return updated, retired


def recompute_trade_metrics(thesis: dict, atr: float | None) -> dict:
    """Deterministic trade metrics for a tracked thesis dict, as a plain dict."""
    entry = thesis.get("entry") or {}
    metrics = compute_trade_metrics(
        entry_price=entry.get("price") if entry.get("type") == "point" else None,
        entry_low=entry.get("low") if entry.get("type") == "range" else None,
        entry_high=entry.get("high") if entry.get("type") == "range" else None,
        take_profit=thesis.get("take_profit"),
        stop_loss=thesis.get("stop_loss"),
        atr=atr,
    )
    return {
        "entry_reference": metrics.entry_reference,
        "risk": metrics.risk,
        "reward": metrics.reward,
        "risk_pct": metrics.risk_pct,
        "reward_pct": metrics.reward_pct,
        "reward_risk": metrics.reward_risk,
        "stop_atr": metrics.stop_atr,
        "target_atr": metrics.target_atr,
    }
