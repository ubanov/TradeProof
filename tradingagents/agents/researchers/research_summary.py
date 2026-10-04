"""Compact, human-readable summary of the structured research-debate state.

The full structured state (setup tags, per-round thesis/review dicts,
conviction history, deterministic metrics) already lives in the debate state
and the saved JSON state log; this renders just enough of it for a human
report without dumping the raw structures (see fork README: "Avoid dumping
huge JSON payloads into the human-readable report").
"""

from __future__ import annotations

from tradingagents.agents.researchers.setup_tags import render_setup_tags
from tradingagents.agents.researchers.trade_metrics import (
    render_trade_metrics,
    trade_metrics_from_dict,
)


def render_research_context_summary(debate: dict) -> str:
    lines = [
        f"**Shared Research Horizon**: {debate.get('research_horizon') or 'not recorded'}",
        "",
        "**Shared Setup Tags**:",
        render_setup_tags(debate.get("setup_tags") or {}),
    ]

    for side in ("bull", "bear"):
        thesis = debate.get(f"{side}_thesis") or {}
        history = debate.get(f"{side}_conviction_history") or []
        trail = " -> ".join(str(value) for value in history) if history else "not recorded"
        risk_unit = thesis.get("suggested_risk_unit", "not recorded")
        metrics = trade_metrics_from_dict(debate.get(f"{side}_trade_metrics"))
        initial_recovery = debate.get(f"{side}_initial_recovery_status") or "NOT_RUN"
        review_recovery = debate.get(f"{side}_review_recovery_status") or "NOT_RUN"
        lines += [
            "",
            f"**{side.title()} Conviction Trail**: {trail} "
            "(self-assessed evidence strength, not a probability)",
            f"**{side.title()} Canonical Recovery**: initial={initial_recovery}, "
            f"review={review_recovery} (NOT_NEEDED/APPLIED = structured or "
            "fallback-recovered state below is authoritative; UNAPPLIED = a "
            "fallback could not be recovered and is excluded from the "
            "structured state below, visible only as narrative text elsewhere "
            "in this report)",
            f"**{side.title()} Suggested Risk Unit**: {risk_unit} "
            "(research-level signal only, not a position size or portfolio percentage)",
            f"**{side.title()} Trade Metrics**:",
            render_trade_metrics(metrics),
        ]

    exceptions = debate.get("new_data_exceptions") or []
    lines += ["", "**NEW_DATA_EXCEPTION usage**:"]
    if exceptions:
        for exc in exceptions:
            lines.append(
                f"- [{exc.get('side', '?')} round {exc.get('round', '?')}] "
                f"{exc.get('datum', '')} -- {exc.get('reason', '')} "
                f"(source: {exc.get('source', '')})"
            )
    else:
        lines.append("none")

    return "\n".join(lines)


def render_repair_status_caveat(debate: dict) -> str:
    """A short caveat when a repair fell back to free text: the structured
    trade plan/conviction below was NOT confirmed by that repair (it still
    reflects the last successful review), so a consumer must not treat it as
    if the repair had validated it.
    """
    notes = []
    for side in ("bull", "bear"):
        if debate.get(f"{side}_repair_structured_status") == "TEXT_ONLY":
            notes.append(
                f"Note: {side.title()}'s repair fell back to free text -- the "
                f"structured trade plan/conviction below reflects the last "
                f"successful review, not the repair prose."
            )
    return "\n".join(notes)


def render_recovery_status_caveat(debate: dict) -> str:
    """A short caveat when a Bull/Bear textual fallback's canonical recovery
    failed: the structured trade plan/conviction below was never updated by
    that fallback (it still reflects the last valid canonical state, or is
    unavailable if the fallback was the INITIAL thesis), so a consumer must
    not treat the fallback's prose as if it were authoritative.
    """
    notes = []
    for side in ("bull", "bear"):
        if debate.get(f"{side}_initial_recovery_status") == "UNAPPLIED":
            notes.append(
                f"Note: {side.title()}'s initial thesis fell back to free text and "
                f"its canonical recovery failed -- no structured trade plan/"
                f"conviction is recorded for {side.title()} from that call."
            )
        if debate.get(f"{side}_review_recovery_status") == "UNAPPLIED":
            notes.append(
                f"Note: {side.title()}'s latest review fell back to free text and "
                f"its canonical recovery failed -- the structured trade plan/"
                f"conviction below reflects the last successful review, not "
                f"that fallback's prose."
            )
    return "\n".join(notes)


def _entry_text(entry: dict) -> str:
    if not entry:
        return "not set"
    if entry.get("type") == "range":
        low = entry.get("low")
        high = entry.get("high")
        return f"{low if low is not None else '?'} - {high if high is not None else '?'} (range)"
    price = entry.get("price")
    return str(price) if price is not None else "not set"


def render_trade_plan_for_trader(debate: dict) -> str:
    """Research team's final, deterministic trade-plan levels for the Trader.

    Distinct from ``render_research_context_summary`` (the report/verifier-
    facing summary, which omits the raw entry/TP/SL numbers and includes
    setup-tag/NEW_DATA_EXCEPTION detail irrelevant to sizing a transaction):
    this shows exactly the levels and deterministic metrics a Trader needs to
    ground its own proposal in, for each side, with no raw JSON.
    """
    caveat = "\n".join(
        note
        for note in (render_repair_status_caveat(debate), render_recovery_status_caveat(debate))
        if note
    )

    if not debate.get("bull_thesis") and not debate.get("bear_thesis"):
        base = (
            "No structured research trade plan is available for this run "
            "(the research team's structured output fell back to free text); "
            "use the investment plan and market report only."
        )
        return f"{caveat}\n\n{base}" if caveat else base

    horizon = debate.get("research_horizon") or "not recorded"
    lines = [f"Shared research horizon: {horizon}", ""]
    if caveat:
        lines = [caveat, ""] + lines

    for side in ("bull", "bear"):
        thesis = debate.get(f"{side}_thesis") or {}
        if not thesis:
            lines.append(f"{side.title()}: no structured trade plan available this run.")
            continue
        metrics = trade_metrics_from_dict(debate.get(f"{side}_trade_metrics"))
        lines += [
            f"{side.title()} ({thesis.get('direction', '?')}):",
            f"  Entry: {_entry_text(thesis.get('entry'))}",
            f"  Entry Reference (deterministic calculation, e.g. a range's midpoint -- "
            "context only, NOT a silent substitute for the Entry above): "
            f"{metrics.entry_reference if metrics.entry_reference is not None else 'N/A'}",
            f"  Take Profit: {thesis.get('take_profit') if thesis.get('take_profit') is not None else 'not set'}",
            f"  Stop Loss: {thesis.get('stop_loss') if thesis.get('stop_loss') is not None else 'not set'}",
            f"  Reward/Risk: {metrics.reward_risk if metrics.reward_risk is not None else 'N/A'}",
            f"  Stop distance (ATR multiples): {metrics.stop_atr if metrics.stop_atr is not None else 'N/A'}",
            f"  Suggested Risk Unit: {thesis.get('suggested_risk_unit', 'not recorded')} "
            "(research-level signal only, not a position size or portfolio percentage)",
            "",
        ]

    return "\n".join(lines).strip()
