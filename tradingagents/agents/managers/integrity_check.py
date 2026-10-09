"""Deterministic, LLM-free integrity check for the Research Manager's output.

The Research Verifier already checks the Bull/Bear research before the
Research Manager sees it, but the manager is itself a free-text-capable LLM
call: nothing stopped it from reintroducing a numeric threshold, sizing rule,
or price level that was never in the verified evidence (observed in real
runs -- a manager inventing a MACD-histogram threshold, a volume threshold, an
allocation percentage, or a yield trigger that no analyst report, verified
thesis, or deterministic calculation ever produced; also a manager
reintroducing a narrative claim Bull/Bear had already corrected during
review, a reward/risk ratio inconsistent with the deterministic trade
metrics, and a simple relational error such as stating one known price is
below another when it is not). This module closes that gap with one cheap,
deterministic pass over the manager's rendered output, run after the
Research Manager and before the Trader. No LLM call is made here, by design
(see fork README: "Research Manager integrity check").

The numeric-extraction/sentence-scanning primitives are shared with the Risk
integrity check (see ``tradingagents.agents.integrity_shared``) so the two do
not drift apart on the same underlying logic.

The check is intentionally narrow and conservative: false positives (a real,
sourced number wrongly flagged) are worse than false negatives (a low-value
new number that slips through). See ``ManagerFindingCategory`` for the full,
fixed set of checks.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum

from tradingagents.agents.integrity_shared import (
    LEVEL_WORDS,
    PROBABILITY_WORDS,
    SIZING_WORDS,
    THRESHOLD_TRIGGERS,
    excerpt,
    extract_corrected_away_numbers,
    extract_numbers,
    find_arithmetic_errors,
    find_numbers_near_triggers,
    find_relational_errors,
    find_withdrawn_values_reused,
    split_sentences,
)

# ---------------------------------------------------------------------------
# Known-good number pool
# ---------------------------------------------------------------------------


def build_known_number_pool(
    debate: dict, *reports: str, extra_sources: tuple[str, ...] = (),
) -> list[float]:
    """Numbers traceable to analyst evidence, verified theses, or deterministic
    calculations -- the only numbers a manager-stated threshold may reuse.

    ``extra_sources`` extends the pool with any further free-text sources a
    caller's own allowed-evidence universe includes but ``reports``/``debate``
    don't already cover -- scanned with the exact same ``extract_numbers``
    call as ``reports``, never a different extraction method. Added for the
    Research Manager Validator (see ``validator_deterministic.py::
    check_manager_integrity_findings``), whose caller must NOT pass the raw
    analyst ``reports`` here (Research Manager's own prompt never shows them)
    but does need bull/bear's free-text narrative counted as supported.
    Defaults to ``()`` so every existing caller is unaffected.
    """
    pool: list[float] = []
    for report in reports:
        pool.extend(extract_numbers(report or ""))
    for source in extra_sources:
        pool.extend(extract_numbers(source or ""))

    for side in ("bull", "bear"):
        thesis = debate.get(f"{side}_thesis") or {}
        entry = thesis.get("entry") or {}
        for value in (entry.get("price"), entry.get("low"), entry.get("high"),
                      thesis.get("take_profit"), thesis.get("stop_loss")):
            if value is not None:
                pool.append(float(value))

        metrics = debate.get(f"{side}_trade_metrics") or {}
        for value in metrics.values():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                pool.append(float(value))

    return pool


def _reward_risk_pool(debate: dict) -> list[float]:
    ratios = []
    for side in ("bull", "bear"):
        metrics = debate.get(f"{side}_trade_metrics") or {}
        ratio = metrics.get("reward_risk")
        if isinstance(ratio, (int, float)) and not isinstance(ratio, bool):
            ratios.append(float(ratio))
    return ratios


# ---------------------------------------------------------------------------
# Finding categories and result
# ---------------------------------------------------------------------------


class ManagerCheckStatus(str, Enum):
    PASS = "PASS"
    WARN = "WARN"


class ManagerFindingCategory(str, Enum):
    NEW_UNSUPPORTED_THRESHOLD = "NEW_UNSUPPORTED_THRESHOLD"
    NEW_UNSUPPORTED_LEVEL = "NEW_UNSUPPORTED_LEVEL"
    UNSUPPORTED_SIZING_RULE = "UNSUPPORTED_SIZING_RULE"
    ARITHMETIC_ERROR = "ARITHMETIC_ERROR"
    RELATIONAL_ERROR = "RELATIONAL_ERROR"
    CONVICTION_AS_PROBABILITY = "CONVICTION_AS_PROBABILITY"
    WITHDRAWN_VALUE_REUSED = "WITHDRAWN_VALUE_REUSED"
    INTERNAL_INCONSISTENCY = "INTERNAL_INCONSISTENCY"


@dataclass(frozen=True)
class ManagerFinding:
    category: str
    detail: str


@dataclass(frozen=True)
class ManagerIntegrityResult:
    status: str
    findings: list[ManagerFinding] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"status": self.status, "findings": [asdict(f) for f in self.findings]}


# ---------------------------------------------------------------------------
# Sentence-level scanning
# ---------------------------------------------------------------------------


def _classify_threshold(sentence_lower: str) -> ManagerFindingCategory:
    if "%" in sentence_lower and any(word in sentence_lower for word in SIZING_WORDS):
        return ManagerFindingCategory.UNSUPPORTED_SIZING_RULE
    if any(word in sentence_lower for word in LEVEL_WORDS):
        return ManagerFindingCategory.NEW_UNSUPPORTED_LEVEL
    return ManagerFindingCategory.NEW_UNSUPPORTED_THRESHOLD


def _find_unsupported_thresholds(text: str, pool: list[float]) -> list[ManagerFinding]:
    return [
        ManagerFinding(_classify_threshold(sentence.lower()).value, sentence)
        for sentence, _value in find_numbers_near_triggers(text, THRESHOLD_TRIGGERS, pool)
    ]


def _find_conviction_as_probability(text: str) -> list[ManagerFinding]:
    findings = []
    for sentence in split_sentences(text):
        lower = sentence.lower()
        if "conviction" in lower and any(word in lower for word in PROBABILITY_WORDS):
            findings.append(
                ManagerFinding(
                    ManagerFindingCategory.CONVICTION_AS_PROBABILITY.value, excerpt(sentence)
                )
            )
    return findings


def _find_withdrawn_values_reused(text: str, withdrawn_values: list[float]) -> list[ManagerFinding]:
    return [
        ManagerFinding(ManagerFindingCategory.WITHDRAWN_VALUE_REUSED.value, detail)
        for detail in find_withdrawn_values_reused(text, withdrawn_values)
    ]


def _find_arithmetic_errors(text: str, debate: dict) -> list[ManagerFinding]:
    return [
        ManagerFinding(ManagerFindingCategory.ARITHMETIC_ERROR.value, sentence)
        for sentence in find_arithmetic_errors(text, _reward_risk_pool(debate))
    ]


def _find_relational_errors(text: str) -> list[ManagerFinding]:
    return [
        ManagerFinding(ManagerFindingCategory.RELATIONAL_ERROR.value, sentence)
        for sentence in find_relational_errors(text)
    ]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def render_integrity_notice_for_trader(status: str, findings: list[dict]) -> str:
    """A notice for a downstream prompt (Trader, or Risk's frozen evidence)
    when the manager's output carried unsupported quantitative claims --
    never silently rewritten (meaning could change), always explicitly
    flagged as unsupported instead.
    """
    if status != ManagerCheckStatus.WARN.value or not findings:
        return ""
    lines = [
        "The following claims in the investment plan above could not be traced to the "
        "supplied evidence, a verified research level, or a deterministic calculation. "
        "Treat them as UNSUPPORTED: do not act on these specific thresholds as if they "
        "were verified input.",
    ]
    lines += [f"- [{finding['category']}] {finding['detail']}" for finding in findings]
    return "\n".join(lines)


def render_manager_integrity_report(status: str, findings: list[dict]) -> str:
    """Compact report-section rendering: a one-line status on PASS, the
    status plus each finding's category and excerpt on WARN."""
    if not status:
        return ""
    lines = [f"Status: {status}"]
    if findings:
        lines += [""] + [f"- {f['category']}: {f['detail']}" for f in findings]
    return "\n".join(lines)


def check_manager_integrity(
    manager_output_text: str,
    debate: dict,
    *reports: str,
    extra_sources: tuple[str, ...] = (),
) -> ManagerIntegrityResult:
    """Run every deterministic check against the Research Manager's rendered
    output and return one PASS/WARN result. Never raises; never calls an LLM.

    ``extra_sources`` is forwarded to ``build_known_number_pool`` as-is (see
    its own docstring) -- defaults to ``()``, so this call's existing
    callers (e.g. the Trader-facing integrity notice, which passes the raw
    analyst ``reports`` instead) are completely unaffected.
    """
    text = manager_output_text or ""
    pool = build_known_number_pool(debate, *reports, extra_sources=extra_sources)
    withdrawn = list(debate.get("withdrawn_values") or []) + extract_corrected_away_numbers(
        (debate.get("review_outcomes") or "").split("\n")
    )

    findings: list[ManagerFinding] = []
    findings += _find_unsupported_thresholds(text, pool)
    findings += _find_conviction_as_probability(text)
    findings += _find_withdrawn_values_reused(text, withdrawn)
    findings += _find_arithmetic_errors(text, debate)
    findings += _find_relational_errors(text)

    status = ManagerCheckStatus.WARN if findings else ManagerCheckStatus.PASS
    return ManagerIntegrityResult(status=status.value, findings=findings)
