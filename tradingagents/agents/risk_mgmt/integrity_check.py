"""Deterministic, LLM-free integrity check for the Risk phase's output.

Risk reviewers evaluate the frozen Trader plan; they must not become a
second, unverified research/trading-system design phase (observed in real
runs -- Risk introducing new thresholds, sizing rules, historical analogies,
and portfolio-percentage rules, then spending later turns correcting claims
it invented itself). This runs once, after the single cross-review round and
before Portfolio Manager, over every Risk reviewer's rendered text. No LLM
call is made here (see fork README: "Risk integrity check").

Shares its numeric-extraction/sentence-scanning primitives with the Research
Manager integrity check (``tradingagents.agents.integrity_shared``) so the
two checkers cannot drift apart on the same underlying logic; the frozen
evidence pool reuses ``managers.integrity_check.build_known_number_pool``
for the same reason.

Conservative by design: false positives (a real, sourced number wrongly
flagged) are worse than false negatives. See ``RiskFindingCategory`` for the
full, fixed set of checks.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum

from tradingagents.agents.integrity_shared import (
    HISTORICAL_WORDS,
    LEVEL_WORDS,
    PORTFOLIO_PERCENT_WORDS,
    PROBABILITY_WORDS,
    SIZING_WORDS,
    THRESHOLD_TRIGGERS,
    excerpt,
    exclude_values_still_active,
    extract_numbers,
    find_arithmetic_errors,
    find_numbers_near_triggers,
    find_numbers_near_words,
    find_relational_errors,
    find_unit_errors,
    find_withdrawn_values_reused,
    numbers_match,
    split_sentences,
)
from tradingagents.agents.managers.integrity_check import build_known_number_pool

_RESEARCH_ATTRIBUTION_WORDS = (
    "verified", "research shows", "the bull thesis", "the bear thesis",
    "research team", "the verifier",
)
_TRADER_ATTRIBUTION_WORDS = ("trader", "the plan", "trader's plan")


class RiskCheckStatus(str, Enum):
    PASS = "PASS"
    WARN = "WARN"


class RiskFindingCategory(str, Enum):
    NEW_UNSUPPORTED_THRESHOLD = "NEW_UNSUPPORTED_THRESHOLD"
    NEW_UNSUPPORTED_LEVEL = "NEW_UNSUPPORTED_LEVEL"
    UNSUPPORTED_SIZING_RULE = "UNSUPPORTED_SIZING_RULE"
    UNSUPPORTED_PORTFOLIO_PERCENTAGE = "UNSUPPORTED_PORTFOLIO_PERCENTAGE"
    UNSUPPORTED_HISTORICAL_CLAIM = "UNSUPPORTED_HISTORICAL_CLAIM"
    UNSUPPORTED_PROBABILITY_OR_FREQUENCY = "UNSUPPORTED_PROBABILITY_OR_FREQUENCY"
    ARITHMETIC_ERROR = "ARITHMETIC_ERROR"
    UNIT_ERROR = "UNIT_ERROR"
    RELATIONAL_ERROR = "RELATIONAL_ERROR"
    WITHDRAWN_VALUE_REUSED = "WITHDRAWN_VALUE_REUSED"
    CONTRADICTION_WITH_TRADER = "CONTRADICTION_WITH_TRADER"
    CONTRADICTION_WITH_VERIFIED_RESEARCH = "CONTRADICTION_WITH_VERIFIED_RESEARCH"


@dataclass(frozen=True)
class RiskFinding:
    category: str
    detail: str


@dataclass(frozen=True)
class RiskIntegrityResult:
    status: str
    findings: list[RiskFinding] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"status": self.status, "findings": [asdict(f) for f in self.findings]}


def build_frozen_evidence_pool(state: dict) -> list[float]:
    """Numbers traceable to the frozen evidence Risk was given: analyst
    reports, verified Bull/Bear theses and deterministic metrics, and the
    Trader's own stated plan -- the only numbers a Risk threshold/level claim
    may reuse."""
    debate = state.get("investment_debate_state") or {}
    pool = build_known_number_pool(
        debate,
        state.get("market_report", ""),
        state.get("sentiment_report", ""),
        state.get("news_report", ""),
        state.get("fundamentals_report", ""),
    )
    pool.extend(extract_numbers(state.get("trader_investment_plan") or ""))
    return pool


def _classify_threshold(sentence_lower: str) -> RiskFindingCategory:
    if "%" in sentence_lower and any(word in sentence_lower for word in PORTFOLIO_PERCENT_WORDS):
        return RiskFindingCategory.UNSUPPORTED_PORTFOLIO_PERCENTAGE
    if "%" in sentence_lower and any(word in sentence_lower for word in SIZING_WORDS):
        return RiskFindingCategory.UNSUPPORTED_SIZING_RULE
    if any(word in sentence_lower for word in _TRADER_ATTRIBUTION_WORDS):
        return RiskFindingCategory.CONTRADICTION_WITH_TRADER
    if any(word in sentence_lower for word in _RESEARCH_ATTRIBUTION_WORDS):
        return RiskFindingCategory.CONTRADICTION_WITH_VERIFIED_RESEARCH
    if any(word in sentence_lower for word in LEVEL_WORDS):
        return RiskFindingCategory.NEW_UNSUPPORTED_LEVEL
    return RiskFindingCategory.NEW_UNSUPPORTED_THRESHOLD


def _find_unsupported_thresholds(text: str, pool: list[float]) -> list[RiskFinding]:
    return [
        RiskFinding(_classify_threshold(sentence.lower()).value, sentence)
        for sentence, _value in find_numbers_near_triggers(text, THRESHOLD_TRIGGERS, pool)
    ]


def _find_level_attribution_mismatches(text: str, pool: list[float]) -> list[RiskFinding]:
    """A LEVEL_WORDS sentence explicitly attributed to the Trader or to
    verified research, naming a number that is not in the frozen pool --
    independent of ``THRESHOLD_TRIGGERS``, since a contradiction like "the
    trader stop should actually be 80.0" carries no exceed/above/below
    wording at all.
    """
    findings = []
    for sentence in split_sentences(text):
        lower = sentence.lower()
        if not any(word in lower for word in LEVEL_WORDS):
            continue
        attributed_to_trader = any(word in lower for word in _TRADER_ATTRIBUTION_WORDS)
        attributed_to_research = any(word in lower for word in _RESEARCH_ATTRIBUTION_WORDS)
        if not (attributed_to_trader or attributed_to_research):
            continue
        category = (
            RiskFindingCategory.CONTRADICTION_WITH_TRADER
            if attributed_to_trader
            else RiskFindingCategory.CONTRADICTION_WITH_VERIFIED_RESEARCH
        )
        for value in extract_numbers(sentence):
            if numbers_match(value, pool):
                continue
            findings.append(RiskFinding(category.value, excerpt(sentence)))
    return findings


def _find_historical_claims(text: str, pool: list[float]) -> list[RiskFinding]:
    findings = []
    for sentence, value in find_numbers_near_words(text, HISTORICAL_WORDS):
        if numbers_match(value, pool):
            continue
        findings.append(RiskFinding(RiskFindingCategory.UNSUPPORTED_HISTORICAL_CLAIM.value, sentence))
    return findings


def _find_probability_claims(text: str, pool: list[float]) -> list[RiskFinding]:
    findings = []
    for sentence, value in find_numbers_near_words(text, PROBABILITY_WORDS):
        if numbers_match(value, pool):
            continue
        findings.append(
            RiskFinding(RiskFindingCategory.UNSUPPORTED_PROBABILITY_OR_FREQUENCY.value, sentence)
        )
    return findings


def _find_withdrawn_values_reused(text: str, withdrawn_values: list[float]) -> list[RiskFinding]:
    return [
        RiskFinding(RiskFindingCategory.WITHDRAWN_VALUE_REUSED.value, detail)
        for detail in find_withdrawn_values_reused(text, withdrawn_values)
    ]


def _find_arithmetic_errors(text: str, state: dict) -> list[RiskFinding]:
    debate = state.get("investment_debate_state") or {}
    ratios = [
        (debate.get(f"{side}_trade_metrics") or {}).get("reward_risk")
        for side in ("bull", "bear")
    ]
    return [
        RiskFinding(RiskFindingCategory.ARITHMETIC_ERROR.value, sentence)
        for sentence in find_arithmetic_errors(text, [r for r in ratios if r is not None])
    ]


def _find_relational_errors(text: str) -> list[RiskFinding]:
    return [
        RiskFinding(RiskFindingCategory.RELATIONAL_ERROR.value, sentence)
        for sentence in find_relational_errors(text)
    ]


def _find_unit_errors(text: str) -> list[RiskFinding]:
    return [RiskFinding(RiskFindingCategory.UNIT_ERROR.value, sentence) for sentence in find_unit_errors(text)]


def check_risk_integrity(text: str, state: dict) -> RiskIntegrityResult:
    """Run every deterministic check against one Risk reviewer's (or the
    whole Risk phase's combined) rendered text. Never raises; never calls an LLM.
    """
    text = text or ""
    pool = build_frozen_evidence_pool(state)
    debate = state.get("investment_debate_state") or {}
    withdrawn = exclude_values_still_active(list(debate.get("withdrawn_values") or []), pool)

    findings: list[RiskFinding] = []
    findings += _find_unsupported_thresholds(text, pool)
    findings += _find_level_attribution_mismatches(text, pool)
    findings += _find_historical_claims(text, pool)
    findings += _find_probability_claims(text, pool)
    findings += _find_withdrawn_values_reused(text, withdrawn)
    findings += _find_arithmetic_errors(text, state)
    findings += _find_relational_errors(text)
    findings += _find_unit_errors(text)

    status = RiskCheckStatus.WARN if findings else RiskCheckStatus.PASS
    return RiskIntegrityResult(status=status.value, findings=findings)


def render_risk_integrity_report(status: str, findings: list[dict]) -> str:
    """Compact report-section rendering: a one-line status on PASS, the
    status plus each finding's category and excerpt on WARN."""
    if not status:
        return ""
    lines = [f"Status: {status}"]
    if findings:
        lines += [""] + [f"- {f['category']}: {f['detail']}" for f in findings]
    return "\n".join(lines)


def render_risk_integrity_notice(status: str, findings: list[dict]) -> str:
    """A notice for the Portfolio Manager's prompt: Risk is an assessment,
    not evidence -- an unsupported claim raised only during Risk must never
    be promoted as if it were verified input."""
    if status != RiskCheckStatus.WARN.value or not findings:
        return ""
    lines = [
        "The following claims appeared only in the Risk discussion and could not be traced "
        "to the frozen evidence, the Trader's plan, or a deterministic calculation. Risk "
        "arguments are assessments, not new evidence: do not treat these as verified input.",
    ]
    lines += [f"- [{finding['category']}] {finding['detail']}" for finding in findings]
    return "\n".join(lines)
