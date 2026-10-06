"""Pydantic schemas used by agents that produce structured output.

The framework's primary artifact is still prose: each agent's natural-language
reasoning is what users read in the saved markdown reports and what the
downstream agents read as context.  Structured output is layered onto the
three decision-making agents (Research Manager, Trader, Portfolio Manager)
so that:

- Their outputs follow consistent section headers across runs and providers
- Each provider's native structured-output mode is used (json_schema for
  OpenAI/xAI, response_schema for Gemini, tool-use for Anthropic)
- Schema field descriptions become the model's output instructions, freeing
  the prompt body to focus on context and the rating-scale guidance
- A render helper turns the parsed Pydantic instance back into the same
  markdown shape the rest of the system already consumes, so display,
  memory log, and saved reports keep working unchanged
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

# LLMs sometimes write a placeholder string ("None", "N/A", ...) into an optional
# numeric field instead of omitting it. Coerce those to None so the structured
# call validates instead of erroring (#1058). Pydantic still parses real numeric
# strings ("189.5") to float.
_NULLISH_FLOAT = {"", "none", "n/a", "na", "null", "nil", "-", "tbd", "unknown"}


def _coerce_optional_float(value):
    """Normalise an LLM-written optional numeric field before validation.

    Three shapes show up in practice: a placeholder string ("None", "N/A") in
    place of an omitted value (#1058); a percentage where a price was asked for
    ("15%", #1288); and a human-formatted price ("$1,234.50"). A percentage
    cannot be salvaged into an absolute level -- reading "15%" as 15 would put a
    stop at $15 on a $600 stock -- so it is dropped like a placeholder, leaving
    one bad field to null out instead of failing the whole proposal. A formatted
    price is reduced to its number.

    Anything that is not a single number is dropped the same way. A range
    ("150-160") or a hedge ("around 150") would otherwise reach pydantic, fail
    validation, and discard the whole decision, losing every field the model got
    right along with the price.
    """
    if not isinstance(value, str):
        return value
    text = value.strip()
    if text.lower() in _NULLISH_FLOAT or text.endswith("%"):
        return None
    cleaned = text.replace(",", "").lstrip("$€£¥").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def _coerce_str_list(value) -> list:
    """Normalise an LLM-written list field before validation.

    A weak structured-output call sometimes serializes a list field as a
    single JSON-encoded string (e.g. ``'["a", "b"]'``) instead of a native
    array. Pydantic's bare ``list[str]`` type otherwise rejects that outright
    (``list_type`` error), discarding every other field the model got right
    along with it -- observed in practice for ``InitialResearchThesis.risks``/
    ``data_gaps`` in a real batch run. ``None`` becomes an empty list, a
    genuine list passes through, a JSON-array-shaped string is parsed, and
    anything else (a plain string, a number) is wrapped as a single item.
    """
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text)
            except (json.JSONDecodeError, ValueError):
                parsed = None
            if isinstance(parsed, list):
                return parsed
        return [value] if value else []
    return [value]


# ---------------------------------------------------------------------------
# Shared rating types
# ---------------------------------------------------------------------------


class PortfolioRating(str, Enum):
    """5-tier rating used by the Research Manager and Portfolio Manager."""

    BUY = "Buy"
    OVERWEIGHT = "Overweight"
    HOLD = "Hold"
    UNDERWEIGHT = "Underweight"
    SELL = "Sell"


class TraderAction(str, Enum):
    """3-tier transaction direction used by the Trader.

    The Trader's job is to translate the Research Manager's investment plan
    into a concrete transaction proposal: should the desk execute a Buy, a
    Sell, or sit on Hold this round.  Position sizing and the nuanced
    Overweight / Underweight calls happen later at the Portfolio Manager.
    """

    BUY = "Buy"
    HOLD = "Hold"
    SELL = "Sell"


# ---------------------------------------------------------------------------
# Research Verifier
# ---------------------------------------------------------------------------


class VerificationStatus(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


class FindingCategory(str, Enum):
    ARITHMETIC_ERROR = "ARITHMETIC_ERROR"
    UNIT_ERROR = "UNIT_ERROR"
    UNSUPPORTED_CLAIM = "UNSUPPORTED_CLAIM"
    SOURCE_MISMATCH = "SOURCE_MISMATCH"
    CONTRADICTION = "CONTRADICTION"
    OVERSTATED_INFERENCE = "OVERSTATED_INFERENCE"


class FindingSeverity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class ArithmeticOperation(str, Enum):
    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    DIVIDE = "divide"
    RATIO = "ratio"
    PERCENTAGE = "percentage"
    PERCENTAGE_CHANGE = "percentage_change"


class ArithmeticClaim(BaseModel):
    """A simple explicit calculation extracted for deterministic checking."""

    operation: ArithmeticOperation
    operands: list[float] = Field(
        description="Operands in calculation order, using only supplied evidence."
    )
    reported_result: float


class VerificationFinding(BaseModel):
    category: FindingCategory
    severity: FindingSeverity
    agent: str = Field(
        default="unknown_research_output",
        description="Research output containing the claim, such as bull_review_2."
    )
    claim: str
    evidence: str = ""
    correction: str = ""
    reason: str = ""
    arithmetic: list[ArithmeticClaim] = Field(
        default_factory=list,
        description=(
            "Machine-checkable arithmetic checks when the finding concerns "
            "explicit supported calculations; otherwise an empty list."
        ),
    )

    @field_validator("category", "severity", mode="before")
    @classmethod
    def _normalise_enum_casing(cls, value):
        # Case/whitespace only -- mirrors ResearchVerification._normalise_status
        # below. A JSON-fallback reconstruction (see research_verifier.py's
        # _parse_freetext_result) is otherwise semantically identical but
        # case-mismatched (e.g. "arithmetic_error" for ARITHMETIC_ERROR) and
        # must not be discarded wholesale over that alone. This is exact-name
        # equivalence, never a fuzzy/semantic mapping: an unrecognised name
        # (e.g. "BAD_MATH_THING") still fails validation normally.
        return value.strip().upper() if isinstance(value, str) else value

    @field_validator("arithmetic", mode="before")
    @classmethod
    def _normalise_arithmetic(cls, value):
        if value is None:
            return []
        return value if isinstance(value, list) else [value]


class ResearchVerification(BaseModel):
    """Quality-control result for the completed Bull/Bear research debate."""

    status: VerificationStatus
    findings: list[VerificationFinding] = Field(default_factory=list)
    verified_points: list[str] = Field(default_factory=list)
    repair_required: bool = False
    repair_targets: list[Literal["bull", "bear"]] = Field(default_factory=list)
    notes: str = ""

    @field_validator("status", mode="before")
    @classmethod
    def _normalise_status(cls, value):
        return value.upper() if isinstance(value, str) else value

    @field_validator("findings", mode="before")
    @classmethod
    def _normalise_findings(cls, value):
        if value is None:
            return []
        if isinstance(value, dict):
            return [value]
        return value

    @field_validator("verified_points", mode="before")
    @classmethod
    def _normalise_verified_points(cls, value):
        if value is None:
            return []
        values = value if isinstance(value, list) else [value]
        return [_verification_text(item) for item in values]

    @field_validator("notes", mode="before")
    @classmethod
    def _normalise_notes(cls, value):
        if value is None:
            return ""
        values = value if isinstance(value, list) else [value]
        return "\n".join(_verification_text(item) for item in values)

    @field_validator("repair_targets", mode="before")
    @classmethod
    def _normalise_repair_targets(cls, value):
        values = value if isinstance(value, list) else ([value] if value else [])
        targets = []
        for item in values:
            text = str(item).lower()
            for target in ("bull", "bear"):
                if target in text and target not in targets:
                    targets.append(target)
        return targets


def _verification_text(value) -> str:
    """Render a loose model-produced note/verified point as compact text."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        preferred = [
            str(value[key]).strip()
            for key in ("claim", "evidence", "reason", "calculation", "correction")
            if value.get(key) not in (None, "")
        ]
        if preferred:
            return " | ".join(preferred)
    return str(value)


# ---------------------------------------------------------------------------
# Bull/Bear structured trade hypothesis and collaborative review
# ---------------------------------------------------------------------------
#
# Bull and Bear each commit to a structured trade hypothesis at the start of
# the debate (InitialResearchThesis), then use ResearchReviewOutcome during
# cross-review to KEEP/REVISE/WITHDRAW its levels and track how conviction
# evolved. See the fork README ("Structured trade hypotheses", "Conviction
# semantics") for the behavioral rules these schemas encode.


class ResearchDirection(str, Enum):
    BULL = "BULL"
    BEAR = "BEAR"


class SuggestedRiskUnit(str, Enum):
    """A research-level signal of how much risk a thesis appears to justify.

    NOT a portfolio percentage, leverage, or position size -- see
    ``suggested_risk_unit_for`` and the fork README.
    """

    NO_TRADE = "NO_TRADE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


def suggested_risk_unit_for(conviction_total: int) -> SuggestedRiskUnit:
    """Deterministic, documented conviction -> suggested-risk-unit mapping.

    Conservative default: <50 NO_TRADE, 50-64 LOW, 65-79 MEDIUM, 80+ HIGH.
    Computed here rather than by the LLM so it can never drift from the
    conviction score it is derived from.
    """
    if conviction_total < 50:
        return SuggestedRiskUnit.NO_TRADE
    if conviction_total < 65:
        return SuggestedRiskUnit.LOW
    if conviction_total < 80:
        return SuggestedRiskUnit.MEDIUM
    return SuggestedRiskUnit.HIGH


class ConvictionScore(BaseModel):
    """Self-assessed evidence-strength score, 0-25 per component, 0-100 total.

    CRITICAL: this is NOT a calibrated probability that the market outcome
    will occur. It answers "how strongly does the supplied evidence support
    this researcher's own thesis?", nothing more.
    """

    evidence_quality: int = Field(
        ge=0, le=25,
        description="0-25: quality, specificity, and relevance of the supporting evidence.",
    )
    internal_consistency: int = Field(
        ge=0, le=25,
        description="0-25: the thesis does not contradict itself or the supplied evidence.",
    )
    robustness_to_challenge: int = Field(
        ge=0, le=25,
        description="0-25: how well the thesis withstands the strongest counterarguments seen so far.",
    )
    trade_plan_coherence: int = Field(
        ge=0, le=25,
        description="0-25: entry/target/stop/horizon form one internally consistent plan.",
    )
    total: int = Field(
        default=0, ge=0, le=100,
        description=(
            "Do not set this yourself: it is always recomputed as the sum of "
            "the four components above."
        ),
    )

    @model_validator(mode="after")
    def _derive_total(self):
        # Recomputed unconditionally so a model's arithmetic mistake here can
        # never desynchronize the total from its components (test: "Conviction
        # total is consistent with component scores").
        self.total = (
            self.evidence_quality
            + self.internal_consistency
            + self.robustness_to_challenge
            + self.trade_plan_coherence
        )
        return self


class EntryLevelType(str, Enum):
    POINT = "point"
    RANGE = "range"


class EntryLevel(BaseModel):
    """A committed entry hypothesis: either one price or a low/high range.

    Reviews and deterministic trade-metric calculations use the midpoint of a
    range as the single reference entry (documented default rule); see
    ``reference``.
    """

    type: EntryLevelType
    price: float | None = Field(default=None, description="Required when type is 'point'.")
    low: float | None = Field(default=None, description="Required when type is 'range'.")
    high: float | None = Field(default=None, description="Required when type is 'range'.")

    @field_validator("price", "low", "high", mode="before")
    @classmethod
    def _nullish_float_to_none(cls, v):
        return _coerce_optional_float(v)

    @property
    def reference(self) -> float | None:
        if self.type is EntryLevelType.POINT:
            return self.price
        if self.low is not None and self.high is not None:
            return (self.low + self.high) / 2
        return self.low if self.low is not None else self.high


class InitialResearchThesis(BaseModel):
    """Structured Bull/Bear initial trade hypothesis.

    Produced once per side at the start of the debate, independently of the
    other side. Later review rounds may KEEP/REVISE/WITHDRAW its entry/
    take_profit/stop_loss fields (see ``ResearchReviewOutcome``) but must not
    replace it with a materially different thesis built on new evidence.
    """

    direction: ResearchDirection
    conviction: ConvictionScore
    horizon: str = Field(
        description="Must match the shared research horizon supplied in the prompt."
    )
    entry: EntryLevel
    take_profit: float | None = Field(default=None)
    stop_loss: float | None = Field(default=None)
    thesis: str = Field(description="The primary thesis, 2-5 sentences.")
    evidence: list[str] = Field(
        default_factory=list, description="Key supporting evidence, from the supplied reports only."
    )
    risks: list[str] = Field(default_factory=list, description="Key risks to this thesis.")
    data_gaps: list[str] = Field(
        default_factory=list, description="Explicit gaps in the available evidence."
    )
    suggested_risk_unit: SuggestedRiskUnit = Field(
        default=SuggestedRiskUnit.NO_TRADE,
        description="Derived automatically from conviction; do not set this yourself.",
    )

    @field_validator("take_profit", "stop_loss", mode="before")
    @classmethod
    def _nullish_float_to_none(cls, v):
        return _coerce_optional_float(v)

    @field_validator("evidence", "risks", "data_gaps", mode="before")
    @classmethod
    def _coerce_list_fields(cls, v):
        return _coerce_str_list(v)

    @model_validator(mode="after")
    def _derive_suggested_risk_unit(self):
        self.suggested_risk_unit = suggested_risk_unit_for(self.conviction.total)
        return self


def render_initial_thesis(thesis: InitialResearchThesis) -> str:
    """Render a structured initial thesis to the markdown prose the rest of
    the system already treats the Bull/Bear output as (debate history,
    verifier, repair, Research Manager all read these fields as text)."""
    entry = thesis.entry
    if entry.type is EntryLevelType.RANGE:
        low = entry.low if entry.low is not None else "?"
        high = entry.high if entry.high is not None else "?"
        entry_text = f"{low} - {high} (range)"
    else:
        entry_text = str(entry.price) if entry.price is not None else "not set"

    lines = [
        f"**Direction**: {thesis.direction.value}",
        f"**Horizon**: {thesis.horizon}",
        "",
        "**Conviction** (self-assessed evidence-strength score, NOT a probability of the market outcome):",
        f"- Evidence quality: {thesis.conviction.evidence_quality}/25",
        f"- Internal consistency: {thesis.conviction.internal_consistency}/25",
        f"- Robustness to challenge: {thesis.conviction.robustness_to_challenge}/25",
        f"- Trade-plan coherence: {thesis.conviction.trade_plan_coherence}/25",
        f"- Total: {thesis.conviction.total}/100",
        "",
        f"**Entry**: {entry_text}",
        f"**Take Profit**: {thesis.take_profit if thesis.take_profit is not None else 'not set'}",
        f"**Stop Loss**: {thesis.stop_loss if thesis.stop_loss is not None else 'not set'}",
        f"**Suggested Risk Unit**: {thesis.suggested_risk_unit.value} "
        "(research-level signal only; not a position size or portfolio percentage)",
        "",
        f"**Thesis**: {thesis.thesis}",
    ]
    if thesis.evidence:
        lines += ["", "**Key Evidence**:"] + [f"- {item}" for item in thesis.evidence]
    if thesis.risks:
        lines += ["", "**Key Risks**:"] + [f"- {item}" for item in thesis.risks]
    if thesis.data_gaps:
        lines += ["", "**Data Gaps**:"] + [f"- {item}" for item in thesis.data_gaps]
    return "\n".join(lines)


class TradePlanAction(str, Enum):
    KEEP = "KEEP"
    REVISE = "REVISE"
    WITHDRAW = "WITHDRAW"


class TradePlanFieldChange(BaseModel):
    action: TradePlanAction = TradePlanAction.KEEP
    old: float | None = None
    new: float | None = None
    reason: str = Field(
        default="",
        description="Required for REVISE/WITHDRAW: why, using only already-supplied evidence.",
    )

    @field_validator("old", "new", mode="before")
    @classmethod
    def _nullish_float_to_none(cls, v):
        return _coerce_optional_float(v)


class TradePlanChanges(BaseModel):
    entry: TradePlanFieldChange = Field(default_factory=TradePlanFieldChange)
    take_profit: TradePlanFieldChange = Field(default_factory=TradePlanFieldChange)
    stop_loss: TradePlanFieldChange = Field(default_factory=TradePlanFieldChange)


class NewDataException(BaseModel):
    """A narrow, explicit exception to the "no new data in reviews" rule.

    Reserved for when a new datum is genuinely necessary to correct a
    material factual misunderstanding; see the fork README.
    """

    datum: str
    reason: str
    source: str = ""


class ResearchReviewOutcome(BaseModel):
    """Structured Bull/Bear cross-review output for one round.

    Normal reviews defend, refute, correct, accept, or withdraw claims
    already present in the initial theses; they must not introduce new
    market facts, sources, targets, thresholds, or catalysts except through
    an explicit, justified ``new_data_exceptions`` entry.
    """

    accepted: list[str] = Field(default_factory=list)
    rejected: list[str] = Field(default_factory=list)
    unresolved: list[str] = Field(default_factory=list)
    arithmetic_corrections: list[str] = Field(default_factory=list)
    trade_plan_changes: TradePlanChanges = Field(default_factory=TradePlanChanges)
    conviction_before: int = Field(ge=0, le=100)
    conviction_after: int = Field(ge=0, le=100)
    conviction_reason: str = Field(
        default="",
        description="Required when conviction changes materially: the evidence that changed it.",
    )
    new_data_exceptions: list[NewDataException] = Field(default_factory=list)
    remaining_disagreements: list[str] = Field(default_factory=list)

    @field_validator(
        "accepted", "rejected", "unresolved", "arithmetic_corrections", "remaining_disagreements",
        mode="before",
    )
    @classmethod
    def _normalise_text_list(cls, value):
        return [_verification_text(item) for item in _coerce_str_list(value)]

    @field_validator("new_data_exceptions", mode="before")
    @classmethod
    def _normalise_new_data_exceptions(cls, value):
        return _coerce_str_list(value)


def render_review_outcome(outcome: ResearchReviewOutcome) -> str:
    """Render a structured review to markdown, preserving the ACCEPTED/
    REJECTED/UNRESOLVED line format the debate history already parses
    (``researchers/review_outcomes.py:extract_review_outcomes``)."""
    lines: list[str] = []
    for item in outcome.accepted:
        lines.append(f"ACCEPTED: {item}")
    for item in outcome.rejected:
        lines.append(f"REJECTED: {item}")
    for item in outcome.unresolved:
        lines.append(f"UNRESOLVED: {item}")
    if not (outcome.accepted or outcome.rejected or outcome.unresolved):
        lines.append("No material disagreements to classify this round.")

    if outcome.arithmetic_corrections:
        lines += ["", "**Arithmetic Corrections**:"]
        lines += [f"- {item}" for item in outcome.arithmetic_corrections]

    lines += ["", "**Trade Plan Changes**:"]
    for label, change in (
        ("Entry", outcome.trade_plan_changes.entry),
        ("Take Profit", outcome.trade_plan_changes.take_profit),
        ("Stop Loss", outcome.trade_plan_changes.stop_loss),
    ):
        detail = f"- {label}: {change.action.value}"
        if change.action is TradePlanAction.REVISE:
            detail += f" ({change.old} -> {change.new})"
        if change.reason:
            detail += f" Reason: {change.reason}"
        lines.append(detail)

    lines += [
        "",
        f"**Conviction**: {outcome.conviction_before} -> {outcome.conviction_after} "
        "(self-assessed evidence strength, not a probability)",
    ]
    if outcome.conviction_reason:
        lines.append(f"Reason: {outcome.conviction_reason}")

    if outcome.new_data_exceptions:
        lines += ["", "**NEW_DATA_EXCEPTION**:"]
        for exc in outcome.new_data_exceptions:
            lines.append(f"- {exc.datum} -- {exc.reason} (source: {exc.source})")

    if outcome.remaining_disagreements:
        lines += ["", "**Remaining Disagreements**:"]
        lines += [f"- {item}" for item in outcome.remaining_disagreements]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Canonical recovery -- minimal patch contracts for a Research textual
# fallback (see ``researchers/canonical_recovery.py``).
#
# Deliberately much smaller than InitialResearchThesis/ResearchReviewOutcome:
# these only carry the fields the rest of the system treats as canonical
# state (bull_thesis/bear_thesis, conviction, trade-plan levels). Narrative
# fields (thesis prose, evidence, risks, data_gaps, accepted/rejected lists,
# conviction_reason, ...) are not reconstructed -- they stay in the free-text
# fallback for audit, never promoted to canonical state.
# ---------------------------------------------------------------------------


class InitialThesisCanonicalPatch(BaseModel):
    """Minimal recovery contract for an INITIAL thesis fallback: just enough
    to create a valid canonical ``bull_thesis``/``bear_thesis`` entry.

    ``suggested_risk_unit`` is optional and deliberately NOT derived from
    ``conviction_total`` the way the normal structured path derives it (see
    ``suggested_risk_unit_for``): a fallback researcher is explicitly told
    not to choose this field itself (see the bull/bear initial prompts), so
    it is normally simply absent from the text, and "high conviction" must
    never be read as "HIGH research risk unit". ``canonical_recovery.py``
    only accepts this value when the fallback text itself explicitly labels
    it (e.g. "Suggested Risk Unit: MEDIUM"); otherwise the canonical value is
    "not recorded", regardless of what this field says.
    """

    direction: ResearchDirection
    conviction_total: int = Field(ge=0, le=100)
    entry: EntryLevel
    take_profit: float | None = None
    stop_loss: float | None = None
    suggested_risk_unit: SuggestedRiskUnit | None = Field(default=None)

    @field_validator("take_profit", "stop_loss", mode="before")
    @classmethod
    def _nullish_float_to_none(cls, v):
        return _coerce_optional_float(v)


class ReviewCanonicalPatch(BaseModel):
    """Minimal recovery contract for a REVIEW fallback: a patch over the
    prior canonical thesis (KEEP/REVISE/WITHDRAW + the new conviction), not a
    reconstruction of the whole thesis."""

    conviction_after: int = Field(ge=0, le=100)
    trade_plan_changes: TradePlanChanges = Field(default_factory=TradePlanChanges)


# ---------------------------------------------------------------------------
# Research Manager
# ---------------------------------------------------------------------------


class ResearchPlan(BaseModel):
    """Structured investment plan produced by the Research Manager.

    Hand-off to the Trader: the recommendation pins the directional view,
    the rationale captures which side of the bull/bear debate carried the
    argument, and the strategic actions translate that into concrete
    instructions the trader can execute against.
    """

    recommendation: PortfolioRating = Field(
        description=(
            "The investment recommendation. Exactly one of Buy / Overweight / "
            "Hold / Underweight / Sell. Conflicting arguments alone are not a "
            "reason to Hold: use the strength of the evidence only to select the "
            "appropriate rating on this scale. Do not translate decisiveness into "
            "portfolio size, allocation, leverage, or notional. Choose Hold only "
            "when the evidence is still balanced after weighing, or too thin to "
            "support a call."
        ),
    )
    rationale: str = Field(
        description=(
            "Conversational summary of the key points from both sides of the "
            "debate, ending with which arguments led to the recommendation. "
            "Speak naturally, as if to a teammate."
        ),
    )
    strategic_actions: str = Field(
        description=(
            "Concrete steps for the trader to implement the recommendation, using "
            "the levels and deterministic metrics already computed above. The "
            "research team does not see the caller's holdings and must not invent "
            "a position-sizing percentage, notional amount, or portfolio-loss "
            "tolerance -- actual position sizing is the trader/portfolio manager's "
            "job, against the real portfolio context they have and this research "
            "layer does not."
        ),
    )


def render_research_plan(plan: ResearchPlan) -> str:
    """Render a ResearchPlan to markdown for storage and the trader's prompt context."""
    return "\n".join([
        f"**Recommendation**: {plan.recommendation.value}",
        "",
        f"**Rationale**: {plan.rationale}",
        "",
        f"**Strategic Actions**: {plan.strategic_actions}",
    ])


# ---------------------------------------------------------------------------
# Trader
# ---------------------------------------------------------------------------


class TraderProposal(BaseModel):
    """Structured transaction proposal produced by the Trader.

    The trader reads the Research Manager's investment plan and the analyst
    reports, then turns them into a concrete transaction: what action to
    take, the reasoning that justifies it, and the practical levels for
    entry, stop-loss, and sizing.
    """

    action: TraderAction = Field(
        description="The transaction direction. Exactly one of Buy / Hold / Sell.",
    )
    reasoning: str = Field(
        description=(
            "The case for this action, anchored in the analysts' reports and "
            "the research plan. Two to four sentences."
        ),
    )
    entry_price: float | None = Field(
        default=None,
        description=(
            "Single absolute entry price in the instrument's quote currency (e.g. 189.5), "
            "never a percentage. Use this ONLY when the verified research plan's entry is "
            "itself a single point (or you are deliberately picking one specific level). "
            "When the verified plan's entry is a RANGE, leave this unset and use "
            "entry_price_low/entry_price_high instead -- do not collapse a verified range "
            "into an invented single price here. Omit entirely if you cannot state a level."
        ),
    )
    entry_price_low: float | None = Field(
        default=None,
        description=(
            "Lower bound of a verified entry RANGE, as an absolute price. Set this (with "
            "entry_price_high) only when the verified research plan's entry is itself a "
            "range; do not invent a range when the verified plan has a single entry."
        ),
    )
    entry_price_high: float | None = Field(
        default=None,
        description="Upper bound of a verified entry RANGE; see entry_price_low.",
    )
    reference_entry: float | None = Field(
        default=None,
        description=(
            "Optional: the deterministic reference entry already computed in code (e.g. "
            "the midpoint of a range) if you want to cite it explicitly as a calculation. "
            "This is context, never a silent substitute for the actual entry/entry range "
            "above -- do not average a range into this unless the verified plan explicitly "
            "asks for that calculation."
        ),
    )
    stop_loss: float | None = Field(
        default=None,
        description=(
            "Optional stop-loss as an absolute price in the instrument's quote "
            "currency (e.g. 172.0), never a percentage. Convert a percentage "
            "distance to the price level it implies, or omit it."
        ),
    )
    position_sizing: str | None = Field(
        default=None,
        description=(
            "Optional sizing guidance. State a concrete percentage or notional "
            "amount ONLY if the supplied portfolio context gives the denominator "
            "(actual holdings/cash) needed to compute it; otherwise state that "
            "sizing is unknown/not determinable rather than inventing a figure. "
            "LOW/MEDIUM/HIGH research risk classification is not a position size."
        ),
    )

    @field_validator(
        "entry_price", "entry_price_low", "entry_price_high", "reference_entry", "stop_loss",
        mode="before",
    )
    @classmethod
    def _nullish_float_to_none(cls, v):
        return _coerce_optional_float(v)


def render_trader_proposal(proposal: TraderProposal) -> str:
    """Render a TraderProposal to markdown.

    The trailing ``FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**`` line is
    preserved for backward compatibility with the analyst stop-signal text
    and any external code that greps for it.
    """
    parts = [
        f"**Action**: {proposal.action.value}",
        "",
        f"**Reasoning**: {proposal.reasoning}",
    ]
    # A verified range stays a range (both bounds), never collapsed into an
    # invented midpoint; a single verified/chosen entry stays a single price.
    if proposal.entry_price_low is not None or proposal.entry_price_high is not None:
        low = proposal.entry_price_low if proposal.entry_price_low is not None else "?"
        high = proposal.entry_price_high if proposal.entry_price_high is not None else "?"
        entry_text = f"{low} - {high}"
    elif proposal.entry_price is not None:
        entry_text = proposal.entry_price
    else:
        entry_text = "not provided"
    parts.extend(["", f"**Entry Price**: {entry_text}"])
    if proposal.reference_entry is not None:
        parts.extend(["", f"**Reference Entry (calculated)**: {proposal.reference_entry}"])
    # Named even when absent, so a reader can tell a level the trader chose not
    # to give from one the schema never asked for.
    for label, value in (("Stop Loss", proposal.stop_loss),
                         ("Position Sizing", proposal.position_sizing)):
        parts.extend(["", f"**{label}**: {value if value is not None and value != '' else 'not provided'}"])
    parts.extend([
        "",
        f"FINAL TRANSACTION PROPOSAL: **{proposal.action.value.upper()}**",
    ])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Portfolio Manager
# ---------------------------------------------------------------------------


class RiskDisposition(str, Enum):
    """A disposition toward the frozen Trader plan -- never a replacement plan
    of its own. Used both by each Risk reviewer's own assessment and by the
    Portfolio Manager's final decision about the Trader's plan specifically
    (a separate concept from ``PortfolioRating``, the investment/directional
    view -- see ``PortfolioDecision.disposition``)."""

    KEEP = "KEEP"
    DEFER = "DEFER"
    REDUCE_RISK = "REDUCE_RISK"
    REJECT_PLAN = "REJECT_PLAN"


class PortfolioDecision(BaseModel):
    """Structured output produced by the Portfolio Manager.

    The model fills every field as part of its primary LLM call; no separate
    extraction pass is required. Field descriptions double as the model's
    output instructions, so the prompt body only needs to convey context and
    the rating-scale guidance.
    """

    rating: PortfolioRating = Field(
        description=(
            "The final position rating: the investment / directional allocation "
            "view. Exactly one of Buy / Overweight / Hold / Underweight / Sell, "
            "picked based on the analysts' debate. Conflicting arguments alone "
            "are not a reason to Hold: use the strength of the evidence only to "
            "select the appropriate rating on this scale. Do not translate "
            "decisiveness into portfolio size, allocation, leverage, or notional. "
            "Choose Hold only when the evidence is still balanced after "
            "weighing, or too thin to support a call."
        ),
    )
    disposition: RiskDisposition = Field(
        description=(
            "What you decide about the TRADER'S PLAN AS CURRENTLY WRITTEN -- a "
            "separate concept from `rating` (the investment view), not derived "
            "from it. KEEP: execute the plan as written. DEFER: hold the "
            "investment view but do not execute this plan yet (wait for more "
            "evidence or a repaired plan). REDUCE_RISK: the plan is directionally "
            "sound but needs less size or tighter protection before executing. "
            "REJECT_PLAN: this plan should not be executed as proposed. A "
            "bullish `rating` can still pair with DEFER or REDUCE_RISK -- e.g. "
            "Overweight+DEFER means a favorable view but do not execute the "
            "current plan until it is repaired; Overweight+REDUCE_RISK means a "
            "favorable view but this specific execution plan needs less risk. "
            "Set this from your own judgment of the plan's executability, never "
            "by mechanically mapping it from `rating`."
        ),
    )
    executive_summary: str = Field(
        description=(
            "A concise action plan covering entry strategy, key risk levels, and time "
            "horizon. Two to four sentences. State position sizing only if the supplied "
            "portfolio context gives the actual holdings/cash needed to compute it; "
            "otherwise do not invent a percentage or notional amount."
        ),
    )
    investment_thesis: str = Field(
        description=(
            "Detailed reasoning anchored in specific evidence from the analysts' "
            "debate. If prior lessons are referenced in the prompt context, "
            "incorporate them; otherwise rely solely on the current analysis."
        ),
    )
    price_target: float | None = Field(
        default=None,
        description="Optional target price in the instrument's quote currency.",
    )
    time_horizon: str | None = Field(
        default=None,
        description="Optional recommended holding period, e.g. '3-6 months'.",
    )

    @field_validator("price_target", mode="before")
    @classmethod
    def _nullish_float_to_none(cls, v):
        return _coerce_optional_float(v)


def render_pm_decision(decision: PortfolioDecision) -> str:
    """Render a PortfolioDecision back to the markdown shape the rest of the system expects.

    Memory log, CLI display, and saved report files all read this markdown,
    so the rendered output preserves the exact section headers (``**Rating**``,
    ``**Executive Summary**``, ``**Investment Thesis**``) that downstream
    parsers and the report writers already handle.
    """
    parts = [
        f"**Rating**: {decision.rating.value}",
        "",
        f"**Disposition**: {decision.disposition.value}",
        "",
        f"**Executive Summary**: {decision.executive_summary}",
        "",
        f"**Investment Thesis**: {decision.investment_thesis}",
    ]
    # Named even when absent: a missing line reads as a field nobody asked for,
    # so a reader cannot tell "no target" from "target not reported".
    target = decision.price_target if decision.price_target is not None else "not provided"
    parts.extend(["", f"**Price Target**: {target}"])
    parts.extend(["", f"**Time Horizon**: {decision.time_horizon or 'not provided'}"])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Risk Management debate
# ---------------------------------------------------------------------------


class RiskStanceAssessment(BaseModel):
    """Structured output for one Risk reviewer's independent assessment or
    cross-review of the frozen Trader plan.

    Risk reviewers evaluate the plan; they do not design a new one. There is
    deliberately no field for a new entry/stop/target/threshold/sizing
    percentage here -- see ``plan_change_required`` for what to do instead
    when the plan genuinely needs a change that cannot be expressed with
    already-verified levels.
    """

    risk_level: Literal["LOW", "MEDIUM", "HIGH"] = Field(
        description=(
            "This reviewer's own assessment of how risky the TRADER's current "
            "plan is, from their assigned perspective -- not a label for "
            "their own archetype."
        )
    )
    disposition: RiskDisposition = Field(
        description=(
            "Recommended disposition toward the Trader's plan: KEEP (no change "
            "needed), DEFER (wait -- evidence is not yet sufficient either way), "
            "REDUCE_RISK (directionally sound but oversized or under-protected), "
            "or REJECT_PLAN (should not be taken as proposed). Not a vote; a "
            "reviewer may conclude KEEP even from a cautious perspective, or "
            "REJECT_PLAN even from an opportunity-seeking one."
        )
    )
    supported_points: list[str] = Field(
        default_factory=list,
        description="Specific points from the frozen evidence or Trader plan this reviewer finds well supported.",
    )
    unresolved_risks: list[str] = Field(
        default_factory=list,
        description=(
            "Risks identified in the frozen evidence, each prefixed with exactly one of "
            "IDENTIFIED_RISK / ALREADY_CONTROLLED / UNRESOLVED / PLAN_CHANGE_REQUIRED. "
            "Do not default to PLAN_CHANGE_REQUIRED merely to seem thorough."
        ),
    )
    corrected_or_withdrawn_claims: list[str] = Field(
        default_factory=list,
        description="A claim this reviewer is correcting or withdrawing from their own earlier turn this run, if any.",
    )
    unsupported_claims_seen: list[str] = Field(
        default_factory=list,
        description=(
            "A specific unsupported number, threshold, sizing rule, historical analogy, or "
            "probability noticed in another reviewer's assessment during cross-review -- "
            "named so it is not silently repeated as if it were valid. Empty during the "
            "independent phase, when no other assessment is visible yet."
        ),
    )
    plan_change_required: bool = Field(
        default=False,
        description=(
            "True only when the Trader plan needs a change that cannot be expressed using "
            "already-verified levels. When true, explain why in rationale rather than "
            "inventing the replacement level yourself."
        ),
    )
    rationale: str = Field(
        description="Concise rationale, 1-3 sentences. No rhetoric, no restating the full case."
    )

    @field_validator(
        "supported_points", "unresolved_risks", "corrected_or_withdrawn_claims",
        "unsupported_claims_seen",
        mode="before",
    )
    @classmethod
    def _normalise_text_list(cls, value):
        return [_verification_text(item) for item in _coerce_str_list(value)]


def render_risk_assessment(assessment: RiskStanceAssessment) -> str:
    """Render one Risk reviewer's structured turn to readable markdown.

    Composes a full-text block from the structured fields (no separate prose
    "argument" blob) so the saved report keeps full text for audit while the
    structure itself keeps the model from inventing a new trading system
    inside free-form rhetoric.
    """
    lines = [
        f"Risk level: {assessment.risk_level}",
        f"Disposition: {assessment.disposition.value}",
    ]
    if assessment.supported_points:
        lines += ["Supported points:"] + [f"- {item}" for item in assessment.supported_points]
    if assessment.unresolved_risks:
        lines += ["Risks:"] + [f"- {item}" for item in assessment.unresolved_risks]
    if assessment.corrected_or_withdrawn_claims:
        lines += ["Corrected/withdrawn:"] + [
            f"- {item}" for item in assessment.corrected_or_withdrawn_claims
        ]
    if assessment.unsupported_claims_seen:
        lines += ["Unsupported claims seen in other assessments:"] + [
            f"- {item}" for item in assessment.unsupported_claims_seen
        ]
    lines.append(f"Plan change required: {'yes' if assessment.plan_change_required else 'no'}")
    lines.append(f"Rationale: {assessment.rationale}")
    return "\n".join(lines)


def render_risk_stance_summary(risk_debate_state: dict) -> str:
    """Each risk reviewer's latest LOW/MEDIUM/HIGH level and disposition, for
    the Portfolio Manager prompt and the saved report."""
    lines = []
    for label, level_key, disposition_key in (
        ("Aggressive", "aggressive_risk_level", "aggressive_disposition"),
        ("Conservative", "conservative_risk_level", "conservative_disposition"),
        ("Neutral", "neutral_risk_level", "neutral_disposition"),
    ):
        level = risk_debate_state.get(level_key) or "not recorded"
        disposition = risk_debate_state.get(disposition_key) or "not recorded"
        lines.append(f"- {label}: risk={level}, disposition={disposition}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Sentiment Analyst
# ---------------------------------------------------------------------------


class SentimentBand(str, Enum):
    """Discrete sentiment direction produced by the Sentiment Analyst.

    Six tiers keep the signal granular enough to be actionable while remaining
    small enough for every provider to map reliably from its JSON output.
    """

    BULLISH = "Bullish"
    MILDLY_BULLISH = "Mildly Bullish"
    NEUTRAL = "Neutral"
    MIXED = "Mixed"
    MILDLY_BEARISH = "Mildly Bearish"
    BEARISH = "Bearish"


# Per-band midpoint of the guideline ranges in SentimentReport.overall_score's
# own description, used only to recover a missing score when the band itself
# was supplied -- never to override a score the model did provide.
_SENTIMENT_BAND_SCORE_MIDPOINT = {
    "bullish": 8.0,
    "mildly bullish": 6.0,
    "neutral": 5.0,
    "mixed": 5.0,
    "mildly bearish": 4.0,
    "bearish": 2.0,
}


class SentimentScoreSource(str, Enum):
    """Where ``SentimentReport.overall_score`` actually came from.

    Always derived deterministically in code (never trusted from the model),
    the same principle already applied to ``ConvictionScore.total`` and
    ``InitialResearchThesis.suggested_risk_unit``: a downstream consumer must
    be able to tell a model-provided score from one this application filled
    in by default, so a midpoint estimate never gets presented as the
    model's own judgment.
    """

    MODEL = "MODEL"
    INFERRED_FROM_BAND = "INFERRED_FROM_BAND"


class SentimentReport(BaseModel):
    """Structured sentiment report produced by the Sentiment Analyst.

    Replaces the previous free-form prose output so downstream consumers
    (dashboards, audit logs, PDF renderers, other agents) can read
    ``overall_band`` and ``overall_score`` without maintaining fragile regex
    fallbacks that drift with every model release. ``narrative`` preserves the
    rich source-by-source analysis; ``render_sentiment_report`` prepends a
    deterministic header so the saved report stays human-readable.
    """

    overall_band: SentimentBand = Field(
        description=(
            "Overall sentiment direction. Exactly one of: "
            "Bullish / Mildly Bullish / Neutral / Mixed / Mildly Bearish / Bearish. "
            "Use Mixed when sources point in clearly different directions. "
            "Use Neutral only when all sources are genuinely silent or non-committal."
        ),
    )
    overall_score: float = Field(
        ge=0.0,
        le=10.0,
        description=(
            "Numeric sentiment intensity on a 0–10 scale. "
            "0 = maximally bearish, 5 = neutral, 10 = maximally bullish. "
            "Guideline for consistency with overall_band: "
            "Bullish ~6.5–10, Mildly Bullish ~5.5–6.4, Neutral/Mixed ~4.5–5.5, "
            "Mildly Bearish ~3.5–4.4, Bearish ~0–3.4. "
            "Only the 0–10 bounds are enforced."
        ),
    )
    confidence: Literal["low", "medium", "high"] = Field(
        description=(
            "Confidence in the assessment based on data quality and sample size. "
            "Use 'low' when one or more sources returned a placeholder or fewer "
            "than 5 data points; 'medium' when data is present but sparse; "
            "'high' when all three sources returned substantive data."
        ),
    )
    narrative: str = Field(
        description=(
            "Full sentiment report covering, in order: "
            "(1) source-by-source breakdown with specific evidence (cite message "
            "counts, ratios, notable posts); "
            "(2) cross-source divergences and alignments; "
            "(3) dominant narrative themes; "
            "(4) catalysts and risks surfaced by the data; "
            "(5) a markdown table summarising key sentiment signals, their "
            "direction, source, and supporting evidence. "
            "Keep it informative and substantive: develop each section thoroughly "
            "with concrete evidence so every point adds new signal for the trader."
        ),
    )
    score_source: SentimentScoreSource = Field(
        default=SentimentScoreSource.MODEL,
        description="Derived automatically; do not set this yourself.",
    )

    @model_validator(mode="before")
    @classmethod
    def _recover_loose_shape(cls, data):
        """Recover from two shapes seen in real runs with a weaker model:
        ``band`` instead of ``overall_band``, and a missing ``overall_score``
        (observed together in a live batch run, 2026-10-01). A band-only
        report still carries real signal; losing it entirely to a missing
        numeric field would discard a correct narrative and direction over a
        guideline number the field description already pins per band.

        ``overall_score`` is checked with ``is None``, not truthiness: a
        falsy-but-valid ``0`` (maximally bearish) must survive unchanged, not
        be mistaken for "missing" and overwritten by the band midpoint.
        ``score_source`` is likewise always set here, in code, rather than
        trusted from the model -- the same principle already applied to
        ``ConvictionScore.total`` -- so an inferred score can never be
        represented as the model's own judgment.
        """
        if not isinstance(data, dict):
            return data
        data = dict(data)
        if "overall_band" not in data and "band" in data:
            data["overall_band"] = data.pop("band")
        if data.get("overall_score") is None:
            band_key = str(data.get("overall_band", "")).strip().lower()
            midpoint = _SENTIMENT_BAND_SCORE_MIDPOINT.get(band_key)
            if midpoint is not None:
                data["overall_score"] = midpoint
                data["score_source"] = SentimentScoreSource.INFERRED_FROM_BAND.value
                return data
        data["score_source"] = SentimentScoreSource.MODEL.value
        return data


def render_sentiment_report(report: SentimentReport) -> str:
    """Render a SentimentReport to the markdown shape the rest of the system expects.

    The structured header (band + score + confidence) is prepended to the
    narrative so the saved report is both human-readable and machine-parseable
    without regex.
    """
    return "\n".join([
        f"**Overall Sentiment:** **{report.overall_band.value}** "
        f"(Score: {report.overall_score:.1f}/10)",
        f"**Confidence:** {report.confidence.capitalize()}",
        "",
        report.narrative,
    ])
