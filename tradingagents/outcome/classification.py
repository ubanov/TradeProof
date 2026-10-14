"""Pure classification of an already-built RecommendationRecord's trade plan.

This is the FIRST block of the future RecommendationOutcome work. Explicitly
NOT implemented here (future work, separate modules): pricing, horizon
dates, execution simulation, MFE/MAE, benchmark comparison, storage, and a
CLI `evaluate` command. This module only answers one question -- "is there
a plan this record actually committed to executing, and if not, why not" --
derived exclusively from the record's own structured fields.

Authority rule (see fork README: "RecommendationRecord"):
``PortfolioDecision.disposition`` is the sole gate on whether the Trader's
plan executes. ``TraderProposal``/``trade_plan`` describes WHAT would
execute if the gate is KEEP. ``PortfolioDecision.rating`` (the investment/
directional view, a separate concept) and the legacy, regex-derived
``result["action"]``/``result["rating"]`` text fields are NEVER consulted --
this module takes a ``RecommendationRecord`` and reads only
``record.decision``/``record.trade_plan``. No LLM call, no market data, no
Markdown, no filesystem, no logging, no globals: same record in always
produces the same classification out.

Works identically for schema 1.0 and 1.1 records: it never reads
``price_basis`` (added in 1.1) or anything else outside ``decision``/
``trade_plan``, both present since schema 1.0.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from tradingagents.recommendation_record import RecommendationRecord, TradePlanRef


class PlanType(str, Enum):
    """Deliberately only two values for v1 -- see module docstring's scope
    note and the task that introduced this module: a third category
    (e.g. a position-management or decision-only state) was considered and
    rejected because every real case so far models cleanly as NO_TRADE plus
    a specific ``NonExecutionReason``."""

    EXECUTABLE_TRADE_PLAN = "EXECUTABLE_TRADE_PLAN"
    NO_TRADE = "NO_TRADE"


class Direction(str, Enum):
    """The Trader's proposed orientation -- NOT, by itself, whether the plan
    executes (see ``classify_recommendation``'s own docstring: a REJECT_PLAN
    Buy still reports ``LONG`` here, so a future signal-quality metric can
    still see what direction was proposed even though nothing executed)."""

    LONG = "LONG"
    SHORT = "SHORT"
    NONE = "NONE"


class NonExecutionReason(str, Enum):
    TRADER_HOLD = "TRADER_HOLD"
    DEFERRED = "DEFERRED"
    REQUIRES_MODIFICATION = "REQUIRES_MODIFICATION"
    REJECTED = "REJECTED"
    INCOMPLETE_PLAN = "INCOMPLETE_PLAN"
    # Not in the task's original four -- added because trade_plan.action or
    # decision.portfolio_disposition can be missing/unrecognized (a record
    # with unexpectedly null structured fields; see module docstring on
    # "never invent"). Distinct from INCOMPLETE_PLAN, which is specifically
    # a KEEP + Buy/Sell with no usable entry -- this is "the gate itself
    # could not be read", a different failure.
    INCOMPLETE_DECISION = "INCOMPLETE_DECISION"


class EntryKind(str, Enum):
    EXACT = "EXACT"
    RANGE = "RANGE"
    NONE = "NONE"


class ClassificationWarning(str, Enum):
    """Small, closed set of codes for entry-field ambiguities worth flagging
    without justifying a whole new state -- never free text."""

    # Exactly one of entry_price_low/entry_price_high is set.
    PARTIAL_ENTRY_RANGE = "PARTIAL_ENTRY_RANGE"
    # entry_price_low > entry_price_high.
    INVERTED_ENTRY_RANGE = "INVERTED_ENTRY_RANGE"
    # entry_price AND a (valid) entry_price_low/high range are both set;
    # resolved by preferring the single point (see _classify_entry).
    ENTRY_POINT_AND_RANGE_BOTH_SET = "ENTRY_POINT_AND_RANGE_BOTH_SET"


@dataclass(frozen=True)
class OutcomeClassification:
    plan_type: PlanType
    direction: Direction
    non_execution_reason: NonExecutionReason | None
    entry_kind: EntryKind
    classification_warnings: tuple[ClassificationWarning, ...] = ()


# The Trader's three possible actions (TraderAction's own string values),
# mapped to the direction they propose -- Hold proposes no orientation.
_ACTION_TO_DIRECTION: dict[str, Direction] = {
    "Buy": Direction.LONG,
    "Sell": Direction.SHORT,
    "Hold": Direction.NONE,
}

# RiskDisposition's own string values -- the only ones PortfolioDecision can
# produce. Anything else means the field could not be read as intended.
_VALID_DISPOSITIONS = frozenset({"KEEP", "DEFER", "REDUCE_RISK", "REJECT_PLAN"})

_DISPOSITION_TO_REASON: dict[str, NonExecutionReason] = {
    "DEFER": NonExecutionReason.DEFERRED,
    "REDUCE_RISK": NonExecutionReason.REQUIRES_MODIFICATION,
    "REJECT_PLAN": NonExecutionReason.REJECTED,
}


def _direction_for_action(action: str | None) -> Direction:
    return _ACTION_TO_DIRECTION.get(action, Direction.NONE)


def _classify_entry(
    trade_plan: TradePlanRef,
) -> tuple[EntryKind, tuple[ClassificationWarning, ...]]:
    """Whether ``trade_plan`` carries a usable entry, and how.

    Usable: a single ``entry_price``, or a valid (``low <= high``)
    ``entry_price_low``/``entry_price_high`` pair. Never inferred from
    ``reference_entry`` -- audited (see ``TraderProposal.reference_entry``'s
    own field description): it is documented as context the Trader may cite
    alongside a real entry, explicitly "never a silent substitute for the
    actual entry/entry range", so it carries no independent entry
    information this function can trust on its own. Never inferred from
    stop_loss, take_profit, market price, or research metrics either --
    those are out of scope for what "entry" means here.
    """
    point = trade_plan.entry_price
    low = trade_plan.entry_price_low
    high = trade_plan.entry_price_high
    has_point = point is not None
    has_low = low is not None
    has_high = high is not None

    if has_point and has_low and has_high:
        if low <= high:
            return EntryKind.EXACT, (ClassificationWarning.ENTRY_POINT_AND_RANGE_BOTH_SET,)
        # The point is still usable on its own; the invalid range alongside
        # it is a second, independent problem worth flagging too.
        return EntryKind.EXACT, (
            ClassificationWarning.ENTRY_POINT_AND_RANGE_BOTH_SET,
            ClassificationWarning.INVERTED_ENTRY_RANGE,
        )
    if has_point:
        return EntryKind.EXACT, ()
    if has_low and has_high:
        if low > high:
            return EntryKind.NONE, (ClassificationWarning.INVERTED_ENTRY_RANGE,)
        return EntryKind.RANGE, ()
    if has_low or has_high:
        return EntryKind.NONE, (ClassificationWarning.PARTIAL_ENTRY_RANGE,)
    return EntryKind.NONE, ()


def classify_recommendation(record: RecommendationRecord) -> OutcomeClassification:
    """Classify ``record``'s trade plan from its structured fields alone.

    See the module docstring for the authority rule and scope. A
    REJECT_PLAN/DEFER/REDUCE_RISK Buy or Sell still reports its proposed
    ``direction`` -- it is simply paired with ``plan_type=NO_TRADE`` and the
    matching ``non_execution_reason``, preserving the signal for any future
    metric that wants to look at proposed-but-not-executed direction without
    treating it as if it had been traded.
    """
    action = record.trade_plan.action
    disposition = record.decision.portfolio_disposition
    direction = _direction_for_action(action)

    if action not in _ACTION_TO_DIRECTION or disposition not in _VALID_DISPOSITIONS:
        return OutcomeClassification(
            plan_type=PlanType.NO_TRADE,
            direction=direction,
            non_execution_reason=NonExecutionReason.INCOMPLETE_DECISION,
            entry_kind=EntryKind.NONE,
        )

    if disposition in _DISPOSITION_TO_REASON:
        return OutcomeClassification(
            plan_type=PlanType.NO_TRADE,
            direction=direction,
            non_execution_reason=_DISPOSITION_TO_REASON[disposition],
            entry_kind=EntryKind.NONE,
        )

    # disposition == "KEEP" from here on -- the only remaining valid value.
    if action == "Hold":
        # direction is already Direction.NONE for "Hold" (see
        # _direction_for_action); reusing it rather than a literal keeps
        # there being exactly one place that maps action -> direction.
        return OutcomeClassification(
            plan_type=PlanType.NO_TRADE,
            direction=direction,
            non_execution_reason=NonExecutionReason.TRADER_HOLD,
            entry_kind=EntryKind.NONE,
        )

    entry_kind, warnings = _classify_entry(record.trade_plan)
    if entry_kind is EntryKind.NONE:
        return OutcomeClassification(
            plan_type=PlanType.NO_TRADE,
            direction=direction,
            non_execution_reason=NonExecutionReason.INCOMPLETE_PLAN,
            entry_kind=entry_kind,
            classification_warnings=warnings,
        )
    return OutcomeClassification(
        plan_type=PlanType.EXECUTABLE_TRADE_PLAN,
        direction=direction,
        non_execution_reason=None,
        entry_kind=entry_kind,
        classification_warnings=warnings,
    )
