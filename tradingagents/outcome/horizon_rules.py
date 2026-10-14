"""Pure resolution of an already-built RecommendationRecord's ``horizon``
into concrete calendar dates.

Second pure block of the future RecommendationOutcome work. Explicitly NOT
implemented here (future work, separate modules): pricing, market fetch,
execution entry/stop/target simulation, MFE/MAE, benchmark comparison, the
full RecommendationOutcome schema, storage, a CLI command, decision_log, or
backtest.

==============================================================================
Contractual rule (see fork README: "RecommendationRecord" -- horizon)
==============================================================================
The ENTIRE evaluation window is anchored to ONE clock:
``RecommendationRecord.run.analysis_date``. There is no second clock keyed
to a (future, not-yet-implemented) entry date. For "5-10 trading sessions",
``min``/``max`` do NOT mean "wait 5 sessions for entry, then hold 10 more
from entry" -- they describe a SINGLE window counted from ``analysis_date``:
the absolute ceiling is always session 10 from ``analysis_date``, regardless
of which session (if any) a later execution layer decides entry happened on.
This module's functions do not even accept an entry date, by design, so
nothing here could depend on one.

``max`` is the absolute evaluation ceiling. ``min`` is the earliest-expected
boundary / metadata for a later execution layer -- it never closes a trade,
never blocks entry after it passes, and never extends ``max``. Resolving
``min`` and ``max`` are two fully independent calls into the same per-unit
logic; neither reads the other's value.

This module consumes ONLY ``RecommendationRecord.horizon`` (``raw``/
``unit``/``min``/``max``/``parse_status``) plus an explicitly-passed anchor
date and, for session-based units, an explicitly-passed list of known
trading session dates. It never re-parses ``horizon.raw``, never calls an
LLM, never reads Markdown, never fetches the market, and never reads a
filesystem or a clock -- same inputs always produce the same
``ResolvedHorizon``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

import pandas as pd

from tradingagents.recommendation_record import Horizon, HorizonParseStatus, HorizonUnit


class HorizonBoundStatus(str, Enum):
    """Resolution status for ONE bound (``min`` or ``max``) -- deliberately
    separate from any future, broader RecommendationOutcome status (see
    module docstring's scope note): this only describes whether a DATE could
    be honestly derived for this bound, nothing about execution or whether
    "now" has passed it."""

    # A concrete date was derived.
    RESOLVED = "RESOLVED"
    # The bound itself was None on Horizon (e.g. a record whose PM text
    # only gave a single value, parsed as min==max, or a horizon with only
    # one bound present) -- absence, not an error.
    NOT_SET = "NOT_SET"
    # unit is TRADING_SESSIONS/TRADING_DAYS and the caller-supplied
    # trading_sessions list does not (yet) contain enough sessions after
    # anchor_date to know this bound's date -- never guessed.
    INSUFFICIENT_SESSIONS = "INSUFFICIENT_SESSIONS"
    # The numeric value itself cannot be honestly turned into a date for
    # this unit (a fractional trading-session/day count, a fractional month
    # count, or a non-positive session count) -- never silently
    # rounded/truncated.
    UNSUPPORTED_VALUE = "UNSUPPORTED_VALUE"
    # horizon.unit is UNKNOWN, or horizon.parse_status is UNPARSEABLE/
    # MISSING -- there is no unit to resolve against at all.
    UNRESOLVABLE_UNIT = "UNRESOLVABLE_UNIT"


@dataclass(frozen=True)
class ResolvedHorizon:
    anchor_date: str
    unit: HorizonUnit
    min_value: float | None
    max_value: float | None
    min_date: str | None
    max_date: str | None
    min_status: HorizonBoundStatus
    max_status: HorizonBoundStatus


# TRADING_SESSIONS and TRADING_DAYS are resolved identically: both name a
# count of market-open days, and neither can be turned into a date from
# calendar arithmetic alone (see module docstring) -- the project's own
# parser (parse_horizon) distinguishes them only by which phrasing produced
# them, not by a different real-world resolution rule.
_SESSION_BASED_UNITS = frozenset({HorizonUnit.TRADING_SESSIONS, HorizonUnit.TRADING_DAYS})


def _is_whole_number(value: float) -> bool:
    return float(value).is_integer()


def _resolve_calendar_days_bound(value: float, anchor: pd.Timestamp) -> tuple[str, HorizonBoundStatus]:
    # analysis_date is day 0: "N calendar days" means anchor_date + N days,
    # so 1 calendar day resolves to the very next day (see module docstring
    # / task spec section 5) -- never anchor_date itself. Fractional days
    # are honest here (e.g. 2.5 days = 60 hours after anchor) -- never
    # truncated.
    return (anchor + pd.Timedelta(days=value)).isoformat(), HorizonBoundStatus.RESOLVED


def _resolve_weeks_bound(value: float, anchor: pd.Timestamp) -> tuple[str, HorizonBoundStatus]:
    # 1 week = anchor_date + 7 calendar days, N weeks = anchor_date + 7*N
    # days -- plain calendar arithmetic, never converted to a trading-session
    # count. Fractional weeks are honest for the same reason as calendar days.
    return (anchor + pd.Timedelta(days=7 * value)).isoformat(), HorizonBoundStatus.RESOLVED


def _resolve_months_bound(value: float, anchor: pd.Timestamp) -> tuple[str | None, HorizonBoundStatus]:
    # Real calendar-month arithmetic (pandas' own DateOffset, already used
    # elsewhere in this codebase for date math -- no new dependency), not
    # 30*N days: Jan 31 + 1 month = Feb 28/29 (clamped to the month's last
    # day when the original day-of-month doesn't exist), Nov 30 + 2 months =
    # Jan 30, which also crosses a year boundary correctly. A fractional
    # month count has no unambiguous calendar meaning (pandas itself raises
    # "Non-integer years and months are ambiguous and not currently
    # supported" for DateOffset) -- reported explicitly rather than rounded.
    if not _is_whole_number(value):
        return None, HorizonBoundStatus.UNSUPPORTED_VALUE
    return (anchor + pd.DateOffset(months=int(value))).isoformat(), HorizonBoundStatus.RESOLVED


def _resolve_session_based_bound(
    value: float, anchor: pd.Timestamp, trading_sessions: Sequence[str] | None,
) -> tuple[str | None, HorizonBoundStatus]:
    # Session N is the Nth session STRICTLY AFTER anchor_date -- anchor_date
    # itself is never session 1, even if the caller's own trading_sessions
    # list happens to include it (filtered out below, not special-cased).
    # Weekends/holidays need no handling here: they simply never appear in
    # a real trading_sessions list.
    if not _is_whole_number(value) or value < 1:
        return None, HorizonBoundStatus.UNSUPPORTED_VALUE
    if not trading_sessions:
        return None, HorizonBoundStatus.INSUFFICIENT_SESSIONS

    sessions_after = sorted({
        session for session in (pd.Timestamp(s).normalize() for s in trading_sessions)
        if session > anchor
    })
    n = int(value)
    if len(sessions_after) < n:
        # Never invent a future date: we simply do not yet know which
        # calendar date session N falls on.
        return None, HorizonBoundStatus.INSUFFICIENT_SESSIONS
    return sessions_after[n - 1].isoformat(), HorizonBoundStatus.RESOLVED


def _resolve_bound(
    value: float | None, unit: HorizonUnit, anchor: pd.Timestamp, trading_sessions: Sequence[str] | None,
) -> tuple[str | None, HorizonBoundStatus]:
    if value is None:
        return None, HorizonBoundStatus.NOT_SET
    if unit in _SESSION_BASED_UNITS:
        return _resolve_session_based_bound(value, anchor, trading_sessions)
    if unit is HorizonUnit.CALENDAR_DAYS:
        return _resolve_calendar_days_bound(value, anchor)
    if unit is HorizonUnit.WEEKS:
        return _resolve_weeks_bound(value, anchor)
    if unit is HorizonUnit.MONTHS:
        return _resolve_months_bound(value, anchor)
    return None, HorizonBoundStatus.UNRESOLVABLE_UNIT  # pragma: no cover -- UNKNOWN is filtered earlier


def resolve_horizon(
    horizon: Horizon,
    anchor_date: str,
    *,
    trading_sessions: Sequence[str] | None = None,
) -> ResolvedHorizon:
    """Resolve ``horizon`` (read as-is from a RecommendationRecord, never
    re-parsed) into concrete dates anchored at ``anchor_date`` -- which must
    be ``record.run.analysis_date``; there is no other clock.

    ``trading_sessions`` is only consulted for ``TRADING_SESSIONS``/
    ``TRADING_DAYS`` horizons; every other unit ignores it. It may contain
    sessions on or before ``anchor_date`` (filtered out), out of order, or
    duplicated -- all handled.

    No side effects: no market fetch, no filesystem, no LLM, no clock (this
    function never reads "today"). The absence of an entry-date parameter is
    deliberate, not an oversight -- see the module docstring's contractual
    rule: nothing here can depend on when/whether an entry happened, by
    construction.
    """
    anchor = pd.Timestamp(anchor_date).normalize()

    if horizon.unit is HorizonUnit.UNKNOWN or horizon.parse_status in (
        HorizonParseStatus.UNPARSEABLE, HorizonParseStatus.MISSING,
    ):
        return ResolvedHorizon(
            anchor_date=anchor_date, unit=horizon.unit,
            min_value=horizon.min, max_value=horizon.max,
            min_date=None, max_date=None,
            min_status=HorizonBoundStatus.UNRESOLVABLE_UNIT,
            max_status=HorizonBoundStatus.UNRESOLVABLE_UNIT,
        )

    min_date, min_status = _resolve_bound(horizon.min, horizon.unit, anchor, trading_sessions)
    max_date, max_status = _resolve_bound(horizon.max, horizon.unit, anchor, trading_sessions)
    return ResolvedHorizon(
        anchor_date=anchor_date, unit=horizon.unit,
        min_value=horizon.min, max_value=horizon.max,
        min_date=min_date, max_date=max_date,
        min_status=min_status, max_status=max_status,
    )
