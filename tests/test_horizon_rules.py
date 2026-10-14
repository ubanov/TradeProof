"""RecommendationOutcome, second pure block: resolve_horizon() turns an
already-built RecommendationRecord's ``horizon`` into concrete calendar
dates, anchored exclusively to ``record.run.analysis_date`` -- never a
second clock keyed to a (not-yet-implemented) entry date, and never a
re-parse of ``horizon.raw`` (see
tradingagents/outcome/horizon_rules.py's own module docstring for the full
contractual rule).

Pricing, market fetch, execution entry/stop/target, MFE/MAE, benchmark, the
full RecommendationOutcome schema, storage, and a CLI command are explicitly
NOT covered here -- future work.
"""

from __future__ import annotations

import inspect

import pytest

from tradingagents.outcome.horizon_rules import HorizonBoundStatus, resolve_horizon
from tradingagents.recommendation_record import Horizon, HorizonParseStatus, HorizonUnit


def _sessions(dates: list[str]) -> list[str]:
    return dates


# A plain Mon-Fri session calendar for January 2026 (2026-01-01 is a
# Thursday) -- weekends are simply absent, exactly as a real trading
# calendar would be; no special handling needed anywhere for that.
_JAN_2026_SESSIONS = _sessions([
    "2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09",
    "2026-01-12", "2026-01-13", "2026-01-14", "2026-01-15", "2026-01-16",
])


def _horizon(
    *, unit: HorizonUnit, min_: float | None, max_: float | None,
    parse_status: HorizonParseStatus = HorizonParseStatus.PARSED_RANGE,
) -> Horizon:
    return Horizon(raw="test", unit=unit, min=min_, max=max_, parse_status=parse_status)


# --- 5-10 trading sessions, the task's own anchor example -------------------


@pytest.mark.unit
def test_5_to_10_trading_sessions_resolves_both_bounds():
    horizon = _horizon(unit=HorizonUnit.TRADING_SESSIONS, min_=5.0, max_=10.0)
    resolved = resolve_horizon(horizon, "2026-01-01", trading_sessions=_JAN_2026_SESSIONS)
    assert resolved.min_status is HorizonBoundStatus.RESOLVED
    assert resolved.max_status is HorizonBoundStatus.RESOLVED
    assert resolved.min_date == "2026-01-08T00:00:00"  # 5th session after 2026-01-01
    assert resolved.max_date == "2026-01-15T00:00:00"  # 10th session after 2026-01-01


@pytest.mark.unit
def test_resolve_horizon_has_no_entry_date_parameter():
    """Structural proof of the contractual rule: there is a single clock
    (analysis_date). An execution layer's later entry date -- whatever
    session it lands on -- cannot change max_date because this function
    does not even accept one."""
    params = set(inspect.signature(resolve_horizon).parameters)
    assert not any("entry" in name for name in params)


@pytest.mark.unit
def test_min_value_never_affects_max_date_resolution():
    """min and max are resolved independently -- varying min (even to None)
    must never change max_date for the same max value."""
    max_only = _horizon(unit=HorizonUnit.TRADING_SESSIONS, min_=None, max_=10.0)
    min_5 = _horizon(unit=HorizonUnit.TRADING_SESSIONS, min_=5.0, max_=10.0)
    min_9 = _horizon(unit=HorizonUnit.TRADING_SESSIONS, min_=9.0, max_=10.0)
    dates = {
        resolve_horizon(h, "2026-01-01", trading_sessions=_JAN_2026_SESSIONS).max_date
        for h in (max_only, min_5, min_9)
    }
    assert len(dates) == 1


# --- analysis_date / session counting ---------------------------------------


@pytest.mark.unit
def test_analysis_date_does_not_count_as_session_one():
    horizon = _horizon(unit=HorizonUnit.TRADING_SESSIONS, min_=None, max_=1.0)
    resolved = resolve_horizon(horizon, "2026-01-01", trading_sessions=_JAN_2026_SESSIONS)
    # 2026-01-01 itself is a session-calendar date in nobody's list; the
    # first session AFTER it is 2026-01-02.
    assert resolved.max_date == "2026-01-02T00:00:00"


@pytest.mark.unit
def test_analysis_date_accidentally_included_in_session_list_is_excluded():
    horizon = _horizon(unit=HorizonUnit.TRADING_SESSIONS, min_=None, max_=1.0)
    with_anchor_in_list = ["2026-01-01", *_JAN_2026_SESSIONS]
    resolved = resolve_horizon(horizon, "2026-01-01", trading_sessions=with_anchor_in_list)
    assert resolved.max_date == "2026-01-02T00:00:00"  # unchanged by the noisy entry


@pytest.mark.unit
def test_weekends_absent_from_session_list_are_simply_skipped():
    # 2026-01-09 (Fri) -> next session is 2026-01-12 (Mon); the weekend
    # 2026-01-10/11 is simply not in the list, no special-casing needed.
    horizon = _horizon(unit=HorizonUnit.TRADING_SESSIONS, min_=None, max_=7.0)
    resolved = resolve_horizon(horizon, "2026-01-01", trading_sessions=_JAN_2026_SESSIONS)
    assert resolved.max_date == "2026-01-12T00:00:00"


@pytest.mark.unit
def test_unordered_and_duplicated_sessions_are_handled():
    shuffled_with_dupes = _JAN_2026_SESSIONS[::-1] + [_JAN_2026_SESSIONS[0]]
    horizon = _horizon(unit=HorizonUnit.TRADING_SESSIONS, min_=5.0, max_=10.0)
    resolved = resolve_horizon(horizon, "2026-01-01", trading_sessions=shuffled_with_dupes)
    assert resolved.min_date == "2026-01-08T00:00:00"
    assert resolved.max_date == "2026-01-15T00:00:00"


# --- Insufficient sessions ----------------------------------------------------


@pytest.mark.unit
def test_insufficient_sessions_for_max_but_sufficient_for_min():
    horizon = _horizon(unit=HorizonUnit.TRADING_SESSIONS, min_=5.0, max_=10.0)
    only_six_sessions = _JAN_2026_SESSIONS[:6]
    resolved = resolve_horizon(horizon, "2026-01-01", trading_sessions=only_six_sessions)
    assert resolved.min_status is HorizonBoundStatus.RESOLVED
    assert resolved.min_date is not None
    assert resolved.max_status is HorizonBoundStatus.INSUFFICIENT_SESSIONS
    assert resolved.max_date is None  # never an invented future date


@pytest.mark.unit
def test_no_trading_sessions_supplied_at_all_is_insufficient_not_a_crash():
    horizon = _horizon(unit=HorizonUnit.TRADING_SESSIONS, min_=5.0, max_=10.0)
    resolved = resolve_horizon(horizon, "2026-01-01", trading_sessions=None)
    assert resolved.min_status is HorizonBoundStatus.INSUFFICIENT_SESSIONS
    assert resolved.max_status is HorizonBoundStatus.INSUFFICIENT_SESSIONS


@pytest.mark.unit
def test_trading_days_unit_is_resolved_the_same_way_as_trading_sessions():
    horizon = _horizon(unit=HorizonUnit.TRADING_DAYS, min_=None, max_=5.0)
    resolved = resolve_horizon(horizon, "2026-01-01", trading_sessions=_JAN_2026_SESSIONS)
    assert resolved.max_date == "2026-01-08T00:00:00"


# --- Calendar days: off-by-one -----------------------------------------------


@pytest.mark.unit
def test_one_calendar_day_lands_on_the_next_day():
    horizon = _horizon(unit=HorizonUnit.CALENDAR_DAYS, min_=None, max_=1.0)
    resolved = resolve_horizon(horizon, "2026-01-01", trading_sessions=None)
    assert resolved.max_date == "2026-01-02T00:00:00"


@pytest.mark.unit
def test_five_calendar_days():
    horizon = _horizon(unit=HorizonUnit.CALENDAR_DAYS, min_=None, max_=5.0)
    resolved = resolve_horizon(horizon, "2026-01-01", trading_sessions=None)
    assert resolved.max_date == "2026-01-06T00:00:00"


# --- Weeks: plain +7*N calendar days, never converted to sessions -----------


@pytest.mark.unit
def test_one_week_is_seven_calendar_days():
    horizon = _horizon(unit=HorizonUnit.WEEKS, min_=None, max_=1.0)
    resolved = resolve_horizon(horizon, "2026-01-01", trading_sessions=None)
    assert resolved.max_date == "2026-01-08T00:00:00"


@pytest.mark.unit
def test_two_weeks_is_fourteen_calendar_days():
    horizon = _horizon(unit=HorizonUnit.WEEKS, min_=None, max_=2.0)
    resolved = resolve_horizon(horizon, "2026-01-01", trading_sessions=None)
    assert resolved.max_date == "2026-01-15T00:00:00"


# --- Months: real calendar-month arithmetic ----------------------------------


@pytest.mark.unit
def test_one_month_from_jan_31_clamps_to_february_end():
    horizon = _horizon(unit=HorizonUnit.MONTHS, min_=None, max_=1.0)
    resolved = resolve_horizon(horizon, "2026-01-31", trading_sessions=None)
    assert resolved.max_date == "2026-02-28T00:00:00"  # 2026 is not a leap year


@pytest.mark.unit
def test_one_month_from_jan_31_in_a_leap_year():
    horizon = _horizon(unit=HorizonUnit.MONTHS, min_=None, max_=1.0)
    resolved = resolve_horizon(horizon, "2024-01-31", trading_sessions=None)
    assert resolved.max_date == "2024-02-29T00:00:00"


@pytest.mark.unit
def test_two_months_crossing_a_year_boundary():
    horizon = _horizon(unit=HorizonUnit.MONTHS, min_=None, max_=2.0)
    resolved = resolve_horizon(horizon, "2026-11-30", trading_sessions=None)
    assert resolved.max_date == "2027-01-30T00:00:00"


# --- min == max ---------------------------------------------------------------


@pytest.mark.unit
def test_min_equals_max_resolves_both_to_the_same_date():
    horizon = _horizon(unit=HorizonUnit.TRADING_SESSIONS, min_=5.0, max_=5.0)
    resolved = resolve_horizon(horizon, "2026-01-01", trading_sessions=_JAN_2026_SESSIONS)
    assert resolved.min_date == resolved.max_date == "2026-01-08T00:00:00"


# --- A missing bound is explicit, never invented -----------------------------


@pytest.mark.unit
def test_missing_min_with_max_present():
    horizon = _horizon(unit=HorizonUnit.TRADING_SESSIONS, min_=None, max_=10.0)
    resolved = resolve_horizon(horizon, "2026-01-01", trading_sessions=_JAN_2026_SESSIONS)
    assert resolved.min_status is HorizonBoundStatus.NOT_SET
    assert resolved.min_date is None
    assert resolved.max_status is HorizonBoundStatus.RESOLVED


@pytest.mark.unit
def test_missing_max_with_min_present():
    horizon = _horizon(unit=HorizonUnit.TRADING_SESSIONS, min_=5.0, max_=None)
    resolved = resolve_horizon(horizon, "2026-01-01", trading_sessions=_JAN_2026_SESSIONS)
    assert resolved.max_status is HorizonBoundStatus.NOT_SET
    assert resolved.max_date is None
    assert resolved.min_status is HorizonBoundStatus.RESOLVED


# --- Invalid/unknown horizons: explicit, never a fallback -------------------


@pytest.mark.unit
def test_unknown_unit_is_unresolvable():
    horizon = _horizon(
        unit=HorizonUnit.UNKNOWN, min_=None, max_=None, parse_status=HorizonParseStatus.UNPARSEABLE,
    )
    resolved = resolve_horizon(horizon, "2026-01-01", trading_sessions=_JAN_2026_SESSIONS)
    assert resolved.min_status is HorizonBoundStatus.UNRESOLVABLE_UNIT
    assert resolved.max_status is HorizonBoundStatus.UNRESOLVABLE_UNIT
    assert resolved.min_date is None and resolved.max_date is None


@pytest.mark.unit
def test_unparseable_parse_status_is_unresolvable_even_with_values_present():
    # Pathological but possible on a hand-edited/legacy record: numeric
    # values present, status says they should not be trusted.
    horizon = _horizon(
        unit=HorizonUnit.TRADING_SESSIONS, min_=5.0, max_=10.0,
        parse_status=HorizonParseStatus.UNPARSEABLE,
    )
    resolved = resolve_horizon(horizon, "2026-01-01", trading_sessions=_JAN_2026_SESSIONS)
    assert resolved.min_status is HorizonBoundStatus.UNRESOLVABLE_UNIT
    assert resolved.max_status is HorizonBoundStatus.UNRESOLVABLE_UNIT


@pytest.mark.unit
def test_missing_parse_status_is_unresolvable():
    horizon = _horizon(
        unit=HorizonUnit.UNKNOWN, min_=None, max_=None, parse_status=HorizonParseStatus.MISSING,
    )
    resolved = resolve_horizon(horizon, "2026-01-01", trading_sessions=None)
    assert resolved.min_status is HorizonBoundStatus.UNRESOLVABLE_UNIT
    assert resolved.max_status is HorizonBoundStatus.UNRESOLVABLE_UNIT


@pytest.mark.unit
def test_no_automatic_5_day_fallback_for_unresolvable_horizons():
    """The previously-floated idea of defaulting to holding_period_days=5
    is deliberately NOT implemented here -- an unresolvable horizon must
    never resolve to a 5-day (or any other) guessed date."""
    horizon = _horizon(
        unit=HorizonUnit.UNKNOWN, min_=None, max_=None, parse_status=HorizonParseStatus.UNPARSEABLE,
    )
    resolved = resolve_horizon(horizon, "2026-01-01")
    assert resolved.max_date is None
    assert resolved.max_date != "2026-01-06T00:00:00"  # what a 5-day fallback would have produced


# --- Fractional values --------------------------------------------------------


@pytest.mark.unit
def test_fractional_trading_sessions_is_unsupported_not_truncated():
    horizon = _horizon(unit=HorizonUnit.TRADING_SESSIONS, min_=5.5, max_=10.0)
    resolved = resolve_horizon(horizon, "2026-01-01", trading_sessions=_JAN_2026_SESSIONS)
    assert resolved.min_status is HorizonBoundStatus.UNSUPPORTED_VALUE
    assert resolved.min_date is None
    assert resolved.max_status is HorizonBoundStatus.RESOLVED  # independent of min


@pytest.mark.unit
def test_non_positive_trading_session_count_is_unsupported():
    horizon = _horizon(unit=HorizonUnit.TRADING_SESSIONS, min_=0.0, max_=10.0)
    resolved = resolve_horizon(horizon, "2026-01-01", trading_sessions=_JAN_2026_SESSIONS)
    assert resolved.min_status is HorizonBoundStatus.UNSUPPORTED_VALUE


@pytest.mark.unit
def test_fractional_weeks_are_supported_with_exact_sub_day_precision():
    horizon = _horizon(unit=HorizonUnit.WEEKS, min_=None, max_=1.5)
    resolved = resolve_horizon(horizon, "2026-01-01", trading_sessions=None)
    assert resolved.max_status is HorizonBoundStatus.RESOLVED
    assert resolved.max_date == "2026-01-11T12:00:00"  # 10.5 days after anchor


@pytest.mark.unit
def test_fractional_calendar_days_are_supported():
    horizon = _horizon(unit=HorizonUnit.CALENDAR_DAYS, min_=None, max_=2.5)
    resolved = resolve_horizon(horizon, "2026-01-01", trading_sessions=None)
    assert resolved.max_date == "2026-01-03T12:00:00"


@pytest.mark.unit
def test_fractional_months_is_unsupported_not_rounded():
    horizon = _horizon(unit=HorizonUnit.MONTHS, min_=None, max_=2.5)
    resolved = resolve_horizon(horizon, "2026-01-31", trading_sessions=None)
    assert resolved.max_status is HorizonBoundStatus.UNSUPPORTED_VALUE
    assert resolved.max_date is None


# --- Purity / determinism ----------------------------------------------------


@pytest.mark.unit
def test_same_input_produces_the_same_output():
    horizon = _horizon(unit=HorizonUnit.TRADING_SESSIONS, min_=5.0, max_=10.0)
    first = resolve_horizon(horizon, "2026-01-01", trading_sessions=_JAN_2026_SESSIONS)
    second = resolve_horizon(horizon, "2026-01-01", trading_sessions=_JAN_2026_SESSIONS)
    assert first == second


@pytest.mark.unit
def test_module_has_no_side_effect_imports():
    import ast
    from pathlib import Path

    import tradingagents.outcome.horizon_rules as mod

    source = Path(mod.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    forbidden = {"os", "logging", "requests", "socket", "json"}
    assert not (imported_modules & forbidden)
