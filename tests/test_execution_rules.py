"""RecommendationOutcome, third pure block: single-bar entry/exit resolution
rules (tradingagents/outcome/execution_rules.py). See that module's own
docstring for the full same-bar-ambiguity policy and fill rules.

Price fetching, walking a full time series, MFE/MAE, returns, benchmark,
the full RecommendationOutcome schema, storage, and a CLI command are
explicitly NOT covered here -- future work. Numbered per the task's own
"CASOS OBLIGATORIOS" list (1-31).
"""

from __future__ import annotations

import pytest

import tradingagents.outcome.classification as classification
from tradingagents.outcome.execution_rules import (
    Direction,
    EntryKind,
    ExitReason,
    InvalidBarError,
    InvalidDirectionError,
    InvalidResolvedEntryError,
    OHLCBar,
    ResolvedEntry,
    SameBarResolution,
    TradeLevelViolation,
    evaluate_entry,
    evaluate_exit,
    evaluate_same_bar_entry_and_exit,
    validate_trade_levels,
)


def _bar(*, date="2026-01-01", open, high, low, close) -> OHLCBar:
    return OHLCBar(date=date, open=open, high=high, low=low, close=close)


# --- ENTRY (1-8) --------------------------------------------------------------


@pytest.mark.unit
def test_1_exact_touch():
    entry = ResolvedEntry(kind=EntryKind.EXACT, exact_price=100.0)
    bar = _bar(open=98.0, high=101.0, low=97.0, close=99.0)
    result = evaluate_entry(entry, bar)
    assert result.touched is True
    assert result.fill_price == 100.0
    assert result.fill_kind is EntryKind.EXACT


@pytest.mark.unit
def test_2_exact_miss():
    entry = ResolvedEntry(kind=EntryKind.EXACT, exact_price=120.0)
    bar = _bar(open=98.0, high=101.0, low=97.0, close=99.0)
    result = evaluate_entry(entry, bar)
    assert result.touched is False
    assert result.fill_price is None


@pytest.mark.unit
def test_3_open_exactly_at_entry():
    entry = ResolvedEntry(kind=EntryKind.EXACT, exact_price=100.0)
    bar = _bar(open=100.0, high=103.0, low=98.0, close=101.0)
    result = evaluate_entry(entry, bar)
    assert result.touched is True
    assert result.fill_price == 100.0


@pytest.mark.unit
def test_4_range_open_inside():
    entry = ResolvedEntry(kind=EntryKind.RANGE, low=98.0, high=102.0)
    bar = _bar(open=100.0, high=103.0, low=97.0, close=101.0)
    result = evaluate_entry(entry, bar)
    assert result.touched is True
    assert result.fill_price == 100.0  # filled at open, since open was already inside
    assert result.fill_kind is EntryKind.RANGE


@pytest.mark.unit
def test_5_range_open_below_and_enters():
    entry = ResolvedEntry(kind=EntryKind.RANGE, low=98.0, high=102.0)
    bar = _bar(open=95.0, high=100.0, low=94.0, close=99.0)
    result = evaluate_entry(entry, bar)
    assert result.touched is True
    assert result.fill_price == 98.0  # first boundary reached moving up from open


@pytest.mark.unit
def test_6_range_open_above_and_enters():
    entry = ResolvedEntry(kind=EntryKind.RANGE, low=98.0, high=102.0)
    bar = _bar(open=106.0, high=107.0, low=100.0, close=101.0)
    result = evaluate_entry(entry, bar)
    assert result.touched is True
    assert result.fill_price == 102.0  # first boundary reached moving down from open


@pytest.mark.unit
def test_7_range_only_touches_low_boundary():
    entry = ResolvedEntry(kind=EntryKind.RANGE, low=98.0, high=102.0)
    bar = _bar(open=95.0, high=98.0, low=94.0, close=96.0)  # high == low_bound exactly
    result = evaluate_entry(entry, bar)
    assert result.touched is True
    assert result.fill_price == 98.0


@pytest.mark.unit
def test_8_range_only_touches_high_boundary():
    entry = ResolvedEntry(kind=EntryKind.RANGE, low=98.0, high=102.0)
    bar = _bar(open=106.0, high=107.0, low=102.0, close=105.0)  # low == high_bound exactly
    result = evaluate_entry(entry, bar)
    assert result.touched is True
    assert result.fill_price == 102.0


@pytest.mark.unit
def test_range_fully_traversed_fills_at_the_near_boundary_not_the_far_one():
    entry = ResolvedEntry(kind=EntryKind.RANGE, low=98.0, high=102.0)
    bar = _bar(open=90.0, high=110.0, low=89.0, close=109.0)  # spans clean through
    result = evaluate_entry(entry, bar)
    assert result.touched is True
    assert result.fill_price == 98.0  # near boundary from open's side (below), never 102


@pytest.mark.unit
def test_entry_none_kind_is_never_touched():
    entry = ResolvedEntry(kind=EntryKind.NONE)
    bar = _bar(open=100.0, high=105.0, low=95.0, close=101.0)
    result = evaluate_entry(entry, bar)
    assert result.touched is False
    assert result.fill_kind is None


# --- LONG EXIT (9-13) ---------------------------------------------------------


@pytest.mark.unit
def test_9_long_stop_normal():
    bar = _bar(open=99.0, high=101.0, low=89.0, close=90.0)
    result = evaluate_exit(Direction.LONG, stop=90.0, target=None, bar=bar)
    assert result.stop_touched is True
    assert result.exit_reason is ExitReason.STOP
    assert result.exit_price == 90.0  # no gap: open (99) was above stop


@pytest.mark.unit
def test_10_long_stop_gap():
    bar = _bar(open=85.0, high=86.0, low=80.0, close=82.0)  # opens below stop
    result = evaluate_exit(Direction.LONG, stop=90.0, target=None, bar=bar)
    assert result.stop_touched is True
    assert result.exit_price == 85.0  # fills at open, the gap, not the stale stop level


@pytest.mark.unit
def test_11_long_target_normal():
    bar = _bar(open=99.0, high=111.0, low=98.0, close=110.0)
    result = evaluate_exit(Direction.LONG, stop=None, target=110.0, bar=bar)
    assert result.target_touched is True
    assert result.exit_reason is ExitReason.TARGET
    assert result.exit_price == 110.0  # no gap: open (99) was below target


@pytest.mark.unit
def test_12_long_target_gap():
    bar = _bar(open=115.0, high=117.0, low=114.0, close=116.0)  # opens above target
    result = evaluate_exit(Direction.LONG, stop=None, target=110.0, bar=bar)
    assert result.target_touched is True
    assert result.exit_price == 115.0  # fills at open


@pytest.mark.unit
def test_13_long_stop_and_target_same_bar_already_open():
    bar = _bar(open=100.0, high=112.0, low=88.0, close=105.0)
    result = evaluate_exit(Direction.LONG, stop=90.0, target=110.0, bar=bar)
    assert result.stop_touched is True
    assert result.target_touched is True
    assert result.exit_reason is ExitReason.STOP  # v1 policy: STOP wins
    assert result.exit_price == 90.0  # no gap (open 100 > stop 90)
    assert result.same_bar_ambiguous is True


# --- SHORT EXIT (14-18) -------------------------------------------------------


@pytest.mark.unit
def test_14_short_stop_normal():
    bar = _bar(open=100.0, high=110.0, low=99.0, close=108.0)
    result = evaluate_exit(Direction.SHORT, stop=110.0, target=None, bar=bar)
    assert result.stop_touched is True
    assert result.exit_price == 110.0  # no gap: open (100) was below stop


@pytest.mark.unit
def test_15_short_stop_gap():
    bar = _bar(open=115.0, high=118.0, low=114.0, close=116.0)  # opens above stop
    result = evaluate_exit(Direction.SHORT, stop=110.0, target=None, bar=bar)
    assert result.stop_touched is True
    assert result.exit_price == 115.0  # fills at the gapped open


@pytest.mark.unit
def test_16_short_target_normal():
    bar = _bar(open=100.0, high=101.0, low=89.0, close=90.0)
    result = evaluate_exit(Direction.SHORT, stop=None, target=90.0, bar=bar)
    assert result.target_touched is True
    assert result.exit_price == 90.0  # no gap: open (100) was above target


@pytest.mark.unit
def test_17_short_target_gap():
    bar = _bar(open=85.0, high=86.0, low=80.0, close=82.0)  # opens below target
    result = evaluate_exit(Direction.SHORT, stop=None, target=90.0, bar=bar)
    assert result.target_touched is True
    assert result.exit_price == 85.0


@pytest.mark.unit
def test_18_short_stop_and_target_same_bar_already_open():
    bar = _bar(open=100.0, high=112.0, low=88.0, close=95.0)
    result = evaluate_exit(Direction.SHORT, stop=110.0, target=90.0, bar=bar)
    assert result.stop_touched is True
    assert result.target_touched is True
    assert result.exit_reason is ExitReason.STOP
    assert result.exit_price == 110.0  # no gap (open 100 < stop 110)
    assert result.same_bar_ambiguous is True


# --- SAME-BAR ENTRY/EXIT, position NOT yet open (19-22) ----------------------


@pytest.mark.unit
def test_19_enters_and_touches_stop_same_day():
    entry = ResolvedEntry(kind=EntryKind.EXACT, exact_price=100.0)
    bar = _bar(open=100.0, high=101.0, low=88.0, close=90.0)
    result = evaluate_same_bar_entry_and_exit(Direction.LONG, entry, stop=90.0, target=None, bar=bar)
    assert result.entry.touched is True
    assert result.stop_touched_same_bar is True
    assert result.target_touched_same_bar is False
    assert result.entry_same_bar_exit_ambiguity is True
    assert result.resolution is SameBarResolution.AMBIGUOUS_INTRADAY_EXECUTION


@pytest.mark.unit
def test_20_enters_and_touches_target_same_day():
    entry = ResolvedEntry(kind=EntryKind.EXACT, exact_price=100.0)
    bar = _bar(open=100.0, high=111.0, low=99.0, close=110.0)
    result = evaluate_same_bar_entry_and_exit(Direction.LONG, entry, stop=None, target=110.0, bar=bar)
    assert result.entry.touched is True
    assert result.target_touched_same_bar is True
    assert result.entry_same_bar_exit_ambiguity is True
    assert result.resolution is SameBarResolution.AMBIGUOUS_INTRADAY_EXECUTION


@pytest.mark.unit
def test_21_enters_and_touches_both_same_day():
    entry = ResolvedEntry(kind=EntryKind.EXACT, exact_price=100.0)
    bar = _bar(open=100.0, high=111.0, low=89.0, close=105.0)
    result = evaluate_same_bar_entry_and_exit(Direction.LONG, entry, stop=90.0, target=110.0, bar=bar)
    assert result.stop_touched_same_bar is True
    assert result.target_touched_same_bar is True
    assert result.entry_same_bar_exit_ambiguity is True
    assert result.resolution is SameBarResolution.AMBIGUOUS_INTRADAY_EXECUTION


@pytest.mark.unit
def test_22_ambiguous_same_bar_never_invents_an_exit():
    """Structural proof: SameBarEntryExitEvaluation has no exit_price/
    exit_reason field at all -- the ambiguous case cannot accidentally
    carry a resolved exit, because the type itself has nowhere to put one."""
    entry = ResolvedEntry(kind=EntryKind.EXACT, exact_price=100.0)
    bar = _bar(open=100.0, high=111.0, low=89.0, close=105.0)
    result = evaluate_same_bar_entry_and_exit(Direction.LONG, entry, stop=90.0, target=110.0, bar=bar)
    assert not hasattr(result, "exit_price")
    assert not hasattr(result, "exit_reason")
    assert result.entry_same_bar_exit_ambiguity is True


@pytest.mark.unit
def test_same_bar_entry_not_touched_is_never_ambiguous():
    entry = ResolvedEntry(kind=EntryKind.EXACT, exact_price=200.0)  # far from the bar
    bar = _bar(open=100.0, high=111.0, low=89.0, close=105.0)
    result = evaluate_same_bar_entry_and_exit(Direction.LONG, entry, stop=90.0, target=110.0, bar=bar)
    assert result.entry.touched is False
    assert result.entry_same_bar_exit_ambiguity is False
    assert result.resolution is SameBarResolution.NOT_TRIGGERED


@pytest.mark.unit
def test_clean_entry_resolution_is_distinct_from_ambiguous():
    """A bar where entry is touched but stop/target do NOT also coincide
    must resolve to CLEAN_ENTRY, a value a caller cannot confuse with
    AMBIGUOUS_INTRADAY_EXECUTION -- the whole point of using an explicit
    enum instead of only a boolean (see module docstring)."""
    entry = ResolvedEntry(kind=EntryKind.EXACT, exact_price=100.0)
    bar = _bar(open=100.0, high=102.0, low=99.0, close=101.0)  # nowhere near stop/target
    result = evaluate_same_bar_entry_and_exit(Direction.LONG, entry, stop=80.0, target=120.0, bar=bar)
    assert result.entry.touched is True
    assert result.entry_same_bar_exit_ambiguity is False
    assert result.resolution is SameBarResolution.CLEAN_ENTRY
    assert result.resolution is not SameBarResolution.AMBIGUOUS_INTRADAY_EXECUTION


@pytest.mark.unit
def test_resolution_enum_has_exactly_three_exhaustive_members():
    """Closed, exhaustive set -- a caller branching on ``resolution`` can
    rely on there being no fourth, unhandled case."""
    assert {member.value for member in SameBarResolution} == {
        "NOT_TRIGGERED", "CLEAN_ENTRY", "AMBIGUOUS_INTRADAY_EXECUTION",
    }


# --- VALIDATION (23-27) -------------------------------------------------------


@pytest.mark.unit
def test_23_invalid_ohlc_raises():
    with pytest.raises(InvalidBarError):
        _bar(open=100.0, high=99.0, low=101.0, close=100.0)  # low > high
    with pytest.raises(InvalidBarError):
        _bar(open=105.0, high=102.0, low=98.0, close=100.0)  # open outside [low, high]
    with pytest.raises(InvalidBarError):
        _bar(open=100.0, high=102.0, low=98.0, close=110.0)  # close outside [low, high]


@pytest.mark.unit
def test_24_long_stop_above_entry_is_a_violation():
    entry = ResolvedEntry(kind=EntryKind.EXACT, exact_price=100.0)
    result = validate_trade_levels(Direction.LONG, entry, stop=105.0, target=None)
    assert result.valid is False
    assert TradeLevelViolation.LONG_STOP_NOT_BELOW_ENTRY in result.violations


@pytest.mark.unit
def test_25_long_target_below_entry_is_a_violation():
    entry = ResolvedEntry(kind=EntryKind.EXACT, exact_price=100.0)
    result = validate_trade_levels(Direction.LONG, entry, stop=None, target=95.0)
    assert result.valid is False
    assert TradeLevelViolation.LONG_TARGET_NOT_ABOVE_ENTRY in result.violations


@pytest.mark.unit
def test_26_short_stop_below_entry_is_a_violation():
    entry = ResolvedEntry(kind=EntryKind.EXACT, exact_price=100.0)
    result = validate_trade_levels(Direction.SHORT, entry, stop=95.0, target=None)
    assert result.valid is False
    assert TradeLevelViolation.SHORT_STOP_NOT_ABOVE_ENTRY in result.violations


@pytest.mark.unit
def test_27_short_target_above_entry_is_a_violation():
    entry = ResolvedEntry(kind=EntryKind.EXACT, exact_price=100.0)
    result = validate_trade_levels(Direction.SHORT, entry, stop=None, target=105.0)
    assert result.valid is False
    assert TradeLevelViolation.SHORT_TARGET_NOT_BELOW_ENTRY in result.violations


@pytest.mark.unit
def test_valid_long_levels_produce_no_violations():
    entry = ResolvedEntry(kind=EntryKind.EXACT, exact_price=100.0)
    result = validate_trade_levels(Direction.LONG, entry, stop=90.0, target=110.0)
    assert result.valid is True
    assert result.violations == ()


@pytest.mark.unit
def test_range_entry_uses_the_near_boundary_as_reference():
    """LONG stop checked against the range's LOW bound: a stop below the
    worst (lowest) possible fill is below every possible fill."""
    entry = ResolvedEntry(kind=EntryKind.RANGE, low=98.0, high=102.0)
    valid = validate_trade_levels(Direction.LONG, entry, stop=97.0, target=None)
    assert valid.valid is True
    invalid = validate_trade_levels(Direction.LONG, entry, stop=99.0, target=None)  # between low/high
    assert invalid.valid is False


# --- Direction NONE must never reach execution -------------------------------


@pytest.mark.unit
def test_invalid_direction_raises_not_assumes_long():
    """Direction.NONE is a real member of the SHARED enum (not a string, not
    a separate duplicated enum) -- passing it must raise, never fall back
    to LONG."""
    entry = ResolvedEntry(kind=EntryKind.EXACT, exact_price=100.0)
    bar = _bar(open=100.0, high=105.0, low=95.0, close=101.0)
    with pytest.raises(InvalidDirectionError):
        evaluate_exit(Direction.NONE, stop=90.0, target=None, bar=bar)
    with pytest.raises(InvalidDirectionError):
        validate_trade_levels(Direction.NONE, entry, stop=90.0, target=None)
    with pytest.raises(InvalidDirectionError):
        evaluate_same_bar_entry_and_exit(Direction.NONE, entry, stop=90.0, target=None, bar=bar)


# --- Single canonical Direction/EntryKind, shared with classification.py ----


@pytest.mark.unit
def test_direction_and_entrykind_are_the_same_objects_as_classification_py():
    """There must be exactly one definition of each enum -- execution_rules
    imports them from classification.py rather than redeclaring."""
    assert Direction is classification.Direction
    assert EntryKind is classification.EntryKind


@pytest.mark.unit
def test_functions_accept_classification_py_enums_directly_no_conversion():
    """classify_recommendation's own Direction/EntryKind values, as they
    come straight out of that function, must work here with zero string
    conversion and zero second enum -- proven by feeding its actual Direction
    member (not execution_rules's own name for it, since there is only one)
    straight into evaluate_exit/validate_trade_levels."""
    direction = classification.Direction.LONG  # exactly what classify_recommendation returns
    entry = ResolvedEntry(kind=classification.EntryKind.EXACT, exact_price=100.0)
    bar = _bar(open=99.0, high=101.0, low=89.0, close=90.0)

    exit_result = evaluate_exit(direction, stop=90.0, target=None, bar=bar)
    assert exit_result.stop_touched is True

    validation = validate_trade_levels(direction, entry, stop=90.0, target=110.0)
    assert validation.valid is True


@pytest.mark.unit
def test_classification_direction_none_is_rejected_here_not_silently_long():
    """classify_recommendation legitimately returns Direction.NONE for a
    NO_TRADE/Hold plan -- if that value ever reached execution_rules (a
    caller bug, since NO_TRADE plans should never reach execution), it must
    raise, never be coerced into LONG."""
    bar = _bar(open=100.0, high=105.0, low=95.0, close=101.0)
    with pytest.raises(InvalidDirectionError):
        evaluate_exit(classification.Direction.NONE, stop=90.0, target=None, bar=bar)


# --- OPTIONAL: no stop / no target / neither (28-30) -------------------------


@pytest.mark.unit
def test_28_no_stop():
    bar = _bar(open=100.0, high=111.0, low=99.0, close=110.0)
    result = evaluate_exit(Direction.LONG, stop=None, target=110.0, bar=bar)
    assert result.stop_touched is False
    assert result.exit_reason is ExitReason.TARGET


@pytest.mark.unit
def test_29_no_target():
    bar = _bar(open=100.0, high=101.0, low=89.0, close=90.0)
    result = evaluate_exit(Direction.LONG, stop=90.0, target=None, bar=bar)
    assert result.target_touched is False
    assert result.exit_reason is ExitReason.STOP


@pytest.mark.unit
def test_30_no_stop_or_target():
    bar = _bar(open=100.0, high=101.0, low=99.0, close=100.5)
    result = evaluate_exit(Direction.LONG, stop=None, target=None, bar=bar)
    assert result.stop_touched is False
    assert result.target_touched is False
    assert result.exit_reason is None
    assert result.exit_price is None
    assert result.same_bar_ambiguous is False


@pytest.mark.unit
def test_entry_with_no_stop_or_target_is_not_invalid_by_itself():
    entry = ResolvedEntry(kind=EntryKind.EXACT, exact_price=100.0)
    result = validate_trade_levels(Direction.LONG, entry, stop=None, target=None)
    assert result.valid is True
    assert result.violations == ()


# --- DETERMINISM (31) ---------------------------------------------------------


@pytest.mark.unit
def test_31_same_input_produces_the_same_output():
    entry = ResolvedEntry(kind=EntryKind.RANGE, low=98.0, high=102.0)
    bar = _bar(open=95.0, high=100.0, low=94.0, close=99.0)
    first = evaluate_entry(entry, bar)
    second = evaluate_entry(entry, bar)
    assert first == second

    exit_bar = _bar(open=100.0, high=112.0, low=88.0, close=105.0)
    first_exit = evaluate_exit(Direction.LONG, 90.0, 110.0, exit_bar)
    second_exit = evaluate_exit(Direction.LONG, 90.0, 110.0, exit_bar)
    assert first_exit == second_exit


# --- Resolved-entry construction validation ----------------------------------


@pytest.mark.unit
def test_resolved_entry_rejects_inverted_range():
    with pytest.raises(InvalidResolvedEntryError):
        ResolvedEntry(kind=EntryKind.RANGE, low=102.0, high=98.0)


@pytest.mark.unit
def test_resolved_entry_exact_requires_a_price():
    with pytest.raises(InvalidResolvedEntryError):
        ResolvedEntry(kind=EntryKind.EXACT, exact_price=None)


# --- Pure module guard --------------------------------------------------------


@pytest.mark.unit
def test_module_has_no_side_effect_imports():
    """No os/pathlib/logging/network/pandas -- and the ONE internal import
    allowed is the shared Direction/EntryKind enums from classification.py
    (see task: "no volver a declararlos"), never RecommendationRecord or
    anything else."""
    import ast
    from pathlib import Path

    import tradingagents.outcome.execution_rules as mod

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
    forbidden_prefixes = ("os", "pathlib", "logging", "requests", "socket", "json", "pandas")
    assert not any(
        module == prefix or module.startswith(prefix + ".")
        for module in imported_modules for prefix in forbidden_prefixes
    )
    tradingagents_imports = {m for m in imported_modules if m.startswith("tradingagents")}
    assert tradingagents_imports == {"tradingagents.outcome.classification"}
