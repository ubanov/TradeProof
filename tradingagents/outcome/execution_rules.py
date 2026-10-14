"""Pure, single-bar entry/exit resolution rules.

Third pure block of the future RecommendationOutcome work. Explicitly NOT
implemented here (future work, separate modules): price fetching (Yahoo or
otherwise), walking a full time series, MFE/MAE, returns, benchmark
comparison, the full RecommendationOutcome schema, storage, a CLI command,
decision_log, or backtest.

Deliberately does NOT import ``RecommendationRecord`` -- this module takes
only small, already-resolved local structures (``ResolvedEntry``,
``Direction``, ``OHLCBar``) as input, so it never re-inspects or
re-interprets a record's own (possibly contradictory) raw fields. A future
orchestration layer is responsible for translating ``classify_recommendation``'s
output into a ``ResolvedEntry`` before calling here -- this module makes
exactly ONE canonical interpretation of entry/stop/target touch-and-fill,
nothing upstream of it.

``Direction`` and ``EntryKind`` ARE imported from
``tradingagents.outcome.classification`` -- there is exactly one canonical
definition of each across both modules, never two independently-declared
enums with the same members. Execution functions still only ever accept
``Direction.LONG``/``Direction.SHORT`` (``Direction.NONE`` is a legitimate
member of the shared enum, meaningful at the classification layer for a
non-executed plan, but is explicitly rejected here -- see
``InvalidDirectionError``/``_require_direction``). A caller can pass
``classify_recommendation(record).direction`` straight into any function
here with no conversion and no second enum.

==============================================================================
Same-bar entry+exit ambiguity (see ``evaluate_same_bar_entry_and_exit``)
==============================================================================
Daily OHLC carries no intraday ordering information. If a position was NOT
yet open at the start of a bar, and that bar's OHLC shows entry being
touched AND stop/target conditions also being satisfiable, there is no way
to know from OHLC alone whether entry happened before or after that
stop/target level was reached. This module takes the conservative option of
NOT inventing that order: it flags the ambiguity and does not resolve an
exit for that bar at all.

Critically: ``entry_same_bar_exit_ambiguity=True`` (equivalently,
``resolution=SameBarResolution.AMBIGUOUS_INTRADAY_EXECUTION``) does NOT mean
"the position survived this bar un-exited" -- it means we cannot know
whether it did. It does NOT authorize a later-bar engine to silently
continue from the next session as if this were a normally, cleanly open
position. A caller MUST branch on ``resolution`` as its own distinct case
(e.g. mark the whole trade's outcome as ambiguous, or apply its own
explicit, separately-justified tie-break policy) -- never fall through the
same code path used for a bar where entry was simply touched cleanly.

For a position that WAS already open before the bar starts (so entry is not
in question), and stop AND target are both touched on the same bar, the
already-agreed v1 policy is used instead: STOP wins (conservative), with
``same_bar_ambiguous=True`` surfaced so the ambiguity is never hidden.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from tradingagents.outcome.classification import Direction, EntryKind

__all__ = [
    "Direction",
    "EntryKind",
    "ExitReason",
    "TradeLevelViolation",
    "SameBarResolution",
    "ResolvedEntry",
    "OHLCBar",
    "EntryEvaluation",
    "ExitEvaluation",
    "SameBarEntryExitEvaluation",
    "TradeLevelsValidation",
    "InvalidBarError",
    "InvalidDirectionError",
    "InvalidResolvedEntryError",
    "evaluate_entry",
    "evaluate_exit",
    "evaluate_same_bar_entry_and_exit",
    "validate_trade_levels",
]


class InvalidBarError(ValueError):
    """Raised when an OHLCBar violates low <= open/close <= high, or
    low <= high -- a data-quality problem that must stop processing, never
    be silently continued through."""


class InvalidDirectionError(ValueError):
    """Raised when a function that requires LONG or SHORT is given anything
    else (including ``Direction.NONE``) -- a NONE direction must never
    reach trade execution, and this module never assumes LONG as a
    fallback."""


class InvalidResolvedEntryError(ValueError):
    """Raised when a ``ResolvedEntry`` is internally inconsistent (e.g.
    ``kind=RANGE`` with ``low > high``, or a kind missing the price fields
    it requires) -- the same "fail fast, never silently continue" policy as
    ``OHLCBar``. ``kind=EntryKind.NONE`` itself is NOT an inconsistency --
    it legitimately means "no entry to check" (see ``evaluate_entry``) and
    constructs without error."""


class ExitReason(str, Enum):
    STOP = "STOP"
    TARGET = "TARGET"


class TradeLevelViolation(str, Enum):
    """Closed set of codes for a stop/target on the wrong side of entry --
    never auto-corrected ("no darle la vuelta"), only flagged."""

    LONG_STOP_NOT_BELOW_ENTRY = "LONG_STOP_NOT_BELOW_ENTRY"
    LONG_TARGET_NOT_ABOVE_ENTRY = "LONG_TARGET_NOT_ABOVE_ENTRY"
    SHORT_STOP_NOT_ABOVE_ENTRY = "SHORT_STOP_NOT_ABOVE_ENTRY"
    SHORT_TARGET_NOT_BELOW_ENTRY = "SHORT_TARGET_NOT_BELOW_ENTRY"


@dataclass(frozen=True)
class ResolvedEntry:
    """The ONE canonical entry interpretation a caller must already have
    resolved (e.g. from ``classification.py``'s ``EntryKind``) before
    calling into this module -- this module never looks at any other,
    possibly-contradictory field to decide entry on its own."""

    kind: EntryKind
    exact_price: float | None = None
    low: float | None = None
    high: float | None = None

    def __post_init__(self) -> None:
        if self.kind is EntryKind.EXACT and self.exact_price is None:
            raise InvalidResolvedEntryError("kind=EXACT requires exact_price")
        if self.kind is EntryKind.RANGE:
            if self.low is None or self.high is None:
                raise InvalidResolvedEntryError("kind=RANGE requires both low and high")
            if self.low > self.high:
                raise InvalidResolvedEntryError(f"low ({self.low}) > high ({self.high})")


@dataclass(frozen=True)
class OHLCBar:
    date: str
    open: float
    high: float
    low: float
    close: float

    def __post_init__(self) -> None:
        if self.low > self.high:
            raise InvalidBarError(f"{self.date}: low ({self.low}) > high ({self.high})")
        if not (self.low <= self.open <= self.high):
            raise InvalidBarError(f"{self.date}: open ({self.open}) outside [low, high]")
        if not (self.low <= self.close <= self.high):
            raise InvalidBarError(f"{self.date}: close ({self.close}) outside [low, high]")


@dataclass(frozen=True)
class EntryEvaluation:
    touched: bool
    fill_price: float | None
    fill_kind: EntryKind | None


@dataclass(frozen=True)
class ExitEvaluation:
    stop_touched: bool
    target_touched: bool
    exit_reason: ExitReason | None
    exit_price: float | None
    same_bar_ambiguous: bool


class SameBarResolution(str, Enum):
    """What a caller is PERMITTED to conclude from a
    ``SameBarEntryExitEvaluation`` -- an explicit, exhaustive label so a
    caller cannot reach ``AMBIGUOUS_INTRADAY_EXECUTION`` by falling through
    the same branch used for a clean entry (see the module docstring's
    "Same-bar entry+exit ambiguity" section for why that distinction
    matters: this is NOT "entered and still open", it is "entered and we
    cannot know whether it is still open")."""

    # entry never touched this bar -- nothing else to resolve.
    NOT_TRIGGERED = "NOT_TRIGGERED"
    # entry touched; stop/target conditions did NOT also coincide on this
    # bar -- safe to treat the position as normally opened going forward.
    CLEAN_ENTRY = "CLEAN_ENTRY"
    # entry touched AND stop and/or target also would have been touched on
    # THIS SAME bar -- intraday order is unknowable from daily OHLC. A
    # caller MUST handle this as its own distinct case, never silently
    # continue to the next bar as if this were CLEAN_ENTRY.
    AMBIGUOUS_INTRADAY_EXECUTION = "AMBIGUOUS_INTRADAY_EXECUTION"


@dataclass(frozen=True)
class SameBarEntryExitEvaluation:
    """Result of checking a bar where the position was NOT yet open at the
    bar's start. No ``exit``/``exit_price`` field exists here by design --
    see the module docstring's "Same-bar entry+exit ambiguity" section:
    when ``resolution`` is ``AMBIGUOUS_INTRADAY_EXECUTION``, this module has
    deliberately NOT attempted to resolve an exit for this bar at all, and
    that resolution must NOT be read as "the position is now cleanly open".

    ``entry_same_bar_exit_ambiguity`` is kept as a convenience boolean
    (``True`` iff ``resolution is AMBIGUOUS_INTRADAY_EXECUTION``) -- the
    ``resolution`` enum is the authoritative signal; the boolean is a
    derived shorthand, never the other way around."""

    entry: EntryEvaluation
    stop_touched_same_bar: bool
    target_touched_same_bar: bool
    resolution: SameBarResolution
    entry_same_bar_exit_ambiguity: bool


@dataclass(frozen=True)
class TradeLevelsValidation:
    valid: bool
    violations: tuple[TradeLevelViolation, ...]


def _require_direction(direction: Direction) -> None:
    if direction not in (Direction.LONG, Direction.SHORT):
        raise InvalidDirectionError(f"execution requires LONG or SHORT, got {direction!r}")


# --- Entry ------------------------------------------------------------------


def evaluate_entry(entry: ResolvedEntry, bar: OHLCBar) -> EntryEvaluation:
    """Whether ``entry`` is touched on ``bar``, and the fill price.

    Direction-independent by construction -- a single, symmetric rule for
    LONG and SHORT alike, since "was this price level touched, and if so at
    what price" is pure price geometry, not a directional question.

    EXACT (price-touched semantics, not a real limit order): touched
    whenever ``low <= price <= high``, and the fill is ALWAYS exactly
    ``price`` when touched -- never a "better" price inferred from where
    ``open`` happened to be relative to it. This is deliberate: modeling
    intrabar order-queue mechanics (e.g. crediting a gap-through fill at
    ``open`` instead) would invent precision daily OHLC cannot support, so
    "touched" always fills at the level itself, symmetrically regardless of
    whether open gapped through it, straddled it, or landed exactly on it
    (open == price is simply one point already inside [low, high]).

    RANGE ``[low, high]``: touched whenever the bar's range intersects it.
    Fill price, conservative and deterministic, never retrospectively the
    best price reachable:
      - ``open`` already inside the range -> fill at ``open``;
      - ``open`` below the range and the bar reaches up into/through it ->
        fill at the range's low bound (the first boundary reached moving
        up from ``open``);
      - ``open`` above the range and the bar reaches down into/through it
        -> fill at the range's high bound (the first boundary reached
        moving down from ``open``).
    This single set of rules already covers a bar that only grazes one
    extreme (the boundary reached IS the fill) and a bar that spans clean
    through the whole range (still the first boundary reached from
    ``open``'s side, never the far one).
    """
    if entry.kind is EntryKind.NONE:
        return EntryEvaluation(touched=False, fill_price=None, fill_kind=None)

    if entry.kind is EntryKind.EXACT:
        price = entry.exact_price
        if bar.low <= price <= bar.high:
            return EntryEvaluation(touched=True, fill_price=price, fill_kind=EntryKind.EXACT)
        return EntryEvaluation(touched=False, fill_price=None, fill_kind=None)

    # EntryKind.RANGE
    low_bound, high_bound = entry.low, entry.high
    if low_bound <= bar.open <= high_bound:
        return EntryEvaluation(touched=True, fill_price=bar.open, fill_kind=EntryKind.RANGE)
    if bar.open < low_bound and bar.high >= low_bound:
        return EntryEvaluation(touched=True, fill_price=low_bound, fill_kind=EntryKind.RANGE)
    if bar.open > high_bound and bar.low <= high_bound:
        return EntryEvaluation(touched=True, fill_price=high_bound, fill_kind=EntryKind.RANGE)
    return EntryEvaluation(touched=False, fill_price=None, fill_kind=None)


# --- Stop / target touch + fill (position already open) ---------------------


def _stop_touched(direction: Direction, stop: float, bar: OHLCBar) -> bool:
    return bar.low <= stop if direction is Direction.LONG else bar.high >= stop


def _stop_exit_price(direction: Direction, stop: float, bar: OHLCBar) -> float:
    if direction is Direction.LONG:
        return bar.open if bar.open < stop else stop
    return bar.open if bar.open > stop else stop


def _target_touched(direction: Direction, target: float, bar: OHLCBar) -> bool:
    return bar.high >= target if direction is Direction.LONG else bar.low <= target


def _target_exit_price(direction: Direction, target: float, bar: OHLCBar) -> float:
    if direction is Direction.LONG:
        return bar.open if bar.open > target else target
    return bar.open if bar.open < target else target


def evaluate_exit(
    direction: Direction, stop: float | None, target: float | None, bar: OHLCBar,
) -> ExitEvaluation:
    """Stop/target resolution for a bar where the position is ALREADY open
    (entry is not this function's concern at all -- see
    ``evaluate_same_bar_entry_and_exit`` for the bar where entry itself is
    in question).

    LONG: stop touched if ``low <= stop``; target touched if
    ``high >= target``. SHORT: stop touched if ``high >= stop``; target
    touched if ``low <= target``.

    Gap handling, symmetric for both: if ``open`` already gapped past the
    level (LONG: ``open < stop`` or ``open > target``; SHORT: ``open >
    stop`` or ``open < target``), exit fills at ``open`` (the level was
    never actually available); otherwise exit fills exactly at the level.

    Both touched on the same bar: STOP wins (the already-agreed
    conservative v1 policy), and ``same_bar_ambiguous=True`` is always set
    in that case so the ambiguity is never hidden behind a clean-looking
    single-reason result. ``stop``/``target`` of ``None`` are simply never
    touched -- a plan missing either (or both) is not invalid on that
    account alone.
    """
    _require_direction(direction)
    stop_touched = stop is not None and _stop_touched(direction, stop, bar)
    target_touched = target is not None and _target_touched(direction, target, bar)

    if stop_touched and target_touched:
        return ExitEvaluation(
            stop_touched=True, target_touched=True, exit_reason=ExitReason.STOP,
            exit_price=_stop_exit_price(direction, stop, bar), same_bar_ambiguous=True,
        )
    if stop_touched:
        return ExitEvaluation(
            stop_touched=True, target_touched=False, exit_reason=ExitReason.STOP,
            exit_price=_stop_exit_price(direction, stop, bar), same_bar_ambiguous=False,
        )
    if target_touched:
        return ExitEvaluation(
            stop_touched=False, target_touched=True, exit_reason=ExitReason.TARGET,
            exit_price=_target_exit_price(direction, target, bar), same_bar_ambiguous=False,
        )
    return ExitEvaluation(
        stop_touched=False, target_touched=False, exit_reason=None,
        exit_price=None, same_bar_ambiguous=False,
    )


# --- Same-bar entry + exit (position NOT yet open at the bar's start) ------


def evaluate_same_bar_entry_and_exit(
    direction: Direction, entry: ResolvedEntry, stop: float | None, target: float | None, bar: OHLCBar,
) -> SameBarEntryExitEvaluation:
    """See the module docstring's "Same-bar entry+exit ambiguity" section.

    No exit is ever resolved by this function -- if entry is touched and
    the bar's OHLC would also touch stop and/or target, that is reported
    only via ``resolution=SameBarResolution.AMBIGUOUS_INTRADAY_EXECUTION``
    (equivalently ``entry_same_bar_exit_ambiguity=True``) plus which of the
    two were touched, never as a resolved ``ExitEvaluation`` (which would
    require assuming an intraday order daily OHLC does not contain). That
    resolution must be treated by the caller as its own distinct case --
    NOT as "entered and now normally open" -- see the module docstring.
    """
    _require_direction(direction)
    entry_eval = evaluate_entry(entry, bar)
    if not entry_eval.touched:
        return SameBarEntryExitEvaluation(
            entry=entry_eval, stop_touched_same_bar=False, target_touched_same_bar=False,
            resolution=SameBarResolution.NOT_TRIGGERED, entry_same_bar_exit_ambiguity=False,
        )

    stop_touched = stop is not None and _stop_touched(direction, stop, bar)
    target_touched = target is not None and _target_touched(direction, target, bar)
    ambiguous = stop_touched or target_touched
    return SameBarEntryExitEvaluation(
        entry=entry_eval, stop_touched_same_bar=stop_touched, target_touched_same_bar=target_touched,
        resolution=(
            SameBarResolution.AMBIGUOUS_INTRADAY_EXECUTION if ambiguous else SameBarResolution.CLEAN_ENTRY
        ),
        entry_same_bar_exit_ambiguity=ambiguous,
    )


# --- Level validation --------------------------------------------------------


def _stop_reference(direction: Direction, entry: ResolvedEntry) -> float | None:
    """The entry price that makes a stop check valid for EVERY possible
    fill within ``entry`` -- the boundary closest to the stop's required
    side, so the check holds regardless of exactly where in a RANGE the
    eventual fill lands."""
    if entry.kind is EntryKind.EXACT:
        return entry.exact_price
    if entry.kind is EntryKind.RANGE:
        return entry.low if direction is Direction.LONG else entry.high
    return None


def _target_reference(direction: Direction, entry: ResolvedEntry) -> float | None:
    if entry.kind is EntryKind.EXACT:
        return entry.exact_price
    if entry.kind is EntryKind.RANGE:
        return entry.high if direction is Direction.LONG else entry.low
    return None


def validate_trade_levels(
    direction: Direction, entry: ResolvedEntry, stop: float | None, target: float | None,
) -> TradeLevelsValidation:
    """Whether ``stop``/``target`` sit on the correct side of ``entry`` for
    ``direction`` -- a signal, never a correction (see module docstring:
    "no darle la vuelta automáticamente"). ``None`` stop/target, or
    ``entry.kind is NONE`` (nothing to compare against), are never
    violations by themselves.

    For a RANGE entry, the reference used is the boundary that makes the
    check valid no matter where in the range the eventual fill lands (see
    ``_stop_reference``/``_target_reference``) -- e.g. a LONG's stop is
    checked against the range's LOW bound, since a stop below the lowest
    possible fill is below every possible fill.
    """
    _require_direction(direction)
    violations: list[TradeLevelViolation] = []

    stop_ref = _stop_reference(direction, entry)
    if stop is not None and stop_ref is not None:
        if direction is Direction.LONG and not stop < stop_ref:
            violations.append(TradeLevelViolation.LONG_STOP_NOT_BELOW_ENTRY)
        if direction is Direction.SHORT and not stop > stop_ref:
            violations.append(TradeLevelViolation.SHORT_STOP_NOT_ABOVE_ENTRY)

    target_ref = _target_reference(direction, entry)
    if target is not None and target_ref is not None:
        if direction is Direction.LONG and not target > target_ref:
            violations.append(TradeLevelViolation.LONG_TARGET_NOT_ABOVE_ENTRY)
        if direction is Direction.SHORT and not target < target_ref:
            violations.append(TradeLevelViolation.SHORT_TARGET_NOT_BELOW_ENTRY)

    return TradeLevelsValidation(valid=not violations, violations=tuple(violations))
