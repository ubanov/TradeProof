"""Deterministic, code-computed trade-plan metrics for the research debate.

The LLM commits to entry/take-profit/stop-loss levels; every number derived
from them (risk, reward, ATR distances) is computed here in plain arithmetic
so it cannot drift from what the model typed -- the same principle already
applied to arithmetic-claim verification (see ``researchers/arithmetic.py``).

Directional geometry (see fork README: "Trade geometry validation" /
NVDA 2026-01-07 regression): the risk/reward formulas below use ``abs()`` and
are intentionally direction-agnostic for MAGNITUDE -- but a trade whose
stop-loss/take-profit are on the WRONG side of entry for its own stated
direction (e.g. a SHORT with its stop below entry instead of above) is not a
"small" or a "large" risk/reward, it is not an executable plan at all.
``abs()`` would otherwise compute a plausible-looking positive risk/reward
from exactly such a broken plan, silently hiding the sign error from every
downstream reader. When a caller supplies ``direction``, geometry is checked
first and invalid metrics are never computed from it (see
``trade_geometry_violations``, ``compute_trade_metrics``'s ``geometry_valid``
field). Omitting ``direction`` (the only way to call this module before this
check existed) preserves the original behavior exactly -- the check is
opt-in per call, never guessed from the numbers alone.
"""

from __future__ import annotations

from dataclasses import dataclass

# Direction labels accepted across the codebase for "this is a LONG/BUY/BULL
# plan" and "this is a SHORT/SELL/BEAR plan" -- the Trader's 3-tier action
# (schemas.TraderAction: "Buy"/"Hold"/"Sell") and the research thesis's
# 2-tier direction (schemas.ResearchDirection: "BULL"/"BEAR") are both
# accepted here, case-insensitively, so one geometry rule serves both.
_LONG_DIRECTIONS = frozenset({"BUY", "LONG", "BULL"})
_SHORT_DIRECTIONS = frozenset({"SELL", "SHORT", "BEAR"})


def resolve_trade_direction(direction: str | None) -> str | None:
    """Normalize a direction label to ``"LONG"``/``"SHORT"``, or ``None`` when
    it names neither (Hold, an unrecognized label, or nothing at all) -- the
    single place every geometry-aware caller resolves this, so "BUY"/"LONG"/
    "BULL" and "SELL"/"SHORT"/"BEAR" are always treated identically."""
    normalized = (direction or "").strip().upper()
    if normalized in _LONG_DIRECTIONS:
        return "LONG"
    if normalized in _SHORT_DIRECTIONS:
        return "SHORT"
    return None


def trade_geometry_violations(
    direction: str | None,
    entry: float | None,
    stop_loss: float | None = None,
    take_profit: float | None = None,
) -> list[str]:
    """Human-readable violations of directional trade geometry.

    LONG/BUY/BULL: ``stop_loss`` must be below ``entry``, ``take_profit``
    above it. SHORT/SELL/BEAR: the reverse. Returns ``[]`` (never invalid)
    when the direction doesn't resolve to LONG/SHORT (e.g. Hold), when
    ``entry`` is unresolvable, or when a given level is simply absent -- a
    missing field is never treated as a violation, and no field is ever
    invented or corrected here.
    """
    resolved = resolve_trade_direction(direction)
    if resolved is None or entry is None:
        return []

    is_long = resolved == "LONG"
    violations = []
    if stop_loss is not None:
        if is_long and stop_loss >= entry:
            violations.append(
                f"stop_loss={stop_loss} is not below entry={entry} for a LONG/BUY/BULL plan"
            )
        elif not is_long and stop_loss <= entry:
            violations.append(
                f"stop_loss={stop_loss} is not above entry={entry} for a SHORT/SELL/BEAR plan"
            )
    if take_profit is not None:
        if is_long and take_profit <= entry:
            violations.append(
                f"take_profit={take_profit} is not above entry={entry} for a LONG/BUY/BULL plan"
            )
        elif not is_long and take_profit >= entry:
            violations.append(
                f"take_profit={take_profit} is not below entry={entry} for a SHORT/SELL/BEAR plan"
            )
    return violations


@dataclass(frozen=True)
class TradeMetrics:
    entry_reference: float | None = None
    risk: float | None = None
    reward: float | None = None
    risk_pct: float | None = None
    reward_pct: float | None = None
    reward_risk: float | None = None
    stop_atr: float | None = None
    target_atr: float | None = None
    # Only populated for a genuine entry RANGE (both low and high given, no
    # single point price) -- the deterministic reward/risk at each endpoint,
    # min/max of whatever is actually computable. ``reward_risk`` above stays
    # exactly what it always was: the ratio at ``entry_reference`` (the
    # range's midpoint) alone, never "the plan's R/R" -- see
    # ``render_trade_metrics``.
    risk_reward_min: float | None = None
    risk_reward_max: float | None = None
    # None when no direction was supplied to compute_trade_metrics (geometry
    # not checked -- see module docstring); True/False when it was. False
    # means every risk/reward/ATR-distance field above is deliberately left
    # None rather than computed via abs(), which would otherwise mask the
    # sign error.
    geometry_valid: bool | None = None


def entry_reference(
    entry_price: float | None, entry_low: float | None, entry_high: float | None
) -> float | None:
    """The single reference entry used for derived calculations.

    A point entry is used directly. A range entry uses the midpoint of
    low/high -- the documented default rule for entry-range handling (see
    fork README) -- and a one-sided range falls back to whichever bound is
    present rather than silently picking one.
    """
    if entry_price is not None:
        return entry_price
    if entry_low is not None and entry_high is not None:
        return (entry_low + entry_high) / 2
    return entry_low if entry_low is not None else entry_high


def _scalar_metrics(
    entry: float, take_profit: float | None, stop_loss: float | None, atr: float | None
) -> tuple[float | None, ...]:
    """Risk/reward at one specific entry price.

    Direction-agnostic by construction (``abs()`` on both legs), the same
    formula already used for the single/reference entry -- reused as-is for
    a range's endpoints rather than duplicated, so a SHORT thesis (TP below
    entry, SL above it) is handled identically to a LONG one.
    """
    risk = abs(entry - stop_loss) if stop_loss is not None else None
    reward = abs(take_profit - entry) if take_profit is not None else None
    risk_pct = (risk / abs(entry) * 100) if risk is not None and entry != 0 else None
    reward_pct = (reward / abs(entry) * 100) if reward is not None and entry != 0 else None
    reward_risk = reward / risk if reward is not None and risk not in (None, 0) else None
    stop_atr = risk / atr if risk is not None and atr not in (None, 0) else None
    target_atr = reward / atr if reward is not None and atr not in (None, 0) else None
    return risk, reward, risk_pct, reward_pct, reward_risk, stop_atr, target_atr


def compute_trade_metrics(
    *,
    direction: str | None = None,
    entry_price: float | None = None,
    entry_low: float | None = None,
    entry_high: float | None = None,
    take_profit: float | None = None,
    stop_loss: float | None = None,
    atr: float | None = None,
) -> TradeMetrics:
    """Compute deterministic trade metrics.

    Any missing or zero input yields ``None`` fields rather than raising --
    a thesis with no stop-loss still gets an entry reference and whatever
    else is computable, instead of losing the whole metrics block.

    ``direction`` ("BUY"/"LONG"/"BULL"/"SELL"/"SHORT"/"BEAR", case-
    insensitive) is optional and, when given, gates everything below it: if
    the resulting stop_loss/take_profit are on the wrong side of entry for
    that direction, NO risk/reward/ATR-distance figure is computed at all
    (``geometry_valid=False``, every such field stays ``None``) -- an
    invalid plan's "risk" and "reward" are not small numbers, they are not
    computable, and ``abs()`` must never be allowed to manufacture a
    plausible-looking ratio from them (see module docstring). Omitting
    ``direction`` skips this check entirely, preserving this function's
    original direction-agnostic behavior for any caller that predates it.

    For a genuine entry RANGE (``entry_low``/``entry_high`` both given, no
    single ``entry_price``), also computes the deterministic reward/risk at
    each endpoint and exposes their min/max (``risk_reward_min``/
    ``risk_reward_max``) -- the reference entry, its own reward_risk, and
    take_profit/stop_loss are unchanged; this never picks an "optimal" entry
    or feeds into any decision, it only makes the range the single
    ``reward_risk`` at the midpoint already hid explicit again.
    """
    entry = entry_reference(entry_price, entry_low, entry_high)
    if entry is None or entry == 0:
        return TradeMetrics(entry_reference=entry)

    geometry_valid = None
    if direction is not None:
        geometry_valid = not trade_geometry_violations(direction, entry, stop_loss, take_profit)
        if not geometry_valid:
            return TradeMetrics(entry_reference=entry, geometry_valid=False)

    risk, reward, risk_pct, reward_pct, reward_risk, stop_atr, target_atr = _scalar_metrics(
        entry, take_profit, stop_loss, atr
    )

    risk_reward_min = risk_reward_max = None
    is_range = entry_price is None and entry_low is not None and entry_high is not None
    if is_range:
        rr_low = _scalar_metrics(entry_low, take_profit, stop_loss, atr)[4]
        rr_high = _scalar_metrics(entry_high, take_profit, stop_loss, atr)[4]
        candidates = [value for value in (rr_low, rr_high) if value is not None]
        if candidates:
            risk_reward_min = min(candidates)
            risk_reward_max = max(candidates)

    return TradeMetrics(
        entry_reference=entry,
        risk=risk,
        reward=reward,
        risk_pct=risk_pct,
        reward_pct=reward_pct,
        reward_risk=reward_risk,
        stop_atr=stop_atr,
        target_atr=target_atr,
        risk_reward_min=risk_reward_min,
        risk_reward_max=risk_reward_max,
        geometry_valid=geometry_valid,
    )


def trade_metrics_from_dict(data: dict | None) -> TradeMetrics:
    """Rehydrate a ``TradeMetrics`` from the plain dict stored in debate state."""
    data = data or {}
    return TradeMetrics(
        entry_reference=data.get("entry_reference"),
        risk=data.get("risk"),
        reward=data.get("reward"),
        risk_pct=data.get("risk_pct"),
        reward_pct=data.get("reward_pct"),
        reward_risk=data.get("reward_risk"),
        stop_atr=data.get("stop_atr"),
        target_atr=data.get("target_atr"),
        risk_reward_min=data.get("risk_reward_min"),
        risk_reward_max=data.get("risk_reward_max"),
        geometry_valid=data.get("geometry_valid"),
    )


def render_trade_metrics(metrics: TradeMetrics) -> str:
    def fmt(value: float | None, suffix: str = "") -> str:
        return f"{value:.2f}{suffix}" if value is not None else "N/A"

    if metrics.geometry_valid is False:
        # Deliberately no Risk/Reward/R:R/ATR-distance lines here -- those
        # would require computing abs(entry - stop_loss) etc., which would
        # manufacture a plausible-looking positive figure from a plan that
        # is not executable at all (see module docstring). Every reader of
        # this text (Verifier, Research Manager, Trader, Risk) sees the
        # same explicit statement instead of a silently-wrong number.
        return "\n".join([
            f"Entry (reference): {fmt(metrics.entry_reference)}",
            "INVALID TRADE GEOMETRY: stop-loss/take-profit are not on the "
            "correct side of entry for this position's own stated direction "
            "(LONG/BUY/BULL needs stop below and target above entry; "
            "SHORT/SELL/BEAR needs the reverse). Risk/reward/ATR-distance "
            "figures are not computed -- this is a structural execution "
            "error, not a minor discrepancy.",
        ])

    has_range = metrics.risk_reward_min is not None or metrics.risk_reward_max is not None
    reward_risk_lines = (
        [
            f"Reference R/R (at reference entry): {fmt(metrics.reward_risk)}",
            f"R/R across entry range: {fmt(metrics.risk_reward_min)} - {fmt(metrics.risk_reward_max)}",
        ]
        if has_range
        else [f"Reward/Risk: {fmt(metrics.reward_risk)}"]
    )

    return "\n".join(
        [
            f"Entry (reference): {fmt(metrics.entry_reference)}",
            f"Risk: {fmt(metrics.risk)} ({fmt(metrics.risk_pct, '%')})",
            f"Reward: {fmt(metrics.reward)} ({fmt(metrics.reward_pct, '%')})",
            *reward_risk_lines,
            f"Stop distance (ATR multiples): {fmt(metrics.stop_atr)}",
            f"Target distance (ATR multiples): {fmt(metrics.target_atr)}",
        ]
    )
