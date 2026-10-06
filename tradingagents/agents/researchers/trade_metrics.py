"""Deterministic, code-computed trade-plan metrics for the research debate.

The LLM commits to entry/take-profit/stop-loss levels; every number derived
from them (risk, reward, ATR distances) is computed here in plain arithmetic
so it cannot drift from what the model typed -- the same principle already
applied to arithmetic-claim verification (see ``researchers/arithmetic.py``).
"""

from __future__ import annotations

from dataclasses import dataclass


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
    )


def render_trade_metrics(metrics: TradeMetrics) -> str:
    def fmt(value: float | None, suffix: str = "") -> str:
        return f"{value:.2f}{suffix}" if value is not None else "N/A"

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
