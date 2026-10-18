"""Directional trade-geometry validation (see fork README: "Trade geometry
validation" / NVDA 2026-01-07 regression).

A directional plan's stop-loss and take-profit must be on the correct side
of entry for its own stated direction:

    LONG / BUY / BULL:   stop_loss < entry_reference < take_profit
    SHORT / SELL / BEAR: take_profit < entry_reference < stop_loss

Real regression: in the NVDA 2026-01-07 run, the Bear's second review round
mistook the Bull's stop-loss (182.90) for its own prior value and proposed
REVISE 182.90 -> 177.80 using LONG-side reasoning ("2x ATR below the
close") applied to a SHORT thesis. The mechanical apply left
``bear_thesis.stop_loss = 177.8`` -- BELOW entry, invalid for a SHORT --
and ``trade_metrics.py``'s ``abs()``-based formulas computed a
plausible-looking ``reward_risk = 0.77`` from it, hiding the sign error
from every downstream reader (Research Verifier, Research Manager, Trader,
Risk). The Research Verifier reviewed this exact claim and cleared it
because it checked only the ATR-distance arithmetic, never geometry.

This suite covers three layers: the pure geometry check
(``trade_metrics.trade_geometry_violations`` / ``validator_deterministic.
check_trade_geometry``), the metrics gate (``compute_trade_metrics``'s
``geometry_valid``), and the structural guard that rejects an invalid
review/repair change before it ever reaches the canonical thesis
(``thesis_flow.apply_validated_trade_plan_changes``).
"""

from __future__ import annotations

import pytest

from tradingagents.agents.researchers.thesis_flow import (
    apply_trade_plan_changes,
    apply_validated_trade_plan_changes,
)
from tradingagents.agents.researchers.trade_metrics import (
    compute_trade_metrics,
    entry_reference,
    render_trade_metrics,
    resolve_trade_direction,
    trade_geometry_violations,
)
from tradingagents.agents.validation_schemas import ValidationCategory
from tradingagents.agents.validator_deterministic import check_trade_geometry

# ---------------------------------------------------------------------------
# resolve_trade_direction / trade_geometry_violations (pure logic)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("label", ["BUY", "buy", " Buy ", "LONG", "BULL", "bull"])
def test_resolve_trade_direction_recognizes_every_long_label(label):
    assert resolve_trade_direction(label) == "LONG"


@pytest.mark.unit
@pytest.mark.parametrize("label", ["SELL", "sell", "SHORT", "BEAR", "bear"])
def test_resolve_trade_direction_recognizes_every_short_label(label):
    assert resolve_trade_direction(label) == "SHORT"


@pytest.mark.unit
@pytest.mark.parametrize("label", ["HOLD", "Hold", None, "", "UNKNOWN", "REVIEW"])
def test_resolve_trade_direction_returns_none_for_non_directional_labels(label):
    assert resolve_trade_direction(label) is None


# --- Valid geometry: PASS --------------------------------------------------


@pytest.mark.unit
def test_long_valid_geometry_passes():
    assert trade_geometry_violations("LONG", entry=100, stop_loss=95, take_profit=110) == []
    assert check_trade_geometry("BUY", entry_price=100, stop_loss=95, take_profit=110) is None


@pytest.mark.unit
def test_short_valid_geometry_passes():
    assert trade_geometry_violations("SHORT", entry=100, stop_loss=105, take_profit=90) == []
    assert check_trade_geometry("SELL", entry_price=100, stop_loss=105, take_profit=90) is None


# --- Invalid geometry: FAIL -------------------------------------------------


@pytest.mark.unit
def test_long_stop_above_entry_fails():
    violations = trade_geometry_violations("LONG", entry=100, stop_loss=105, take_profit=110)
    assert violations
    finding = check_trade_geometry("BUY", entry_price=100, stop_loss=105, take_profit=110)
    assert finding is not None
    assert finding.category == ValidationCategory.TRADE_GEOMETRY_ERROR


@pytest.mark.unit
def test_long_target_below_entry_fails():
    violations = trade_geometry_violations("LONG", entry=100, stop_loss=95, take_profit=90)
    assert violations
    finding = check_trade_geometry("BULL", entry_price=100, stop_loss=95, take_profit=90)
    assert finding is not None
    assert finding.category == ValidationCategory.TRADE_GEOMETRY_ERROR


@pytest.mark.unit
def test_short_stop_below_entry_fails():
    """The exact shape of the NVDA regression: SHORT with stop UNDER entry."""
    violations = trade_geometry_violations("SHORT", entry=100, stop_loss=95, take_profit=90)
    assert violations
    finding = check_trade_geometry("BEAR", entry_price=100, stop_loss=95, take_profit=90)
    assert finding is not None
    assert finding.category == ValidationCategory.TRADE_GEOMETRY_ERROR


@pytest.mark.unit
def test_short_target_above_entry_fails():
    violations = trade_geometry_violations("SHORT", entry=100, stop_loss=105, take_profit=110)
    assert violations
    finding = check_trade_geometry("SELL", entry_price=100, stop_loss=105, take_profit=110)
    assert finding is not None
    assert finding.category == ValidationCategory.TRADE_GEOMETRY_ERROR


# --- Entry ranges, using the same entry_reference() as production ----------


@pytest.mark.unit
def test_bull_entry_range_valid_geometry_passes():
    entry = entry_reference(None, 186.4, 189.0)  # production BULL range
    assert trade_geometry_violations("BULL", entry, stop_loss=182.9, take_profit=195.03) == []
    finding = check_trade_geometry(
        "BULL", entry_low=186.4, entry_high=189.0, stop_loss=182.9, take_profit=195.03,
    )
    assert finding is None


@pytest.mark.unit
def test_bear_entry_range_valid_geometry_passes():
    entry = entry_reference(None, 189.0, 190.5)  # production BEAR range
    assert trade_geometry_violations("BEAR", entry, stop_loss=195.2, take_profit=180.6) == []
    finding = check_trade_geometry(
        "BEAR", entry_low=189.0, entry_high=190.5, stop_loss=195.2, take_profit=180.6,
    )
    assert finding is None


@pytest.mark.unit
def test_bear_entry_range_invalid_geometry_fails():
    """The exact NVDA shape with a range entry instead of a pre-resolved reference."""
    finding = check_trade_geometry(
        "BEAR", entry_low=189.0, entry_high=190.5, stop_loss=177.8, take_profit=180.6,
    )
    assert finding is not None
    assert finding.category == ValidationCategory.TRADE_GEOMETRY_ERROR


@pytest.mark.unit
def test_reference_entry_takes_precedence_over_point_or_range():
    """Trader's own reference_entry (e.g. a calculated range reference)
    overrides entry_price/entry_low/entry_high when given."""
    finding = check_trade_geometry(
        "SELL", entry_price=50, reference_entry=100, stop_loss=95, take_profit=90,
    )
    assert finding is not None  # fails against reference_entry=100, not entry_price=50
    assert check_trade_geometry(
        "SELL", entry_price=50, reference_entry=100, stop_loss=105, take_profit=90,
    ) is None


# --- Null fields / Hold / no direction: never a false positive --------------


@pytest.mark.unit
def test_missing_stop_loss_is_not_checked():
    assert trade_geometry_violations("LONG", entry=100, stop_loss=None, take_profit=110) == []


@pytest.mark.unit
def test_missing_take_profit_is_not_checked():
    assert trade_geometry_violations("SHORT", entry=100, stop_loss=105, take_profit=None) == []


@pytest.mark.unit
def test_missing_entry_is_not_checked_never_invented():
    assert trade_geometry_violations("LONG", entry=None, stop_loss=105, take_profit=90) == []
    assert check_trade_geometry("BUY", stop_loss=105, take_profit=90) is None


@pytest.mark.unit
def test_hold_never_produces_a_finding_regardless_of_levels():
    assert trade_geometry_violations("HOLD", entry=100, stop_loss=105, take_profit=90) == []
    assert check_trade_geometry("Hold", entry_price=100, stop_loss=105, take_profit=90) is None


@pytest.mark.unit
def test_no_direction_at_all_never_produces_a_finding():
    assert trade_geometry_violations(None, entry=100, stop_loss=105, take_profit=90) == []
    assert check_trade_geometry(None, entry_price=100, stop_loss=105, take_profit=90) is None


# ---------------------------------------------------------------------------
# compute_trade_metrics: geometry gate (section 5 / 11 -- never silently
# present a plausible reward_risk for an invalid plan).
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_metrics_unaffected_when_direction_omitted():
    """Backward compatibility: a caller that never passes direction (every
    caller before this feature existed) gets exactly the old behavior."""
    metrics = compute_trade_metrics(entry_price=100, stop_loss=105, take_profit=90, atr=5)
    assert metrics.geometry_valid is None
    assert metrics.reward_risk is not None  # still computed, unchanged


@pytest.mark.unit
def test_metrics_valid_geometry_computes_normally():
    metrics = compute_trade_metrics(
        direction="BULL", entry_price=100, stop_loss=95, take_profit=110, atr=5,
    )
    assert metrics.geometry_valid is True
    assert metrics.risk == pytest.approx(5)
    assert metrics.reward == pytest.approx(10)
    assert metrics.reward_risk == pytest.approx(2.0)


@pytest.mark.unit
def test_metrics_invalid_geometry_never_computes_a_plausible_reward_risk():
    """The exact NVDA regression numbers: entry ~189.75, take_profit 180.60,
    stop_loss 177.80 for a BEAR -- abs()-based math alone would produce
    reward_risk ~0.77, masking the sign error. It must not."""
    metrics = compute_trade_metrics(
        direction="BEAR", entry_price=189.75, stop_loss=177.8, take_profit=180.6, atr=5.45,
    )
    assert metrics.geometry_valid is False
    assert metrics.reward_risk is None
    assert metrics.risk is None
    assert metrics.reward is None
    assert metrics.risk_pct is None
    assert metrics.reward_pct is None
    assert metrics.stop_atr is None
    assert metrics.target_atr is None
    # entry_reference is still reported -- it is independently correct and
    # useful, unlike the risk/reward figures that depend on the broken pair.
    assert metrics.entry_reference == pytest.approx(189.75)


@pytest.mark.unit
def test_render_trade_metrics_states_invalid_geometry_explicitly():
    metrics = compute_trade_metrics(
        direction="BEAR", entry_price=189.75, stop_loss=177.8, take_profit=180.6,
    )
    rendered = render_trade_metrics(metrics)
    assert "INVALID TRADE GEOMETRY" in rendered
    assert "0.7" not in rendered  # no masked reward_risk anywhere in the text
    assert "Reward/Risk" not in rendered


@pytest.mark.unit
def test_render_trade_metrics_unchanged_for_valid_geometry():
    metrics = compute_trade_metrics(
        direction="BULL", entry_price=100, stop_loss=95, take_profit=110, atr=5,
    )
    rendered = render_trade_metrics(metrics)
    assert "INVALID TRADE GEOMETRY" not in rendered
    assert "Reward/Risk: 2.00" in rendered


@pytest.mark.unit
def test_trade_metrics_from_dict_roundtrips_geometry_valid():
    from tradingagents.agents.researchers.trade_metrics import trade_metrics_from_dict

    metrics = trade_metrics_from_dict({"entry_reference": 189.75, "geometry_valid": False})
    assert metrics.geometry_valid is False


# ---------------------------------------------------------------------------
# apply_validated_trade_plan_changes: reject-and-revert at the point of
# application (section 3 / 4) -- the structural fix for the actual bug.
# ---------------------------------------------------------------------------


def _bear_thesis(stop_loss=195.2, take_profit=180.6):
    return {
        "direction": "BEAR",
        "entry": {"type": "range", "price": None, "low": 189.0, "high": 190.5},
        "take_profit": take_profit,
        "stop_loss": stop_loss,
        "conviction": {"total": 62},
    }


def _revise_stop_review(new_stop, old_stop=182.9):
    """Shape of the real NVDA review_dict (call 009_Bear_Researcher_structured)
    that mistook the Bull's stop (182.90) for Bear's own and revised to a
    LONG-side level."""
    return {
        "trade_plan_changes": {
            "entry": {"action": "KEEP", "new": None, "old": None, "reason": ""},
            "stop_loss": {
                "action": "REVISE", "new": new_stop, "old": old_stop,
                "reason": "2x ATR below the close",
            },
            "take_profit": {"action": "KEEP", "new": None, "old": None, "reason": ""},
        },
        "conviction_after": 62,
    }


@pytest.mark.unit
def test_nvda_regression_invalid_revise_is_rejected_not_consolidated():
    """The exact regression: BEAR, entry range 189.0-190.5 (reference
    189.75), take_profit=180.6, original stop=195.2, proposed revised
    stop=177.8. Must produce TRADE_GEOMETRY_ERROR / an equivalent FAIL
    finding, and 177.8 must never end up as bear_thesis.stop_loss."""
    thesis = _bear_thesis(stop_loss=195.2, take_profit=180.6)
    review = _revise_stop_review(new_stop=177.8)

    updated, retired, finding = apply_validated_trade_plan_changes(thesis, review)

    assert finding is not None
    assert finding.category == ValidationCategory.TRADE_GEOMETRY_ERROR
    assert updated["stop_loss"] == 195.2  # the last known geometrically-valid value
    assert updated["stop_loss"] != 177.8
    assert updated["take_profit"] == 180.6
    assert retired == []  # nothing was actually retired -- 195.2 stays active


@pytest.mark.unit
def test_rejected_candidate_proves_the_raw_apply_would_have_corrupted_state():
    """Sanity check that the scenario is real: the unguarded
    apply_trade_plan_changes (still used for other purposes) DOES consolidate
    177.8, confirming the guard is actually doing something."""
    thesis = _bear_thesis(stop_loss=195.2, take_profit=180.6)
    review = _revise_stop_review(new_stop=177.8)
    raw_updated, _ = apply_trade_plan_changes(thesis, review)
    assert raw_updated["stop_loss"] == 177.8


@pytest.mark.unit
def test_valid_revise_is_consolidated_normally():
    thesis = _bear_thesis(stop_loss=195.2, take_profit=180.6)
    review = _revise_stop_review(new_stop=196.5)  # still above entry -- valid for SHORT

    updated, retired, finding = apply_validated_trade_plan_changes(thesis, review)

    assert finding is None
    assert updated["stop_loss"] == 196.5
    assert retired == [195.2]  # genuinely replaced -- correctly retired


@pytest.mark.unit
def test_repair_reproducing_the_same_invalid_geometry_is_rejected_again():
    """Section 4: if repair proposes the SAME broken geometry again, it must
    be rejected the same way -- never accepted on a second attempt, and the
    last known-valid state is what's kept."""
    thesis = _bear_thesis(stop_loss=195.2, take_profit=180.6)
    repair_review = _revise_stop_review(new_stop=177.8)

    updated, retired, finding = apply_validated_trade_plan_changes(thesis, repair_review)

    assert finding is not None
    assert updated["stop_loss"] == 195.2
    assert retired == []


@pytest.mark.unit
def test_keep_action_is_never_flagged():
    """A round that proposes no trade-plan change at all must never trigger
    a spurious geometry finding."""
    thesis = _bear_thesis(stop_loss=195.2, take_profit=180.6)
    review = {
        "trade_plan_changes": {
            "entry": {"action": "KEEP", "new": None, "old": None, "reason": ""},
            "stop_loss": {"action": "KEEP", "new": None, "old": None, "reason": ""},
            "take_profit": {"action": "KEEP", "new": None, "old": None, "reason": ""},
        },
        "conviction_after": 62,
    }
    updated, retired, finding = apply_validated_trade_plan_changes(thesis, review)
    assert finding is None
    assert updated["stop_loss"] == 195.2
    assert updated["take_profit"] == 180.6


@pytest.mark.unit
def test_no_direction_thesis_is_never_checked():
    """A thesis with no direction at all (should not normally happen, but
    must never crash or false-positive) skips the check entirely."""
    thesis = {
        "direction": None,
        "entry": {"type": "point", "price": 100, "low": None, "high": None},
        "take_profit": 90, "stop_loss": 95, "conviction": {"total": 50},
    }
    review = {
        "trade_plan_changes": {
            "entry": {"action": "KEEP", "new": None, "old": None, "reason": ""},
            "stop_loss": {"action": "REVISE", "new": 80, "old": 95, "reason": "x"},
            "take_profit": {"action": "KEEP", "new": None, "old": None, "reason": ""},
        },
        "conviction_after": 50,
    }
    updated, retired, finding = apply_validated_trade_plan_changes(thesis, review)
    assert finding is None
    assert updated["stop_loss"] == 80
