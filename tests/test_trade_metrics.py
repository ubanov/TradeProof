"""Deterministic trade-plan metrics (fork README: Deterministic trade calculations)."""

from __future__ import annotations

import pytest

from tradingagents.agents.researchers.trade_metrics import (
    compute_trade_metrics,
    entry_reference,
    render_trade_metrics,
)


@pytest.mark.unit
def test_worked_example_matches_spec_numbers():
    metrics = compute_trade_metrics(
        entry_price=83000, take_profit=87000, stop_loss=80500, atr=2192.41
    )
    assert metrics.entry_reference == 83000
    assert metrics.risk == pytest.approx(2500)
    assert metrics.reward == pytest.approx(4000)
    assert metrics.reward_risk == pytest.approx(1.6)
    assert metrics.stop_atr == pytest.approx(2500 / 2192.41)
    assert metrics.target_atr == pytest.approx(4000 / 2192.41)
    assert metrics.risk_pct == pytest.approx(2500 / 83000 * 100)
    assert metrics.reward_pct == pytest.approx(4000 / 83000 * 100)


@pytest.mark.unit
def test_entry_range_uses_midpoint_as_reference():
    assert entry_reference(None, 82900, 83300) == pytest.approx(83100)
    metrics = compute_trade_metrics(entry_low=82900, entry_high=83300, take_profit=87360, stop_loss=80920)
    assert metrics.entry_reference == pytest.approx(83100)


@pytest.mark.unit
def test_one_sided_range_falls_back_to_the_present_bound():
    assert entry_reference(None, 100, None) == 100
    assert entry_reference(None, None, 200) == 200
    assert entry_reference(None, None, None) is None


@pytest.mark.unit
def test_point_entry_takes_precedence_over_range_fields():
    assert entry_reference(150, 100, 200) == 150


@pytest.mark.unit
def test_missing_entry_tp_sl_does_not_crash():
    metrics = compute_trade_metrics()
    assert metrics.entry_reference is None
    assert metrics.risk is None
    assert metrics.reward is None
    assert metrics.reward_risk is None
    assert metrics.stop_atr is None
    assert metrics.target_atr is None

    metrics = compute_trade_metrics(entry_price=100)
    assert metrics.entry_reference == 100
    assert metrics.risk is None
    assert metrics.reward is None

    metrics = compute_trade_metrics(entry_price=100, stop_loss=90)
    assert metrics.risk == pytest.approx(10)
    assert metrics.reward is None
    assert metrics.reward_risk is None  # no reward => no ratio, not a crash


@pytest.mark.unit
def test_zero_entry_and_zero_atr_do_not_crash():
    metrics = compute_trade_metrics(entry_price=0, take_profit=10, stop_loss=-10)
    assert metrics.entry_reference == 0
    assert metrics.risk is None
    assert metrics.reward is None

    metrics = compute_trade_metrics(entry_price=100, take_profit=110, stop_loss=90, atr=0)
    assert metrics.risk == pytest.approx(10)
    assert metrics.stop_atr is None
    assert metrics.target_atr is None


@pytest.mark.unit
def test_render_trade_metrics_handles_missing_values():
    metrics = compute_trade_metrics()
    rendered = render_trade_metrics(metrics)
    assert "N/A" in rendered
    assert "Reward/Risk" in rendered


# --- Entry-range R/R: reference R/R unchanged, deterministic min/max added --
# (see fork README: "Deterministic trade calculations" -- real XOM run, a
# single reference-entry R/R of 1.30 read as "the plan's R/R" when the actual
# range implied 0.93-1.86 depending on where the limit order fills).


@pytest.mark.unit
def test_single_entry_plan_is_completely_unaffected():
    """Scenario 1: a point entry must behave exactly as before -- no range
    fields populated, no change to any existing field."""
    metrics = compute_trade_metrics(
        entry_price=83000, take_profit=87000, stop_loss=80500, atr=2192.41
    )
    assert metrics.reward_risk == pytest.approx(1.6)
    assert metrics.risk_reward_min is None
    assert metrics.risk_reward_max is None


@pytest.mark.unit
def test_long_entry_range_xom_like_exposes_reference_and_range_rr():
    """Scenario 2: the real XOM case from the task spec."""
    metrics = compute_trade_metrics(
        entry_low=159.85, entry_high=161.97, take_profit=167.96, stop_loss=155.50
    )
    assert metrics.entry_reference == pytest.approx(160.91)
    assert metrics.reward_risk == pytest.approx(1.30, abs=0.01)
    # low=159.85 -> risk 4.35, reward 8.11 -> R/R ~1.86
    # high=161.97 -> risk 6.47, reward 5.99 -> R/R ~0.93
    assert metrics.risk_reward_max == pytest.approx(1.86, abs=0.01)
    assert metrics.risk_reward_min == pytest.approx(0.93, abs=0.01)


@pytest.mark.unit
def test_short_entry_range_uses_the_same_direction_agnostic_formula():
    """Scenario 3: SHORT (TP below entry, SL above) -- reuses the exact same
    abs()-based formula as LONG, no separate direction-aware code path."""
    metrics = compute_trade_metrics(
        entry_low=95, entry_high=105, take_profit=80, stop_loss=110
    )
    assert metrics.entry_reference == pytest.approx(100)
    assert metrics.reward_risk == pytest.approx(2.0)  # risk 10, reward 20
    # low=95 -> risk 15, reward 15 -> R/R 1.0
    # high=105 -> risk 5, reward 25 -> R/R 5.0
    assert metrics.risk_reward_min == pytest.approx(1.0)
    assert metrics.risk_reward_max == pytest.approx(5.0)


@pytest.mark.unit
def test_min_max_are_not_naively_assigned_to_low_and_high():
    """Scenario 4: in the XOM-like case above, the LOW endpoint produces the
    MAX ratio and the HIGH endpoint produces the MIN -- min/max must come
    from actually comparing both values, never assuming low=min."""
    metrics = compute_trade_metrics(
        entry_low=159.85, entry_high=161.97, take_profit=167.96, stop_loss=155.50
    )
    rr_at_low = (167.96 - 159.85) / (159.85 - 155.50)
    rr_at_high = (167.96 - 161.97) / (161.97 - 155.50)
    assert rr_at_low > rr_at_high  # the naive "low -> min" assumption would be wrong
    assert metrics.risk_reward_max == pytest.approx(rr_at_low)
    assert metrics.risk_reward_min == pytest.approx(rr_at_high)


@pytest.mark.unit
def test_no_entry_range_does_not_create_range_metrics():
    """Scenario 5: a single entry_price must never get range fields, even
    when take_profit/stop_loss would make a range calculation possible."""
    metrics = compute_trade_metrics(entry_price=100, take_profit=120, stop_loss=90)
    assert metrics.risk_reward_min is None
    assert metrics.risk_reward_max is None

    # A one-sided range (only one bound known) is not a real range either.
    metrics = compute_trade_metrics(entry_low=100, take_profit=120, stop_loss=90)
    assert metrics.risk_reward_min is None
    assert metrics.risk_reward_max is None


@pytest.mark.unit
def test_no_trade_entry_range_behaves_as_before():
    """Scenario 6: no entry/TP/SL at all (NO_TRADE-style) -- unchanged, and
    no range metrics invented from nothing."""
    metrics = compute_trade_metrics()
    assert metrics.entry_reference is None
    assert metrics.reward_risk is None
    assert metrics.risk_reward_min is None
    assert metrics.risk_reward_max is None


@pytest.mark.unit
def test_entry_range_endpoint_equal_to_stop_does_not_divide_by_zero():
    """Scenario 7: one endpoint exactly at the stop (risk=0 there) must not
    crash and must not contribute a 0/invented ratio -- only the other,
    genuinely computable endpoint feeds min/max."""
    metrics = compute_trade_metrics(
        entry_low=100, entry_high=120, take_profit=140, stop_loss=100
    )
    # low == stop_loss -> risk 0 there -> that endpoint's R/R is None, not 0.
    # high -> risk 20, reward 20 -> R/R 1.0 -- the only valid candidate.
    assert metrics.risk_reward_min == pytest.approx(1.0)
    assert metrics.risk_reward_max == pytest.approx(1.0)


@pytest.mark.unit
def test_entry_range_zero_endpoint_does_not_divide_by_zero():
    """An endpoint of exactly 0 must not raise computing its own risk_pct/
    reward_pct (division by the endpoint itself), independent of whether the
    reference entry is zero."""
    metrics = compute_trade_metrics(
        entry_low=0, entry_high=100, take_profit=150, stop_loss=-50
    )
    assert metrics.risk_reward_min is not None
    assert metrics.risk_reward_max is not None


@pytest.mark.unit
def test_render_shows_unambiguous_reference_and_range_labels_for_a_range():
    """Scenario 8: for a genuine range, the reference R/R must be clearly
    labelled as the reference entry's own ratio, distinct from the range."""
    metrics = compute_trade_metrics(
        entry_low=159.85, entry_high=161.97, take_profit=167.96, stop_loss=155.50
    )
    rendered = render_trade_metrics(metrics)
    assert "Reference R/R (at reference entry): 1.30" in rendered
    assert "R/R across entry range: 0.93 - 1.86" in rendered
    # The bare, unqualified "Reward/Risk:" label must not appear for a range
    # -- it is exactly the ambiguous presentation this feature replaces.
    assert "Reward/Risk:" not in rendered


@pytest.mark.unit
def test_render_single_entry_keeps_the_original_unqualified_label():
    """A single-entry plan keeps today's exact wording -- nothing to
    disambiguate when there is only one entry."""
    metrics = compute_trade_metrics(entry_price=100, take_profit=120, stop_loss=90)
    rendered = render_trade_metrics(metrics)
    assert "Reward/Risk: 2.00" in rendered
    assert "Reference R/R" not in rendered
    assert "R/R across entry range" not in rendered


@pytest.mark.unit
def test_trade_metrics_from_dict_roundtrips_range_fields():
    """The range fields must survive the plain-dict roundtrip used to store
    bull_trade_metrics/bear_trade_metrics in debate state."""
    from tradingagents.agents.researchers.trade_metrics import trade_metrics_from_dict

    data = {"entry_reference": 160.91, "reward_risk": 1.30, "risk_reward_min": 0.93,
            "risk_reward_max": 1.86}
    metrics = trade_metrics_from_dict(data)
    assert metrics.risk_reward_min == pytest.approx(0.93)
    assert metrics.risk_reward_max == pytest.approx(1.86)


@pytest.mark.unit
def test_risk_frozen_evidence_receives_the_range_metrics_too():
    """Scenario 9 (Risk/Portfolio Manager side): render_frozen_evidence reuses
    the same render_trade_metrics/trade_metrics_from_dict helpers, so the
    range line reaches Risk with no change needed in frozen_evidence.py."""
    from tradingagents.agents.risk_mgmt.frozen_evidence import render_frozen_evidence

    metrics = compute_trade_metrics(
        entry_low=159.85, entry_high=161.97, take_profit=167.96, stop_loss=155.50
    )
    state = {
        "company_of_interest": "XOM", "asset_type": "stock",
        "instrument_context": "INSTRUMENT", "portfolio_context": "",
        "market_report": "", "sentiment_report": "", "news_report": "",
        "fundamentals_report": "", "investment_plan": "", "trader_investment_plan": "",
        "investment_debate_state": {
            "bull_thesis": {}, "bear_thesis": {},
            "bull_trade_metrics": {
                "entry_reference": metrics.entry_reference,
                "reward_risk": metrics.reward_risk,
                "risk_reward_min": metrics.risk_reward_min,
                "risk_reward_max": metrics.risk_reward_max,
            },
            "bear_trade_metrics": {},
        },
    }
    rendered = render_frozen_evidence(state)
    assert "Reference R/R (at reference entry): 1.30" in rendered
    assert "R/R across entry range: 0.93 - 1.86" in rendered
