"""Risk/Manager integrity false-positive fixes from the first real NVDA Risk
architecture validation run (Risk Integrity: WARN with 11 findings, several
spurious). Three independent issues, all in the shared
``tradingagents.agents.integrity_shared`` module:

A. WITHDRAWN_VALUE_REUSED's tolerance was loose enough (0.1% relative) to
   treat a canonical replacement a cent away from a retired value (240.65
   retired -> 240.64 canonical) as the same retired value.
B. UNIT_ERROR fired on any LEVEL_WORDS sentence sharing space with any
   percentage, even a legitimate distance/share-of-profit/ratio percentage.
C. NEW_UNSUPPORTED_THRESHOLD fired on a comparison-style trigger word
   ("above", "below", "exceeds", ...) even in purely descriptive/
   comparison/arithmetic prose with no conditional "new rule" semantics.
"""

from __future__ import annotations

import pytest

from tradingagents.agents.integrity_shared import (
    find_unit_errors,
    find_withdrawn_values_reused,
)
from tradingagents.agents.managers.integrity_check import check_manager_integrity
from tradingagents.agents.risk_mgmt.integrity_check import check_risk_integrity


def _debate(**overrides):
    base = {
        "bull_thesis": {}, "bear_thesis": {},
        "bull_trade_metrics": {}, "bear_trade_metrics": {},
        "withdrawn_values": [],
    }
    base.update(overrides)
    return base


def _risk_state(**overrides):
    base = {
        "investment_debate_state": _debate(),
        "trader_investment_plan": "**Action**: Buy\n**Entry Price**: 234.69\n**Stop Loss**: 211.28",
        "market_report": "", "sentiment_report": "", "news_report": "", "fundamentals_report": "",
    }
    base.update(overrides)
    return base


# --- A: canonical replacement vs withdrawn value -----------------------------


@pytest.mark.unit
def test_direct_tolerance_distinguishes_a_cent_apart_canonical_values():
    # Retired 240.65; canonical replacement 240.64 must not match.
    assert find_withdrawn_values_reused("The level is 240.64.", [240.65]) == []
    # Exact reuse of the retired value is still caught.
    assert find_withdrawn_values_reused("The level is 240.65.", [240.65]) != []


@pytest.mark.unit
def test_manager_reusing_canonical_replacement_after_tiny_correction_passes():
    debate = _debate(withdrawn_values=[240.65])
    manager_text = "The invalidation level is 240.64, consistent with the corrected research plan."
    result = check_manager_integrity(manager_text, debate, "", "", "", "")
    assert result.status == "PASS"


@pytest.mark.unit
def test_manager_reusing_the_actual_retired_value_still_warns():
    debate = _debate(withdrawn_values=[240.65])
    manager_text = "The invalidation level remains 240.65 as originally proposed."
    result = check_manager_integrity(manager_text, debate, "", "", "", "")
    assert result.status == "WARN"


@pytest.mark.unit
def test_risk_reusing_canonical_replacement_after_tiny_correction_passes():
    # 240.64 is the corrected canonical value, now part of the verified
    # research evidence Risk was given (not just a bare number nobody sourced).
    state = _risk_state(
        investment_debate_state=_debate(
            withdrawn_values=[240.65], bull_trade_metrics={"stop_atr": 240.64},
        ),
    )
    text = "Invalidation sits at 240.64, the corrected level."
    assert check_risk_integrity(text, state).status == "PASS"


@pytest.mark.unit
def test_sign_normalization_of_8_8_and_negative_8_8_still_works_after_tolerance_fix():
    """The tolerance tightening must not break safe sign normalization."""
    assert find_withdrawn_values_reused("declined -8.8% sequentially", [-8.8]) != []
    assert find_withdrawn_values_reused("grew 8.8% sequentially", [8.8]) != []
    # And the two signs remain distinct from each other.
    assert find_withdrawn_values_reused("grew 8.8% sequentially", [-8.8]) == []


# --- B: UNIT_ERROR false positives -------------------------------------------


@pytest.mark.unit
def test_percentage_distance_between_two_known_prices_is_not_a_unit_error():
    text = "A move from the stop 211.28 to entry 217.23 allows ~2.6% additional movement."
    assert find_unit_errors(text) == []


@pytest.mark.unit
def test_percentage_of_quarterly_profit_is_not_a_unit_error():
    text = "The target exposure represents roughly 10.9% of quarterly profit at risk."
    assert find_unit_errors(text) == []


@pytest.mark.unit
def test_percentage_change_near_a_level_word_is_not_a_unit_error():
    text = "The entry zone reflects a 2.6% change from the prior session close."
    assert find_unit_errors(text) == []


@pytest.mark.unit
def test_percentage_stated_as_the_level_itself_is_still_a_unit_error():
    text = "The stop should be set at 5% as an absolute level."
    assert find_unit_errors(text) != []


@pytest.mark.unit
def test_risk_integrity_does_not_misfire_on_legitimate_distance_percentage():
    state = _risk_state()
    text = (
        "The distance from stop 211.28 to entry 217.23 allows ~2.6% additional movement "
        "before invalidation, roughly 10.9% of quarterly profit at risk."
    )
    assert check_risk_integrity(text, state).status == "PASS"


# --- C: NEW_UNSUPPORTED_THRESHOLD false positives ----------------------------


@pytest.mark.unit
def test_restating_an_existing_level_with_a_comparison_word_passes():
    state = _risk_state()
    # "211.28 is the existing stop" -- a quotation of the Trader's own stop,
    # phrased with a comparison word but no conditional "new rule" language.
    text = "Price remains above 211.28, the existing stop, with no new conditions proposed."
    assert check_risk_integrity(text, state).status == "PASS"


@pytest.mark.unit
def test_arithmetic_derivation_sentence_does_not_trigger_a_threshold_warning():
    state = _risk_state()
    text = "240.64 is derived as 234.69 plus the 5.95 ATR buffer, a calculation, not a new rule."
    assert check_risk_integrity(text, state).status == "PASS"


@pytest.mark.unit
def test_percentage_distance_restatement_does_not_trigger_a_threshold_warning():
    state = _risk_state()
    text = "217.23 to 211.28 is approximately 2.6%, confirming the existing risk distance."
    assert check_risk_integrity(text, state).status == "PASS"


@pytest.mark.unit
def test_explicit_new_conditional_threshold_still_warns():
    state = _risk_state()
    text = "Only proceed if the 10Y yield exceeds 5.50%, a level not seen in the evidence."
    result = check_risk_integrity(text, state)
    assert result.status == "WARN"


@pytest.mark.unit
def test_explicit_new_conditional_rsi_rule_still_warns():
    state = _risk_state()
    text = "Reduce exposure when RSI falls below 42, a threshold not present in the research."
    result = check_risk_integrity(text, state)
    assert result.status == "WARN"


@pytest.mark.unit
def test_manager_volume_requirement_without_if_when_still_warns():
    """Regression guard: the conditional-language gate must still recognise
    'requires ... before' phrasing, not just literal if/when, since this
    existing case predates the false-positive fix."""
    manager_text = "The plan requires volume to exceed 120M before entering."
    result = check_manager_integrity(manager_text, _debate(), "", "", "", "")
    assert result.status == "WARN"


# --- D: a header/bullet block glued by bare "\n" must not cascade false
#        positives (real qwen3.8-flash-next/XOM run: the same
#        NEW_UNSUPPORTED_THRESHOLD finding repeated 8 times) ----------------


@pytest.mark.unit
def test_disposition_header_glued_to_an_unrelated_bullet_is_not_flagged():
    """``render_risk_assessment`` (schemas.py) joins "Risk level: HIGH",
    "Disposition: REDUCE_RISK", "Supported points:", and each bullet with a
    single "\n" -- never trailing whitespace. The previous sentence splitter
    only split after ".!?;", so this whole block stayed one giant "sentence"
    and "REDUCE_RISK" alone (matching the "reduce" threshold trigger) caused
    an unrelated number in a later bullet -- with no threshold language of
    its own -- to be flagged as a new, unsupported operational threshold.
    """
    state = _risk_state()
    text = (
        "Aggressive Reviewer (Cross-Review): Risk level: HIGH\n"
        "Disposition: REDUCE_RISK\n"
        "Supported points:\n"
        "- WTI oil reference moved to 96.16 dollars per barrel this quarter."
    )
    result = check_risk_integrity(text, state)
    assert result.status == "PASS"


@pytest.mark.unit
def test_header_bullet_block_does_not_produce_duplicate_looking_findings():
    """The real regression: once glued into one "sentence", every unrelated
    number anywhere in the block generated its own finding, all sharing the
    same (truncated) excerpt -- indistinguishable duplicates in the report.
    Splitting on every bare newline keeps each bullet independent, so a
    genuine new threshold in one bullet is reported once, not once per
    unrelated number elsewhere in the same block.
    """
    state = _risk_state()
    text = (
        "Aggressive Reviewer (Cross-Review): Risk level: HIGH\n"
        "Disposition: REDUCE_RISK\n"
        "Supported points:\n"
        "- WTI oil reference moved to 96.16 dollars per barrel this quarter.\n"
        "- Only proceed if the 10Y yield exceeds 5.50%, a level not seen in the evidence."
    )
    result = check_risk_integrity(text, state)
    assert result.status == "WARN"
    assert len(result.findings) == 1
    assert "5.50" in result.findings[0].detail
    assert "96.16" not in result.findings[0].detail
