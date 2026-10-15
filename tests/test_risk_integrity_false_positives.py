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
    exclude_values_still_active,
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


# --- E: WITHDRAWN_VALUE_REUSED false positives (real NVDA/2026-09-30 run) --
#
# Audited the actual mechanism (NOT the originally-suspected
# "Corrected/withdrawn:" section-boundary parser -- RiskStanceAssessment's
# own ``corrected_or_withdrawn_claims`` field is rendered but never
# programmatically read anywhere; verified by grep). The real cause:
# Bull's review genuinely retired its own superseded levels (221 -> 217.23
# stop, 236 -> 234.69 take-profit; see call 014's real trade_plan_changes),
# and Bear's review (via canonical recovery, since Bear's round-1 structured
# attempt used the wrong JSON key "trade_plan" instead of "trade_plan_
# changes" and was silently dropped by the schema's lenient default --- a
# separate, adjacent issue not fixed here, flagged for a follow-up task)
# later genuinely retired ITS OWN stale take-profit of 217.23. But 217.23
# is ALSO, completely independently, Bull's own CURRENT, never-retired,
# verified stop-loss (the real SMA50 price level) -- so every mention of
# 217.23 anywhere in Risk's text was flagged, including Bull's own
# unchanged, still-active plan. Reproduced exactly with the real run's
# risk reports: 36 findings (33 WITHDRAWN_VALUE_REUSED) before this fix,
# 3 (0 WITHDRAWN_VALUE_REUSED) after.


@pytest.mark.unit
def test_e1_a_value_retired_in_one_role_but_not_currently_active_anywhere_is_still_flagged():
    """Retiring a VALUE with nothing else currently vouching for it still
    blocks its reuse -- the baseline behavior this fix must not weaken."""
    debate = _debate(withdrawn_values=[221.0])
    state = _risk_state(investment_debate_state=debate)  # pool has nothing at 221.0
    text = "The stop remains at 221.00 as originally proposed."
    result = check_risk_integrity(text, state)
    assert result.status == "WARN"
    assert any(f.category == "WITHDRAWN_VALUE_REUSED" for f in result.findings)


@pytest.mark.unit
def test_e2_a_value_still_active_as_a_different_sides_current_field_is_not_flagged():
    """The exact real regression, minimally reproduced: 217.23 was Bear's
    own superseded take-profit (correctly retired), but it is ALSO Bull's
    CURRENT, never-retired stop-loss -- citing it as Bull's stop must not
    be read as reusing Bear's withdrawn claim."""
    debate = _debate(
        withdrawn_values=[217.23],
        bull_thesis={"entry": {}, "stop_loss": 217.23, "take_profit": 234.69},
        bear_thesis={"entry": {}, "stop_loss": None, "take_profit": 222.84},
    )
    state = _risk_state(investment_debate_state=debate)
    text = "Bull's stop remains at 217.23, unchanged from before."
    result = check_risk_integrity(text, state)
    assert result.status == "PASS"
    assert result.findings == []


@pytest.mark.unit
def test_e3_withdrawing_an_interpretation_does_not_blacklist_a_still_active_number():
    """'withdraw my interpretation of pattern X' must not retire every
    number mentioned around it -- only an actual trade-plan-field
    withdrawal (tracked structurally, never by scanning prose) adds to
    withdrawn_values in the first place; this confirms the exclusion layer
    doesn't need a value to be "mentioned in Rationale" to stay usable."""
    debate = _debate(
        withdrawn_values=[],  # no structural field was ever withdrawn
        bull_thesis={"entry": {}, "stop_loss": 217.23, "take_profit": 234.69},
    )
    state = _risk_state(investment_debate_state=debate)
    text = (
        "Supported points:\n"
        "- Bull's stop at 217.23 remains valid.\n"
        "Corrected/withdrawn:\n"
        "- I withdraw my earlier reading of the breakout pattern as confirmed; "
        "it was an interpretation, not a level.\n"
        "Plan change required: no\n"
        "Rationale: 217.23 is still the operative stop; nothing here retires it."
    )
    result = check_risk_integrity(text, state)
    assert result.status == "PASS"


@pytest.mark.unit
def test_e4_corrected_withdrawn_section_followed_by_plan_change_and_rationale():
    """The originally-suspected shape (Corrected/withdrawn: -> Plan change
    required: -> Rationale:), reproduced with a value retired in ONE role
    (236.0, Bull's own superseded take-profit, not currently active
    anywhere) and a DIFFERENT value (234.69, Bull's CURRENT take-profit)
    cited only in Rationale -- the current value must never be caught by
    whatever extracted the retired one, regardless of section adjacency."""
    debate = _debate(
        withdrawn_values=[236.0],
        bull_thesis={"entry": {}, "stop_loss": 217.23, "take_profit": 234.69},
    )
    state = _risk_state(investment_debate_state=debate)
    text = (
        "Supported points:\n"
        "- Trend remains intact per the verified indicators.\n"
        "Corrected/withdrawn:\n"
        "- The prior target of 236.00 is no longer supported; superseded.\n"
        "Plan change required: no\n"
        "Rationale: The current take-profit at 234.69 is unaffected by that correction."
    )
    result = check_risk_integrity(text, state)
    assert result.status == "PASS"


@pytest.mark.unit
def test_e5_a_value_cited_only_in_rationale_and_never_withdrawn_is_never_flagged():
    debate = _debate(withdrawn_values=[])
    state = _risk_state(investment_debate_state=debate)
    text = "Rationale: the deterministic target 234.69 is unaffected."
    assert check_risk_integrity(text, state).status == "PASS"


@pytest.mark.unit
def test_e6_exclude_values_still_active_unit_behavior():
    assert exclude_values_still_active([217.23, 236.0], [217.23, 300.0]) == [236.0]
    assert exclude_values_still_active([217.23, 236.0], [300.0]) == [217.23, 236.0]
    assert exclude_values_still_active([], [217.23]) == []
    assert exclude_values_still_active([217.23], []) == [217.23]


@pytest.mark.unit
def test_e7_manager_integrity_also_respects_still_active_values():
    """check_manager_integrity shares the same withdrawn-value plumbing
    (plus extract_corrected_away_numbers) -- the fix must apply there too,
    not just to Risk's own check."""
    debate = _debate(
        withdrawn_values=[217.23],
        bull_thesis={"entry": {}, "stop_loss": 217.23, "take_profit": 234.69},
    )
    text = "**Recommendation**: Buy\n**Rationale**: Bull's stop at 217.23 still governs.\n**Strategic Actions**: Enter per plan."
    result = check_manager_integrity(text, debate, "", "", "", "")
    assert result.status == "PASS"


@pytest.mark.unit
def test_e8_real_run_fixture_reduces_36_findings_to_zero_withdrawn_value_reused():
    """Direct reproduction with the audited NVDA/2026-09-30 run's own real
    withdrawn_values and current bull/bear theses (221/236 retired by
    Bull's own review round 1; 217.23 retired by Bear's review round 2 --
    see this section's own header comment): a risk reviewer's real-shaped
    text citing Bull's unchanged, currently-active levels must not warn."""
    debate = _debate(
        withdrawn_values=[221.0, 236.0, 217.23],
        bull_thesis={"entry": {"low": 225.45, "high": 229.0}, "stop_loss": 217.23, "take_profit": 234.69},
        bear_thesis={"entry": {"price": 228.38}, "stop_loss": None, "take_profit": 222.84},
    )
    state = _risk_state(investment_debate_state=debate)
    text = (
        "Supported points:\n"
        "- Trend alignment verified: close 228.38 > EMA10 225.45 > SMA50 217.23 > SMA200 199.81.\n"
        "- Plan arithmetic verified: 227.225 - 217.23 = 10.00 (4.40%).\n"
        "Risks:\n"
        "- The exact stop at 217.23 (SMA50) is a widely-watched level, a candidate for noise sweeps.\n"
        "Rationale: Maintaining REDUCE_RISK; the stop at 217.23 and target 234.69 are both still current."
    )
    result = check_risk_integrity(text, state)
    assert result.status == "PASS"
    assert result.findings == []


@pytest.mark.unit
def test_e9_real_run_fixture_keeps_exactly_its_three_non_withdrawn_findings():
    """Point 5 of this fix's own review: the fixture's 3 surviving findings
    (1 NEW_UNSUPPORTED_THRESHOLD + 2 RELATIONAL_ERROR, the EXACT categories
    the real run's own result.json recorded alongside the 33 false
    WITHDRAWN_VALUE_REUSED positives) must still be produced -- this fix
    only removes the false positives, it never silences anything else."""
    debate = _debate(
        withdrawn_values=[221.0, 236.0, 217.23],
        bull_thesis={"entry": {"low": 225.45, "high": 229.0}, "stop_loss": 217.23, "take_profit": 234.69},
        bear_thesis={"entry": {"price": 228.38}, "stop_loss": None, "take_profit": 222.84},
    )
    state = _risk_state(investment_debate_state=debate)
    text = (
        "Mantengo REDUCE_RISK sobre KEEP: la ejecucion admite llenados con R/R = 0.48 "
        "en el borde alto de la banda, y no hay ningun nivel que lo evite.\n"
        "Alineacion de tendencia verificada por tags: PRICE_VS_EMA10/SMA20/SMA50 ABOVE, "
        "SMA50_VS_SMA200 ABOVE, TREND_STRUCTURE BULLISH, con cierre 228.38 > EMA10 230.00."
    )
    result = check_risk_integrity(text, state)
    assert result.status == "WARN"
    categories = sorted(f.category for f in result.findings)
    assert categories.count("NEW_UNSUPPORTED_THRESHOLD") == 1
    assert categories.count("RELATIONAL_ERROR") == 2
    assert "WITHDRAWN_VALUE_REUSED" not in categories


# --- Known boundary: this fix operates on VALUES, not on CLAIM identity ----
#
# exclude_values_still_active compares bare numbers against the current
# frozen-evidence pool -- it has no notion of WHICH claim/field/side a
# number is being cited for. So once 217.23 is excluded (because it is
# Bull's current, active stop), ANY mention of 217.23 stops being flagged
# as withdrawn-value reuse -- including a hypothetical restatement of
# Bear's own stale claim ("Bear's take-profit is still 217.23"), not just
# the legitimate Bull-stop citations this fix was written to stop
# misflagging. This is an accepted, pre-existing architectural limit (the
# whole withdrawn-value/frozen-evidence-pool mechanism is a deterministic
# bag-of-numbers check with no claim/field attribution anywhere in it, not
# something this fix introduces), consistent with the module's own
# documented design principle: false positives are worse than false
# negatives here. Documented explicitly, not silently assumed.


@pytest.mark.unit
def test_e10_known_boundary_reusing_the_stale_claim_itself_is_not_caught_once_excluded():
    """Documents the exact edge this fix does not (and, without a real
    claim-identity model, cannot cheaply) resolve: restating Bear's own
    withdrawn take-profit claim verbatim, once 217.23 is excluded for being
    Bull's current stop, no longer warns -- a real miss, accepted per the
    module's own "false positives are worse than false negatives" design
    principle, not a silent regression introduced here."""
    debate = _debate(
        withdrawn_values=[217.23],  # Bear's own stale take-profit, genuinely retired
        bull_thesis={"entry": {}, "stop_loss": 217.23, "take_profit": 234.69},
        bear_thesis={"entry": {}, "stop_loss": None, "take_profit": 222.84},
    )
    state = _risk_state(investment_debate_state=debate)
    # Restates the EXACT withdrawn claim (Bear's take-profit), not Bull's
    # stop -- a genuine stale-value reuse this check cannot distinguish
    # from the legitimate Bull-stop citation, given only a bare number.
    text = "Bear's take-profit remains 217.23, unchanged from the initial thesis."
    result = check_risk_integrity(text, state)
    assert result.status == "PASS"  # known limitation, not a crash or a silent new heuristic
