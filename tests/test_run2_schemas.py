"""Structural validation for run2's binary First Barrier stage schemas (fork
README: "run2" -- binary First Barrier experiment). ``BarrierSignal`` is a
FORCED BINARY CHOICE -- LONG/SHORT only, NO_POSITION is not a valid
predictor output at all (rejected at the Pydantic level, so it is routed
into the same repair-then-degrade path a malformed JSON shape would be).
``confidence`` is kept purely as diagnostic information (see ``pipeline.py``
-- it never gates the final decision). ``FinalDecision.action`` adds
``UNRESOLVED`` for a technical/structural pipeline failure -- a separate
vocabulary from the predictor's own.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from tradingagents.run2.schemas import BarrierSignal, FinalDecision, RegimeAnalysis


def _signal(**overrides):
    base = {
        "direction": "LONG", "confidence": 0.55, "thesis": "t",
        "evidence_for": [], "evidence_against": [], "invalidation_conditions": [],
    }
    base.update(overrides)
    return BarrierSignal(**base)


@pytest.mark.unit
def test_1_valid_long_signal_parses_regardless_of_low_confidence():
    s = _signal(direction="LONG", confidence=0.01)
    assert s.direction == "LONG"
    assert s.confidence == 0.01


@pytest.mark.unit
def test_2_valid_short_signal_parses_regardless_of_low_confidence():
    s = _signal(direction="SHORT", confidence=0.01)
    assert s.direction == "SHORT"
    assert s.confidence == 0.01


@pytest.mark.unit
def test_3_no_position_is_not_a_valid_predictor_output():
    """Fork spec: "run2" binary simplification -- NO_POSITION is removed as
    a predictor output entirely (not merely discouraged by the prompt), so
    the LLM can never silently reintroduce it."""
    with pytest.raises(ValidationError, match="LONG.*SHORT|SHORT.*LONG"):
        _signal(direction="NO_POSITION")


@pytest.mark.unit
def test_4_signal_has_no_expected_return_range_field():
    s = _signal()
    assert not hasattr(s, "expected_return_range_pct")


@pytest.mark.unit
def test_5_confidence_out_of_range_rejected():
    with pytest.raises(ValidationError):
        _signal(confidence=1.5)
    with pytest.raises(ValidationError):
        _signal(confidence=-0.1)


@pytest.mark.unit
def test_6_invalid_direction_literal_rejected():
    with pytest.raises(ValidationError):
        _signal(direction="BUY")


@pytest.mark.unit
def test_7_regime_analysis_valid_shape():
    r = RegimeAnalysis(
        primary_trend="UP", secondary_trend="SIDEWAYS", momentum="POSITIVE",
        volatility_regime="NORMAL", trend_strength=0.6, regime_confidence=0.5,
    )
    assert r.key_levels.support == []
    assert r.observations == []


@pytest.mark.unit
def test_8_regime_analysis_rejects_out_of_range_strength():
    with pytest.raises(ValidationError):
        RegimeAnalysis(
            primary_trend="UP", secondary_trend="UP", momentum="POSITIVE",
            volatility_regime="NORMAL", trend_strength=1.5, regime_confidence=0.5,
        )


@pytest.mark.unit
def test_9_regime_analysis_has_no_no_position_option():
    """The Regime Analyzer still only describes the market -- it never
    gained a NO_POSITION-shaped escape hatch of its own (fork spec: "run2"
    binary simplification -- "No anadir una opcion NO_POSITION al regimen")."""
    assert "NO_POSITION" not in RegimeAnalysis.model_fields["primary_trend"].annotation.__args__
    assert "NO_POSITION" not in RegimeAnalysis.model_fields["secondary_trend"].annotation.__args__


@pytest.mark.unit
def test_10_final_decision_accepts_long_short_and_unresolved():
    for action in ("LONG", "SHORT", "UNRESOLVED"):
        d = FinalDecision(action=action, reason="x")
        assert d.action == action


@pytest.mark.unit
def test_11_final_decision_rejects_no_position():
    """UNRESOLVED replaces NO_POSITION entirely for a technical failure --
    NO_POSITION must not be constructible as a FinalDecision.action at all."""
    with pytest.raises(ValidationError):
        FinalDecision(action="NO_POSITION", reason="regime_stage_failed")


@pytest.mark.unit
def test_12_final_decision_rejects_invalid_action():
    with pytest.raises(ValidationError):
        FinalDecision(action="BUY", reason="x")


@pytest.mark.unit
def test_13_final_decision_is_action_and_reason_only():
    d = FinalDecision(action="UNRESOLVED", reason="regime_stage_failed")
    assert not hasattr(d, "raw_direction")
    assert not hasattr(d, "challenger_verdict")
