"""Structured repair: a repaired Bull/Bear position updates the canonical
bull_thesis/bear_thesis the same way a normal review does, reusing
ResearchReviewOutcome + apply_trade_plan_changes/recompute_trade_metrics --
no separate parsing of free-form repair prose.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tradingagents.agents.managers.integrity_check import check_manager_integrity
from tradingagents.agents.managers.research_manager import create_research_manager
from tradingagents.agents.researchers.research_repair import (
    create_bear_repair,
    create_bull_repair,
)
from tradingagents.agents.schemas import (
    PortfolioRating,
    ResearchPlan,
    ResearchReviewOutcome,
)
from tradingagents.agents.trader.trader import create_trader
from tradingagents.graph.propagation import Propagator


class _StructuredLLM:
    """Returns queued schema instances; mimics ``with_structured_output``."""

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.prompts: list[str] = []

    def with_structured_output(self, schema):
        return self

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return self.outputs.pop(0)


def _state_with_bull_thesis(**bull_overrides):
    state = Propagator().create_initial_state(
        "NVDA", "2026-09-01", atr_reference=2192.41
    )
    bull_thesis = {
        "direction": "BULL",
        "conviction": {
            "evidence_quality": 18, "internal_consistency": 21,
            "robustness_to_challenge": 17, "trade_plan_coherence": 19, "total": 75,
        },
        "horizon": "5-10 trading sessions",
        "entry": {"type": "point", "price": 83000, "low": None, "high": None},
        "take_profit": 87360,
        "stop_loss": 80920,
        "suggested_risk_unit": "MEDIUM",
    }
    bull_thesis.update(bull_overrides)
    state["instrument_context"] = "INSTRUMENT"
    state["market_report"] = "MARKET"
    state["sentiment_report"] = ""
    state["news_report"] = ""
    state["fundamentals_report"] = ""
    state["investment_debate_state"].update(
        {
            "bull_initial": "Bull Analyst (Initial): original",
            "bear_initial": "Bear Analyst (Initial): original",
            "current_bull_response": "Bull Analyst (Initial): original",
            "current_bear_response": "Bear Analyst (Initial): original",
            "bull_thesis": bull_thesis,
            "bull_conviction_history": [75],
            "bull_trade_metrics": {
                "entry_reference": 83000, "risk": 2080, "reward": 4360,
                "reward_risk": 2.096, "stop_atr": 0.949, "target_atr": 1.989,
            },
            "verification_history": [
                {
                    "status": "FAIL", "findings": [], "verified_points": [],
                    "repair_required": True, "repair_targets": ["bull"], "notes": "",
                }
            ],
        }
    )
    return state


def _apply(state, node):
    state["investment_debate_state"] = node(state)["investment_debate_state"]


def _review(conviction_before=75, conviction_after=68, **changes):
    trade_plan_changes = {
        "entry": changes.get("entry", {"action": "KEEP"}),
        "take_profit": changes.get("take_profit", {"action": "KEEP"}),
        "stop_loss": changes.get("stop_loss", {"action": "KEEP"}),
    }
    return ResearchReviewOutcome(
        accepted=["Corrected the subtraction the verifier flagged."],
        conviction_before=conviction_before,
        conviction_after=conviction_after,
        conviction_reason="Material evidence changed.",
        trade_plan_changes=trade_plan_changes,
    )


@pytest.mark.unit
def test_structured_bull_repair_can_revise_entry():
    state = _state_with_bull_thesis()
    review = _review(stop_loss={"action": "KEEP"}, entry={"action": "REVISE", "new": 81500})
    _apply(state, create_bull_repair(_StructuredLLM([review])))
    debate = state["investment_debate_state"]
    assert debate["bull_thesis"]["entry"]["price"] == 81500
    assert debate["bull_repair_structured_status"] == "APPLIED"


@pytest.mark.unit
def test_structured_bear_repair_can_revise_stop():
    state = _state_with_bull_thesis()
    state["investment_debate_state"]["bear_thesis"] = {
        "direction": "BEAR", "entry": {"type": "point", "price": 83000, "low": None, "high": None},
        "take_profit": 78000, "stop_loss": 86000, "suggested_risk_unit": "LOW",
        "conviction": {"evidence_quality": 10, "internal_consistency": 10,
                       "robustness_to_challenge": 10, "trade_plan_coherence": 10, "total": 40},
    }
    state["investment_debate_state"]["bear_conviction_history"] = [40]
    review = _review(conviction_before=40, conviction_after=35, stop_loss={"action": "REVISE", "new": 84500})
    _apply(state, create_bear_repair(_StructuredLLM([review])))
    debate = state["investment_debate_state"]
    assert debate["bear_thesis"]["stop_loss"] == 84500
    assert debate["bear_repair_structured_status"] == "APPLIED"


@pytest.mark.unit
def test_structured_repair_can_withdraw_take_profit():
    state = _state_with_bull_thesis()
    review = _review(take_profit={"action": "WITHDRAW"})
    _apply(state, create_bull_repair(_StructuredLLM([review])))
    debate = state["investment_debate_state"]
    assert debate["bull_thesis"]["take_profit"] is None
    assert 87360 in debate["withdrawn_values"]


@pytest.mark.unit
def test_canonical_thesis_reflects_repaired_values():
    state = _state_with_bull_thesis()
    review = _review(stop_loss={"action": "REVISE", "new": 79000})
    _apply(state, create_bull_repair(_StructuredLLM([review])))
    assert state["investment_debate_state"]["bull_thesis"]["stop_loss"] == 79000


@pytest.mark.unit
def test_deterministic_trade_metrics_recomputed_after_repaired_levels():
    state = _state_with_bull_thesis()
    review = _review(stop_loss={"action": "REVISE", "new": 79000})
    _apply(state, create_bull_repair(_StructuredLLM([review])))
    metrics = state["investment_debate_state"]["bull_trade_metrics"]
    assert metrics["risk"] == pytest.approx(83000 - 79000)


@pytest.mark.unit
def test_old_revised_value_no_longer_active_provenance():
    state = _state_with_bull_thesis()
    review = _review(stop_loss={"action": "REVISE", "new": 79000})
    _apply(state, create_bull_repair(_StructuredLLM([review])))
    debate = state["investment_debate_state"]
    assert 80920 in debate["withdrawn_values"]
    assert debate["bull_thesis"]["stop_loss"] != 80920


@pytest.mark.unit
def test_old_withdrawn_value_no_longer_active_provenance():
    state = _state_with_bull_thesis()
    review = _review(take_profit={"action": "WITHDRAW"})
    _apply(state, create_bull_repair(_StructuredLLM([review])))
    debate = state["investment_debate_state"]
    assert debate["bull_thesis"]["take_profit"] is None
    assert 87360 in debate["withdrawn_values"]


@pytest.mark.unit
def test_keep_preserves_the_active_value():
    state = _state_with_bull_thesis()
    review = _review()  # all KEEP
    _apply(state, create_bull_repair(_StructuredLLM([review])))
    debate = state["investment_debate_state"]
    assert debate["bull_thesis"]["stop_loss"] == 80920
    assert debate["bull_thesis"]["take_profit"] == 87360
    assert debate.get("withdrawn_values", []) == []


@pytest.mark.unit
def test_repair_cannot_introduce_an_unrelated_external_level():
    """Repair is validated through the exact same schema/apply path as a
    normal review -- there is no free-form parser that could let an
    unrelated number slip into a structured field undetected."""
    state = _state_with_bull_thesis()
    # The schema only accepts a REVISE as an explicit numeric `new` value on
    # a named field (entry/take_profit/stop_loss); there is no mechanism for
    # a repair to set an arbitrary new field or level outside that shape.
    review = _review(entry={"action": "REVISE", "new": 81000})
    _apply(state, create_bull_repair(_StructuredLLM([review])))
    debate = state["investment_debate_state"]
    assert set(debate["bull_thesis"].keys()) == {
        "direction", "conviction", "horizon", "entry", "take_profit",
        "stop_loss", "suggested_risk_unit",
    }


@pytest.mark.unit
def test_text_only_fallback_preserves_prose_without_guessing_structured_state():
    state = _state_with_bull_thesis()

    class _PlainLLM:
        def invoke(self, prompt):
            return SimpleNamespace(
                content="ACCEPTED: corrected subtraction; stop now at 79000"
            )

    _apply(state, create_bull_repair(_PlainLLM()))
    debate = state["investment_debate_state"]
    assert "stop now at 79000" in debate["bull_repair"]
    assert debate["bull_repair_structured_status"] == "TEXT_ONLY"
    # Structured state is untouched, not guess-parsed from the prose above.
    assert debate["bull_thesis"]["stop_loss"] == 80920


@pytest.mark.unit
def test_text_only_status_is_visible_to_research_manager_and_trader():
    state = _state_with_bull_thesis()

    class _PlainLLM:
        def invoke(self, prompt):
            return SimpleNamespace(content="free text repair")

    _apply(state, create_bull_repair(_PlainLLM()))
    assert state["investment_debate_state"]["bull_repair_structured_status"] == "TEXT_ONLY"

    class _ManagerLLM:
        def __init__(self):
            self.prompts = []

        def with_structured_output(self, _schema):
            return self

        def invoke(self, prompt):
            self.prompts.append(prompt)
            return ResearchPlan(
                recommendation=PortfolioRating.HOLD, rationale="x", strategic_actions="y"
            )

    manager_llm = _ManagerLLM()
    create_research_manager(manager_llm)(state)
    assert "fell back to free text" in manager_llm.prompts[0]

    captured = {"prompts": []}

    class _PlainTraderLLM:
        # No with_structured_output -- Trader's own call falls back to free
        # text, and (since that text then PASSes validation with no
        # structured object behind it) a second, canonical-recovery call
        # now follows too (see validate_and_repair's early-return fix,
        # 2026-10-16) -- capture every call and check the FIRST one, which
        # is Trader's own real rendered prompt (a message list), never the
        # recovery call's own plain-string prompt.
        def invoke(self, prompt):
            captured["prompts"].append(prompt)
            return SimpleNamespace(
                content="**Action**: Hold\n\nFINAL TRANSACTION PROPOSAL: **HOLD**"
            )

    state["investment_plan"] = "Hold plan"
    create_trader(_PlainTraderLLM())(state)
    user = " ".join(m["content"] for m in captured["prompts"][0] if m["role"] == "user")
    assert "fell back to free text" in user


@pytest.mark.unit
def test_research_manager_receives_repaired_structured_state_after_applied_repair():
    state = _state_with_bull_thesis()
    review = _review(stop_loss={"action": "REVISE", "new": 79000})
    _apply(state, create_bull_repair(_StructuredLLM([review])))

    class _ManagerLLM:
        def __init__(self):
            self.prompts = []

        def with_structured_output(self, _schema):
            return self

        def invoke(self, prompt):
            self.prompts.append(prompt)
            return ResearchPlan(
                recommendation=PortfolioRating.BUY, rationale="x", strategic_actions="y"
            )

    manager_llm = _ManagerLLM()
    create_research_manager(manager_llm)(state)
    assert "79000" in manager_llm.prompts[0]


@pytest.mark.unit
def test_trader_receives_repaired_structured_trade_plan_values():
    state = _state_with_bull_thesis()
    review = _review(stop_loss={"action": "REVISE", "new": 79000})
    _apply(state, create_bull_repair(_StructuredLLM([review])))
    state["investment_plan"] = "Buy plan"

    captured = {"prompts": []}

    class _PlainTraderLLM:
        # See the sibling test above: a second, canonical-recovery call now
        # follows Trader's own fallback -- capture every call and check the
        # FIRST one (Trader's real rendered prompt).
        def invoke(self, prompt):
            captured["prompts"].append(prompt)
            return SimpleNamespace(content="**Action**: Buy\n\nFINAL TRANSACTION PROPOSAL: **BUY**")

    create_trader(_PlainTraderLLM())(state)
    user = " ".join(m["content"] for m in captured["prompts"][0] if m["role"] == "user")
    assert "Stop Loss: 79000" in user


@pytest.mark.unit
def test_manager_integrity_check_builds_provenance_from_repaired_state():
    state = _state_with_bull_thesis()
    review = _review(stop_loss={"action": "REVISE", "new": 79123})
    _apply(state, create_bull_repair(_StructuredLLM([review])))
    debate = state["investment_debate_state"]

    # A manager citing the NEW repaired stop must pass (it's in the pool).
    result = check_manager_integrity(
        "A close below 79123 would invalidate the bullish thesis.", debate, "", "", "", ""
    )
    assert result.status == "PASS"

    # A manager citing the OLD, now-retired stop must be flagged.
    result = check_manager_integrity(
        "The stop remains at 80920 as before.", debate, "", "", "", ""
    )
    assert result.status == "WARN"
    assert result.findings[0].category == "WITHDRAWN_VALUE_REUSED"
