"""TraderProposal can now preserve a take_profit the Trader itself produced
(see fork README: "RecommendationRecord" -- trade_plan). Fixes a real
prompt/schema inconsistency: trader/system.txt already told the Trader the
research team computed a take-profit level, but neither the structured
output contract nor TraderProposal itself ever asked for or kept one.

Optional by design: a Trader that leaves it unset stays fully valid, and
RecommendationRecord.trade_plan.take_profit must come ONLY from
TraderProposal.take_profit -- never backfilled from PortfolioDecision's
different, different-purpose price_target.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.test_recommendation_record import _build, _execution, _full_final_state
from tradingagents.agents.schemas import (
    PortfolioDecision,
    PortfolioRating,
    RiskDisposition,
    TraderAction,
    TraderProposal,
    render_trader_proposal,
)
from tradingagents.recommendation_record import build_recommendation_record, sha256_file

# --- 1: TraderProposal without take_profit stays valid ----------------------


@pytest.mark.unit
def test_trader_proposal_without_take_profit_is_still_valid():
    proposal = TraderProposal(action=TraderAction.BUY, reasoning="No defensible target.")
    assert proposal.take_profit is None
    rendered = render_trader_proposal(proposal)
    assert "**Take Profit**: not provided" in rendered


# --- 2: TraderProposal with take_profit preserves it ------------------------


@pytest.mark.unit
def test_trader_proposal_with_take_profit_preserves_it():
    proposal = TraderProposal(
        action=TraderAction.BUY, reasoning="Research target still holds.",
        entry_price=100.0, stop_loss=90.0, take_profit=120.0,
    )
    assert proposal.take_profit == 120.0
    rendered = render_trader_proposal(proposal)
    assert "**Take Profit**: 120.0" in rendered


@pytest.mark.unit
def test_trader_proposal_take_profit_coerces_nullish_strings():
    proposal = TraderProposal(action=TraderAction.HOLD, reasoning="x", take_profit="not provided")
    assert proposal.take_profit is None


# --- 3: structured result.json preserves take_profit, no regex parsing -----


@pytest.mark.unit
def test_structured_result_preserves_trader_take_profit(tmp_path: Path):
    proposal = TraderProposal(
        action=TraderAction.BUY, reasoning="x",
        entry_price=100.0, stop_loss=90.0, take_profit=120.0,
    )
    final_state = _full_final_state(trader_proposal=proposal)
    execution = _execution(final_state)
    from cli.result_document import result_document

    result = result_document(execution, tmp_path)
    assert result["structured"]["trader_proposal"]["take_profit"] == 120.0


# --- 4/5: RecommendationRecord copies TraderProposal.take_profit exactly,--
# --- null stays null ---------------------------------------------------------


@pytest.mark.unit
def test_record_copies_trader_take_profit_exactly(tmp_path: Path):
    proposal = TraderProposal(
        action=TraderAction.BUY, reasoning="x",
        entry_price=100.0, stop_loss=90.0, take_profit=120.0,
    )
    execution, result, result_path, report_path = _build(tmp_path, trader_proposal=proposal)
    record = build_recommendation_record(
        execution=execution, result=result, config={}, result_path="result.json",
        result_sha256=sha256_file(result_path), report_path="complete_report.md",
        report_sha256=sha256_file(report_path),
    )
    assert record.trade_plan.take_profit == 120.0


@pytest.mark.unit
def test_record_take_profit_stays_null_when_trader_leaves_it_unset(tmp_path: Path):
    proposal = TraderProposal(action=TraderAction.BUY, reasoning="x", entry_price=100.0, stop_loss=90.0)
    execution, result, result_path, report_path = _build(tmp_path, trader_proposal=proposal)
    record = build_recommendation_record(
        execution=execution, result=result, config={}, result_path="result.json",
        result_sha256=sha256_file(result_path), report_path="complete_report.md",
        report_sha256=sha256_file(report_path),
    )
    assert record.trade_plan.take_profit is None


# --- 6: PortfolioDecision.price_target is never used as a silent fallback --


@pytest.mark.unit
def test_portfolio_price_target_is_never_used_as_a_fallback(tmp_path: Path):
    proposal = TraderProposal(action=TraderAction.BUY, reasoning="x", entry_price=100.0, stop_loss=90.0)
    decision = PortfolioDecision(
        rating=PortfolioRating.BUY, disposition=RiskDisposition.KEEP,
        executive_summary="x", investment_thesis="y",
        price_target=999.0,  # deliberately different from any take_profit
    )
    execution, result, result_path, report_path = _build(
        tmp_path, trader_proposal=proposal, portfolio_decision=decision,
    )
    record = build_recommendation_record(
        execution=execution, result=result, config={}, result_path="result.json",
        result_sha256=sha256_file(result_path), report_path="complete_report.md",
        report_sha256=sha256_file(report_path),
    )
    assert record.trade_plan.take_profit is None
    # price_target survives as its own, separate signal -- never merged in.
    assert result["structured"]["portfolio_decision"]["price_target"] == 999.0


# --- 7: prompts ask for take_profit explicitly without forcing invention ---


@pytest.mark.unit
def test_prompts_ask_for_take_profit_without_forcing_invention():
    system_text = Path("tradingagents/prompts/trader/system.txt").read_text(encoding="utf-8")
    user_text = Path("tradingagents/prompts/trader/user.txt").read_text(encoding="utf-8")

    assert "take-profit" in system_text.lower()
    assert "take profit" in user_text.lower()
    # Both explicitly guard against inventing one just because the field exists.
    assert "invent" in system_text.lower()
    assert "invent" in user_text.lower()
