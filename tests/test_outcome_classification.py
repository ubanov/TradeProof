"""RecommendationOutcome, first pure block: classify_recommendation() derives
a plan's EXECUTABLE_TRADE_PLAN/NO_TRADE status exclusively from an already-
built RecommendationRecord's structured decision/trade_plan fields -- no
LLM, no market data, no Markdown, no legacy result["action"]/rating text
(see fork README / tradingagents/outcome/classification.py's own docstring
for the authority rule: PortfolioDecision.disposition is the sole execution
gate).

Pricing, horizon dates, execution simulation, MFE/MAE, benchmark, storage,
and a CLI `evaluate` command are explicitly NOT covered here -- future work.
"""

from __future__ import annotations

import pytest

from tests.test_recommendation_record import _build
from tradingagents.agents.schemas import (
    PortfolioDecision,
    PortfolioRating,
    RiskDisposition,
    TraderAction,
    TraderProposal,
)
from tradingagents.outcome.classification import (
    ClassificationWarning,
    Direction,
    EntryKind,
    NonExecutionReason,
    OutcomeClassification,
    PlanType,
    classify_recommendation,
)
from tradingagents.recommendation_record import (
    RecommendationRecord,
    build_recommendation_record,
    sha256_file,
)


def _record(
    tmp_path,
    *,
    action: TraderAction,
    disposition: RiskDisposition,
    entry_price: float | None = None,
    entry_price_low: float | None = None,
    entry_price_high: float | None = None,
    stop_loss: float | None = 90.0,
    take_profit: float | None = None,
    rating: PortfolioRating = PortfolioRating.HOLD,  # deliberately uncorrelated with disposition
) -> RecommendationRecord:
    trader_proposal = TraderProposal(
        action=action, reasoning="Structured trader proposal for classification testing.",
        entry_price=entry_price, entry_price_low=entry_price_low, entry_price_high=entry_price_high,
        stop_loss=stop_loss, take_profit=take_profit,
    )
    portfolio_decision = PortfolioDecision(
        rating=rating, disposition=disposition,
        executive_summary="x", investment_thesis="y",
    )
    execution, result, result_path, report_path = _build(
        tmp_path, trader_proposal=trader_proposal, portfolio_decision=portfolio_decision,
    )
    return build_recommendation_record(
        execution=execution, result=result, config={}, result_path="result.json",
        result_sha256=sha256_file(result_path), report_path="complete_report.md",
        report_sha256=sha256_file(report_path),
    )


# --- Disposition x action matrix (task spec section 8) ----------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "action, disposition, entry_price, expected",
    [
        (
            TraderAction.BUY, RiskDisposition.KEEP, 100.0,
            OutcomeClassification(PlanType.EXECUTABLE_TRADE_PLAN, Direction.LONG, None, EntryKind.EXACT),
        ),
        (
            TraderAction.SELL, RiskDisposition.KEEP, 100.0,
            OutcomeClassification(PlanType.EXECUTABLE_TRADE_PLAN, Direction.SHORT, None, EntryKind.EXACT),
        ),
        (
            TraderAction.HOLD, RiskDisposition.KEEP, None,
            OutcomeClassification(PlanType.NO_TRADE, Direction.NONE, NonExecutionReason.TRADER_HOLD, EntryKind.NONE),
        ),
        (
            TraderAction.BUY, RiskDisposition.DEFER, 100.0,
            OutcomeClassification(PlanType.NO_TRADE, Direction.LONG, NonExecutionReason.DEFERRED, EntryKind.NONE),
        ),
        (
            TraderAction.SELL, RiskDisposition.DEFER, 100.0,
            OutcomeClassification(PlanType.NO_TRADE, Direction.SHORT, NonExecutionReason.DEFERRED, EntryKind.NONE),
        ),
        (
            TraderAction.BUY, RiskDisposition.REDUCE_RISK, 100.0,
            OutcomeClassification(
                PlanType.NO_TRADE, Direction.LONG, NonExecutionReason.REQUIRES_MODIFICATION, EntryKind.NONE
            ),
        ),
        (
            TraderAction.SELL, RiskDisposition.REDUCE_RISK, 100.0,
            OutcomeClassification(
                PlanType.NO_TRADE, Direction.SHORT, NonExecutionReason.REQUIRES_MODIFICATION, EntryKind.NONE
            ),
        ),
        (
            TraderAction.BUY, RiskDisposition.REJECT_PLAN, 100.0,
            OutcomeClassification(PlanType.NO_TRADE, Direction.LONG, NonExecutionReason.REJECTED, EntryKind.NONE),
        ),
        (
            TraderAction.SELL, RiskDisposition.REJECT_PLAN, 100.0,
            OutcomeClassification(PlanType.NO_TRADE, Direction.SHORT, NonExecutionReason.REJECTED, EntryKind.NONE),
        ),
    ],
    ids=[
        "KEEP+Buy+entry", "KEEP+Sell+entry", "KEEP+Hold",
        "DEFER+Buy", "DEFER+Sell",
        "REDUCE_RISK+Buy", "REDUCE_RISK+Sell",
        "REJECT_PLAN+Buy", "REJECT_PLAN+Sell",
    ],
)
def test_disposition_action_matrix(tmp_path, action, disposition, entry_price, expected):
    record = _record(tmp_path, action=action, disposition=disposition, entry_price=entry_price)
    assert classify_recommendation(record) == expected


@pytest.mark.unit
def test_keep_buy_without_entry_is_incomplete_plan(tmp_path):
    record = _record(tmp_path, action=TraderAction.BUY, disposition=RiskDisposition.KEEP)
    result = classify_recommendation(record)
    assert result.plan_type is PlanType.NO_TRADE
    assert result.direction is Direction.LONG
    assert result.non_execution_reason is NonExecutionReason.INCOMPLETE_PLAN
    assert result.entry_kind is EntryKind.NONE


@pytest.mark.unit
def test_keep_sell_with_incomplete_range_is_incomplete_plan(tmp_path):
    record = _record(
        tmp_path, action=TraderAction.SELL, disposition=RiskDisposition.KEEP,
        entry_price_low=100.0,  # high deliberately unset
    )
    result = classify_recommendation(record)
    assert result.plan_type is PlanType.NO_TRADE
    assert result.direction is Direction.SHORT
    assert result.non_execution_reason is NonExecutionReason.INCOMPLETE_PLAN
    assert ClassificationWarning.PARTIAL_ENTRY_RANGE in result.classification_warnings


@pytest.mark.unit
def test_keep_buy_with_inverted_range_is_incomplete_plan(tmp_path):
    record = _record(
        tmp_path, action=TraderAction.BUY, disposition=RiskDisposition.KEEP,
        entry_price_low=110.0, entry_price_high=90.0,  # low > high
    )
    result = classify_recommendation(record)
    assert result.plan_type is PlanType.NO_TRADE
    assert result.direction is Direction.LONG
    assert result.non_execution_reason is NonExecutionReason.INCOMPLETE_PLAN
    assert ClassificationWarning.INVERTED_ENTRY_RANGE in result.classification_warnings


# --- Entry-kind cases --------------------------------------------------------


@pytest.mark.unit
def test_exact_entry_is_executable(tmp_path):
    record = _record(tmp_path, action=TraderAction.BUY, disposition=RiskDisposition.KEEP, entry_price=100.0)
    result = classify_recommendation(record)
    assert result.plan_type is PlanType.EXECUTABLE_TRADE_PLAN
    assert result.entry_kind is EntryKind.EXACT


@pytest.mark.unit
def test_valid_range_entry_is_executable(tmp_path):
    record = _record(
        tmp_path, action=TraderAction.BUY, disposition=RiskDisposition.KEEP,
        entry_price_low=98.0, entry_price_high=102.0,
    )
    result = classify_recommendation(record)
    assert result.plan_type is PlanType.EXECUTABLE_TRADE_PLAN
    assert result.entry_kind is EntryKind.RANGE
    assert result.classification_warnings == ()


@pytest.mark.unit
def test_equal_bound_range_is_a_valid_degenerate_range(tmp_path):
    record = _record(
        tmp_path, action=TraderAction.BUY, disposition=RiskDisposition.KEEP,
        entry_price_low=100.0, entry_price_high=100.0,
    )
    result = classify_recommendation(record)
    assert result.plan_type is PlanType.EXECUTABLE_TRADE_PLAN
    assert result.entry_kind is EntryKind.RANGE


@pytest.mark.unit
def test_partial_range_low_only_is_not_usable(tmp_path):
    record = _record(
        tmp_path, action=TraderAction.BUY, disposition=RiskDisposition.KEEP, entry_price_low=98.0,
    )
    result = classify_recommendation(record)
    assert result.entry_kind is EntryKind.NONE
    assert result.non_execution_reason is NonExecutionReason.INCOMPLETE_PLAN
    assert ClassificationWarning.PARTIAL_ENTRY_RANGE in result.classification_warnings


@pytest.mark.unit
def test_partial_range_high_only_is_not_usable(tmp_path):
    record = _record(
        tmp_path, action=TraderAction.SELL, disposition=RiskDisposition.KEEP, entry_price_high=102.0,
    )
    result = classify_recommendation(record)
    assert result.entry_kind is EntryKind.NONE
    assert result.non_execution_reason is NonExecutionReason.INCOMPLETE_PLAN
    assert ClassificationWarning.PARTIAL_ENTRY_RANGE in result.classification_warnings


@pytest.mark.unit
def test_inverted_range_is_not_usable(tmp_path):
    record = _record(
        tmp_path, action=TraderAction.SELL, disposition=RiskDisposition.KEEP,
        entry_price_low=105.0, entry_price_high=95.0,
    )
    result = classify_recommendation(record)
    assert result.entry_kind is EntryKind.NONE
    assert ClassificationWarning.INVERTED_ENTRY_RANGE in result.classification_warnings


@pytest.mark.unit
def test_no_entry_at_all_is_not_usable(tmp_path):
    record = _record(tmp_path, action=TraderAction.SELL, disposition=RiskDisposition.KEEP)
    result = classify_recommendation(record)
    assert result.entry_kind is EntryKind.NONE
    assert result.classification_warnings == ()  # absence alone needs no warning code


@pytest.mark.unit
def test_point_and_valid_range_both_set_prefers_the_point(tmp_path):
    record = _record(
        tmp_path, action=TraderAction.BUY, disposition=RiskDisposition.KEEP,
        entry_price=100.0, entry_price_low=98.0, entry_price_high=102.0,
    )
    result = classify_recommendation(record)
    assert result.plan_type is PlanType.EXECUTABLE_TRADE_PLAN
    assert result.entry_kind is EntryKind.EXACT
    assert ClassificationWarning.ENTRY_POINT_AND_RANGE_BOTH_SET in result.classification_warnings


@pytest.mark.unit
def test_reference_entry_alone_is_never_a_substitute_for_entry(tmp_path):
    """Audited: TraderProposal.reference_entry is documented as context the
    Trader may cite, never a silent substitute for entry_price/range (see
    its own field description). A plan with only reference_entry set must
    still be INCOMPLETE_PLAN, not treated as if it had a real entry."""
    trader_proposal = TraderProposal(
        action=TraderAction.BUY, reasoning="Citing a computed reference only.",
        reference_entry=100.0, stop_loss=90.0,
    )
    portfolio_decision = PortfolioDecision(
        rating=PortfolioRating.BUY, disposition=RiskDisposition.KEEP,
        executive_summary="x", investment_thesis="y",
    )
    execution, result, result_path, report_path = _build(
        tmp_path, trader_proposal=trader_proposal, portfolio_decision=portfolio_decision,
    )
    record = build_recommendation_record(
        execution=execution, result=result, config={}, result_path="result.json",
        result_sha256=sha256_file(result_path), report_path="complete_report.md",
        report_sha256=sha256_file(report_path),
    )
    assert record.trade_plan.reference_entry == 100.0  # confirms it WAS preserved upstream
    outcome = classify_recommendation(record)
    assert outcome.plan_type is PlanType.NO_TRADE
    assert outcome.non_execution_reason is NonExecutionReason.INCOMPLETE_PLAN


# --- stop_loss / take_profit never gate classification -----------------------


@pytest.mark.unit
def test_executable_plan_without_stop_loss(tmp_path):
    record = _record(
        tmp_path, action=TraderAction.BUY, disposition=RiskDisposition.KEEP,
        entry_price=100.0, stop_loss=None,
    )
    assert classify_recommendation(record).plan_type is PlanType.EXECUTABLE_TRADE_PLAN


@pytest.mark.unit
def test_executable_plan_without_take_profit(tmp_path):
    record = _record(
        tmp_path, action=TraderAction.BUY, disposition=RiskDisposition.KEEP,
        entry_price=100.0, take_profit=None,
    )
    assert record.trade_plan.take_profit is None
    assert classify_recommendation(record).plan_type is PlanType.EXECUTABLE_TRADE_PLAN


@pytest.mark.unit
def test_executable_plan_without_stop_loss_or_take_profit(tmp_path):
    record = _record(
        tmp_path, action=TraderAction.SELL, disposition=RiskDisposition.KEEP,
        entry_price=100.0, stop_loss=None, take_profit=None,
    )
    assert classify_recommendation(record).plan_type is PlanType.EXECUTABLE_TRADE_PLAN


# --- Schema 1.0 / 1.1 compatibility ------------------------------------------


@pytest.mark.unit
def test_works_on_a_schema_1_1_record_with_price_basis(tmp_path):
    record = _record(tmp_path, action=TraderAction.BUY, disposition=RiskDisposition.KEEP, entry_price=100.0)
    # Current schema, bumped past 1.1 since this test was written (see
    # tests/test_decision_memory_isolation.py, tests/test_pre_pm_snapshot.py)
    # -- still exercises the same 1.1-introduced price_basis compatibility
    # path this test is named for.
    assert record.schema_version == "1.3"
    assert record.price_basis is None  # no price_basis captured by _build's fixture state
    assert classify_recommendation(record).plan_type is PlanType.EXECUTABLE_TRADE_PLAN


@pytest.mark.unit
def test_works_on_a_legacy_1_0_shaped_record_missing_price_basis_entirely(tmp_path):
    record = _record(tmp_path, action=TraderAction.SELL, disposition=RiskDisposition.KEEP, entry_price=100.0)
    legacy_payload = record.model_dump(mode="json")
    del legacy_payload["price_basis"]  # a real 1.0 record on disk never has this key at all
    legacy_payload["schema_version"] = "1.0"
    legacy_record = RecommendationRecord.model_validate(legacy_payload)
    assert classify_recommendation(legacy_record) == classify_recommendation(record)


# --- Missing/unreadable decision fields: never invent -----------------------
#
# DecisionRef.portfolio_disposition and TradePlanRef.action are both
# Optional[str] on the record schema itself (not required), so a record
# loaded from a malformed/legacy file on disk can have either as None, or
# as a value outside the enum RecommendationRecord's own producers ever
# write. Built via model_copy on a real, fully-built record rather than a
# hand-rolled stub, so classify_recommendation is exercised against an
# actual RecommendationRecord instance, matching its real type contract.


@pytest.mark.unit
def test_missing_disposition_is_incomplete_decision_not_invented(tmp_path):
    record = _record(tmp_path, action=TraderAction.BUY, disposition=RiskDisposition.KEEP, entry_price=100.0)
    mutated = record.model_copy(update={
        "decision": record.decision.model_copy(update={"portfolio_disposition": None}),
    })
    result = classify_recommendation(mutated)
    assert result.plan_type is PlanType.NO_TRADE
    assert result.non_execution_reason is NonExecutionReason.INCOMPLETE_DECISION
    assert result.direction is Direction.LONG  # action was still readable


@pytest.mark.unit
def test_missing_action_is_incomplete_decision_not_invented(tmp_path):
    record = _record(tmp_path, action=TraderAction.BUY, disposition=RiskDisposition.KEEP, entry_price=100.0)
    mutated = record.model_copy(update={
        "trade_plan": record.trade_plan.model_copy(update={"action": None}),
    })
    result = classify_recommendation(mutated)
    assert result.plan_type is PlanType.NO_TRADE
    assert result.non_execution_reason is NonExecutionReason.INCOMPLETE_DECISION
    assert result.direction is Direction.NONE


@pytest.mark.unit
def test_unrecognized_disposition_value_is_incomplete_decision_not_invented(tmp_path):
    record = _record(tmp_path, action=TraderAction.SELL, disposition=RiskDisposition.KEEP, entry_price=100.0)
    mutated = record.model_copy(update={
        "decision": record.decision.model_copy(update={"portfolio_disposition": "SOMETHING_UNEXPECTED"}),
    })
    result = classify_recommendation(mutated)
    assert result.non_execution_reason is NonExecutionReason.INCOMPLETE_DECISION
    assert result.direction is Direction.SHORT


# --- Legacy/contradictory fields never influence the gate --------------------


@pytest.mark.unit
def test_legacy_result_action_contradicting_structured_plan_does_not_affect_classification(tmp_path):
    """build_recommendation_record already ignores legacy result["action"]/
    final_trade_decision text in favor of structured fields (see
    test_recommendation_record.py tests 8/30); classify_recommendation reads
    only the resulting record, so a contradictory legacy text never reaches
    it at all. Exercised through the real result_document()/result pipeline,
    not just the record object, to prove the whole chain is immune."""
    trader_proposal = TraderProposal(
        action=TraderAction.BUY, reasoning="Structured says Buy.", entry_price=100.0, stop_loss=90.0,
    )
    portfolio_decision = PortfolioDecision(
        rating=PortfolioRating.BUY, disposition=RiskDisposition.KEEP,
        executive_summary="x", investment_thesis="y",
    )
    execution, result, result_path, report_path = _build(
        tmp_path, trader_proposal=trader_proposal, portfolio_decision=portfolio_decision,
        legacy_trader_text="**Action**: Sell\n**Entry Price**: 50.0",  # deliberately contradicts
        legacy_final_decision="**Rating**: Sell",
    )
    assert result["action"] == "Sell"  # the legacy, regex-derived field really does disagree
    record = build_recommendation_record(
        execution=execution, result=result, config={}, result_path="result.json",
        result_sha256=sha256_file(result_path), report_path="complete_report.md",
        report_sha256=sha256_file(report_path),
    )
    outcome = classify_recommendation(record)
    assert outcome.plan_type is PlanType.EXECUTABLE_TRADE_PLAN
    assert outcome.direction is Direction.LONG  # structured Buy wins, not the legacy "Sell"


@pytest.mark.unit
def test_contradictory_rating_does_not_affect_the_execution_gate(tmp_path):
    """PortfolioDecision.rating is a separate, non-gating concept (see
    module docstring's authority rule) -- a bearish rating paired with a
    KEEP disposition on a Buy plan must still classify as executable."""
    record = _record(
        tmp_path, action=TraderAction.BUY, disposition=RiskDisposition.KEEP,
        entry_price=100.0, rating=PortfolioRating.SELL,  # contradicts the Buy/KEEP plan
    )
    assert classify_recommendation(record).plan_type is PlanType.EXECUTABLE_TRADE_PLAN


# --- Purity / determinism ----------------------------------------------------


@pytest.mark.unit
def test_classification_is_pure_and_deterministic(tmp_path):
    record = _record(tmp_path, action=TraderAction.SELL, disposition=RiskDisposition.KEEP, entry_price=100.0)
    first = classify_recommendation(record)
    second = classify_recommendation(record)
    assert first == second
    assert first is not second  # a fresh value each call, not a cached singleton


@pytest.mark.unit
def test_classify_recommendation_has_no_side_effect_tools_available():
    """No filesystem/logging/network import anywhere in the module -- a
    structural guard, matching the module's own "pure" claim."""
    import ast
    from pathlib import Path

    import tradingagents.outcome.classification as mod

    source = Path(mod.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    forbidden = {"os", "pathlib", "logging", "requests", "socket", "json"}
    assert not (imported_modules & forbidden)
