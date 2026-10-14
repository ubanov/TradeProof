"""RecommendationRecord v1: an immutable ledger entry per final recommendation
(see fork README: "RecommendationRecord"). Structured state is the source of
truth; a record is built and written exactly once, only for a successful run
whose portfolio_decision parsed structurally.

Numbered per the task's own "TESTS" list (1-30); horizon/hash-specific cases
get descriptive names alongside the numbered ones.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import cli.run as cli_run
from cli.headless import run_single_analysis
from cli.result_document import result_document
from tests.test_llm_trace_integration import _FakeTradingAgentsGraph
from tradingagents.agents.schemas import (
    ConvictionScore,
    EntryLevel,
    EntryLevelType,
    InitialResearchThesis,
    PortfolioDecision,
    PortfolioRating,
    ResearchDirection,
    ResearchPlan,
    RiskDisposition,
    TraderAction,
    TraderProposal,
)
from tradingagents.recommendation_record import (
    HorizonParseStatus,
    HorizonUnit,
    IncompleteRecommendationError,
    ReproducibilityStatus,
    build_recommendation_record,
    compute_run_fingerprint,
    parse_horizon,
    record_recommendation,
    resolve_horizon,
    sha256_file,
    verify_recommendation_record,
    write_recommendation_record,
)
from tradingagents.run_provenance import build_run_provenance


def _conviction(total_each: int = 15) -> ConvictionScore:
    return ConvictionScore(
        evidence_quality=total_each, internal_consistency=total_each,
        robustness_to_challenge=total_each, trade_plan_coherence=total_each,
    )


def _thesis(
    *, direction=ResearchDirection.BULL, entry_price=100.0, take_profit=120.0,
    stop_loss=90.0, conviction_total_each=15, data_gaps=None,
) -> dict:
    thesis = InitialResearchThesis(
        direction=direction, conviction=_conviction(conviction_total_each),
        horizon="5-10 trading sessions",
        entry=EntryLevel(type=EntryLevelType.POINT, price=entry_price),
        take_profit=take_profit, stop_loss=stop_loss,
        thesis="The evidence supports this side of the debate.",
        evidence=["Supporting data point."],
        risks=["Earnings surprise"], data_gaps=data_gaps or [],
    )
    return thesis.model_dump(mode="json")


def _trade_metrics(reward_risk: float = 2.0, stop_atr: float = 1.0, target_atr: float = 2.0) -> dict:
    return {
        "entry_reference": 100.0, "risk": 10.0, "reward": 20.0,
        "risk_pct": 10.0, "reward_pct": 20.0, "reward_risk": reward_risk,
        "stop_atr": stop_atr, "target_atr": target_atr,
        "risk_reward_min": None, "risk_reward_max": None,
    }


def _run_provenance(**overrides) -> dict:
    config = {
        "llm_provider": "ollama",
        "generation_llm": "gen-model", "generation_reasoning_effort": "low",
        "review_llm": "review-model", "review_reasoning_effort": "max",
        "validator_llm": "validator-model", "validator_reasoning_effort": "medium",
        "validation_enabled": True,
        "output_language": "English", "max_risk_discuss_rounds": 3,
    }
    provenance = build_run_provenance(
        selected_analysts=["market", "social"], config=config,
        configured_seed=0, base_seed=184736291, trace_llm=False,
        git_info={"commit": "abc123", "branch": "main", "dirty": False},
        package_version="0.9.1", prompt_hashes={"global_policy.txt": "h1", "analysts/market.txt": "h2"},
    )
    provenance.update(overrides)
    return provenance


def _full_final_state(
    *, portfolio_decision=None, trader_proposal=None, research_plan=None,
    legacy_trader_text=None, legacy_final_decision=None,
    run_provenance=None, bull_data_gaps=None, bear_data_gaps=None,
) -> dict:
    if portfolio_decision is None:
        portfolio_decision = PortfolioDecision(
            rating=PortfolioRating.OVERWEIGHT, disposition=RiskDisposition.KEEP,
            executive_summary="Enter per the trader's plan.",
            investment_thesis="Bull case carried the debate.",
            price_target=220.0, time_horizon="5-10 trading sessions",
        )
    if trader_proposal is None:
        trader_proposal = TraderProposal(
            action=TraderAction.BUY, reasoning="Bull case is stronger.",
            entry_price_low=98.0, entry_price_high=102.0, reference_entry=100.0,
            stop_loss=90.0,
        )
    if research_plan is None:
        research_plan = ResearchPlan(
            recommendation=PortfolioRating.OVERWEIGHT,
            rationale="Bull carried the debate.", strategic_actions="Buy on confirmation.",
        )
    return {
        "research_plan_structured": research_plan.model_dump(mode="json"),
        "trader_proposal_structured": trader_proposal.model_dump(mode="json"),
        "portfolio_decision_structured": portfolio_decision.model_dump(mode="json"),
        "trader_investment_plan": legacy_trader_text or "**Action**: Hold\n**Entry Price**: not provided",
        "final_trade_decision": legacy_final_decision or "**Rating**: Hold",
        "investment_debate_state": {
            "bull_thesis": _thesis(direction=ResearchDirection.BULL, data_gaps=bull_data_gaps),
            "bear_thesis": _thesis(direction=ResearchDirection.BEAR, entry_price=100.0,
                                    take_profit=80.0, stop_loss=110.0, data_gaps=bear_data_gaps),
            "bull_trade_metrics": _trade_metrics(2.0),
            "bear_trade_metrics": _trade_metrics(1.5),
            "bull_conviction_history": [55, 60],
            "bear_conviction_history": [50],
            "verification_status": "PASS",
            "verifier_pass_count": 1,
            "repair_rounds": 0,
            "repaired_agents": [],
            "manager_integrity_status": "PASS",
            "manager_integrity_findings": [],
            "research_horizon": "5-10 trading sessions",
        },
        "risk_debate_state": {
            "risk_integrity_status": "PASS",
            "risk_integrity_findings": [],
            "portfolio_disposition": "KEEP",
        },
        "source_coverage": {"news": {"status": "SUCCESS_WITH_DATA"}},
        "validation_log": [],
        "run_provenance": run_provenance if run_provenance is not None else _run_provenance(),
    }


def _execution(final_state, **overrides) -> SimpleNamespace:
    defaults = {
        "ticker": "NVDA", "analysis_date": "2026-09-01", "rating": "Buy",
        "report_path": None, "duration_seconds": 1.0,
        "stats": {"llm_calls": 1, "tool_calls": 0, "tokens_in": 1, "tokens_out": 1},
        "fallback_warnings": 0, "base_seed": 184736291,
    }
    defaults.update(overrides)
    return SimpleNamespace(final_state=final_state, **defaults)


def _write_artifacts(tmp_path: Path, result: dict) -> tuple[Path, Path]:
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    report_path = tmp_path / "complete_report.md"
    report_path.write_text("# Report\n\nHold.\n", encoding="utf-8")
    return result_path, report_path


def _build(tmp_path: Path, **state_overrides):
    final_state = _full_final_state(**state_overrides)
    execution = _execution(final_state)
    result = result_document(execution, tmp_path)
    result_path, report_path = _write_artifacts(tmp_path, result)
    return execution, result, result_path, report_path


# --- 1/2: created on success, not on partial/failed run --------------------


@pytest.mark.unit
def test_1_record_created_on_successful_run(tmp_path):
    execution, result, result_path, report_path = _build(tmp_path)
    written = record_recommendation(
        execution=execution, result=result, config={},
        result_json_path=result_path, report_path=report_path,
    )
    assert written is not None
    assert written.exists()
    assert written.parent == (tmp_path / "ledger") / "recommendations" / "NVDA"


@pytest.mark.unit
def test_2_no_record_on_partial_or_failed_run(monkeypatch, tmp_path):
    class _BoomGraphCore:
        def stream(self, state, **kwargs):
            yield {"market_report": "partial"}
            raise RuntimeError("boom mid-run")

    class _BoomGraph(_FakeTradingAgentsGraph):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.graph = _BoomGraphCore()

    from types import SimpleNamespace as _SimpleNamespace

    monkeypatch.setattr(cli_run, "TradingAgentsGraph", _BoomGraph)
    monkeypatch.setattr(
        cli_run, "build_analyst_execution_plan", lambda analysts: _SimpleNamespace(specs=[])
    )
    result = run_single_analysis(
        ticker="NVDA", analysis_date="2026-09-30", analysts=["market"], depth_value=3,
        language="English", output_dir=tmp_path,
    )
    assert result["status"] == "error"
    assert not ((tmp_path / "ledger") / "recommendations").exists()


@pytest.mark.unit
def test_2b_no_record_when_portfolio_decision_never_parsed(tmp_path):
    final_state = _full_final_state()
    final_state["portfolio_decision_structured"] = None
    execution = _execution(final_state)
    result = result_document(execution, tmp_path)
    result_path, report_path = _write_artifacts(tmp_path, result)
    written = record_recommendation(
        execution=execution, result=result, config={},
        result_json_path=result_path, report_path=report_path,
    )
    assert written is None
    assert not ((tmp_path / "ledger") / "recommendations").exists()

    with pytest.raises(IncompleteRecommendationError):
        build_recommendation_record(
            execution=execution, result=result, config={}, result_path="result.json",
            result_sha256=sha256_file(result_path), report_path="complete_report.md",
            report_sha256=sha256_file(report_path),
        )


# --- 3: recommendation_id is unique -----------------------------------------


@pytest.mark.unit
def test_3_recommendation_id_is_unique(tmp_path):
    execution, result, result_path, report_path = _build(tmp_path)
    record_a = build_recommendation_record(
        execution=execution, result=result, config={}, result_path="result.json",
        result_sha256=sha256_file(result_path), report_path="complete_report.md",
        report_sha256=sha256_file(report_path),
    )
    record_b = build_recommendation_record(
        execution=execution, result=result, config={}, result_path="result.json",
        result_sha256=sha256_file(result_path), report_path="complete_report.md",
        report_sha256=sha256_file(report_path),
    )
    assert record_a.recommendation_id != record_b.recommendation_id


# --- 4/5/6/7: run_fingerprint stability -------------------------------------


@pytest.mark.unit
def test_4_run_fingerprint_stable_with_same_config():
    provenance = _run_provenance()
    fp1 = compute_run_fingerprint(ticker="NVDA", analysis_date="2026-09-01", run_provenance=provenance)
    fp2 = compute_run_fingerprint(ticker="NVDA", analysis_date="2026-09-01", run_provenance=provenance)
    assert fp1 == fp2


@pytest.mark.unit
def test_5_run_fingerprint_changes_when_model_changes():
    base = _run_provenance()
    changed = _run_provenance()
    changed["models"]["generation"]["name"] = "a-different-model"
    fp_base = compute_run_fingerprint(ticker="NVDA", analysis_date="2026-09-01", run_provenance=base)
    fp_changed = compute_run_fingerprint(ticker="NVDA", analysis_date="2026-09-01", run_provenance=changed)
    assert fp_base != fp_changed


@pytest.mark.unit
def test_6_run_fingerprint_changes_when_prompt_hash_changes():
    base = _run_provenance()
    changed = _run_provenance()
    changed["prompts"]["hashes"]["global_policy.txt"] = "a-different-hash"
    fp_base = compute_run_fingerprint(ticker="NVDA", analysis_date="2026-09-01", run_provenance=base)
    fp_changed = compute_run_fingerprint(ticker="NVDA", analysis_date="2026-09-01", run_provenance=changed)
    assert fp_base != fp_changed


@pytest.mark.unit
def test_7_run_fingerprint_changes_when_base_seed_changes():
    base = _run_provenance()
    changed = _run_provenance()
    changed["seed"]["base_used"] = 999999999
    fp_base = compute_run_fingerprint(ticker="NVDA", analysis_date="2026-09-01", run_provenance=base)
    fp_changed = compute_run_fingerprint(ticker="NVDA", analysis_date="2026-09-01", run_provenance=changed)
    assert fp_base != fp_changed


# --- 8: structured trader is source of truth, legacy text ignored ----------


@pytest.mark.unit
def test_8_structured_trader_is_source_of_truth_not_legacy_text(tmp_path):
    proposal = TraderProposal(action=TraderAction.BUY, reasoning="Structured says Buy.", stop_loss=90.0)
    execution, result, result_path, report_path = _build(
        tmp_path, trader_proposal=proposal,
        legacy_trader_text="**Action**: Sell\n**Entry Price**: 50.0",  # deliberately contradicts
    )
    record = build_recommendation_record(
        execution=execution, result=result, config={}, result_path="result.json",
        result_sha256=sha256_file(result_path), report_path="complete_report.md",
        report_sha256=sha256_file(report_path),
    )
    assert record.trade_plan.action == "Buy"
    assert record.decision.trader_action == "Buy"


# --- 9: full entry range is preserved ---------------------------------------


@pytest.mark.unit
def test_9_entry_range_is_fully_preserved(tmp_path):
    proposal = TraderProposal(
        action=TraderAction.BUY, reasoning="Range entry.",
        entry_price_low=98.0, entry_price_high=102.0, reference_entry=100.0, stop_loss=90.0,
    )
    execution, result, result_path, report_path = _build(tmp_path, trader_proposal=proposal)
    record = build_recommendation_record(
        execution=execution, result=result, config={}, result_path="result.json",
        result_sha256=sha256_file(result_path), report_path="complete_report.md",
        report_sha256=sha256_file(report_path),
    )
    assert record.trade_plan.entry_price_low == 98.0
    assert record.trade_plan.entry_price_high == 102.0
    assert record.trade_plan.reference_entry == 100.0
    assert record.trade_plan.entry_price is None


# --- 10/11: research plan / portfolio decision structured are preserved ----


@pytest.mark.unit
def test_10_research_plan_structured_is_preserved(tmp_path):
    plan = ResearchPlan(
        recommendation=PortfolioRating.UNDERWEIGHT, rationale="Bear carried it.",
        strategic_actions="Reduce exposure.",
    )
    execution, result, result_path, report_path = _build(tmp_path, research_plan=plan)
    record = build_recommendation_record(
        execution=execution, result=result, config={}, result_path="result.json",
        result_sha256=sha256_file(result_path), report_path="complete_report.md",
        report_sha256=sha256_file(report_path),
    )
    assert record.decision.research_manager_recommendation == "Underweight"


@pytest.mark.unit
def test_11_portfolio_decision_structured_is_preserved(tmp_path):
    decision = PortfolioDecision(
        rating=PortfolioRating.SELL, disposition=RiskDisposition.REJECT_PLAN,
        executive_summary="Exit.", investment_thesis="Bear case won.",
    )
    execution, result, result_path, report_path = _build(tmp_path, portfolio_decision=decision)
    record = build_recommendation_record(
        execution=execution, result=result, config={}, result_path="result.json",
        result_sha256=sha256_file(result_path), report_path="complete_report.md",
        report_sha256=sha256_file(report_path),
    )
    assert record.decision.portfolio_rating == "Sell"
    assert record.decision.portfolio_disposition == "REJECT_PLAN"


# --- 12/13: source_coverage and run_provenance are preserved ---------------


@pytest.mark.unit
def test_12_source_coverage_is_preserved(tmp_path):
    execution, result, result_path, report_path = _build(tmp_path)
    record = build_recommendation_record(
        execution=execution, result=result, config={}, result_path="result.json",
        result_sha256=sha256_file(result_path), report_path="complete_report.md",
        report_sha256=sha256_file(report_path),
    )
    assert record.evidence.source_coverage == {"news": {"status": "SUCCESS_WITH_DATA"}}


@pytest.mark.unit
def test_13_run_provenance_is_preserved(tmp_path):
    provenance = _run_provenance()
    execution, result, result_path, report_path = _build(tmp_path, run_provenance=provenance)
    record = build_recommendation_record(
        execution=execution, result=result, config={}, result_path="result.json",
        result_sha256=sha256_file(result_path), report_path="complete_report.md",
        report_sha256=sha256_file(report_path),
    )
    assert record.run.tradeproof_version == provenance["tradeproof_version"]
    assert record.run.git.commit == provenance["git"]["commit"]
    assert record.run.seed.base_used == provenance["seed"]["base_used"]
    assert record.artifacts.prompt_hashes == provenance["prompts"]["hashes"]


# --- 14/15/16: horizon parsing ----------------------------------------------


@pytest.mark.unit
def test_14_horizon_range_is_parsed_correctly():
    horizon = parse_horizon("5-10 trading sessions")
    assert horizon.parse_status == HorizonParseStatus.PARSED_RANGE
    assert horizon.unit == HorizonUnit.TRADING_SESSIONS
    assert horizon.min == 5.0
    assert horizon.max == 10.0
    assert horizon.raw == "5-10 trading sessions"


@pytest.mark.unit
def test_15_horizon_single_is_parsed():
    horizon = parse_horizon("10 trading days")
    assert horizon.parse_status == HorizonParseStatus.PARSED_SINGLE
    assert horizon.unit == HorizonUnit.TRADING_DAYS
    assert horizon.min == 10.0
    assert horizon.max == 10.0


@pytest.mark.unit
def test_16_horizon_unrecognizable_is_unparseable():
    horizon = parse_horizon("soon, once the picture clears up")
    assert horizon.parse_status == HorizonParseStatus.UNPARSEABLE
    assert horizon.unit == HorizonUnit.UNKNOWN
    assert horizon.min is None and horizon.max is None
    assert horizon.raw == "soon, once the picture clears up"


# --- Regression: a real run_provenance validation with output_language=
# Spanish produced "5-10 sesiones" from the Portfolio Manager, which the
# (then English-only) parser read as UNPARSEABLE. parse_horizon() itself
# must understand the Spanish terms this project's own output_language
# setting can actually produce -- deterministic keyword matching, no LLM
# translation.


@pytest.mark.unit
def test_spanish_regression_5_10_sesiones_parses_as_a_range():
    horizon = parse_horizon("5-10 sesiones")
    assert horizon.raw == "5-10 sesiones"
    assert horizon.min == 5.0
    assert horizon.max == 10.0
    assert horizon.unit == HorizonUnit.TRADING_SESSIONS
    assert horizon.parse_status == HorizonParseStatus.PARSED_RANGE


@pytest.mark.unit
def test_spanish_single_10_sesiones_parses_correctly():
    horizon = parse_horizon("10 sesiones")
    assert horizon.min == 10.0
    assert horizon.max == 10.0
    assert horizon.unit == HorizonUnit.TRADING_SESSIONS
    assert horizon.parse_status == HorizonParseStatus.PARSED_SINGLE


@pytest.mark.unit
def test_english_range_5_10_trading_sessions_still_parses():
    horizon = parse_horizon("5-10 trading sessions")
    assert horizon.min == 5.0 and horizon.max == 10.0
    assert horizon.unit == HorizonUnit.TRADING_SESSIONS
    assert horizon.parse_status == HorizonParseStatus.PARSED_RANGE


@pytest.mark.unit
def test_english_single_10_trading_days_still_parses():
    horizon = parse_horizon("10 trading days")
    assert horizon.min == 10.0 and horizon.max == 10.0
    assert horizon.unit == HorizonUnit.TRADING_DAYS
    assert horizon.parse_status == HorizonParseStatus.PARSED_SINGLE


@pytest.mark.unit
def test_genuinely_unknown_spanish_text_is_still_unparseable():
    horizon = parse_horizon("pronto, cuando el panorama se aclare")
    assert horizon.unit == HorizonUnit.UNKNOWN
    assert horizon.min is None and horizon.max is None
    assert horizon.parse_status == HorizonParseStatus.UNPARSEABLE
    assert horizon.raw == "pronto, cuando el panorama se aclare"


@pytest.mark.unit
def test_spanish_trading_days_and_weeks_and_months_variants():
    assert parse_horizon("5 dias de negociacion").unit == HorizonUnit.TRADING_DAYS
    assert parse_horizon("5 días hábiles").unit == HorizonUnit.TRADING_DAYS
    assert parse_horizon("2 semanas").unit == HorizonUnit.WEEKS
    assert parse_horizon("3 meses").unit == HorizonUnit.MONTHS
    assert parse_horizon("10 dias naturales").unit == HorizonUnit.CALENDAR_DAYS


# --- resolve_horizon: structured, non-localized source wins for unit/min/max,
# raw always preserves exactly what the recommendation itself said ----------


@pytest.mark.unit
def test_resolve_horizon_prefers_structured_source_over_localized_pm_text():
    """The exact real-world bug: output_language=Spanish made the Portfolio
    Manager write "5-10 sesiones"; the shared (always-English, never
    LLM-touched) debate horizon is used for unit/min/max instead of parsing
    that translated text, while raw still shows the PM's own Spanish text."""
    horizon = resolve_horizon(
        portfolio_decision={"time_horizon": "5-10 sesiones"},
        debate={"research_horizon": "5-10 trading sessions"},
        config={"research_horizon": "5-10 trading sessions"},
    )
    assert horizon.raw == "5-10 sesiones"
    assert horizon.unit == HorizonUnit.TRADING_SESSIONS
    assert horizon.min == 5.0
    assert horizon.max == 10.0
    assert horizon.parse_status == HorizonParseStatus.PARSED_RANGE


@pytest.mark.unit
def test_resolve_horizon_falls_back_to_config_when_debate_lacks_research_horizon():
    horizon = resolve_horizon(
        portfolio_decision={}, debate={}, config={"research_horizon": "5-10 trading sessions"},
    )
    assert horizon.raw == "5-10 trading sessions"
    assert horizon.unit == HorizonUnit.TRADING_SESSIONS
    assert horizon.parse_status == HorizonParseStatus.PARSED_RANGE


@pytest.mark.unit
def test_resolve_horizon_parses_pm_text_directly_when_no_structured_source_exists():
    horizon = resolve_horizon(
        portfolio_decision={"time_horizon": "10 sesiones"}, debate={}, config={},
    )
    assert horizon.raw == "10 sesiones"
    assert horizon.unit == HorizonUnit.TRADING_SESSIONS
    assert horizon.min == 10.0 and horizon.max == 10.0
    assert horizon.parse_status == HorizonParseStatus.PARSED_SINGLE


# --- 17/18: artifact hashes --------------------------------------------------


@pytest.mark.unit
def test_17_result_sha256_is_correct(tmp_path):
    execution, result, result_path, report_path = _build(tmp_path)
    written = record_recommendation(
        execution=execution, result=result, config={},
        result_json_path=result_path, report_path=report_path,
    )
    saved = json.loads(written.read_text(encoding="utf-8"))
    import hashlib

    expected = hashlib.sha256(result_path.read_bytes()).hexdigest()
    assert saved["artifacts"]["result_sha256"] == expected


@pytest.mark.unit
def test_18_report_sha256_is_correct(tmp_path):
    execution, result, result_path, report_path = _build(tmp_path)
    written = record_recommendation(
        execution=execution, result=result, config={},
        result_json_path=result_path, report_path=report_path,
    )
    saved = json.loads(written.read_text(encoding="utf-8"))
    import hashlib

    expected = hashlib.sha256(report_path.read_bytes()).hexdigest()
    assert saved["artifacts"]["report_sha256"] == expected


# --- 19/20: record_hash is verifiable, tampering breaks it ------------------


@pytest.mark.unit
def test_19_record_hash_is_verifiable(tmp_path):
    execution, result, result_path, report_path = _build(tmp_path)
    written = record_recommendation(
        execution=execution, result=result, config={},
        result_json_path=result_path, report_path=report_path,
    )
    assert verify_recommendation_record(written) is True


@pytest.mark.unit
def test_20_tampering_with_the_json_breaks_verify(tmp_path):
    execution, result, result_path, report_path = _build(tmp_path)
    written = record_recommendation(
        execution=execution, result=result, config={},
        result_json_path=result_path, report_path=report_path,
    )
    payload = json.loads(written.read_text(encoding="utf-8"))
    payload["decision"]["portfolio_rating"] = "Sell"  # tamper
    written.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    assert verify_recommendation_record(written) is False


# --- 21: an existing record is never overwritten ----------------------------


@pytest.mark.unit
def test_21_existing_record_is_never_overwritten(tmp_path):
    execution, result, result_path, report_path = _build(tmp_path)
    record = build_recommendation_record(
        execution=execution, result=result, config={}, result_path="result.json",
        result_sha256=sha256_file(result_path), report_path="complete_report.md",
        report_sha256=sha256_file(report_path),
    )
    first_path = write_recommendation_record(record, tmp_path)
    # Same record object written again: identical content, idempotent no-op.
    second_path = write_recommendation_record(record, tmp_path)
    assert first_path == second_path

    tampered = record.model_copy(update={
        "decision": record.decision.model_copy(update={"portfolio_rating": "Sell"})
    })
    tampered = tampered.model_copy(update={"recommendation_id": record.recommendation_id})
    with pytest.raises(FileExistsError):
        write_recommendation_record(tampered, tmp_path)


# --- 22: git dirty produces the correct reproducibility_status -------------


@pytest.mark.unit
def test_22_git_dirty_produces_correct_reproducibility_status(tmp_path):
    for dirty, commit, expected in (
        (True, "abc123", ReproducibilityStatus.DIRTY_WORKTREE),
        (False, "abc123", ReproducibilityStatus.CLEAN_COMMIT),
        (None, None, ReproducibilityStatus.UNKNOWN),
    ):
        provenance = _run_provenance(git={"commit": commit, "branch": "main", "dirty": dirty})
        execution, result, result_path, report_path = _build(tmp_path, run_provenance=provenance)
        record = build_recommendation_record(
            execution=execution, result=result, config={}, result_path="result.json",
            result_sha256=sha256_file(result_path), report_path="complete_report.md",
            report_sha256=sha256_file(report_path),
        )
        assert record.run.reproducibility_status == expected


# --- 23: trace manifest path is null when tracing is off -------------------


@pytest.mark.unit
def test_23_trace_manifest_path_is_null_without_trace(tmp_path):
    execution, result, result_path, report_path = _build(tmp_path)
    record = build_recommendation_record(
        execution=execution, result=result, config={}, result_path="result.json",
        result_sha256=sha256_file(result_path), report_path="complete_report.md",
        report_sha256=sha256_file(report_path),
    )
    assert record.artifacts.trace_manifest_path is None


# --- 24/25: no coupling with decision_log / backtest ------------------------


@pytest.mark.unit
def test_24_does_not_touch_decision_log():
    import ast

    import tradingagents.recommendation_record as rr

    source = Path(rr.__file__).read_text(encoding="utf-8")
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
    assert not any("decision_log" in module for module in imported_modules)


@pytest.mark.unit
def test_25_does_not_touch_backtest():
    import ast

    import tradingagents.recommendation_record as rr

    source = Path(rr.__file__).read_text(encoding="utf-8")
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
    assert not any("backtest" in module for module in imported_modules)


# --- 26: the report file is never modified by the ledger write -------------


@pytest.mark.unit
def test_26_report_md_is_never_modified(tmp_path):
    execution, result, result_path, report_path = _build(tmp_path)
    original_bytes = report_path.read_bytes()
    record_recommendation(
        execution=execution, result=result, config={},
        result_json_path=result_path, report_path=report_path,
    )
    assert report_path.read_bytes() == original_bytes


# --- 27/28: JSON-serializable, schema_version ------------------------------


@pytest.mark.unit
def test_27_final_json_is_serializable(tmp_path):
    execution, result, result_path, report_path = _build(tmp_path)
    record = build_recommendation_record(
        execution=execution, result=result, config={}, result_path="result.json",
        result_sha256=sha256_file(result_path), report_path="complete_report.md",
        report_sha256=sha256_file(report_path),
    )
    json.dumps(record.model_dump(mode="json"))


@pytest.mark.unit
def test_28_schema_version_is_1_0(tmp_path):
    execution, result, result_path, report_path = _build(tmp_path)
    record = build_recommendation_record(
        execution=execution, result=result, config={}, result_path="result.json",
        result_sha256=sha256_file(result_path), report_path="complete_report.md",
        report_sha256=sha256_file(report_path),
    )
    assert record.schema_version == "1.0"


# --- 29: reruns of the same ticker/date can coexist -------------------------


@pytest.mark.unit
def test_29_reruns_of_same_ticker_date_coexist(tmp_path):
    execution, result, result_path, report_path = _build(tmp_path)
    path_a = record_recommendation(
        execution=execution, result=result, config={},
        result_json_path=result_path, report_path=report_path,
    )
    path_b = record_recommendation(
        execution=execution, result=result, config={},
        result_json_path=result_path, report_path=report_path,
    )
    assert path_a != path_b
    assert path_a.exists() and path_b.exists()


# --- 30: legacy fields are not used when structured data exists ------------


@pytest.mark.unit
def test_30_legacy_fields_unused_when_structured_exists(tmp_path):
    decision = PortfolioDecision(
        rating=PortfolioRating.BUY, disposition=RiskDisposition.KEEP,
        executive_summary="x", investment_thesis="y",
    )
    execution, result, result_path, report_path = _build(
        tmp_path, portfolio_decision=decision,
        legacy_final_decision="**Rating**: Sell",  # deliberately contradicts structured
    )
    record = build_recommendation_record(
        execution=execution, result=result, config={}, result_path="result.json",
        result_sha256=sha256_file(result_path), report_path="complete_report.md",
        report_sha256=sha256_file(report_path),
    )
    assert record.decision.portfolio_rating == "Buy"
