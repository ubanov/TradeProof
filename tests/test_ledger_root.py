"""RecommendationRecord storage lives in a global ``ledger_root``, separate
from any run's own ``run_output_dir``/``--output-dir`` -- so the ledger is
not fragmented across however many ad-hoc output directories a user has
ever pointed the CLI at (see fork README: "RecommendationRecord").

Default/env-override coverage for ``ledger_root`` itself lives in
tests/test_env_overrides.py (next to the other path-like config keys);
backtest/benchmark-replay/direct-``propagate()`` non-coupling is already
covered by tests/test_recommendation_record_persistence.py's AST checks
(tests 10/11/12) and is unaffected by this split, so it is not duplicated
here.

Numbered per the task's own "TESTS" list (1-16); the ones covered elsewhere
are noted rather than duplicated.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import cli.run as cli_run
from cli.headless import run_batch_analysis, run_single_analysis
from tests.test_llm_trace_integration import _FakeTradingAgentsGraph
from tests.test_recommendation_record import _full_final_state
from tests.test_recommendation_record_persistence import (
    _recommendation_files,
    _RichFakeGraph,
    _run_interactive,
)
from tradingagents.recommendation_record import (
    ArtifactsRef,
    RecommendationRecord,
    verify_recommendation_record,
)

# 1/2 (default ledger_root stable / env override): see
# tests/test_env_overrides.py::test_empty_path_value_keeps_the_default_path
# and ::test_ledger_root_env_override.


@pytest.fixture
def structured_graph(monkeypatch):
    """Headless equivalent of test_recommendation_record_persistence.py's
    ``fake_graph_structured`` -- a dedicated local copy (rather than an
    imported fixture used as a parameter) to avoid the ruff F811 a
    cross-module fixture import causes when also used by name."""

    class _StructuredGraphCore:
        def stream(self, state, **kwargs):
            yield {"messages": [], **_full_final_state()}

    class _StructuredTradingAgentsGraph(_FakeTradingAgentsGraph):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.graph = _StructuredGraphCore()

    monkeypatch.setattr(cli_run, "TradingAgentsGraph", _StructuredTradingAgentsGraph)
    monkeypatch.setattr(
        cli_run, "build_analyst_execution_plan", lambda analysts: SimpleNamespace(specs=[])
    )
    return _StructuredTradingAgentsGraph


# --- 3: interactive writes to ledger_root, never inside the run dir --------


@pytest.mark.unit
def test_3_interactive_writes_to_ledger_root_not_the_run_dir(monkeypatch, tmp_path):
    _run_interactive(monkeypatch, tmp_path, _RichFakeGraph(), save_answer="N")
    records = _recommendation_files(tmp_path)
    assert len(records) == 1
    ledger_root = (tmp_path / "ledger")
    assert records[0].is_relative_to(ledger_root)
    assert not records[0].is_relative_to(tmp_path / "results")


# --- 4: `run` writes to ledger_root independent of --output-dir -----------


@pytest.mark.unit
def test_4_run_writes_to_ledger_root_regardless_of_output_dir(structured_graph, tmp_path):
    out_a = tmp_path / "out_a"
    out_b = tmp_path / "out_b"
    run_single_analysis(
        ticker="NVDA", analysis_date="2026-09-30", analysts=["market"], depth_value=3,
        language="English", output_dir=out_a,
    )
    run_single_analysis(
        ticker="NVDA", analysis_date="2026-09-30", analysts=["market"], depth_value=3,
        language="English", output_dir=out_b,
    )
    ledger_root = (tmp_path / "ledger")
    records = list((ledger_root / "recommendations" / "NVDA").glob("*.json"))
    assert len(records) == 2  # same ledger_root, two different --output-dir
    assert not (out_a / "recommendations").exists()
    assert not (out_b / "recommendations").exists()


# --- 5: batch writes every ticker under the SAME ledger_root --------------


@pytest.mark.unit
def test_5_batch_writes_every_ticker_under_the_same_ledger_root(structured_graph, tmp_path):
    results, _, _ = run_batch_analysis(
        tickers=["NVDA", "AAPL"], analysis_date="2026-09-30", analysts=["market"],
        depth_name="shallow", depth_value=1, language="English",
        output_dir=tmp_path / "batch_out", progress=lambda *a, **k: None,
    )
    assert all(r["status"] == "ok" for r in results)
    ledger_root = (tmp_path / "ledger")
    for ticker in ("NVDA", "AAPL"):
        records = list((ledger_root / "recommendations" / ticker).glob("*.json"))
        assert len(records) == 1


# --- 6/7/8: artifacts stay put, record references them unambiguously, -----
# --- hashes still verify ----------------------------------------------------


@pytest.mark.unit
def test_6_7_8_artifacts_unchanged_record_unambiguous_hashes_verify(structured_graph, tmp_path):
    run_single_analysis(
        ticker="NVDA", analysis_date="2026-09-30", analysts=["market"], depth_value=3,
        language="English", output_dir=tmp_path,
    )
    # 6: artifacts remain exactly where run/batch always put them.
    assert (tmp_path / "result.json").exists()
    assert (tmp_path / "complete_report.md").exists()

    ledger_root = (tmp_path / "ledger")
    records = list((ledger_root / "recommendations" / "NVDA").glob("*.json"))
    assert len(records) == 1
    payload = json.loads(records[0].read_text(encoding="utf-8"))

    # 7: unambiguous -- absolute paths, no "../.." convention needed.
    result_path = Path(payload["artifacts"]["result_path"])
    report_path = Path(payload["artifacts"]["report_path"])
    assert result_path.is_absolute() and result_path == (tmp_path / "result.json").resolve()
    assert report_path.is_absolute() and report_path == (tmp_path / "complete_report.md").resolve()

    # 8: hashes still verify.
    assert verify_recommendation_record(records[0]) is True


# --- 9: same ticker/date can coexist (two UUIDs, no collision) ------------


@pytest.mark.unit
def test_9_same_ticker_date_can_coexist(structured_graph, tmp_path):
    run_single_analysis(
        ticker="NVDA", analysis_date="2026-09-30", analysts=["market"], depth_value=3,
        language="English", output_dir=tmp_path / "run1",
    )
    run_single_analysis(
        ticker="NVDA", analysis_date="2026-09-30", analysts=["market"], depth_value=3,
        language="English", output_dir=tmp_path / "run2",
    )
    ledger_root = (tmp_path / "ledger")
    records = list((ledger_root / "recommendations" / "NVDA").glob("*.json"))
    assert len(records) == 2
    ids = {r.stem for r in records}
    assert len(ids) == 2  # distinct recommendation_ids, no collision


# --- 10: a different --output-dir never fragments the ledger --------------


@pytest.mark.unit
def test_10_different_output_dir_does_not_fragment_the_ledger(structured_graph, tmp_path):
    """The exact gap this task closes: two invocations with UNRELATED
    --output-dir trees (as a real user switching projects/dates would do)
    must still land in ONE enumerable ledger location."""
    first_root = tmp_path / "project_a" / "2026-09-30"
    second_root = tmp_path / "somewhere_else_entirely" / "batch_runs" / "20261013"
    run_single_analysis(
        ticker="NVDA", analysis_date="2026-09-30", analysts=["market"], depth_value=3,
        language="English", output_dir=first_root,
    )
    run_single_analysis(
        ticker="NVDA", analysis_date="2026-09-30", analysts=["market"], depth_value=3,
        language="English", output_dir=second_root,
    )
    ledger_root = (tmp_path / "ledger")
    # One glob, one root, both recommendations -- no need to know first_root
    # or second_root existed at all to enumerate them.
    records = list((ledger_root / "recommendations" / "NVDA").glob("*.json"))
    assert len(records) == 2


# --- 11/12/13: backtest / benchmark / library -- see
# tests/test_recommendation_record_persistence.py tests 10/11/12 (AST-based,
# unaffected by this split: those modules still import neither
# recommendation_record nor result_document).


# --- 14: a ledger write failure warns, loses nothing, fakes nothing -------


@pytest.mark.unit
def test_14_ledger_write_failure_warns_and_keeps_artifacts(monkeypatch, structured_graph, tmp_path, capsys):
    import tradingagents.recommendation_record as rr

    def _boom(record, ledger_root):
        raise PermissionError("simulated permission denied")

    monkeypatch.setattr(rr, "write_recommendation_record", _boom)
    result = run_single_analysis(
        ticker="NVDA", analysis_date="2026-09-30", analysts=["market"], depth_value=3,
        language="English", output_dir=tmp_path,
    )
    assert result["status"] == "ok"
    assert (tmp_path / "result.json").exists()
    assert (tmp_path / "complete_report.md").exists()
    ledger_root = (tmp_path / "ledger")
    assert not (ledger_root / "recommendations").exists()
    captured = capsys.readouterr()
    assert "[WARN]" in captured.out
    assert "simulated permission denied" in captured.out


# --- 15: schema fields unchanged except artifact-path semantics -----------


@pytest.mark.unit
def test_15_schema_fields_unchanged():
    assert set(ArtifactsRef.model_fields) == {
        "result_path", "result_sha256", "report_path", "report_sha256",
        "prompt_hashes", "trace_manifest_path",
    }
    assert set(RecommendationRecord.model_fields) == {
        "schema_version", "recommendation_id", "created_at", "record_hash",
        "run_fingerprint", "run", "models", "decision", "trade_plan",
        "horizon", "research", "evidence", "artifacts", "portfolio_context",
        # Added in schema 1.1 (see tradingagents.recommendation_record):
        # optional, backward-compatible with 1.0 records lacking this key.
        "price_basis",
        # Added in schema 1.3 (see tests/test_pre_pm_snapshot.py): optional,
        # backward-compatible with records lacking this key.
        "pre_pm_snapshot",
    }


# --- 16: prior RecommendationRecord tests still pass -- see full-suite run.
