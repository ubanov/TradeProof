"""CLI output now shows where the RecommendationRecord ledger lives, not
just the run's own result.json/complete_report.md -- see fork README:
"RecommendationRecord". ``persist_successful_run`` (shared by interactive,
`run`, and `batch`) prints "Ledger root: ..." unconditionally and
"Recommendation record: <path>" (the exact path record_recommendation()
itself returned) or "Recommendation record: not written" -- the ledger's
own storage/resolution logic is untouched; this only adds visibility.

Numbered per the task's own "TESTS" list (1-7).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

import cli.main as m
import cli.run as cli_run
from cli.headless import run_batch_analysis, run_single_analysis
from tests.test_llm_trace_integration import _FakeTradingAgentsGraph
from tests.test_recommendation_record import _full_final_state
from tests.test_recommendation_record_persistence import _RichFakeGraph, _run_interactive
from tradingagents.recommendation_record import resolve_ledger_root, verify_recommendation_record


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def structured_graph(monkeypatch):
    """A real end-to-end run (through execute_analysis) whose one streamed
    chunk carries a full, structurally-valid final decision -- a local copy
    rather than an imported fixture used as a parameter, to avoid the ruff
    F811 that causes (same precedent as the other CLI/ledger test files)."""

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


def _ledger_record(tmp_path: Path, ticker: str) -> Path:
    records = list((tmp_path / "ledger" / "recommendations" / ticker).glob("*.json"))
    assert len(records) == 1, records
    return records[0]


# --- 1/2: run success prints Ledger root and Recommendation record with ----
# --- the real path ----------------------------------------------------------


@pytest.mark.unit
def test_1_2_successful_run_prints_ledger_root_and_record_path(runner, structured_graph, tmp_path):
    result = runner.invoke(m.app, ["run", "--ticker", "NVDA", "--date", "2026-09-30"])
    assert result.exit_code == 0, result.output

    ledger_root_line = next(
        line for line in result.output.splitlines() if "Ledger root:" in line
    )
    assert str(resolve_ledger_root({})) in ledger_root_line

    record_line = next(
        line for line in result.output.splitlines() if "Recommendation record:" in line
    )
    printed_path = Path(record_line.split("Recommendation record:", 1)[1].strip())
    assert printed_path.exists()
    assert printed_path == _ledger_record(tmp_path, "NVDA")


# --- 3: a record that is NOT written keeps the existing WARN, plus a clear -
# --- "not written" line ------------------------------------------------------


@pytest.mark.unit
def test_3_record_not_written_keeps_warning_and_says_so_clearly(runner, monkeypatch):
    # _FakeTradingAgentsGraph's bare final_trade_decision text never produces
    # a parsed portfolio_decision, so record_recommendation() is a no-op.
    monkeypatch.setattr(cli_run, "TradingAgentsGraph", _FakeTradingAgentsGraph)
    monkeypatch.setattr(
        cli_run, "build_analyst_execution_plan", lambda analysts: SimpleNamespace(specs=[])
    )
    result = runner.invoke(m.app, ["run", "--ticker", "NVDA", "--date", "2026-09-30"])
    assert result.exit_code == 0, result.output
    assert "[WARN] RecommendationRecord not written for NVDA" in result.output
    assert any(
        "Recommendation record: not written" in line for line in result.output.splitlines()
    )


# --- 4: interactive also shows the ledger root and record path -------------


@pytest.mark.unit
def test_4_interactive_shows_ledger_root_and_record_path(monkeypatch, tmp_path):
    buffer = _run_interactive(monkeypatch, tmp_path, _RichFakeGraph(), save_answer="N")
    system_messages = [text for _, kind, text in buffer.messages if kind == "System"]
    assert any("Ledger root:" in text for text in system_messages)
    record_messages = [text for text in system_messages if "Recommendation record:" in text]
    assert len(record_messages) == 1
    printed_path = Path(record_messages[0].split("Recommendation record:", 1)[1].strip())
    assert printed_path.exists()
    assert printed_path == _ledger_record(tmp_path, "NVDA")


# --- 5: batch shows one Recommendation record line per ticker --------------


@pytest.mark.unit
def test_5_batch_shows_one_record_line_per_ticker(structured_graph, tmp_path):
    lines = []
    results, _, _ = run_batch_analysis(
        tickers=["NVDA", "AAPL"], analysis_date="2026-09-30", analysts=["market"],
        depth_name="shallow", depth_value=1, language="English",
        output_dir=tmp_path / "batch_out", progress=lines.append,
    )
    assert all(r["status"] == "ok" for r in results)
    record_lines = {line for line in lines if "Recommendation record:" in line}
    assert len(record_lines) == 2
    assert any(line.startswith("[1/2] NVDA - Recommendation record:") for line in record_lines)
    assert any(line.startswith("[2/2] AAPL - Recommendation record:") for line in record_lines)
    for ticker in ("NVDA", "AAPL"):
        assert _ledger_record(tmp_path, ticker).exists()


# --- 6: result.json / complete_report.md locations are unchanged -----------


@pytest.mark.unit
def test_6_result_and_report_paths_are_unchanged(structured_graph, tmp_path):
    result = run_single_analysis(
        ticker="NVDA", analysis_date="2026-09-30", analysts=["market"], depth_value=3,
        language="English", output_dir=tmp_path,
    )
    assert result["status"] == "ok"
    assert (tmp_path / "result.json").exists()
    assert (tmp_path / "complete_report.md").exists()
    assert result["report_path"] == "complete_report.md"


# --- 7: ledger storage itself (location, hashes, verifiability) is ---------
# --- unchanged ----------------------------------------------------------------


@pytest.mark.unit
def test_7_ledger_storage_is_unchanged(structured_graph, tmp_path):
    result = run_single_analysis(
        ticker="NVDA", analysis_date="2026-09-30", analysts=["market"], depth_value=3,
        language="English", output_dir=tmp_path,
    )
    record_path = _ledger_record(tmp_path, "NVDA")
    assert record_path.parent == resolve_ledger_root({}) / "recommendations" / "NVDA"
    payload = json.loads(record_path.read_text(encoding="utf-8"))
    assert payload["artifacts"]["result_path"] == str((tmp_path / "result.json").resolve())
    assert payload["artifacts"]["result_sha256"] == hashlib.sha256(
        (tmp_path / "result.json").read_bytes()
    ).hexdigest()
    assert verify_recommendation_record(record_path) is True
    assert result["status"] == "ok"
