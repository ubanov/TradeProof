"""RecommendationRecord persistence is a single shared path
(``cli/result_document.py::persist_successful_run``) across every normal
frontend that produces a final recommendation -- interactive
(``cli/run.py::run_analysis``), single (``cli/headless.py::run_single_analysis``),
and batch (which calls single per ticker) -- and deliberately absent from
backtest, benchmark-replay, and direct ``TradingAgentsGraph.propagate()``
library usage, none of which this task may give new side effects.

A successful interactive recommendation is recorded UNCONDITIONALLY,
regardless of the user's answer to "Save report?" -- that prompt controls
only an optional additional export copy, never whether the recommendation
enters the ledger (see fork README: "RecommendationRecord" -- avoiding
post-hoc selection bias in a future longitudinal evaluation).

Numbered per the task's own "TESTS" list (1-20).
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import cli.headless as cli_headless
import cli.main as cli_main
import cli.result_document as cli_result_document
import cli.run as cli_run
from cli.headless import run_single_analysis
from tests.test_llm_trace_integration import _FakeTradingAgentsGraph
from tests.test_recommendation_record import _full_final_state
from tradingagents.recommendation_record import verify_recommendation_record


class _NullLive:
    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakeBuffer:
    def __init__(self):
        self.messages = []
        self.tool_calls = []
        self.report_sections = {}
        self.agent_status = {}
        self.selected_analysts = []
        self._processed_message_ids = set()

    def init_for_analysis(self, selected_analysts):
        self.selected_analysts = [a.lower() for a in selected_analysts]

    def add_message(self, kind, content):
        self.messages.append((0.0, kind, content))

    def add_tool_call(self, name, args):
        self.tool_calls.append((0.0, name, args))

    def update_report_section(self, *a):
        pass

    def update_agent_status(self, agent, status):
        self.agent_status[agent] = status


class _RichFakeGraph:
    """Same lifecycle shape as test_cli_decision_log.py's ``_FakeGraph``, but
    its one streamed chunk carries a full, structurally-valid final decision
    (research/trader/portfolio-manager structured objects) -- the interactive
    equivalent of test_llm_trace_integration.py's ``_FakeTradingAgentsGraph``,
    needed here because a RecommendationRecord requires a parsed
    ``portfolio_decision``, which that simpler fixture never produces."""

    def __init__(self):
        self.calls = []
        self.graph = self
        self.propagator = self

    def create_run_state(self, ticker, trade_date, asset_type="stock", portfolio=None):
        self.calls.append(("create_run_state", ticker, trade_date))
        return {"messages": [], "company_of_interest": ticker}

    def process_signal(self, text):
        from tradingagents.agents.rating import parse_rating
        return parse_rating(text)

    def record_decision(self, ticker, trade_date, final_state):
        self.calls.append(("record_decision", ticker, trade_date, final_state.get("final_trade_decision")))

    def get_graph_args(self, callbacks=None):
        return {}

    def begin_checkpoint(self, *a, **k):
        return None

    def checkpoint_input(self, state):
        return state

    def clear_checkpoint_on_success(self, *a, **k):
        self.calls.append(("clear_checkpoint",))

    def end_checkpoint(self):
        pass

    def stream(self, graph_input, **kwargs):
        yield {"messages": [], **_full_final_state()}


class _PartialFakeGraph(_RichFakeGraph):
    def stream(self, graph_input, **kwargs):
        yield {"messages": [], "market_report": "partial market report"}
        raise RuntimeError("boom mid-run")


def _prompt_responder(save_answer="N"):
    def _prompt(text, default=None, **kwargs):
        if text.startswith("Save report?"):
            return save_answer
        if text.startswith("Save path"):
            return default
        return "N"  # "Display full report?" and anything else
    return _prompt


def _run_interactive(monkeypatch, tmp_path, fake, *, save_answer="N"):
    """Drive the real ``cli.run.run_analysis()`` against ``fake``, exactly
    the pattern test_cli_decision_log.py's ``_run_cli`` already established
    for the decision log, extended to control the "Save report?" answer."""
    from cli.models import AnalystType

    buffer = _FakeBuffer()
    monkeypatch.setattr(cli_run, "TradingAgentsGraph", lambda *a, **k: fake)
    monkeypatch.setattr(cli_run, "message_buffer", buffer)
    monkeypatch.setattr(cli_run, "create_layout", lambda: None)
    monkeypatch.setattr(cli_run, "update_display", lambda *a, **k: None)
    monkeypatch.setattr(cli_run, "Live", _NullLive)
    monkeypatch.setattr(cli_run, "get_user_selections", lambda: {
        "ticker": "NVDA", "analysis_date": "2026-01-10",
        "analysts": [AnalystType.MARKET], "asset_type": "stock",
    })
    monkeypatch.setattr(cli_run, "_build_run_config", lambda selections, checkpoint: {
        "data_cache_dir": str(tmp_path / "cache"), "results_dir": str(tmp_path / "results"),
        "llm_provider": "ollama",
    })
    monkeypatch.setattr(cli_main.typer, "prompt", _prompt_responder(save_answer))
    cli_run.run_analysis()
    return buffer


@pytest.fixture
def fake_graph_structured(monkeypatch):
    """Like test_llm_trace_integration.py's ``fake_graph`` fixture, but its
    one streamed chunk carries a full, structurally-valid final decision --
    needed because a RecommendationRecord requires a parsed
    ``portfolio_decision``, which that simpler fixture's bare
    ``final_trade_decision`` text never produces."""

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


def _recommendation_files(tmp_path: Path) -> list[Path]:
    """Every record in the global ledger for NVDA -- a FLAT location, no
    longer nested inside any particular run's own output directory (see
    fork README: "RecommendationRecord" -- ledger_root)."""
    ledger_root = (tmp_path / "ledger")
    recommendations_dir = ledger_root / "recommendations" / "NVDA"
    if not recommendations_dir.exists():
        return []
    return sorted(recommendations_dir.glob("*.json"))


# --- 1/2: interactive success always records, regardless of Save report ----


@pytest.mark.unit
def test_1_interactive_success_creates_a_record(monkeypatch, tmp_path):
    _run_interactive(monkeypatch, tmp_path, _RichFakeGraph(), save_answer="N")
    assert len(_recommendation_files(tmp_path)) == 1


@pytest.mark.unit
def test_2_interactive_save_report_no_still_records(monkeypatch, tmp_path):
    _run_interactive(monkeypatch, tmp_path, _RichFakeGraph(), save_answer="N")
    records = _recommendation_files(tmp_path)
    assert len(records) == 1
    payload = json.loads(records[0].read_text(encoding="utf-8"))
    assert payload["decision"]["portfolio_rating"] is not None


# --- 3/4: Save report = YES -> exactly one record, hashes unaffected -------


@pytest.mark.unit
def test_3_interactive_save_report_yes_creates_exactly_one_record(monkeypatch, tmp_path):
    _run_interactive(monkeypatch, tmp_path, _RichFakeGraph(), save_answer="Y")
    assert len(_recommendation_files(tmp_path)) == 1


@pytest.mark.unit
def test_4_exported_report_copy_does_not_change_record_hashes(monkeypatch, tmp_path):
    _run_interactive(monkeypatch, tmp_path, _RichFakeGraph(), save_answer="Y")
    records = _recommendation_files(tmp_path)
    assert len(records) == 1
    payload = json.loads(records[0].read_text(encoding="utf-8"))
    canonical_result_path = Path(payload["artifacts"]["result_path"])
    assert canonical_result_path.is_absolute()
    expected_result_sha256 = hashlib.sha256(canonical_result_path.read_bytes()).hexdigest()
    assert payload["artifacts"]["result_sha256"] == expected_result_sha256
    assert verify_recommendation_record(records[0]) is True
    # The export prompt wrote a SEPARATE copy under results/.../reports/,
    # which must not be what the record's hashes point to.
    assert "reports" not in canonical_result_path.parts
    assert canonical_result_path.name == "result.json"


# --- 5/6: interactive partial/error never records --------------------------


@pytest.mark.unit
def test_5_interactive_partial_creates_no_record(monkeypatch, tmp_path):
    with pytest.raises(cli_run.PartialExecutionError):
        _run_interactive(monkeypatch, tmp_path, _PartialFakeGraph(), save_answer="N")
    assert _recommendation_files(tmp_path) == []


@pytest.mark.unit
def test_6_interactive_error_creates_no_record(monkeypatch, tmp_path):
    class _BoomGraph(_RichFakeGraph):
        def stream(self, graph_input, **kwargs):
            raise RuntimeError("immediate failure")
            yield {}  # pragma: no cover - never reached, makes this a generator

    with pytest.raises(cli_run.PartialExecutionError):
        _run_interactive(monkeypatch, tmp_path, _BoomGraph(), save_answer="N")
    assert _recommendation_files(tmp_path) == []


# --- 7/8: headless/batch still create exactly one record each --------------


@pytest.mark.unit
def test_7_headless_success_creates_exactly_one_record(fake_graph_structured, tmp_path):
    result = run_single_analysis(
        ticker="NVDA", analysis_date="2026-09-30", analysts=["market"], depth_value=3,
        language="English", output_dir=tmp_path,
    )
    assert result["status"] == "ok"
    records = list(((tmp_path / "ledger") / "recommendations" / "NVDA").glob("*.json"))
    assert len(records) == 1
    # The ledger is global, NOT nested inside --output-dir.
    assert not (tmp_path / "recommendations").exists()


@pytest.mark.unit
def test_8_batch_creates_exactly_one_record_per_ticker(fake_graph_structured, tmp_path):
    from cli.headless import run_batch_analysis

    results, _, _ = run_batch_analysis(
        tickers=["NVDA", "AAPL"], analysis_date="2026-09-30", analysts=["market"],
        depth_name="shallow", depth_value=1, language="English", output_dir=tmp_path,
        progress=lambda *a, **k: None,
    )
    assert all(r["status"] == "ok" for r in results)
    ledger_root = (tmp_path / "ledger")
    for ticker in ("NVDA", "AAPL"):
        records = list((ledger_root / "recommendations" / ticker).glob("*.json"))
        assert len(records) == 1
        assert not (tmp_path / ticker / "recommendations").exists()


# --- 9: no double-write after the refactor ----------------------------------


@pytest.mark.unit
def test_9_persist_successful_run_called_exactly_once(monkeypatch, fake_graph_structured, tmp_path):
    calls = []
    original = cli_headless.persist_successful_run

    def _spy(**kwargs):
        calls.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(cli_headless, "persist_successful_run", _spy)
    run_single_analysis(
        ticker="NVDA", analysis_date="2026-09-30", analysts=["market"], depth_value=3,
        language="English", output_dir=tmp_path,
    )
    assert len(calls) == 1


# --- 10/11/12: backtest / benchmark-replay / direct propagate() never ------
# --- import the ledger, so they cannot have gained this side effect --------


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {
        alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names
    }
    names |= {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module
    }
    return names


@pytest.mark.unit
def test_10_backtest_does_not_import_the_ledger():
    import tradingagents.backtest as backtest

    modules = _imported_modules(Path(backtest.__file__))
    assert not any("recommendation_record" in m or "result_document" in m for m in modules)


@pytest.mark.unit
def test_11_benchmark_replay_does_not_import_the_ledger():
    import tradingagents.benchmark_replay as benchmark_replay

    modules = _imported_modules(Path(benchmark_replay.__file__))
    assert not any("recommendation_record" in m or "result_document" in m for m in modules)


@pytest.mark.unit
def test_12_trading_graph_propagate_does_not_import_the_ledger():
    import tradingagents.graph.trading_graph as trading_graph

    modules = _imported_modules(Path(trading_graph.__file__))
    assert not any("recommendation_record" in m or "result_document" in m for m in modules)


# --- 13/14: canonical result.json / complete_report.md exist (interactive) -


@pytest.mark.unit
def test_13_canonical_result_json_exists_in_interactive(monkeypatch, tmp_path):
    _run_interactive(monkeypatch, tmp_path, _RichFakeGraph(), save_answer="N")
    records = _recommendation_files(tmp_path)
    payload = json.loads(records[0].read_text(encoding="utf-8"))
    assert Path(payload["artifacts"]["result_path"]).exists()


@pytest.mark.unit
def test_14_canonical_complete_report_exists_in_interactive(monkeypatch, tmp_path):
    _run_interactive(monkeypatch, tmp_path, _RichFakeGraph(), save_answer="N")
    records = _recommendation_files(tmp_path)
    payload = json.loads(records[0].read_text(encoding="utf-8"))
    assert Path(payload["artifacts"]["report_path"]).exists()


# --- 15: record hashes point at the canonical artifacts ---------------------


@pytest.mark.unit
def test_15_record_hashes_point_at_canonical_artifacts(monkeypatch, tmp_path):
    _run_interactive(monkeypatch, tmp_path, _RichFakeGraph(), save_answer="N")
    records = _recommendation_files(tmp_path)
    payload = json.loads(records[0].read_text(encoding="utf-8"))
    assert payload["artifacts"]["result_sha256"] == hashlib.sha256(
        Path(payload["artifacts"]["result_path"]).read_bytes()
    ).hexdigest()
    assert payload["artifacts"]["report_sha256"] == hashlib.sha256(
        Path(payload["artifacts"]["report_path"]).read_bytes()
    ).hexdigest()


# --- 16: two runs of the same ticker/date coexist without overwriting ------


@pytest.mark.unit
def test_16_two_interactive_runs_same_ticker_date_do_not_overwrite(monkeypatch, tmp_path):
    _run_interactive(monkeypatch, tmp_path, _RichFakeGraph(), save_answer="N")
    _run_interactive(monkeypatch, tmp_path, _RichFakeGraph(), save_answer="N")
    records = _recommendation_files(tmp_path)
    assert len(records) == 2
    payloads = [json.loads(r.read_text(encoding="utf-8")) for r in records]
    canonical_paths = {p["artifacts"]["result_path"] for p in payloads}
    assert len(canonical_paths) == 2  # distinct runs/<timestamp> directories
    for record_path in records:
        assert verify_recommendation_record(record_path)


# --- 17: Save report = NO does not delete canonical artifacts --------------


@pytest.mark.unit
def test_17_save_report_no_does_not_delete_canonical_artifacts(monkeypatch, tmp_path):
    _run_interactive(monkeypatch, tmp_path, _RichFakeGraph(), save_answer="N")
    records = _recommendation_files(tmp_path)
    payload = json.loads(records[0].read_text(encoding="utf-8"))
    assert Path(payload["artifacts"]["result_path"]).exists()
    assert Path(payload["artifacts"]["report_path"]).exists()


# --- 18: decision log behavior is unchanged ---------------------------------


@pytest.mark.unit
def test_18_decision_log_still_records_the_decision(monkeypatch, tmp_path):
    fake = _RichFakeGraph()
    _run_interactive(monkeypatch, tmp_path, fake, save_answer="N")
    assert ("record_decision", "NVDA", "2026-01-10", "**Rating**: Hold") in fake.calls
    assert ("clear_checkpoint",) in fake.calls


# --- 19: interactive/headless build result.json the same way ---------------


@pytest.mark.unit
def test_19_interactive_and_headless_share_the_same_persistence_function():
    assert cli_run.persist_successful_run is cli_headless.persist_successful_run
    assert cli_run.persist_successful_run is cli_result_document.persist_successful_run


# --- 20: no new circular imports --------------------------------------------


@pytest.mark.unit
def test_20_result_document_module_imports_neither_run_nor_headless():
    modules = _imported_modules(Path(cli_result_document.__file__))
    assert not any(m in ("cli.run", "cli.headless") for m in modules)
    # And the modules themselves import cleanly together either order.
    import importlib

    importlib.reload(cli_result_document)
    importlib.reload(cli_run)
    importlib.reload(cli_headless)
