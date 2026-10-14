"""`tradingagents run`: --ticker is the only required flag. Every other
option falls back to an explicit flag, then an environment variable (where
one applies), then a hardcoded/config-derived default -- the same
precedence `batch` already established (see fork README: "Non-interactive
and batch CLI execution" / "RecommendationRecord").

Numbered per the task's own "TESTS" list (1-20); a few share one test.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
from typer.testing import CliRunner

import cli.main as m
import cli.run as cli_run
from cli.display import ANALYST_ORDER
from tests.test_llm_trace_integration import _FakeTradingAgentsGraph
from tests.test_recommendation_record import _full_final_state

RUN_RESULT = {
    "status": "ok",
    "ticker": "LLY",
    "run_duration_seconds": 1.0,
    "fallback_warnings": 0,
    "stats": {"llm_calls": 1, "tool_calls": 0, "tokens_in": 1, "tokens_out": 1},
    "research_execution": {
        "verification_status": "PASS", "verifier_passes": 1, "repair_rounds": 0,
        "bull_initial_calls": 1, "bear_initial_calls": 1,
        "bull_review_calls": 0, "bear_review_calls": 0,
    },
}


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def mocked_run(monkeypatch):
    calls = []
    monkeypatch.setattr(m, "run_single_analysis", lambda **kw: calls.append(kw) or RUN_RESULT)
    return calls


@pytest.fixture
def structured_graph(monkeypatch):
    """A real end-to-end run (through execute_analysis) whose one streamed
    chunk carries a full, structurally-valid final decision -- needed for
    the provenance/ledger tests, which read an actual result.json and
    RecommendationRecord off disk rather than a mocked run_single_analysis."""

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


# --- 1: only --ticker parses and runs ---------------------------------------


@pytest.mark.unit
def test_1_only_ticker_parses_correctly(runner, mocked_run):
    result = runner.invoke(m.app, ["run", "--ticker", "LLY"])
    assert result.exit_code == 0, result.output
    assert mocked_run[0]["ticker"] == "LLY"


# --- 2/3: date omitted -> today, explicit date wins -------------------------


@pytest.mark.unit
def test_2_date_omitted_defaults_to_today(runner, mocked_run):
    result = runner.invoke(m.app, ["run", "--ticker", "LLY"])
    assert result.exit_code == 0, result.output
    assert mocked_run[0]["analysis_date"] == datetime.now().strftime("%Y-%m-%d")


@pytest.mark.unit
def test_3_explicit_date_wins(runner, mocked_run):
    result = runner.invoke(m.app, ["run", "--ticker", "LLY", "--date", "2026-09-30"])
    assert result.exit_code == 0, result.output
    assert mocked_run[0]["analysis_date"] == "2026-09-30"


# --- 4/5: analysts omitted -> all, explicit analysts win --------------------


@pytest.mark.unit
def test_4_analysts_omitted_defaults_to_all(runner, mocked_run):
    result = runner.invoke(m.app, ["run", "--ticker", "LLY"])
    assert result.exit_code == 0, result.output
    assert mocked_run[0]["analysts"] == list(ANALYST_ORDER)


@pytest.mark.unit
def test_5_explicit_analysts_win(runner, mocked_run):
    result = runner.invoke(m.app, ["run", "--ticker", "LLY", "--analysts", "market"])
    assert result.exit_code == 0, result.output
    assert mocked_run[0]["analysts"] == ["market"]


# --- 6/7: depth omitted -> medium, explicit depth wins ----------------------


@pytest.mark.unit
def test_6_depth_omitted_defaults_to_medium(runner, mocked_run):
    result = runner.invoke(m.app, ["run", "--ticker", "LLY"])
    assert result.exit_code == 0, result.output
    assert mocked_run[0]["depth_value"] == 3


@pytest.mark.unit
def test_7_explicit_depth_wins(runner, mocked_run):
    result = runner.invoke(m.app, ["run", "--ticker", "LLY", "--depth", "shallow"])
    assert result.exit_code == 0, result.output
    assert mocked_run[0]["depth_value"] == 1


# --- 8/9/10: language omitted -> config, config falls back to English, ----
# --- explicit language wins -------------------------------------------------


@pytest.mark.unit
def test_8_language_omitted_uses_configured_output_language(runner, monkeypatch, mocked_run):
    monkeypatch.delenv("TRADINGAGENTS_OUTPUT_LANGUAGE", raising=False)
    patched = dict(m.DEFAULT_CONFIG, output_language="French")
    with mock.patch.object(m, "DEFAULT_CONFIG", patched):
        result = runner.invoke(m.app, ["run", "--ticker", "LLY"])
    assert result.exit_code == 0, result.output
    assert mocked_run[0]["language"] == "French"


@pytest.mark.unit
def test_9_language_omitted_without_config_falls_back_to_english(runner, monkeypatch, mocked_run):
    monkeypatch.delenv("TRADINGAGENTS_OUTPUT_LANGUAGE", raising=False)
    patched = dict(m.DEFAULT_CONFIG, output_language="English")
    with mock.patch.object(m, "DEFAULT_CONFIG", patched):
        result = runner.invoke(m.app, ["run", "--ticker", "LLY"])
    assert result.exit_code == 0, result.output
    assert mocked_run[0]["language"] == "English"


@pytest.mark.unit
def test_10_explicit_language_wins_over_config(runner, monkeypatch, mocked_run):
    patched = dict(m.DEFAULT_CONFIG, output_language="French")
    with mock.patch.object(m, "DEFAULT_CONFIG", patched):
        result = runner.invoke(m.app, ["run", "--ticker", "LLY", "--language", "Spanish"])
    assert result.exit_code == 0, result.output
    assert mocked_run[0]["language"] == "Spanish"


# --- 11/12/13: output-dir omitted -> automatic unique path, two runs -------
# --- differ, explicit output-dir is respected literally --------------------


@pytest.mark.unit
def test_11_output_dir_omitted_is_automatic(runner, mocked_run):
    result = runner.invoke(m.app, ["run", "--ticker", "LLY", "--date", "2026-09-30"])
    assert result.exit_code == 0, result.output
    output_dir = mocked_run[0]["output_dir"]
    assert output_dir is not None
    parts = output_dir.parts
    assert "LLY" in parts and "2026-09-30" in parts and "runs" in parts


@pytest.mark.unit
def test_12_two_runs_same_ticker_date_get_different_paths(runner, mocked_run):
    result_a = runner.invoke(m.app, ["run", "--ticker", "LLY", "--date", "2026-09-30"])
    result_b = runner.invoke(m.app, ["run", "--ticker", "LLY", "--date", "2026-09-30"])
    assert result_a.exit_code == 0 and result_b.exit_code == 0
    assert mocked_run[0]["output_dir"] != mocked_run[1]["output_dir"]


@pytest.mark.unit
def test_13_explicit_output_dir_is_respected_literally(runner, mocked_run, tmp_path):
    chosen = tmp_path / "foo"
    result = runner.invoke(m.app, ["run", "--ticker", "LLY", "--output-dir", str(chosen)])
    assert result.exit_code == 0, result.output
    assert mocked_run[0]["output_dir"] == chosen


# --- 14/15/16/17: real end-to-end -- ledger independent of output_dir, ----
# --- provenance carries the resolved analysts/depth/language --------------


@pytest.mark.unit
def test_14_15_16_17_ledger_and_provenance_reflect_resolved_defaults(
    runner, structured_graph, tmp_path
):
    result = runner.invoke(m.app, ["run", "--ticker", "NVDA", "--date", "2026-09-30"])
    assert result.exit_code == 0, result.output

    output_dir_line = next(
        line for line in result.output.splitlines() if "Output directory:" in line
    )
    output_dir = Path(output_dir_line.split("Output directory:", 1)[1].strip())
    saved = json.loads((output_dir / "result.json").read_text(encoding="utf-8"))
    provenance = saved["run_provenance"]

    # 15: resolved analysts (all four, since --analysts was omitted).
    assert provenance["execution"]["selected_analysts"] == list(ANALYST_ORDER)
    # 16: resolved depth (medium).
    assert provenance["execution"]["depth"] == 3
    # 17: resolved language (whatever this environment's configured default
    # is -- never null just because --language was omitted).
    assert provenance["execution"]["language"]

    # 14: the ledger record exists independently of the automatic output_dir,
    # under the isolated ledger_root (see tests/conftest.py).
    ledger_root = tmp_path / "ledger"
    records = list((ledger_root / "recommendations" / "NVDA").glob("*.json"))
    assert len(records) == 1
    record_payload = json.loads(records[0].read_text(encoding="utf-8"))
    assert Path(record_payload["artifacts"]["result_path"]) == (output_dir / "result.json").resolve()


# --- 18: --help marks only --ticker as required -----------------------------


@pytest.mark.unit
def test_18_help_marks_only_ticker_as_required(runner):
    result = runner.invoke(m.app, ["run", "--help"])
    assert result.exit_code == 0, result.output
    # Only rows inside the bordered options table start with "│" --
    # the docstring prose above it ("Only --ticker is required...") also
    # contains these flag names as plain text and must not be matched.
    option_lines = [line for line in result.output.splitlines() if line.startswith("│")]

    def _is_required(flag: str) -> bool:
        line = next(line for line in option_lines if flag in line)
        return "*" in line.split(flag)[0]

    assert _is_required("--ticker") is True
    for flag in ("--date", "--analysts", "--depth", "--language", "--output-dir"):
        assert _is_required(flag) is False, flag


# --- 19: batch is not broken -------------------------------------------------


@pytest.mark.unit
def test_19_batch_still_works_with_only_tickers(runner, monkeypatch, tmp_path):
    calls = []

    def fake_batch(**kw):
        calls.append(kw)
        return [RUN_RESULT], tmp_path / "summary.md", tmp_path / "summary.json"

    monkeypatch.setattr(m, "run_batch_analysis", fake_batch)
    result = runner.invoke(m.app, ["batch", "--tickers", "NVDA"])
    assert result.exit_code == 0, result.output
    assert calls[0]["tickers"] == ["NVDA"]
    assert calls[0]["depth_value"] == 3
    assert calls[0]["analysts"] == list(ANALYST_ORDER)


# --- 20: RecommendationRecord persistence tests still pass -- see the
# full-suite run (tests/test_recommendation_record*.py, tests/test_ledger_root.py).
