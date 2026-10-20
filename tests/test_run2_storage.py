"""run2's own, separate storage root and run-directory layout (fork README:
"run2": "No debe mezclar resultados con run"). Mirrors
``tests/test_automatic_run_dir.py``'s own collision-safety coverage for the
duplicated ``automatic_run2_dir`` algorithm.
"""

from __future__ import annotations

import datetime as _datetime_module

import pytest

import tradingagents.run2.storage as storage
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.recommendation_record import resolve_ledger_root


@pytest.mark.unit
def test_1_default_root_env_var_is_its_own_namespace(monkeypatch):
    monkeypatch.delenv("TRADINGAGENTS_RUN2_ROOT", raising=False)
    root = storage.default_run2_root()
    assert root.name == "run2"
    assert root.parent.name == ".tradingagents"


@pytest.mark.unit
def test_2_default_root_respects_its_own_env_var(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADINGAGENTS_RUN2_ROOT", str(tmp_path / "custom_run2"))
    assert storage.default_run2_root() == tmp_path / "custom_run2"


@pytest.mark.unit
def test_3_default_root_is_independent_of_results_dir_and_ledger_root():
    run2_root = storage.default_run2_root()
    assert run2_root != resolve_ledger_root(None)
    assert str(run2_root) != DEFAULT_CONFIG["results_dir"]


@pytest.mark.unit
def test_4_runs_base_layout_is_root_ticker_date_runs(tmp_path):
    base = storage.run2_runs_base(tmp_path, "NVDA", "2026-01-07")
    assert base == tmp_path / "NVDA" / "2026-01-07" / "runs"


@pytest.mark.unit
def test_5_automatic_run2_dir_creates_and_never_collides(tmp_path, monkeypatch):
    class _Frozen(_datetime_module.datetime):
        @classmethod
        def now(cls, tz=None):
            return _datetime_module.datetime(2026, 1, 7, 12, 0, 0, 1)

    monkeypatch.setattr(storage.datetime, "datetime", _Frozen)
    base = tmp_path / "runs"
    first = storage.automatic_run2_dir(base)
    second = storage.automatic_run2_dir(base)
    assert first != second
    assert first.exists() and second.exists()
    assert second.name == first.name + "_01"


@pytest.mark.unit
def test_6_write_result_is_atomic_and_round_trips(tmp_path):
    payload = {"schema_version": "1.0", "ticker": "NVDA"}
    path = storage.write_result(tmp_path, payload)
    assert path.name == "result.json"
    assert storage.load_result(path) == payload
    assert not path.with_suffix(".tmp").exists()


@pytest.mark.unit
def test_7_write_report_writes_markdown(tmp_path):
    path = storage.write_report(tmp_path, "# hello")
    assert path.name == "complete_report.md"
    assert path.read_text(encoding="utf-8") == "# hello"


@pytest.mark.unit
def test_8_write_debug_call_writes_prompt_and_response(tmp_path):
    call_dir = storage.write_debug_call(tmp_path, 1, "Market Regime Analyzer", "PROMPT TEXT", "RESPONSE TEXT")
    assert (call_dir / "prompt.txt").read_text(encoding="utf-8") == "PROMPT TEXT"
    assert (call_dir / "response.txt").read_text(encoding="utf-8") == "RESPONSE TEXT"
    assert call_dir.name == "001_Market_Regime_Analyzer"
