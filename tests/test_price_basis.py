"""price_basis: a structured snapshot of the OHLCV basis a run's Market
Analyst actually saw for its own analysis_date (see fork README:
"RecommendationRecord" -- price_basis; schema 1.0 -> 1.1).

Section A tests the capture mechanism itself, at the one shared, cached
load point every yfinance-backed path funnels through
(``tradingagents.dataflows.vendors.yahoo.ohlcv.load_ohlcv``). Section B
tests how that snapshot propagates into a RecommendationRecord. Section C
is a robustness follow-up: contextvar lifecycle (clean start/end, no
cross-run or cross-ticker leakage, explicit on exception), fetched_at's
real-fetch-instant semantics, and the fingerprint's no-artificial-rounding
precision policy.

Numbered 1-15 per the task's own original "TESTS" list; Section C is the
follow-up robustness review's own list (not renumbered against Section A/B).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import cli.run as cli_run
from tests.test_llm_trace_integration import _FakeGraphCore, _FakeTradingAgentsGraph
from tests.test_recommendation_record import _build, _execution, _full_final_state
from tradingagents.dataflows import price_basis as price_basis_module
from tradingagents.dataflows.price_basis import (
    PRICE_MODE_YAHOO_AUTO_ADJUSTED,
    PriceBasisCollector,
    compute_dataset_fingerprint,
    get_price_basis_snapshot,
    set_active_price_basis_collector,
)
from tradingagents.dataflows.vendors.yahoo import ohlcv
from tradingagents.recommendation_record import (
    build_recommendation_record,
    record_recommendation,
    sha256_file,
    verify_recommendation_record,
    write_recommendation_record,
)


def _stamp(path, ts):
    """Set ``path``'s mtime to the wall-clock ``ts`` (local time, matching how
    ``_cache_is_fresh`` reads it back)."""
    t = ts.to_pydatetime().timestamp()
    os.utime(path, (t, t))


def _seed_cache(monkeypatch, tmp_path, frame, curr_date, symbol="AAPL"):
    monkeypatch.setattr(ohlcv, "get_config", lambda: {"data_cache_dir": str(tmp_path)})
    today = pd.Timestamp(curr_date)
    monkeypatch.setattr(ohlcv.pd.Timestamp, "today", staticmethod(lambda: today))
    cache_file = tmp_path / f"{symbol}-YFin-data.csv"
    cache_file.write_text(frame.to_csv(index=False))
    _stamp(cache_file, today)
    return cache_file


def _basic_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "Date": ["2026-05-06", "2026-05-07", "2026-05-08"],
        "Open": [100.0, 101.0, 102.0], "High": [101.0, 102.0, 103.0],
        "Low": [99.0, 100.0, 101.0], "Close": [100.5, 101.5, 102.5],
        "Volume": [1_000_000, 1_100_000, 1_200_000],
    })


# --- Section A: capture mechanism (1-7) -------------------------------------


@pytest.mark.unit
def test_1_snapshot_captures_provider_mode_symbol_date(monkeypatch, tmp_path):
    _seed_cache(monkeypatch, tmp_path, _basic_frame(), "2026-05-08")
    monkeypatch.setattr(ohlcv.yf, "download", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("should use the seeded cache, not download")))
    set_active_price_basis_collector(PriceBasisCollector())
    try:
        ohlcv.load_ohlcv("AAPL", "2026-05-08", fill_gaps=False)
        snapshot = get_price_basis_snapshot("AAPL", "2026-05-08")
    finally:
        set_active_price_basis_collector(None)
    assert snapshot is not None
    assert snapshot["provider"] == "yfinance"
    assert snapshot["price_mode"] == PRICE_MODE_YAHOO_AUTO_ADJUSTED
    assert snapshot["symbol"] == "AAPL"
    assert snapshot["requested_analysis_date"] == "2026-05-08"


@pytest.mark.unit
def test_2_reference_bar_preserves_exact_ohlcv(monkeypatch, tmp_path):
    _seed_cache(monkeypatch, tmp_path, _basic_frame(), "2026-05-08")
    set_active_price_basis_collector(PriceBasisCollector())
    try:
        ohlcv.load_ohlcv("AAPL", "2026-05-08", fill_gaps=False)
        snapshot = get_price_basis_snapshot("AAPL", "2026-05-08")
    finally:
        set_active_price_basis_collector(None)
    assert snapshot["reference_bar"] == {
        "date": "2026-05-08", "open": 102.0, "high": 103.0,
        "low": 101.0, "close": 102.5, "volume": 1_200_000.0,
    }


@pytest.mark.unit
def test_3_no_exact_bar_is_explicit_not_silent(monkeypatch, tmp_path):
    # 2026-05-09 is a Saturday with no bar; the nearest prior session
    # (2026-05-08) is used, but as an EXPLICIT mismatch, never silently
    # presented as if it were the requested date's own bar.
    _seed_cache(monkeypatch, tmp_path, _basic_frame(), "2026-05-09")
    set_active_price_basis_collector(PriceBasisCollector())
    try:
        ohlcv.load_ohlcv("AAPL", "2026-05-09", fill_gaps=False)
        snapshot = get_price_basis_snapshot("AAPL", "2026-05-09")
    finally:
        set_active_price_basis_collector(None)
    assert snapshot["requested_analysis_date"] == "2026-05-09"
    assert snapshot["reference_bar_date"] == "2026-05-08"
    assert snapshot["reference_bar_matches_requested_date"] is False


@pytest.mark.unit
def test_4_fingerprint_stable_with_identical_data():
    frame = _basic_frame()
    fp1 = compute_dataset_fingerprint(
        provider="yfinance", symbol="AAPL", price_mode=PRICE_MODE_YAHOO_AUTO_ADJUSTED, data=frame
    )
    fp2 = compute_dataset_fingerprint(
        provider="yfinance", symbol="AAPL", price_mode=PRICE_MODE_YAHOO_AUTO_ADJUSTED, data=frame.copy()
    )
    assert fp1 == fp2
    assert fp1 is not None


@pytest.mark.unit
def test_5_fingerprint_normalizes_row_order():
    frame = _basic_frame()
    shuffled = frame.iloc[::-1].reset_index(drop=True)
    fp_original = compute_dataset_fingerprint(
        provider="yfinance", symbol="AAPL", price_mode=PRICE_MODE_YAHOO_AUTO_ADJUSTED, data=frame
    )
    fp_shuffled = compute_dataset_fingerprint(
        provider="yfinance", symbol="AAPL", price_mode=PRICE_MODE_YAHOO_AUTO_ADJUSTED, data=shuffled
    )
    assert fp_original == fp_shuffled


@pytest.mark.unit
def test_6_fingerprint_normalizes_nan_regardless_of_spelling():
    import numpy as np

    frame_np_nan = pd.DataFrame({
        "Date": ["2026-05-08"], "Open": [100.0], "High": [101.0],
        "Low": [99.0], "Close": [100.5], "Volume": [np.nan],
    })
    frame_none = pd.DataFrame({
        "Date": ["2026-05-08"], "Open": [100.0], "High": [101.0],
        "Low": [99.0], "Close": [100.5], "Volume": [None],
    })
    fp_np_nan = compute_dataset_fingerprint(
        provider="yfinance", symbol="AAPL", price_mode=PRICE_MODE_YAHOO_AUTO_ADJUSTED, data=frame_np_nan
    )
    fp_none = compute_dataset_fingerprint(
        provider="yfinance", symbol="AAPL", price_mode=PRICE_MODE_YAHOO_AUTO_ADJUSTED, data=frame_none
    )
    assert fp_np_nan == fp_none


@pytest.mark.unit
def test_7_capturing_price_basis_triggers_no_second_fetch(monkeypatch, tmp_path):
    _seed_cache(monkeypatch, tmp_path, _basic_frame(), "2026-05-08")
    download_calls = []
    monkeypatch.setattr(
        ohlcv.yf, "download",
        lambda *a, **k: download_calls.append((a, k)) or (_ for _ in ()).throw(
            AssertionError("a fresh download must not happen; the cache is fresh")
        ),
    )
    set_active_price_basis_collector(PriceBasisCollector())
    try:
        # Same (symbol, curr_date) called twice in one run, as a real run's
        # get_indicators (fill_gaps=True) and get_verified_market_snapshot
        # (fill_gaps=False) both would -- both reuse the one cached frame.
        ohlcv.load_ohlcv("AAPL", "2026-05-08", fill_gaps=True)
        ohlcv.load_ohlcv("AAPL", "2026-05-08", fill_gaps=False)
    finally:
        set_active_price_basis_collector(None)
    assert download_calls == []


# --- Section B: RecommendationRecord integration (8-15) ---------------------


def _price_basis_payload(**overrides) -> dict:
    base = {
        "provider": "yfinance",
        "price_mode": PRICE_MODE_YAHOO_AUTO_ADJUSTED,
        "symbol": "NVDA",
        "requested_analysis_date": "2026-09-01",
        "reference_bar_date": "2026-09-01",
        "reference_bar_matches_requested_date": True,
        "reference_bar": {
            "date": "2026-09-01", "open": 100.0, "high": 105.0,
            "low": 99.0, "close": 104.0, "volume": 1_000_000.0,
        },
        "dataset_fingerprint": "f" * 64,
        "fetched_at": "2026-09-01T12:00:00+00:00",
    }
    base.update(overrides)
    return base


@pytest.mark.unit
def test_8_snapshot_reaches_result_document(tmp_path):
    from cli.result_document import result_document

    price_basis = _price_basis_payload()
    final_state = _full_final_state()
    final_state["price_basis"] = price_basis
    execution = _execution(final_state)
    result = result_document(execution, tmp_path)
    assert result["price_basis"] == price_basis


@pytest.mark.unit
def test_9_new_records_use_schema_version_1_1(tmp_path):
    execution, result, result_path, report_path = _build(tmp_path)
    record = build_recommendation_record(
        execution=execution, result=result, config={}, result_path="result.json",
        result_sha256=sha256_file(result_path), report_path="complete_report.md",
        report_sha256=sha256_file(report_path),
    )
    assert record.schema_version == "1.1"


@pytest.mark.unit
def test_10_a_1_0_shaped_record_without_price_basis_still_validates(tmp_path):
    from tradingagents.recommendation_record import RecommendationRecord

    execution, result, result_path, report_path = _build(tmp_path)
    record = build_recommendation_record(
        execution=execution, result=result, config={}, result_path="result.json",
        result_sha256=sha256_file(result_path), report_path="complete_report.md",
        report_sha256=sha256_file(report_path),
    )
    legacy_payload = record.model_dump(mode="json")
    del legacy_payload["price_basis"]
    legacy_payload["schema_version"] = "1.0"
    revalidated = RecommendationRecord.model_validate(legacy_payload)
    assert revalidated.price_basis is None
    assert revalidated.schema_version == "1.0"


@pytest.mark.unit
def test_11_1_1_record_hash_verifies(tmp_path):
    final_state = _full_final_state()
    final_state["price_basis"] = _price_basis_payload()
    execution = _execution(final_state)
    from cli.result_document import result_document

    result = result_document(execution, tmp_path)
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    report_path = tmp_path / "complete_report.md"
    report_path.write_text("# Report\n\nHold.\n", encoding="utf-8")

    written = record_recommendation(
        execution=execution, result=result, config={},
        result_json_path=result_path, report_path=report_path,
    )
    saved = json.loads(written.read_text(encoding="utf-8"))
    assert saved["price_basis"] == _price_basis_payload()
    assert verify_recommendation_record(written) is True


@pytest.mark.unit
def test_12_tampering_with_price_basis_breaks_verify(tmp_path):
    final_state = _full_final_state()
    final_state["price_basis"] = _price_basis_payload()
    execution = _execution(final_state)
    from cli.result_document import result_document

    result = result_document(execution, tmp_path)
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    report_path = tmp_path / "complete_report.md"
    report_path.write_text("# Report\n\nHold.\n", encoding="utf-8")

    written = record_recommendation(
        execution=execution, result=result, config={},
        result_json_path=result_path, report_path=report_path,
    )
    payload = json.loads(written.read_text(encoding="utf-8"))
    payload["price_basis"]["dataset_fingerprint"] = "0" * 64  # tamper
    written.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    assert verify_recommendation_record(written) is False


@pytest.mark.unit
def test_13_no_snapshot_yields_null_price_basis_without_failing(tmp_path):
    execution, result, result_path, report_path = _build(tmp_path)
    assert result.get("price_basis") is None  # legacy/non-structured run shape
    record = build_recommendation_record(
        execution=execution, result=result, config={}, result_path="result.json",
        result_sha256=sha256_file(result_path), report_path="complete_report.md",
        report_sha256=sha256_file(report_path),
    )
    assert record.price_basis is None
    written = write_recommendation_record(record, tmp_path)
    assert verify_recommendation_record(written) is True


@pytest.mark.unit
def test_14_recommendation_record_never_parses_market_report_markdown():
    """Only docstrings/comments may mention "market_report" (explaining why
    it is deliberately avoided) -- no actual string literal or dict/attr
    access in real code may reference it."""
    import ast

    import tradingagents.recommendation_record as rr

    source = Path(rr.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            # A module/function/class docstring is its own Expr statement;
            # any other string constant (a dict key, a .get() argument) is
            # real code and must not reference the report field.
            assert "market_report" not in node.value or _is_docstring(node, tree)


def _is_docstring(node: object, tree: object) -> bool:
    import ast

    for parent in ast.walk(tree):
        body = getattr(parent, "body", None)
        if body and isinstance(body[0], ast.Expr) and body[0].value is node:
            return True
    return False


@pytest.mark.unit
def test_15_take_profit_from_the_prior_commit_stays_intact(tmp_path):
    from tradingagents.agents.schemas import TraderAction, TraderProposal

    proposal = TraderProposal(
        action=TraderAction.BUY, reasoning="Bull case.", stop_loss=90.0, take_profit=125.0,
    )
    execution, result, result_path, report_path = _build(tmp_path, trader_proposal=proposal)
    record = build_recommendation_record(
        execution=execution, result=result, config={}, result_path="result.json",
        result_sha256=sha256_file(result_path), report_path="complete_report.md",
        report_sha256=sha256_file(report_path),
    )
    assert record.trade_plan.take_profit == 125.0


# --- Section C: contextvar lifecycle, fetched_at semantics, fingerprint
# precision (robustness follow-up review) ------------------------------------


@pytest.mark.unit
def test_lifecycle_run_starts_and_ends_clean():
    assert price_basis_module._active_collector.get() is None
    set_active_price_basis_collector(PriceBasisCollector())
    assert price_basis_module._active_collector.get() is not None
    set_active_price_basis_collector(None)
    assert price_basis_module._active_collector.get() is None


@pytest.mark.unit
def test_lifecycle_two_runs_do_not_leak_price_basis(monkeypatch, tmp_path):
    _seed_cache(monkeypatch, tmp_path, _basic_frame(), "2026-05-08")
    set_active_price_basis_collector(PriceBasisCollector())
    ohlcv.load_ohlcv("AAPL", "2026-05-08", fill_gaps=False)
    assert get_price_basis_snapshot("AAPL", "2026-05-08") is not None
    set_active_price_basis_collector(None)  # run A ends

    # Run B starts with its OWN fresh collector -- run A's entry must not
    # be visible even though it queries the exact same (symbol, date).
    set_active_price_basis_collector(PriceBasisCollector())
    assert get_price_basis_snapshot("AAPL", "2026-05-08") is None
    set_active_price_basis_collector(None)


class _BoomOnInitGraph:
    """Raises during graph construction -- BEFORE the stream try/except in
    execute_analysis is ever reached, the exact gap this review closes."""

    def __init__(self, *a, **k):
        raise RuntimeError("boom during graph construction")


@pytest.mark.unit
def test_lifecycle_resets_on_exception_before_the_stream_even_starts(monkeypatch, tmp_path):
    monkeypatch.setattr(cli_run, "TradingAgentsGraph", _BoomOnInitGraph)
    monkeypatch.setattr(
        cli_run, "build_analyst_execution_plan", lambda analysts: SimpleNamespace(specs=[])
    )
    with pytest.raises(RuntimeError, match="boom during graph construction"):
        cli_run.execute_analysis(
            ticker="NVDA", analysis_date="2026-09-30", selected_analysts=["market"],
            asset_type="stock",
            config={"max_risk_discuss_rounds": 3, "deep_think_llm": "A", "quick_think_llm": "B",
                    "llm_provider": "ollama", "output_language": "English"},
            output_dir=tmp_path,
        )
    assert price_basis_module._active_collector.get() is None


class _LoadsOhlcvGraphCore(_FakeGraphCore):
    """Like the shared fake graph, but its stream also performs a real
    (cache-seeded, no-network) load_ohlcv(fill_gaps=False) call, standing in
    for the Market Analyst's own get_verified_market_snapshot tool call."""

    def __init__(self, symbol: str, curr_date: str):
        self._symbol = symbol
        self._curr_date = curr_date

    def stream(self, state, **kwargs):
        ohlcv.load_ohlcv(self._symbol, self._curr_date, fill_gaps=False)
        yield from super().stream(state, **kwargs)


@pytest.mark.unit
def test_lifecycle_two_sequential_tickers_in_one_process_do_not_cross_contaminate(
    monkeypatch, tmp_path
):
    """Simulates a batch run: ticker A's Market Analyst call captures a real
    price_basis entry; ticker B (processed next, same thread/process) must
    get its own clean collector, never A's leftover entry -- even though
    this test deliberately reuses A's own (symbol, date) pair for B's
    lookup, to prove isolation is by RUN, not by (symbol, date) alone."""
    _seed_cache(monkeypatch, tmp_path, _basic_frame(), "2026-05-08", symbol="AAPL")
    monkeypatch.setattr(
        cli_run, "build_analyst_execution_plan", lambda analysts: SimpleNamespace(specs=[])
    )

    class _GraphA(_FakeTradingAgentsGraph):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.graph = _LoadsOhlcvGraphCore("AAPL", "2026-05-08")

    class _GraphB(_FakeTradingAgentsGraph):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.graph = _FakeGraphCore()  # no load_ohlcv call at all

    base_config = {
        "max_risk_discuss_rounds": 3, "deep_think_llm": "A", "quick_think_llm": "B",
        "llm_provider": "ollama", "output_language": "English",
    }

    monkeypatch.setattr(cli_run, "TradingAgentsGraph", _GraphA)
    execution_a = cli_run.execute_analysis(
        ticker="AAPL", analysis_date="2026-05-08", selected_analysts=["market"],
        asset_type="stock", config=base_config, output_dir=tmp_path / "a",
    )
    assert execution_a.final_state["price_basis"] is not None
    assert execution_a.final_state["price_basis"]["symbol"] == "AAPL"

    monkeypatch.setattr(cli_run, "TradingAgentsGraph", _GraphB)
    execution_b = cli_run.execute_analysis(
        ticker="AAPL", analysis_date="2026-05-08", selected_analysts=["market"],
        asset_type="stock", config=base_config, output_dir=tmp_path / "b",
    )
    assert execution_b.final_state["price_basis"] is None


@pytest.mark.unit
def test_run_without_market_analyst_produces_null_price_basis(monkeypatch, tmp_path):
    monkeypatch.setattr(cli_run, "TradingAgentsGraph", _FakeTradingAgentsGraph)
    monkeypatch.setattr(
        cli_run, "build_analyst_execution_plan", lambda analysts: SimpleNamespace(specs=[])
    )
    execution = cli_run.execute_analysis(
        ticker="NVDA", analysis_date="2026-09-30", selected_analysts=["social"],
        asset_type="stock",
        config={"max_risk_discuss_rounds": 3, "deep_think_llm": "A", "quick_think_llm": "B",
                "llm_provider": "ollama", "output_language": "English"},
        output_dir=tmp_path,
    )
    assert execution.final_state["price_basis"] is None


@pytest.mark.unit
def test_symbol_date_mismatch_is_never_silently_substituted(monkeypatch, tmp_path):
    _seed_cache(monkeypatch, tmp_path, _basic_frame(), "2026-05-08", symbol="AAPL")
    set_active_price_basis_collector(PriceBasisCollector())
    try:
        ohlcv.load_ohlcv("AAPL", "2026-05-08", fill_gaps=False)
        assert get_price_basis_snapshot("AAPL", "2026-05-08") is not None
        assert get_price_basis_snapshot("MSFT", "2026-05-08") is None  # wrong symbol
        assert get_price_basis_snapshot("AAPL", "2026-05-07") is None  # wrong date
    finally:
        set_active_price_basis_collector(None)


@pytest.mark.unit
def test_build_recommendation_record_drops_a_mismatched_price_basis(tmp_path):
    """Defense in depth: even if a caller handed build_recommendation_record a
    result["price_basis"] for a different symbol/date than THIS record's own
    run (never happens via cli.run's own wiring, which looks the snapshot up
    by this exact pair), it must be dropped to null, never kept."""
    final_state = _full_final_state()
    final_state["price_basis"] = _price_basis_payload(
        symbol="AAPL", requested_analysis_date="2099-01-01",  # mismatches NVDA/2026-09-01
    )
    execution = _execution(final_state)  # ticker=NVDA, analysis_date=2026-09-01
    from cli.result_document import result_document

    result = result_document(execution, tmp_path)
    result_path, report_path = _write_artifacts_for(tmp_path, result)
    record = build_recommendation_record(
        execution=execution, result=result, config={}, result_path="result.json",
        result_sha256=sha256_file(result_path), report_path="complete_report.md",
        report_sha256=sha256_file(report_path),
    )
    assert record.price_basis is None


@pytest.mark.unit
def test_fill_gaps_true_never_captures_even_with_an_active_collector(monkeypatch, tmp_path):
    _seed_cache(monkeypatch, tmp_path, _basic_frame(), "2026-05-08")
    collector = PriceBasisCollector()
    set_active_price_basis_collector(collector)
    try:
        ohlcv.load_ohlcv("AAPL", "2026-05-08", fill_gaps=True)
        assert collector.entry_for("AAPL", "2026-05-08") is None
    finally:
        set_active_price_basis_collector(None)


@pytest.mark.unit
def test_cache_hit_fetched_at_is_the_files_real_mtime_not_now(monkeypatch, tmp_path):
    _seed_cache(monkeypatch, tmp_path, _basic_frame(), "2026-05-08")
    cache_file = tmp_path / "AAPL-YFin-data.csv"
    # A real vendor-fetch instant earlier the same day -- if fetched_at were
    # ever datetime.now() standing in for an unknown real fetch time, it
    # would not match this file's own, earlier mtime. Compared in raw epoch
    # seconds (what os.path.getmtime and pd.Timestamp.timestamp() both use),
    # never in wall-clock strings, so the comparison is timezone-invariant.
    _stamp(cache_file, pd.Timestamp("2026-05-08 09:05:00"))
    expected_epoch = os.path.getmtime(cache_file)
    monkeypatch.setattr(ohlcv.yf, "download", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("should use the seeded cache, not download")))
    set_active_price_basis_collector(PriceBasisCollector())
    try:
        ohlcv.load_ohlcv("AAPL", "2026-05-08", fill_gaps=False)
        snapshot = get_price_basis_snapshot("AAPL", "2026-05-08")
    finally:
        set_active_price_basis_collector(None)
    assert pd.Timestamp(snapshot["fetched_at"]).timestamp() == pytest.approx(expected_epoch)


@pytest.mark.unit
def test_fresh_download_fetched_at_matches_the_cache_files_write_instant(monkeypatch, tmp_path):
    monkeypatch.setattr(ohlcv, "get_config", lambda: {"data_cache_dir": str(tmp_path)})
    today = pd.Timestamp("2026-05-08 12:00:00")
    monkeypatch.setattr(ohlcv.pd.Timestamp, "today", staticmethod(lambda: today))
    frame = _basic_frame()
    monkeypatch.setattr(ohlcv.yf, "download", lambda *a, **k: frame.set_index("Date"))
    set_active_price_basis_collector(PriceBasisCollector())
    try:
        ohlcv.load_ohlcv("AAPL", "2026-05-08", fill_gaps=False)
        snapshot = get_price_basis_snapshot("AAPL", "2026-05-08")
    finally:
        set_active_price_basis_collector(None)
    cache_file = tmp_path / "AAPL-YFin-data.csv"
    actual_mtime = pd.Timestamp.fromtimestamp(os.path.getmtime(cache_file), tz="UTC")
    assert pd.Timestamp(snapshot["fetched_at"]) == actual_mtime


@pytest.mark.unit
def test_fingerprint_precision_only_bit_identical_values_are_equivalent():
    """The fingerprint's precision policy, made explicit: two occurrences of
    the EXACT same float are equivalent (same fingerprint, regardless of row
    order/NaN spelling -- tests 4/5/6); a genuinely different value, however
    small the difference, is NOT -- no 6-decimal rounding hides it, since
    that is exactly the kind of change (a Yahoo retroactive re-adjustment)
    this fingerprint exists to catch."""
    base = _basic_frame()
    tiny_change = base.copy()
    # Previously would have rounded to the same 6-decimal value (102.123456
    # vs 102.123457 differ only in the 7th decimal) -- must NOT be equivalent.
    tiny_change.loc[tiny_change.index[-1], "Close"] = 102.1234565
    base.loc[base.index[-1], "Close"] = 102.1234564

    fp_base = compute_dataset_fingerprint(
        provider="yfinance", symbol="AAPL", price_mode=PRICE_MODE_YAHOO_AUTO_ADJUSTED, data=base
    )
    fp_tiny_change = compute_dataset_fingerprint(
        provider="yfinance", symbol="AAPL", price_mode=PRICE_MODE_YAHOO_AUTO_ADJUSTED, data=tiny_change
    )
    assert fp_base != fp_tiny_change

    identical_repeat = compute_dataset_fingerprint(
        provider="yfinance", symbol="AAPL", price_mode=PRICE_MODE_YAHOO_AUTO_ADJUSTED, data=base.copy()
    )
    assert fp_base == identical_repeat


def _write_artifacts_for(tmp_path: Path, result: dict) -> tuple[Path, Path]:
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    report_path = tmp_path / "complete_report.md"
    report_path.write_text("# Report\n\nHold.\n", encoding="utf-8")
    return result_path, report_path
