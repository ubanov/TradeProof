"""Orchestration: multi-ticker, NEWS/SOCIAL independence, one provider
failure never aborting the rest, append-only writes, explicit provider
recording, UTC timestamps, and the structural "no LLM anywhere" guarantee.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

import tradingagents.evidence_archive as pkg
from tradingagents.evidence_archive import orchestrator, providers
from tradingagents.evidence_archive.schema import EMPTY, ERROR, NEWS, SOCIAL, SUCCESS
from tradingagents.evidence_archive.storage import load_capture


@pytest.mark.unit
class TestParsing:
    def test_evidence_types_default_order_validate_and_dedupe(self):
        assert orchestrator.parse_evidence_types("news, social, NEWS") == [NEWS, SOCIAL]

    def test_evidence_types_rejects_unknown(self):
        with pytest.raises(ValueError, match="unknown evidence type"):
            orchestrator.parse_evidence_types("global")

    def test_evidence_types_requires_at_least_one(self):
        with pytest.raises(ValueError):
            orchestrator.parse_evidence_types("")

    def test_providers_rejects_unknown(self):
        with pytest.raises(ValueError, match="unknown social provider"):
            orchestrator.parse_providers("twitter", SOCIAL)

    def test_providers_accepts_valid_list(self):
        assert orchestrator.parse_providers("reddit,stocktwits", SOCIAL) == ["reddit", "stocktwits"]

    def test_defaults_are_the_conservative_single_provider_per_type(self):
        # Alpha Vantage's small free-tier quota and Reddit's per-IP rate
        # limit make them unsuitable as defaults for a large ticker universe
        # -- both require explicit opt-in via --news-providers/--social-providers.
        assert orchestrator.DEFAULT_PROVIDERS_BY_TYPE[NEWS] == ("yfinance",)
        assert orchestrator.DEFAULT_PROVIDERS_BY_TYPE[SOCIAL] == ("stocktwits",)


@pytest.mark.unit
class TestRunArchiveEvidence:
    def test_multi_ticker_and_news_social_independence(self, tmp_path, monkeypatch):
        monkeypatch.setattr(providers, "_fetch_raw_news_items", lambda c, n: [])
        monkeypatch.setattr(providers, "_fetch_raw_messages", lambda t, timeout=10.0: [])
        outcomes = orchestrator.run_archive_evidence(["NVDA", "MSFT"], root=tmp_path)
        kinds = {(o.ticker, o.evidence_type, o.provider) for o in outcomes}
        assert kinds == {
            ("NVDA", NEWS, "yfinance"), ("NVDA", SOCIAL, "stocktwits"),
            ("MSFT", NEWS, "yfinance"), ("MSFT", SOCIAL, "stocktwits"),
        }

    def test_one_provider_failure_never_aborts_the_rest(self, tmp_path, monkeypatch):
        monkeypatch.setattr(providers, "_fetch_raw_news_items", lambda c, n: (_ for _ in ()).throw(RuntimeError("boom")))
        monkeypatch.setattr(providers, "_fetch_raw_messages", lambda t, timeout=10.0: [])
        outcomes = orchestrator.run_archive_evidence(["NVDA", "MSFT"], root=tmp_path)
        status_by = {(o.ticker, o.evidence_type): o.status for o in outcomes}
        assert status_by[("NVDA", NEWS)] == ERROR
        assert status_by[("MSFT", NEWS)] == ERROR
        assert status_by[("NVDA", SOCIAL)] == EMPTY
        assert status_by[("MSFT", SOCIAL)] == EMPTY
        assert len(outcomes) == 4  # the news failures never skipped the social captures

    def test_missing_alpha_vantage_key_does_not_abort_other_providers(self, tmp_path, monkeypatch):
        from tradingagents.dataflows.vendors.alpha_vantage.common import (
            AlphaVantageNotConfiguredError,
        )

        def boom(fn, params):
            raise AlphaVantageNotConfiguredError("no key")

        monkeypatch.setattr(providers, "_make_api_request", boom)
        monkeypatch.setattr(providers, "_fetch_raw_news_items", lambda c, n: [])
        monkeypatch.setattr(providers, "_fetch_raw_messages", lambda t, timeout=10.0: [])

        outcomes = orchestrator.run_archive_evidence(
            ["NVDA"], evidence_types=[NEWS, SOCIAL],
            news_providers=["yfinance", "alpha_vantage"], social_providers=["stocktwits"],
            root=tmp_path,
        )
        status_by_provider = {o.provider: o.status for o in outcomes}
        assert status_by_provider["alpha_vantage"] == ERROR
        assert status_by_provider["yfinance"] == EMPTY
        assert status_by_provider["stocktwits"] == EMPTY

    def test_append_only_two_runs_produce_two_distinct_captures(self, tmp_path, monkeypatch):
        monkeypatch.setattr(providers, "_fetch_raw_news_items", lambda c, n: [])
        monkeypatch.setattr(providers, "_fetch_raw_messages", lambda t, timeout=10.0: [])
        first = orchestrator.run_archive_evidence(["NVDA"], root=tmp_path)
        second = orchestrator.run_archive_evidence(["NVDA"], root=tmp_path)
        first_paths = {o.path for o in first}
        second_paths = {o.path for o in second}
        assert first_paths.isdisjoint(second_paths)
        for path in first_paths | second_paths:
            assert path.exists()

    def test_on_capture_callback_streams_every_outcome(self, tmp_path, monkeypatch):
        monkeypatch.setattr(providers, "_fetch_raw_news_items", lambda c, n: [])
        monkeypatch.setattr(providers, "_fetch_raw_messages", lambda t, timeout=10.0: [])
        seen = []
        orchestrator.run_archive_evidence(["NVDA"], root=tmp_path, on_capture=seen.append)
        assert len(seen) == 2  # one NEWS/yfinance + one SOCIAL/stocktwits outcome

    def test_capture_file_records_exactly_the_provider_requested(self, tmp_path, monkeypatch):
        monkeypatch.setattr(providers, "_fetch_raw_news_items", lambda c, n: [])
        outcomes = orchestrator.run_archive_evidence(
            ["NVDA"], evidence_types=[NEWS], news_providers=["yfinance"], root=tmp_path,
        )
        capture = load_capture(outcomes[0].path)
        assert capture["provider"] == "yfinance"
        assert capture["ticker"] == "NVDA"
        assert capture["evidence_type"] == NEWS

    def test_timestamps_are_utc_iso8601(self, tmp_path, monkeypatch):
        monkeypatch.setattr(providers, "_fetch_raw_news_items", lambda c, n: [])
        outcomes = orchestrator.run_archive_evidence(["NVDA"], evidence_types=[NEWS], root=tmp_path)
        capture = load_capture(outcomes[0].path)
        assert capture["requested_at"].endswith("Z")
        assert capture["retrieved_at"].endswith("Z")

    def test_a_provider_function_raising_an_unexpected_exception_still_writes_an_error_capture(
        self, tmp_path, monkeypatch,
    ):
        def explode(ticker, config):
            raise ValueError("unexpected shape")

        monkeypatch.setitem(orchestrator.PROVIDERS_BY_TYPE[NEWS], "yfinance", explode)
        outcomes = orchestrator.run_archive_evidence(["NVDA"], evidence_types=[NEWS], root=tmp_path)
        assert outcomes[0].status == ERROR
        capture = load_capture(outcomes[0].path)
        assert capture["status"] == ERROR
        assert capture["error"]["error_type"] == "ValueError"


@pytest.mark.unit
def test_malformed_item_shape_becomes_error_never_empty_end_to_end(tmp_path, monkeypatch):
    """A raw item yfinance returns in an unrecognized shape must end up as
    an ERROR capture, never classified as EMPTY -- EMPTY must mean only
    "fetch valido y entendible, 0 items," never "something we could not
    parse." ``capture_news_yfinance`` itself raises on the bad shape (no
    per-item try/except); this confirms the orchestrator's own catch-all
    around the whole provider call is what turns that into an explicit,
    saved ERROR capture rather than an uncaught crash or a silent EMPTY."""
    monkeypatch.setattr(providers, "_fetch_raw_news_items", lambda c, n: ["not-a-dict-item"])
    outcomes = orchestrator.run_archive_evidence(["NVDA"], evidence_types=[NEWS], root=tmp_path)
    assert outcomes[0].status == ERROR
    assert outcomes[0].status != EMPTY
    capture = load_capture(outcomes[0].path)
    assert capture["status"] == ERROR
    assert capture["error"]["error_type"] == "AttributeError"


@pytest.mark.unit
def test_summarize_counts_every_status():
    from tradingagents.evidence_archive.orchestrator import CaptureOutcome, summarize
    from tradingagents.evidence_archive.schema import UNSUPPORTED

    outcomes = [
        CaptureOutcome("NVDA", NEWS, "yfinance", SUCCESS, 3, None),
        CaptureOutcome("NVDA", SOCIAL, "stocktwits", EMPTY, 0, None),
        CaptureOutcome("MSFT", NEWS, "yfinance", ERROR, 0, None, "boom"),
        CaptureOutcome("MSFT", SOCIAL, "stocktwits", UNSUPPORTED, 0, None),
    ]
    summary = summarize(outcomes)
    assert summary == {
        "tickers": 2, "captures": 4, "success": 1, "empty": 1, "unsupported": 1, "error": 1,
    }


@pytest.mark.unit
def test_no_llm_imports_anywhere_in_the_evidence_archive_package():
    """Structural guarantee: this package never imports an LLM/agent-graph
    module -- the task is capture-only, never analysis."""
    package_dir = pathlib.Path(pkg.__file__).parent
    forbidden_prefixes = ("langchain", "tradingagents.agents", "tradingagents.graph")
    for py_file in package_dir.glob("*.py"):
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                assert not name.startswith(forbidden_prefixes), f"{py_file.name} imports {name!r}"
