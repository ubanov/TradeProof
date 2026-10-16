"""``tradingagents archive-evidence``: CLI argument wiring only (the capture
logic itself is tested in test_evidence_archive_orchestrator.py). Patches
``cli.main.run_archive_evidence`` directly, so these never touch a real
provider or the network.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

import cli.main as m
from tradingagents.evidence_archive import NEWS, SOCIAL
from tradingagents.evidence_archive.orchestrator import CaptureOutcome


@pytest.mark.unit
class TestArchiveEvidenceCLI:
    def test_tickers_flag_and_defaults_wire_through(self, monkeypatch):
        captured = {}

        def fake_run(tickers, **kwargs):
            captured["tickers"] = tickers
            captured.update(kwargs)
            return []

        monkeypatch.setattr(m, "run_archive_evidence", fake_run)
        result = CliRunner().invoke(m.app, ["archive-evidence", "--tickers", "NVDA,MSFT"])
        assert result.exit_code == 0, result.output
        assert captured["tickers"] == ["NVDA", "MSFT"]
        assert captured["evidence_types"] == [NEWS, SOCIAL]
        assert captured["news_providers"] == ["yfinance"]
        assert captured["social_providers"] == ["stocktwits"]

    def test_tickers_file_flag(self, monkeypatch, tmp_path):
        tickers_file = tmp_path / "tickers.txt"
        tickers_file.write_text("NVDA\nMSFT\n", encoding="utf-8")
        captured = {}

        def fake_run(tickers, **kwargs):
            captured["tickers"] = tickers
            return []

        monkeypatch.setattr(m, "run_archive_evidence", fake_run)
        result = CliRunner().invoke(m.app, ["archive-evidence", "--tickers-file", str(tickers_file)])
        assert result.exit_code == 0, result.output
        assert captured["tickers"] == ["NVDA", "MSFT"]

    def test_explicit_provider_selection_is_recorded_exactly_as_asked(self, monkeypatch):
        captured = {}

        def fake_run(tickers, **kwargs):
            captured.update(kwargs)
            return []

        monkeypatch.setattr(m, "run_archive_evidence", fake_run)
        result = CliRunner().invoke(m.app, [
            "archive-evidence", "--tickers", "NVDA",
            "--news-providers", "yfinance,alpha_vantage",
            "--social-providers", "reddit",
        ])
        assert result.exit_code == 0, result.output
        assert captured["news_providers"] == ["yfinance", "alpha_vantage"]
        assert captured["social_providers"] == ["reddit"]

    def test_types_flag_restricts_to_one_evidence_type(self, monkeypatch):
        captured = {}

        def fake_run(tickers, **kwargs):
            captured.update(kwargs)
            return []

        monkeypatch.setattr(m, "run_archive_evidence", fake_run)
        result = CliRunner().invoke(m.app, ["archive-evidence", "--tickers", "NVDA", "--types", "news"])
        assert result.exit_code == 0, result.output
        assert captured["evidence_types"] == [NEWS]
        assert captured["social_providers"] is None

    def test_exit_code_1_when_any_capture_errors(self, monkeypatch):
        def fake_run(tickers, **kwargs):
            return [CaptureOutcome("NVDA", NEWS, "yfinance", "ERROR", 0, None, "boom")]

        monkeypatch.setattr(m, "run_archive_evidence", fake_run)
        result = CliRunner().invoke(m.app, ["archive-evidence", "--tickers", "NVDA"])
        assert result.exit_code == 1

    def test_exit_code_0_when_everything_succeeds_or_is_empty(self, monkeypatch):
        def fake_run(tickers, **kwargs):
            return [
                CaptureOutcome("NVDA", NEWS, "yfinance", "SUCCESS", 3, None),
                CaptureOutcome("NVDA", SOCIAL, "stocktwits", "EMPTY", 0, None),
            ]

        monkeypatch.setattr(m, "run_archive_evidence", fake_run)
        result = CliRunner().invoke(m.app, ["archive-evidence", "--tickers", "NVDA"])
        assert result.exit_code == 0, result.output

    def test_no_tickers_given_fails_cleanly(self):
        result = CliRunner().invoke(m.app, ["archive-evidence"])
        assert result.exit_code == 1
        assert "ticker" in result.output.lower()

    def test_unknown_provider_fails_cleanly_before_any_capture(self, monkeypatch):
        called = []
        monkeypatch.setattr(m, "run_archive_evidence", lambda *a, **k: called.append(1))
        result = CliRunner().invoke(
            m.app, ["archive-evidence", "--tickers", "NVDA", "--news-providers", "bloomberg"]
        )
        assert result.exit_code == 1
        assert not called
