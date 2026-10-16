"""Evidence archive storage: layout, root resolution, and append-only,
atomically-created writes (never ``exists()`` then write -- see
tradingagents/evidence_archive/storage.py)."""

from __future__ import annotations

import pytest

from tradingagents.evidence_archive import storage
from tradingagents.evidence_archive.schema import EMPTY, NEWS, SUCCESS, build_capture


@pytest.mark.unit
class TestDefaultRoot:
    def test_env_override_wins(self, monkeypatch, tmp_path):
        custom = tmp_path / "custom-evidence"
        monkeypatch.setenv("TRADINGAGENTS_EVIDENCE_ROOT", str(custom))
        assert storage.default_evidence_root() == custom

    def test_falls_back_to_tradingagents_home(self, monkeypatch):
        monkeypatch.delenv("TRADINGAGENTS_EVIDENCE_ROOT", raising=False)
        root = storage.default_evidence_root()
        assert root.name == "evidence"
        assert root.parent.name == ".tradingagents"


@pytest.mark.unit
class TestCapturePath:
    def test_layout_is_root_date_ticker_type_provider_file(self, tmp_path):
        path = storage.capture_path(
            tmp_path, "NVDA", NEWS, "yfinance", "2026-10-15T22:15:03.123456Z", "abc123"
        )
        assert path.parent.name == "yfinance"
        assert path.parent.parent.name == "news"
        assert path.parent.parent.parent.name == "NVDA"
        assert path.parent.parent.parent.parent.name == "2026-10-15"
        assert path.parent.parent.parent.parent.parent == tmp_path
        assert path.name == "20261015T221503123456Z_abc123.json"

    def test_capture_date_comes_from_retrieved_at_not_an_items_published_at(self, tmp_path):
        # The function takes no item data at all -- proof by construction
        # that the date can only come from retrieved_at.
        path = storage.capture_path(tmp_path, "NVDA", NEWS, "yfinance", "2030-01-01T00:00:00Z", "x")
        assert path.parent.parent.parent.parent.name == "2030-01-01"

    def test_rejects_unsafe_ticker_component(self, tmp_path):
        with pytest.raises(ValueError):
            storage.capture_path(tmp_path, "../../etc", NEWS, "yfinance", "2026-10-15T00:00:00Z", "x")


@pytest.mark.unit
class TestWriteAndLoad:
    def test_round_trip(self, tmp_path):
        capture = build_capture(
            ticker="NVDA", evidence_type=NEWS, provider="yfinance",
            requested_at="2026-10-15T22:15:00Z", retrieved_at="2026-10-15T22:15:03Z",
            requested_query={"ticker": "NVDA"}, status=SUCCESS,
        )
        path = storage.write_capture(tmp_path, "NVDA", NEWS, "yfinance", capture)
        assert path.exists()
        assert storage.load_capture(path) == capture

    def test_never_overwrites_an_existing_capture_file(self, tmp_path):
        capture = build_capture(
            ticker="NVDA", evidence_type=NEWS, provider="yfinance",
            requested_at="t", retrieved_at="2026-10-15T22:15:03Z",
            requested_query={}, status=SUCCESS,
        )
        path = storage.write_capture(tmp_path, "NVDA", NEWS, "yfinance", capture)
        original = path.read_bytes()

        colliding = dict(capture)  # same retrieved_at + same capture_id -> same path
        colliding["status"] = EMPTY
        with pytest.raises(FileExistsError):
            storage.write_capture(tmp_path, "NVDA", NEWS, "yfinance", colliding)

        # The real create used O_EXCL -- an atomic check-and-create, never an
        # exists() check followed by a separate write -- so the original
        # file's bytes must still be exactly what the first write produced.
        assert path.read_bytes() == original

    def test_two_captures_in_the_same_microsecond_never_collide(self, tmp_path):
        same_instant = "2026-10-15T22:15:03.123456Z"
        first = build_capture(
            ticker="NVDA", evidence_type=NEWS, provider="yfinance",
            requested_at="t", retrieved_at=same_instant, requested_query={}, status=SUCCESS,
        )
        second = build_capture(
            ticker="NVDA", evidence_type=NEWS, provider="yfinance",
            requested_at="t", retrieved_at=same_instant, requested_query={}, status=SUCCESS,
        )
        p1 = storage.write_capture(tmp_path, "NVDA", NEWS, "yfinance", first)
        p2 = storage.write_capture(tmp_path, "NVDA", NEWS, "yfinance", second)
        assert p1 != p2
        assert p1.exists() and p2.exists()

    def test_write_creates_parent_directories(self, tmp_path):
        capture = build_capture(
            ticker="MSFT", evidence_type=NEWS, provider="alpha_vantage",
            requested_at="t", retrieved_at="2026-01-01T00:00:00Z", requested_query={}, status=EMPTY,
        )
        root = tmp_path / "does" / "not" / "exist" / "yet"
        path = storage.write_capture(root, "MSFT", NEWS, "alpha_vantage", capture)
        assert path.exists()
