"""Capture schema: deterministic hashing, integrity verification, exact-id
dedupe (never fuzzy), secret scrubbing, and "missing field is null, never
fabricated" (see tradingagents/evidence_archive/schema.py)."""

from __future__ import annotations

import pytest

from tradingagents.evidence_archive.schema import (
    EMPTY,
    EMPTY_UNVERIFIED,
    NEWS,
    SUCCESS,
    CaptureItem,
    build_capture,
    compute_content_hash,
    dedupe_items,
    normalized_item,
    scrub_secrets,
    unverifiable_empty_metadata,
    verify_capture_integrity,
)


def _item(provider_item_id=None, title="A", payload=None) -> CaptureItem:
    return CaptureItem(
        normalized=normalized_item(provider_item_id=provider_item_id, title=title),
        provider_payload=payload or {},
    )


@pytest.mark.unit
class TestContentHash:
    def test_hash_is_deterministic_and_verifiable(self):
        capture = build_capture(
            ticker="NVDA", evidence_type=NEWS, provider="yfinance",
            requested_at="2026-10-15T22:15:00Z", retrieved_at="2026-10-15T22:15:03Z",
            requested_query={"ticker": "NVDA"}, status=SUCCESS, items=[_item("1")],
        )
        assert verify_capture_integrity(capture)
        assert compute_content_hash(capture) == capture["content_hash"]

    def test_tampering_invalidates_the_hash(self):
        capture = build_capture(
            ticker="NVDA", evidence_type=NEWS, provider="yfinance",
            requested_at="t", retrieved_at="t", requested_query={}, status=EMPTY,
        )
        tampered = dict(capture)
        tampered["status"] = SUCCESS
        assert not verify_capture_integrity(tampered)

    def test_hash_excludes_itself_from_its_own_input(self):
        capture = build_capture(
            ticker="NVDA", evidence_type=NEWS, provider="yfinance",
            requested_at="t", retrieved_at="t", requested_query={}, status=EMPTY,
        )
        # Recomputing over the capture as-is (content_hash included) must give
        # the same result as the field already stored -- compute_content_hash
        # itself strips the field before hashing, never hashing its own output.
        assert compute_content_hash(capture) == compute_content_hash({**capture, "content_hash": "bogus"})


@pytest.mark.unit
def test_capture_id_is_never_the_content_hash():
    capture = build_capture(
        ticker="NVDA", evidence_type=NEWS, provider="yfinance",
        requested_at="t", retrieved_at="t", requested_query={}, status=EMPTY,
    )
    assert capture["capture_id"] != capture["content_hash"]


@pytest.mark.unit
def test_two_captures_with_identical_items_get_distinct_physical_identities():
    item = _item("1")
    first = build_capture(
        ticker="NVDA", evidence_type=NEWS, provider="yfinance",
        requested_at="t", retrieved_at="t", requested_query={}, status=SUCCESS, items=[item],
    )
    second = build_capture(
        ticker="NVDA", evidence_type=NEWS, provider="yfinance",
        requested_at="t", retrieved_at="t", requested_query={}, status=SUCCESS, items=[item],
    )
    assert first["capture_id"] != second["capture_id"]
    assert first["content_hash"] != second["content_hash"]  # capture_id feeds the hash


@pytest.mark.unit
class TestDedupeWithinOneCapture:
    def test_exact_provider_item_id_match_is_deduped(self):
        a = _item("1", title="Apple earnings")
        b = _item("1", title="Apple earnings (duplicate copy)")
        assert dedupe_items([a, b]) == [a]

    def test_no_unambiguous_id_is_never_fuzzy_deduped(self):
        # Same title, no provider_item_id at all (e.g. Reddit's RSS feed) --
        # both must be kept; this is explicitly NOT fuzzy title matching.
        a = _item(None, title="NVDA earnings beat")
        b = _item(None, title="NVDA earnings beat")
        assert dedupe_items([a, b]) == [a, b]

    def test_mixed_ids_and_missing_ids(self):
        a = _item("1", title="x")
        b = _item("1", title="y")  # dropped: same id as a
        c = _item(None, title="z")
        d = _item("2", title="w")
        assert dedupe_items([a, b, c, d]) == [a, c, d]


@pytest.mark.unit
def test_scrub_secrets_redacts_key_like_fields_recursively_but_not_normal_fields():
    payload = {
        "api_key": "SECRET123",
        "nested": {"auth_token": "abc", "title": "fine"},
        "items": [{"password": "hunter2"}, {"title": "also fine"}],
        "title": "fine",
    }
    scrubbed = scrub_secrets(payload)
    assert scrubbed["api_key"] == "***"
    assert scrubbed["nested"]["auth_token"] == "***"
    assert scrubbed["nested"]["title"] == "fine"
    assert scrubbed["items"][0]["password"] == "***"
    assert scrubbed["items"][1]["title"] == "also fine"
    assert scrubbed["title"] == "fine"


@pytest.mark.unit
def test_scrub_secrets_does_not_false_positive_on_substring_matches():
    """A whole-token check, never a bare substring search: "authors" contains
    "auth" and must NOT be redacted -- it is Alpha Vantage's real, non-secret
    byline field. Likewise "keyword"/"authority" must survive untouched."""
    payload = {"authors": ["Jane Analyst"], "keyword": "macro", "authority": "SEC"}
    scrubbed = scrub_secrets(payload)
    assert scrubbed["authors"] == ["Jane Analyst"]
    assert scrubbed["keyword"] == "macro"
    assert scrubbed["authority"] == "SEC"


@pytest.mark.unit
def test_build_capture_rejects_unknown_status_or_evidence_type():
    with pytest.raises(ValueError):
        build_capture(
            ticker="X", evidence_type=NEWS, provider="p",
            requested_at="t", retrieved_at="t", requested_query={}, status="BOGUS",
        )
    with pytest.raises(ValueError):
        build_capture(
            ticker="X", evidence_type="GLOBAL", provider="p",
            requested_at="t", retrieved_at="t", requested_query={}, status=SUCCESS,
        )


@pytest.mark.unit
def test_normalized_item_absent_fields_are_null_never_fabricated():
    item = normalized_item(title="only a title")
    assert item["provider_item_id"] is None
    assert item["published_at"] is None
    assert item["text"] is None
    assert item["author_or_source"] is None
    assert item["url"] is None
    assert item["engagement"] is None


@pytest.mark.unit
def test_unverifiable_empty_metadata_shape():
    metadata = unverifiable_empty_metadata("some provider-specific explanation")
    assert metadata == {
        "empty_semantics": EMPTY_UNVERIFIED,
        "observability_limitation": "some provider-specific explanation",
    }


@pytest.mark.unit
def test_build_capture_with_unverifiable_empty_metadata_still_hashes_and_verifies():
    capture = build_capture(
        ticker="NVDA", evidence_type=NEWS, provider="yfinance",
        requested_at="t", retrieved_at="t", requested_query={}, status=EMPTY,
        provider_metadata=unverifiable_empty_metadata("explanation"),
    )
    assert verify_capture_integrity(capture)
    assert capture["provider_metadata"]["empty_semantics"] == EMPTY_UNVERIFIED


@pytest.mark.unit
def test_capture_carries_the_point_in_time_notice():
    capture = build_capture(
        ticker="NVDA", evidence_type=NEWS, provider="yfinance",
        requested_at="t", retrieved_at="t", requested_query={}, status=EMPTY,
    )
    assert "retrieved_at" in capture["point_in_time_notice"]
    assert capture["status"] == EMPTY
