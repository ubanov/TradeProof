"""Capture schema for the point-in-time NEWS/SOCIAL evidence archive.

A capture means exactly one thing: "this is what this provider returned at
``retrieved_at``." It never means "these were all the news/posts that
existed at that moment" -- a provider's own feed may only reach a few days
back, may rank by relevance rather than time, or may simply not have
surfaced an item that existed. That limitation is a property of the
provider, not something this module can see past, so it is recorded
(``status``) rather than hidden.

Two levels per item, when the provider allows it:
  - ``normalized``: the common NEWS/SOCIAL contract (title/text/author/url/
    engagement/timestamps), the same shape regardless of provider.
  - ``provider_payload``: the raw dict received from the provider, kept at
    maximum fidelity (secrets scrubbed), for anything the normalized shape
    does not carry.

A field the provider does not supply is ``null`` -- never invented.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

SCHEMA_VERSION = 1

NEWS = "NEWS"
SOCIAL = "SOCIAL"
EVIDENCE_TYPES = (NEWS, SOCIAL)

SUCCESS = "SUCCESS"
EMPTY = "EMPTY"
ERROR = "ERROR"
UNSUPPORTED = "UNSUPPORTED"
STATUSES = (SUCCESS, EMPTY, ERROR, UNSUPPORTED)

# A capture's EMPTY status always means "fetch completed without raising, 0
# items" -- but for a provider whose underlying library can itself swallow
# an upstream failure into an empty result (confirmed for yfinance's
# Ticker.get_news(): it only raises for a 429, never for other non-2xx
# responses -- see EvidenceArchive's 2026-10-16 investigation), that EMPTY
# cannot be told apart from a genuine absence of items without reimplementing
# that library's private HTTP client, which this project deliberately never
# does. EMPTY_UNVERIFIED names that specific, narrower case; it is added to
# ``provider_metadata`` ONLY for a capture already classified EMPTY, by the
# one provider function that knows it applies to itself (``capture_news_*``
# functions for every other provider never attach it, since their own fetch
# already raises/returns None on a real transport or HTTP error -- see each
# provider's own raw-fetch primitive).
EMPTY_UNVERIFIED = "EMPTY_UNVERIFIED"


def unverifiable_empty_metadata(observability_limitation: str) -> dict:
    """``provider_metadata`` for an EMPTY capture whose "0 items" cannot be
    verified against the real upstream outcome (see ``EMPTY_UNVERIFIED``).

    Never asserts a specific HTTP status for THIS capture -- only describes
    the known CLASS of failure mode the underlying library can silently
    absorb, since this project has no way to observe what actually happened
    for any single capture without reimplementing that library's internals.
    """
    return {
        "empty_semantics": EMPTY_UNVERIFIED,
        "observability_limitation": observability_limitation,
    }

# Defensive secret scrubbing: none of the providers this module reuses embed
# a credential inside a response item today (the API key travels as a request
# parameter, never as response content), but a capture is a permanent
# artifact a future reprocessing step will read -- this never trusts that to
# stay true forever.
#
# Matched on whole snake_case/kebab-case/camelCase-insensitive TOKENS, never
# a bare substring search: a substring match on "auth" would also redact
# Alpha Vantage's entirely non-secret "authors" field (#observed while
# building this module's own example captures -- "authors" contains "auth").
_SECRET_KEY_WORDS = frozenset({
    "key", "apikey", "token", "secret", "password", "passwd",
    "credential", "credentials", "auth", "authorization",
})
_KEY_TOKEN_SPLIT_RE = re.compile(r"[^a-zA-Z0-9]+")


def _looks_like_secret_key(key: str) -> bool:
    tokens = _KEY_TOKEN_SPLIT_RE.split(str(key).lower())
    return any(token in _SECRET_KEY_WORDS for token in tokens if token)


def utcnow_iso() -> str:
    """Current time as UTC ISO-8601 with an explicit ``Z`` suffix."""
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def new_capture_id() -> str:
    """A UUID4 hex string identifying this physical capture -- never the
    content hash, so two captures with identical items still get distinct
    identities."""
    return uuid.uuid4().hex


def scrub_secrets(value: Any) -> Any:
    """Recursively redact any dict value whose key looks secret-shaped."""
    if isinstance(value, dict):
        return {
            key: ("***" if _looks_like_secret_key(key) else scrub_secrets(val))
            for key, val in value.items()
        }
    if isinstance(value, list):
        return [scrub_secrets(val) for val in value]
    return value


def _canonical_json(obj: Any) -> bytes:
    return json.dumps(
        obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str
    ).encode("utf-8")


def compute_content_hash(capture: dict) -> str:
    """Deterministic SHA256 over ``capture``'s content, excluding
    ``content_hash`` itself (the field it computes can't be part of its own
    input)."""
    payload = {key: val for key, val in capture.items() if key != "content_hash"}
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def verify_capture_integrity(capture: dict) -> bool:
    """Whether ``capture["content_hash"]`` matches a fresh recomputation."""
    return capture.get("content_hash") == compute_content_hash(capture)


def normalized_item(
    *,
    provider_item_id: str | None = None,
    published_at: str | None = None,
    title: str | None = None,
    text: str | None = None,
    author_or_source: str | None = None,
    url: str | None = None,
    engagement: dict | None = None,
) -> dict:
    """The common NEWS/SOCIAL item contract. A field the provider does not
    carry is ``null``, never fabricated."""
    return {
        "provider_item_id": provider_item_id,
        "published_at": published_at,
        "title": title,
        "text": text,
        "author_or_source": author_or_source,
        "url": url,
        "engagement": engagement,
    }


@dataclass
class CaptureItem:
    normalized: dict
    provider_payload: dict

    def to_dict(self) -> dict:
        return {
            "normalized": self.normalized,
            "provider_payload": scrub_secrets(self.provider_payload),
        }


@dataclass
class CaptureError:
    error_type: str
    message: str

    def to_dict(self) -> dict:
        return {"error_type": self.error_type, "message": self.message}


def dedupe_items(items: list[CaptureItem]) -> list[CaptureItem]:
    """Drop a later item ONLY when its ``provider_item_id`` exactly matches
    an earlier one's -- never a fuzzy match on title/text. An item with no
    ``provider_item_id`` (``None``) is never deduped against anything,
    since the provider gave no unambiguous identity to compare."""
    seen: set[str] = set()
    deduped: list[CaptureItem] = []
    for item in items:
        provider_item_id = item.normalized.get("provider_item_id")
        if provider_item_id is not None:
            if provider_item_id in seen:
                continue
            seen.add(provider_item_id)
        deduped.append(item)
    return deduped


_POINT_IN_TIME_NOTICE = (
    "This capture reflects what this provider returned at retrieved_at. "
    "It is not a claim that these were all the items that existed at that "
    "moment -- see status for whether this provider's own coverage even "
    "reached this request."
)


def build_capture(
    *,
    ticker: str,
    evidence_type: str,
    provider: str,
    requested_at: str,
    retrieved_at: str,
    requested_query: dict,
    status: str,
    items: list[CaptureItem] | None = None,
    error: CaptureError | None = None,
    provider_metadata: dict | None = None,
) -> dict:
    """Assemble one capture dict (with its own ``content_hash``) ready to
    write to disk. Never mutates its inputs."""
    if evidence_type not in EVIDENCE_TYPES:
        raise ValueError(f"unknown evidence_type: {evidence_type!r}")
    if status not in STATUSES:
        raise ValueError(f"unknown capture status: {status!r}")

    capture = {
        "schema_version": SCHEMA_VERSION,
        "capture_id": new_capture_id(),
        "ticker": ticker,
        "evidence_type": evidence_type,
        "provider": provider,
        "requested_at": requested_at,
        "retrieved_at": retrieved_at,
        "requested_query": scrub_secrets(requested_query),
        "status": status,
        "error": error.to_dict() if error is not None else None,
        "items": [item.to_dict() for item in (items or [])],
        "provider_metadata": scrub_secrets(provider_metadata or {}),
        "point_in_time_notice": _POINT_IN_TIME_NOTICE,
    }
    capture["content_hash"] = compute_content_hash(capture)
    return capture
