"""Point-in-time NEWS/SOCIAL evidence archive.

Downloads evidence from the existing NEWS/SOCIAL dataflows, preserves it
exactly as the provider returned it (before any Markdown/text rendering),
and writes it append-only to disk for later inspection/reprocessing.

Deliberately NOT wired into ``tradingagents run``: no LLM call, no
RecommendationRecord, no Outcome. See ``orchestrator.run_archive_evidence``
for the entry point and ``schema`` for the capture contract.
"""

from tradingagents.evidence_archive.orchestrator import (
    CaptureOutcome,
    parse_evidence_types,
    parse_providers,
    run_archive_evidence,
    summarize,
)
from tradingagents.evidence_archive.schema import (
    EMPTY,
    EMPTY_UNVERIFIED,
    ERROR,
    NEWS,
    SOCIAL,
    SUCCESS,
    UNSUPPORTED,
    unverifiable_empty_metadata,
    verify_capture_integrity,
)
from tradingagents.evidence_archive.storage import default_evidence_root, load_capture

__all__ = [
    "CaptureOutcome",
    "parse_evidence_types",
    "parse_providers",
    "run_archive_evidence",
    "summarize",
    "EMPTY",
    "EMPTY_UNVERIFIED",
    "ERROR",
    "NEWS",
    "SOCIAL",
    "SUCCESS",
    "UNSUPPORTED",
    "unverifiable_empty_metadata",
    "verify_capture_integrity",
    "default_evidence_root",
    "load_capture",
]
