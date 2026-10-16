"""Orchestrates evidence capture across tickers, evidence types, and
providers.

Sequential by design (see fork spec: capture is rate-limit-sensitive,
especially Reddit and Alpha Vantage's free tier) -- no concurrency here. One
provider/ticker failure is always caught and recorded as an ``ERROR``
capture; it never aborts the rest of the run.

Deliberately standalone from ``tradingagents run``'s own config/pipeline:
reads ``get_config()`` only for the handful of existing, shared knobs
(``news_article_limit``) that the reused dataflows already key off of, not
for anything run-specific (vendor fallback chains, LLM settings, etc).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.symbols import safe_ticker_component
from tradingagents.evidence_archive.providers import NEWS_PROVIDERS, SOCIAL_PROVIDERS
from tradingagents.evidence_archive.schema import (
    EMPTY,
    ERROR,
    NEWS,
    SOCIAL,
    SUCCESS,
    UNSUPPORTED,
    CaptureError,
    build_capture,
    utcnow_iso,
)
from tradingagents.evidence_archive.storage import default_evidence_root, write_capture

DEFAULT_NEWS_PROVIDERS = ("yfinance",)
DEFAULT_SOCIAL_PROVIDERS = ("stocktwits",)

PROVIDERS_BY_TYPE = {NEWS: NEWS_PROVIDERS, SOCIAL: SOCIAL_PROVIDERS}
DEFAULT_PROVIDERS_BY_TYPE = {NEWS: DEFAULT_NEWS_PROVIDERS, SOCIAL: DEFAULT_SOCIAL_PROVIDERS}


def parse_evidence_types(value: str) -> list[str]:
    requested = [item.strip().upper() for item in value.split(",") if item.strip()]
    invalid = sorted(set(requested) - set(PROVIDERS_BY_TYPE))
    if invalid:
        raise ValueError(
            f"unknown evidence type(s): {', '.join(invalid)}; choose from: news, social"
        )
    if not requested:
        raise ValueError("at least one evidence type must be selected")
    seen = set()
    return [t for t in requested if not (t in seen or seen.add(t))]


def parse_providers(value: str, evidence_type: str) -> list[str]:
    allowed = PROVIDERS_BY_TYPE[evidence_type]
    requested = [item.strip().lower() for item in value.split(",") if item.strip()]
    invalid = sorted(set(requested) - set(allowed))
    if invalid:
        raise ValueError(
            f"unknown {evidence_type.lower()} provider(s): {', '.join(invalid)}; "
            f"choose from: {', '.join(allowed)}"
        )
    if not requested:
        raise ValueError(f"at least one {evidence_type.lower()} provider must be selected")
    seen = set()
    return [p for p in requested if not (p in seen or seen.add(p))]


@dataclass
class CaptureOutcome:
    ticker: str
    evidence_type: str
    provider: str
    status: str
    item_count: int
    path: Path | None
    error_message: str | None = None


def _run_one(ticker: str, evidence_type: str, provider: str, config: dict, root: Path) -> CaptureOutcome:
    fn = PROVIDERS_BY_TYPE[evidence_type][provider]
    requested_at = utcnow_iso()
    try:
        raw = fn(ticker, config)
    except Exception as exc:  # last-resort guard: a provider function must never abort the batch
        retrieved_at = utcnow_iso()
        capture = build_capture(
            ticker=ticker, evidence_type=evidence_type, provider=provider,
            requested_at=requested_at, retrieved_at=retrieved_at,
            requested_query={"ticker": ticker}, status=ERROR,
            items=[],
            error=CaptureError(type(exc).__name__, str(exc)),
            provider_metadata={},
        )
        path = write_capture(root, ticker, evidence_type, provider, capture)
        return CaptureOutcome(ticker, evidence_type, provider, ERROR, 0, path, str(exc))

    retrieved_at = utcnow_iso()
    capture = build_capture(
        ticker=ticker, evidence_type=evidence_type, provider=provider,
        requested_at=requested_at, retrieved_at=retrieved_at,
        requested_query=raw.requested_query, status=raw.status,
        items=raw.items, error=raw.error, provider_metadata=raw.provider_metadata,
    )
    path = write_capture(root, ticker, evidence_type, provider, capture)
    return CaptureOutcome(
        ticker, evidence_type, provider, raw.status, len(raw.items), path,
        raw.error.message if raw.error else None,
    )


def run_archive_evidence(
    tickers: list[str],
    *,
    evidence_types: list[str] | None = None,
    news_providers: list[str] | None = None,
    social_providers: list[str] | None = None,
    root: Path | None = None,
    on_capture: Callable[[CaptureOutcome], None] | None = None,
) -> list[CaptureOutcome]:
    """Capture NEWS/SOCIAL evidence for every ticker, sequentially.

    Each ``(ticker, evidence_type, provider)`` combination is captured and
    written independently; a failure in one never skips the rest. Returns
    every outcome in the order captured, so a caller can both stream
    progress (``on_capture``) and build a final summary from the list.
    """
    evidence_types = evidence_types or [NEWS, SOCIAL]
    providers_by_type = {
        NEWS: news_providers or list(DEFAULT_PROVIDERS_BY_TYPE[NEWS]),
        SOCIAL: social_providers or list(DEFAULT_PROVIDERS_BY_TYPE[SOCIAL]),
    }
    config = get_config()
    evidence_root = root or default_evidence_root()

    outcomes: list[CaptureOutcome] = []
    for ticker in tickers:
        # Tickers are expected to already be canonical/path-safe (the CLI's
        # own parse_tickers already runs normalize_ticker_symbol +
        # safe_ticker_component) -- re-validated here defensively since this
        # function is also a public, directly-callable entry point.
        safe_ticker_component(ticker)
        for evidence_type in evidence_types:
            for provider in providers_by_type[evidence_type]:
                outcome = _run_one(ticker, evidence_type, provider, config, evidence_root)
                outcomes.append(outcome)
                if on_capture is not None:
                    on_capture(outcome)
    return outcomes


def summarize(outcomes: list[CaptureOutcome]) -> dict:
    tickers = sorted({o.ticker for o in outcomes})
    return {
        "tickers": len(tickers),
        "captures": len(outcomes),
        "success": sum(o.status == SUCCESS for o in outcomes),
        "empty": sum(o.status == EMPTY for o in outcomes),
        "unsupported": sum(o.status == UNSUPPORTED for o in outcomes),
        "error": sum(o.status == ERROR for o in outcomes),
    }
