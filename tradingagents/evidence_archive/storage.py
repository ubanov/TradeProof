"""Append-only, atomically-created storage for evidence captures.

Layout: ``<root>/<capture-date>/<ticker>/<evidence_type>/<provider>/<file>.json``

``capture-date`` is derived from the capture's own ``retrieved_at`` (when this
archiver actually asked the provider), never from an item's ``published_at``
-- a capture is dated by when it happened, not by what it found.

Each filename embeds a UUID4 (the capture_id), so two captures made in the
same microsecond never collide; the write itself still uses an atomic
``O_CREAT | O_EXCL`` open (the same primitive already relied on for
``automatic_run_dir``) rather than an ``exists()`` check followed by a write,
so a capture is never silently overwritten even under a theoretical name
collision.
"""

from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path

from tradingagents.dataflows.symbols import safe_ticker_component

_ENV_ROOT = "TRADINGAGENTS_EVIDENCE_ROOT"
_TRADINGAGENTS_HOME = os.path.join(os.path.expanduser("~"), ".tradingagents")


def default_evidence_root() -> Path:
    """``$TRADINGAGENTS_EVIDENCE_ROOT``, else ``~/.tradingagents/evidence`` --
    the same ``_TRADINGAGENTS_HOME`` convention ``results_dir``/``ledger_root``/
    ``data_cache_dir`` already use, kept independent of all three (this
    archiver is not wired into ``tradingagents run``)."""
    return Path(os.getenv(_ENV_ROOT) or os.path.join(_TRADINGAGENTS_HOME, "evidence"))


def _compact_timestamp(retrieved_at_iso: str) -> str:
    """``retrieved_at`` with path-unfriendly separators stripped, for a
    readable-but-safe filename prefix."""
    return retrieved_at_iso.replace("-", "").replace(":", "").replace(".", "")


def capture_path(
    root: Path, ticker: str, evidence_type: str, provider: str, retrieved_at_iso: str, capture_id: str
) -> Path:
    """The on-disk path for a capture -- pure, no filesystem access."""
    safe_ticker_component(ticker)
    capture_date = retrieved_at_iso[:10]  # the ISO string's own YYYY-MM-DD prefix
    filename = f"{_compact_timestamp(retrieved_at_iso)}_{capture_id}.json"
    return Path(root) / capture_date / ticker / evidence_type.lower() / provider.lower() / filename


def write_capture(root: Path, ticker: str, evidence_type: str, provider: str, capture: dict) -> Path:
    """Write ``capture`` to its canonical path, atomically and append-only.

    Raises ``FileExistsError`` rather than overwriting if the exact path is
    already taken -- this should not happen given the UUID4 in every
    filename, but the write is structured so it could never silently merge
    into an existing capture even if it did.
    """
    path = capture_path(root, ticker, evidence_type, provider, capture["retrieved_at"], capture["capture_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(capture, handle, indent=2, sort_keys=True, ensure_ascii=False)
    except Exception:
        with contextlib.suppress(OSError):
            os.remove(path)
        raise
    return path


def load_capture(path: Path) -> dict:
    """Read a capture file back for inspection/reprocessing."""
    return json.loads(Path(path).read_text(encoding="utf-8"))
