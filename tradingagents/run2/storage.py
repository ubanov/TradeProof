"""run2's own, fully separate persistence -- never mixed with
`tradingagents run`'s ``results_dir``/``ledger_root`` (fork spec: "No debe
mezclar resultados con run").

Layout: ``<root>/<ticker>/<analysis_date>/runs/<run_id>/`` -- deliberately
the same shape as ``cli/run.py``'s own ``<results_dir>/<ticker>/<date>/runs/
<run_id>`` (see ``_run_directory``/``automatic_run_dir``), so the convention
is familiar to anyone who already knows `run`'s output layout, but rooted
under run2's own, independent directory.
"""

from __future__ import annotations

import datetime
import json
import os
from pathlib import Path
from typing import Any

from tradingagents.dataflows.symbols import safe_ticker_component

_ENV_ROOT = "TRADINGAGENTS_RUN2_ROOT"
_TRADINGAGENTS_HOME = os.path.join(os.path.expanduser("~"), ".tradingagents")


def default_run2_root() -> Path:
    """``$TRADINGAGENTS_RUN2_ROOT``, else ``~/.tradingagents/run2`` -- the
    same ``_TRADINGAGENTS_HOME`` convention ``results_dir``/``ledger_root``/
    ``evidence_root`` already use (see ``tradingagents/evidence_archive/
    storage.py::default_evidence_root``), kept fully independent of all of
    them."""
    return Path(os.getenv(_ENV_ROOT) or os.path.join(_TRADINGAGENTS_HOME, "run2"))


def run2_runs_base(root: Path, ticker: str, analysis_date: str) -> Path:
    """``<root>/<ticker>/<analysis_date>/runs`` -- pure, no filesystem access."""
    safe_ticker = safe_ticker_component(ticker)
    return Path(root) / safe_ticker / analysis_date / "runs"


def automatic_run2_dir(runs_base: Path) -> Path:
    """Atomically reserve a fresh, already-created run directory under
    ``runs_base``. Identical collision-safe ``mkdir(exist_ok=False)`` + retry
    algorithm as ``cli/run.py::automatic_run_dir``, duplicated here (not
    imported) because run2 is a structurally separate pipeline that must
    never import from ``cli/`` (see fork spec: run2 must not touch `run`'s
    own code)."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    candidate = Path(runs_base) / timestamp
    suffix = 0
    while True:
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            suffix += 1
            candidate = Path(runs_base) / f"{timestamp}_{suffix:02d}"


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    data = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def write_result(run_dir: Path, result: dict[str, Any]) -> Path:
    path = Path(run_dir) / "result.json"
    _atomic_write_json(path, result)
    return path


def write_report(run_dir: Path, markdown: str) -> Path:
    path = Path(run_dir) / "complete_report.md"
    path.write_text(markdown, encoding="utf-8")
    return path


def write_debug_call(run_dir: Path, index: int, stage: str, prompt: str, response_text: str) -> Path:
    """One raw prompt/response pair under ``<run_dir>/debug/calls/NNN_<stage>/``
    -- a deliberately minimal, bespoke dump (not the main pipeline's
    ``LLMTraceCallbackHandler``/``run_manifest.json`` machinery), since run2
    only needs this for `--trace-llm` auditability, not full trace-mode
    parity."""
    call_dir = Path(run_dir) / "debug" / "calls" / f"{index:03d}_{stage.replace(' ', '_')}"
    call_dir.mkdir(parents=True, exist_ok=True)
    (call_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    (call_dir / "response.txt").write_text(response_text, encoding="utf-8")
    return call_dir


def load_result(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
