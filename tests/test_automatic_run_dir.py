"""``automatic_run_dir`` (cli/run.py): a fresh, unique, ALREADY-CREATED run
directory, even when the clock returns the exact same instant twice.

Real-run regression: ``automatic_run_dir`` used to only COMPUTE a path
string (``results_dir/runs/<microsecond timestamp>``) for someone else to
``mkdir`` much later (deep inside ``write_report_tree``, with
``exist_ok=True``). Two CLI invocations issued close enough together could
read the identical microsecond timestamp from the clock; neither would see
the other's not-yet-created directory, and the later, lenient
``mkdir(exist_ok=True)`` would then silently let BOTH runs write into the
same directory -- a real correctness bug, not just a test flake (confirmed:
``tests/test_cli_run_defaults.py::test_12_two_runs_same_ticker_date_get_different_paths``
could fail even in isolation, because ``CliRunner.invoke`` can call
``automatic_run_dir`` twice within the same microsecond).

Fix: ``automatic_run_dir`` now ATOMICALLY CREATES the directory itself via
``Path.mkdir(exist_ok=False)`` -- an OS-guaranteed indivisible check-and-
create on both POSIX and Windows -- retrying with ``_01``, ``_02``, ... on
a collision. No sleep, no dependence on the clock advancing, no shared/
global counter, never overwrites an existing run's directory.
"""

from __future__ import annotations

import datetime as _datetime_module

import pytest

import cli.run as cli_run


class _FrozenDatetime(_datetime_module.datetime):
    """Stands in for ``datetime.datetime`` with a ``now()`` that always
    returns the exact same instant -- the scenario a real clock can
    produce for two invocations issued close enough together."""

    _frozen_at: _datetime_module.datetime

    @classmethod
    def now(cls, tz=None):
        return cls._frozen_at


def _freeze_clock(monkeypatch, instant: _datetime_module.datetime) -> None:
    frozen = type("_Frozen", (_FrozenDatetime,), {"_frozen_at": instant})
    monkeypatch.setattr(cli_run.datetime, "datetime", frozen)


# --- 1: two calls with an identical clock reading -> distinct paths --------


@pytest.mark.unit
def test_1_identical_clock_reading_still_produces_distinct_paths(tmp_path, monkeypatch):
    _freeze_clock(monkeypatch, _datetime_module.datetime(2026, 9, 30, 12, 0, 0, 123456))
    first = cli_run.automatic_run_dir(tmp_path)
    second = cli_run.automatic_run_dir(tmp_path)
    assert first != second
    assert first.exists() and second.exists()


# --- 2: multiple consecutive collisions -------------------------------------


@pytest.mark.unit
def test_2_multiple_consecutive_collisions_all_resolve_to_distinct_paths(tmp_path, monkeypatch):
    _freeze_clock(monkeypatch, _datetime_module.datetime(2026, 9, 30, 12, 0, 0, 123456))
    paths = [cli_run.automatic_run_dir(tmp_path) for _ in range(5)]
    assert len(set(paths)) == 5  # all distinct
    assert all(p.exists() for p in paths)
    # Deterministic, readable suffix sequence -- never a random/opaque id.
    names = sorted(p.name for p in paths)
    timestamp = "20260930_120000_123456"
    assert names == [
        timestamp, f"{timestamp}_01", f"{timestamp}_02", f"{timestamp}_03", f"{timestamp}_04",
    ]


@pytest.mark.unit
def test_2b_no_sleep_no_clock_advance_needed_for_many_collisions(tmp_path, monkeypatch):
    """20 calls under one frozen instant must all succeed immediately --
    proving the resolution never waits for the clock to tick."""
    _freeze_clock(monkeypatch, _datetime_module.datetime(2026, 9, 30, 12, 0, 0, 999999))
    paths = [cli_run.automatic_run_dir(tmp_path) for _ in range(20)]
    assert len(set(paths)) == 20


# --- 3: layout preserved -----------------------------------------------------


@pytest.mark.unit
def test_3_layout_is_results_dir_runs_run_id(tmp_path, monkeypatch):
    _freeze_clock(monkeypatch, _datetime_module.datetime(2026, 9, 30, 12, 0, 0, 1))
    results_dir = tmp_path / "LLY" / "2026-09-30"  # mirrors _run_directory's own shape
    run_dir = cli_run.automatic_run_dir(results_dir)
    assert run_dir.parent.parent == results_dir
    assert run_dir.parent.name == "runs"
    assert run_dir.name == "20260930_120000_000001"


@pytest.mark.unit
def test_3b_readable_timestamp_remains_the_run_ids_base(tmp_path, monkeypatch):
    instant = _datetime_module.datetime(2024, 1, 31, 23, 59, 59, 500000)
    _freeze_clock(monkeypatch, instant)
    run_dir = cli_run.automatic_run_dir(tmp_path)
    assert run_dir.name == "20240131_235959_500000"


# --- 4: explicit --output-dir is untouched ----------------------------------


@pytest.mark.unit
def test_4_automatic_run_dir_is_never_consulted_when_output_dir_is_explicit(tmp_path, monkeypatch):
    """Structural: run_non_interactive only calls automatic_run_dir inside
    the `if output_dir is None:` branch (cli/main.py) -- confirmed here by
    making automatic_run_dir raise if invoked at all, then driving the CLI
    with an explicit --output-dir."""
    from typer.testing import CliRunner

    import cli.main as m

    def _boom(*_a, **_k):
        raise AssertionError("automatic_run_dir must not be called with an explicit --output-dir")

    monkeypatch.setattr(m, "automatic_run_dir", _boom)
    monkeypatch.setattr(
        m, "run_single_analysis", lambda **kw: {
            "status": "ok", "ticker": kw["ticker"], "run_duration_seconds": 1.0,
            "fallback_warnings": 0, "stats": {"llm_calls": 1, "tool_calls": 0, "tokens_in": 1, "tokens_out": 1},
            "research_execution": {
                "verification_status": "PASS", "verifier_passes": 1, "repair_rounds": 0,
                "bull_initial_calls": 1, "bear_initial_calls": 1, "bull_review_calls": 0, "bear_review_calls": 0,
            },
        },
    )
    chosen = tmp_path / "explicit"
    result = CliRunner().invoke(m.app, ["run", "--ticker", "LLY", "--output-dir", str(chosen)])
    assert result.exit_code == 0, result.output


# --- 5: atomic reservation is safe against a concurrent reservation --------


@pytest.mark.unit
def test_5_a_pre_existing_slot_from_a_concurrent_caller_is_never_reused_or_overwritten(tmp_path, monkeypatch):
    """Simulates the real race: another process (or an earlier call this
    same process lost a race to) already atomically reserved the bare
    timestamp slot before this call even runs -- automatic_run_dir must
    skip straight to the next free suffix, never reuse or touch that
    directory's contents."""
    _freeze_clock(monkeypatch, _datetime_module.datetime(2026, 9, 30, 12, 0, 0, 42))
    pre_existing = tmp_path / "runs" / "20260930_120000_000042"
    pre_existing.mkdir(parents=True)
    (pre_existing / "result.json").write_text('{"owner": "concurrent caller"}', encoding="utf-8")

    run_dir = cli_run.automatic_run_dir(tmp_path)

    assert run_dir != pre_existing
    assert run_dir.name == "20260930_120000_000042_01"
    # The concurrent caller's own file is completely untouched.
    assert (pre_existing / "result.json").read_text(encoding="utf-8") == '{"owner": "concurrent caller"}'


@pytest.mark.unit
def test_5b_mkdir_itself_is_the_only_synchronization_no_global_state(tmp_path, monkeypatch):
    """No module-level counter/lock is consulted -- two independent,
    freshly-imported-style calls (nothing shared between them but the
    filesystem) still resolve correctly, proving the OS's own atomic
    mkdir is sufficient."""
    _freeze_clock(monkeypatch, _datetime_module.datetime(2026, 9, 30, 12, 0, 0, 7))
    first = cli_run.automatic_run_dir(tmp_path)
    second = cli_run.automatic_run_dir(tmp_path)
    third = cli_run.automatic_run_dir(tmp_path)
    assert {first, second, third} == {
        tmp_path / "runs" / "20260930_120000_000007",
        tmp_path / "runs" / "20260930_120000_000007_01",
        tmp_path / "runs" / "20260930_120000_000007_02",
    }


@pytest.mark.unit
def test_5c_genuinely_concurrent_threads_under_a_frozen_clock_never_collide(tmp_path, monkeypatch):
    """Real concurrency, not simulated: many threads racing to call
    automatic_run_dir at the exact same frozen instant must all succeed
    with distinct paths and no exception -- the OS's own atomic mkdir is
    what makes this safe, not any Python-level lock this test could
    accidentally rely on."""
    import concurrent.futures

    _freeze_clock(monkeypatch, _datetime_module.datetime(2026, 9, 30, 12, 0, 0, 55))
    thread_count = 16

    with concurrent.futures.ThreadPoolExecutor(max_workers=thread_count) as pool:
        results = list(pool.map(lambda _: cli_run.automatic_run_dir(tmp_path), range(thread_count)))

    assert len(set(results)) == thread_count  # no two threads got the same directory
    assert all(p.exists() for p in results)


@pytest.mark.unit
def test_distinct_clock_readings_still_use_the_bare_timestamp_no_unnecessary_suffix(
    tmp_path, monkeypatch,
):
    """No collision -- the common case -- must not gain a spurious suffix.

    Previously used the real, unmocked clock and asserted ``"_01" not in
    run_dir.name`` -- a substring check that is a false positive whenever
    the real wall-clock hour, minute, or second happens to contain "01"
    (e.g. run at 01:23:45, or at 10:01:23): that "01" is legitimately part
    of the ``HHMMSS`` block, not an appended collision suffix, and the test
    failed with no actual bug. Every other test in this file already
    freezes the clock via ``_freeze_clock`` -- this one simply never did.
    Frozen to an instant whose hour is deliberately 01 (matching the exact
    failure mode) and asserting the full, exact expected name: a suffix
    would appear only as a trailing ``_NN`` the frozen microsecond timestamp
    does not itself contain, so exact equality is unambiguous regardless of
    what digits appear inside the timestamp.
    """
    _freeze_clock(monkeypatch, _datetime_module.datetime(2026, 1, 19, 1, 34, 7, 480406))
    run_dir = cli_run.automatic_run_dir(tmp_path)
    assert run_dir.name == "20260119_013407_480406"
    assert run_dir.exists()
    # Structural, not substring-based: a real suffix is always a trailing
    # "_" + exactly two digits appended AFTER the full timestamp -- "01"
    # inside HHMMSS (positions 9-14) can never match this.
    import re

    assert re.search(r"_\d{2}$", run_dir.name) is None
