"""Decision Memory isolation (see fork README: "Decision memory
isolation").

The first historical Market-only pilot batch found that Decision
Memory/the decision log could leak look-ahead information into a
historical/backtest run: a reflection written from a later (real-time)
session could carry a realised_return/alpha that was not actually known as
of the simulated analysis date. ``resolve_decision_memory_config`` is the
one place ``decision_memory_enabled``/``decision_memory_root`` are
consulted, resolving them into the ``memory_log_path`` ``TradingMemoryLog``
reads -- so a run can go completely stateless with respect to past/future
sessions (``decision_memory_enabled=False``), or use a store fully isolated
from the normal one (``decision_memory_root``), without scattering
``if no_decision_memory:`` checks through the pipeline.

This module does NOT implement true point-in-time memory (``available_from
<= analysis_date``) -- that remains future work. It only proves the two
new escape hatches are airtight: a run with Decision Memory disabled never
reads, never writes, and never leaves a trace.
"""

from __future__ import annotations

import pytest

from tradingagents.decision_log import TradingMemoryLog
from tradingagents.default_config import resolve_decision_memory_config
from tradingagents.recommendation_record import RunRef
from tradingagents.run_provenance import build_run_provenance

FUTURE_LEAK_SENTINEL = "FUTURE_LEAK_SENTINEL_12345"


def _seed_future_leak(path):
    """Write a resolved memory-log entry carrying an unmistakable sentinel
    and unrealistic realised_return/alpha, simulating the exact shape of
    leaked future knowledge the pilot found."""
    log = TradingMemoryLog({"memory_log_path": str(path)})
    log.store_decision("NVDA", "2026-01-07", f"Rating: Buy\n{FUTURE_LEAK_SENTINEL}")
    log.update_with_outcome(
        "NVDA", "2026-01-07",
        raw_return=0.99, alpha_return=0.88, holding_days=10,
        reflection=f"{FUTURE_LEAK_SENTINEL} realised_return=+99% alpha=+88%",
        resolution_date="2026-01-21",
    )
    return log


# --- resolve_decision_memory_config: default / no-op behavior --------------


@pytest.mark.unit
def test_default_leaves_memory_log_path_untouched():
    config = {"memory_log_path": "/normal/path/trading_memory.md"}
    resolved = resolve_decision_memory_config(config)
    assert resolved["memory_log_path"] == "/normal/path/trading_memory.md"


@pytest.mark.unit
def test_enabled_true_with_no_root_is_a_no_op():
    config = {
        "decision_memory_enabled": True,
        "decision_memory_root": None,
        "memory_log_path": "/normal/path/trading_memory.md",
    }
    resolved = resolve_decision_memory_config(config)
    assert resolved["memory_log_path"] == "/normal/path/trading_memory.md"


@pytest.mark.unit
def test_missing_keys_default_to_enabled_no_root():
    """A config built before these keys existed (e.g. a bare dict a test or
    programmatic caller constructs) must behave exactly as before."""
    config = {"memory_log_path": "/normal/path/trading_memory.md"}
    resolved = resolve_decision_memory_config(config)
    assert resolved["memory_log_path"] == "/normal/path/trading_memory.md"


# --- --no-decision-memory: read=false, write=false --------------------------


@pytest.mark.unit
def test_no_decision_memory_sets_memory_log_path_to_none():
    config = {
        "decision_memory_enabled": False,
        "decision_memory_root": None,
        "memory_log_path": "/normal/path/trading_memory.md",
    }
    resolved = resolve_decision_memory_config(config)
    assert resolved["memory_log_path"] is None


@pytest.mark.unit
def test_no_decision_memory_never_reads_the_future_leak_sentinel(tmp_path):
    mem_path = tmp_path / "trading_memory.md"
    _seed_future_leak(mem_path)

    config = resolve_decision_memory_config(
        {"decision_memory_enabled": False, "decision_memory_root": None,
         "memory_log_path": str(mem_path)}
    )
    log = TradingMemoryLog(config)

    # 1. not recovered from load_entries at all.
    assert log.load_entries() == []
    # 2. never reaches the past-context string fed into agent prompts.
    past_context = log.get_past_context("NVDA", as_of="2026-01-15")
    assert FUTURE_LEAK_SENTINEL not in past_context
    assert past_context == ""
    # Also unfiltered (live-run-shaped) lookups must not see it either.
    assert FUTURE_LEAK_SENTINEL not in log.get_past_context("NVDA")


@pytest.mark.unit
def test_no_decision_memory_never_writes_a_new_decision(tmp_path):
    mem_path = tmp_path / "trading_memory.md"
    _seed_future_leak(mem_path)
    before = mem_path.read_text(encoding="utf-8")

    config = resolve_decision_memory_config(
        {"decision_memory_enabled": False, "decision_memory_root": None,
         "memory_log_path": str(mem_path)}
    )
    log = TradingMemoryLog(config)
    log.store_decision("NVDA", "2026-01-15", "Rating: Hold\nnew decision text")

    after = mem_path.read_text(encoding="utf-8")
    assert after == before  # 4. the store on disk is byte-for-byte unchanged


@pytest.mark.unit
def test_no_decision_memory_never_updates_an_outcome(tmp_path):
    mem_path = tmp_path / "trading_memory.md"
    log_normal = TradingMemoryLog({"memory_log_path": str(mem_path)})
    log_normal.store_decision("NVDA", "2026-01-15", "Rating: Hold\npending decision")
    before = mem_path.read_text(encoding="utf-8")

    config = resolve_decision_memory_config(
        {"decision_memory_enabled": False, "decision_memory_root": None,
         "memory_log_path": str(mem_path)}
    )
    log_disabled = TradingMemoryLog(config)
    log_disabled.update_with_outcome(
        "NVDA", "2026-01-15", raw_return=0.5, alpha_return=0.4,
        holding_days=5, reflection="should never be written",
    )
    log_disabled.batch_update_with_outcomes([{
        "ticker": "NVDA", "trade_date": "2026-01-15", "raw_return": 0.5,
        "alpha_return": 0.4, "holding_days": 5, "reflection": "should never be written",
    }])

    after = mem_path.read_text(encoding="utf-8")
    assert after == before  # 4. no effect on the existing store either


@pytest.mark.unit
def test_decision_memory_before_equals_after_when_disabled(tmp_path):
    """The task's own acceptance check: Decision Memory before == after."""
    mem_path = tmp_path / "trading_memory.md"
    _seed_future_leak(mem_path)
    before = mem_path.read_bytes()

    config = resolve_decision_memory_config(
        {"decision_memory_enabled": False, "decision_memory_root": None,
         "memory_log_path": str(mem_path)}
    )
    log = TradingMemoryLog(config)
    log.load_entries()
    log.get_past_context("NVDA")
    log.store_decision("MSFT", "2026-02-01", "Rating: Buy\nunrelated run")
    log.update_with_outcome("MSFT", "2026-02-01", 0.1, 0.1, 5, "reflection")

    after = mem_path.read_bytes()
    assert after == before


@pytest.mark.unit
def test_no_decision_memory_does_not_create_a_store_where_none_existed(tmp_path):
    """store_decision normally creates the parent directory on first use;
    disabled mode must not even do that."""
    mem_path = tmp_path / "does" / "not" / "exist" / "trading_memory.md"
    config = resolve_decision_memory_config(
        {"decision_memory_enabled": False, "decision_memory_root": None,
         "memory_log_path": str(mem_path)}
    )
    log = TradingMemoryLog(config)
    log.store_decision("NVDA", "2026-01-15", "Rating: Hold\nsomething")
    assert not mem_path.exists()
    assert not mem_path.parent.exists()


# --- --decision-memory-root: isolated store ----------------------------------


@pytest.mark.unit
def test_decision_memory_root_points_at_root_slash_trading_memory_md(tmp_path):
    root = tmp_path / "experiment-001" / "memory"
    config = resolve_decision_memory_config(
        {"decision_memory_enabled": True, "decision_memory_root": str(root),
         "memory_log_path": "/should/be/overridden/trading_memory.md"}
    )
    assert config["memory_log_path"] == str(root / "trading_memory.md")


@pytest.mark.unit
def test_three_roots_are_fully_isolated_from_each_other(tmp_path):
    normal_path = tmp_path / "normal-memory" / "trading_memory.md"
    root_a = tmp_path / "experiment-A"
    root_b = tmp_path / "experiment-B"

    TradingMemoryLog({"memory_log_path": str(normal_path)}).store_decision(
        "NVDA", "2026-01-01", "Rating: Hold\nNORMAL_SENTINEL"
    )
    config_a = resolve_decision_memory_config(
        {"decision_memory_enabled": True, "decision_memory_root": str(root_a)}
    )
    TradingMemoryLog(config_a).store_decision(
        "NVDA", "2026-01-01", "Rating: Buy\nEXPERIMENT_A_SENTINEL"
    )
    config_b = resolve_decision_memory_config(
        {"decision_memory_enabled": True, "decision_memory_root": str(root_b)}
    )
    TradingMemoryLog(config_b).store_decision(
        "NVDA", "2026-01-01", "Rating: Sell\nEXPERIMENT_B_SENTINEL"
    )

    normal_text = normal_path.read_text(encoding="utf-8")
    a_text = (root_a / "trading_memory.md").read_text(encoding="utf-8")
    b_text = (root_b / "trading_memory.md").read_text(encoding="utf-8")

    # 1/2: each store holds only its own sentinel.
    assert "NORMAL_SENTINEL" in normal_text
    assert "EXPERIMENT_A_SENTINEL" not in normal_text
    assert "EXPERIMENT_B_SENTINEL" not in normal_text

    assert "EXPERIMENT_A_SENTINEL" in a_text
    assert "NORMAL_SENTINEL" not in a_text
    assert "EXPERIMENT_B_SENTINEL" not in a_text

    assert "EXPERIMENT_B_SENTINEL" in b_text
    assert "NORMAL_SENTINEL" not in b_text
    assert "EXPERIMENT_A_SENTINEL" not in b_text

    # 5: reading back through TradingMemoryLog confirms the same isolation.
    assert TradingMemoryLog(config_a).get_pending_entries()[0]["decision"] == (
        "Rating: Buy\nEXPERIMENT_A_SENTINEL"
    )


@pytest.mark.unit
def test_an_existing_normal_memory_is_not_visible_through_an_alternate_root(tmp_path):
    normal_path = tmp_path / "normal-memory" / "trading_memory.md"
    _seed_future_leak(normal_path)

    root = tmp_path / "experiment-001"
    config = resolve_decision_memory_config(
        {"decision_memory_enabled": True, "decision_memory_root": str(root)}
    )
    log = TradingMemoryLog(config)
    assert log.load_entries() == []
    assert FUTURE_LEAK_SENTINEL not in log.get_past_context("NVDA")


@pytest.mark.unit
def test_an_existing_alternate_root_memory_can_be_recovered(tmp_path):
    root = tmp_path / "experiment-001"
    config = resolve_decision_memory_config(
        {"decision_memory_enabled": True, "decision_memory_root": str(root)}
    )
    log = TradingMemoryLog(config)
    log.store_decision("NVDA", "2026-01-01", "Rating: Buy\nEXPERIMENT_A_SENTINEL")
    log.update_with_outcome(
        "NVDA", "2026-01-01", 0.05, 0.03, 5, "EXPERIMENT_A_SENTINEL reflection",
    )

    # A fresh TradingMemoryLog pointed at the SAME root recovers it.
    reopened = TradingMemoryLog(resolve_decision_memory_config(
        {"decision_memory_enabled": True, "decision_memory_root": str(root)}
    ))
    assert "EXPERIMENT_A_SENTINEL" in reopened.get_past_context("NVDA")


@pytest.mark.unit
def test_writes_under_an_alternate_root_never_touch_the_normal_root(tmp_path):
    normal_path = tmp_path / "normal-memory" / "trading_memory.md"
    TradingMemoryLog({"memory_log_path": str(normal_path)}).store_decision(
        "NVDA", "2026-01-01", "Rating: Hold\nNORMAL_SENTINEL"
    )
    before = normal_path.read_text(encoding="utf-8")

    root = tmp_path / "experiment-001"
    config = resolve_decision_memory_config(
        {"decision_memory_enabled": True, "decision_memory_root": str(root)}
    )
    TradingMemoryLog(config).store_decision("NVDA", "2026-01-02", "Rating: Buy\nsomething else")

    after = normal_path.read_text(encoding="utf-8")
    assert after == before


# --- Precedence / incompatible combination -----------------------------------


@pytest.mark.unit
def test_enabled_false_and_root_together_raises_clearly(tmp_path):
    config = {
        "decision_memory_enabled": False,
        "decision_memory_root": str(tmp_path / "experiment-001"),
    }
    with pytest.raises(ValueError, match="--no-decision-memory.*--decision-memory-root"):
        resolve_decision_memory_config(config)


@pytest.mark.unit
def test_disabled_wins_is_never_reached_because_the_conflict_raises_first(tmp_path):
    """Belt-and-suspenders: even if a future caller somehow bypassed the
    raise, disabled must never silently defer to root (no silent pick)."""
    config = {"decision_memory_enabled": False, "decision_memory_root": None}
    resolved = resolve_decision_memory_config(config)
    assert resolved["memory_log_path"] is None


# --- Default behavior is unchanged ------------------------------------------


@pytest.mark.unit
def test_default_tradingagents_graph_config_resolves_cleanly():
    from tradingagents.default_config import DEFAULT_CONFIG

    config = resolve_decision_memory_config(DEFAULT_CONFIG.copy())
    assert config["memory_log_path"] == DEFAULT_CONFIG["memory_log_path"]


# --- Provenance / RecommendationRecord auditability -------------------------


_NO_GIT = {"commit": None, "branch": None, "dirty": None}


@pytest.mark.unit
def test_run_provenance_records_decision_memory_mode():
    provenance = build_run_provenance(
        selected_analysts=["market"],
        config={"decision_memory_enabled": False, "decision_memory_root": None},
        git_info=_NO_GIT, package_version="test", prompt_hashes={},
    )
    assert provenance["execution"]["decision_memory_enabled"] is False
    assert provenance["execution"]["decision_memory_root"] is None


@pytest.mark.unit
def test_run_provenance_records_decision_memory_root():
    provenance = build_run_provenance(
        selected_analysts=["market"],
        config={"decision_memory_enabled": True, "decision_memory_root": "/some/root"},
        git_info=_NO_GIT, package_version="test", prompt_hashes={},
    )
    assert provenance["execution"]["decision_memory_enabled"] is True
    assert provenance["execution"]["decision_memory_root"] == "/some/root"


@pytest.mark.unit
def test_run_provenance_defaults_to_enabled_when_config_omits_it():
    provenance = build_run_provenance(
        selected_analysts=["market"], config={},
        git_info=_NO_GIT, package_version="test", prompt_hashes={},
    )
    assert provenance["execution"]["decision_memory_enabled"] is True
    assert provenance["execution"]["decision_memory_root"] is None


@pytest.mark.unit
def test_runref_defaults_when_provenance_predates_the_field():
    """A RunRef built from older provenance (no decision_memory_* keys at
    all in its execution block) must default to enabled -- never guessed
    as disabled -- so an old RecommendationRecord read back doesn't look
    like it used an unusual memory mode it never actually had."""
    ref = RunRef(
        ticker="NVDA", analysis_date="2026-01-07",
        git={"commit": None, "branch": None, "dirty": None},
        seed={"configured": None, "base_used": None},
    )
    assert ref.decision_memory_enabled is True
    assert ref.decision_memory_root is None
