"""PrePMSnapshot (see fork README: "PrePMSnapshot"): an immutable,
append-only record of exactly the state the Portfolio Manager node
receives, frozen the instant before it runs -- so a later experiment can
replay only the Portfolio Manager without re-running Analysts, the
research debate, the Trader, or Risk.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from tradingagents.agents.managers.portfolio_manager import create_portfolio_manager
from tradingagents.agents.schemas import PortfolioDecision, PortfolioRating
from tradingagents.graph.propagation import Propagator
from tradingagents.pre_pm_snapshot import (
    IncompletePrePMSnapshotError,
    PrePMSnapshotIntegrityError,
    build_pre_pm_snapshot,
    load_pre_pm_snapshot,
    resolve_pre_pm_root,
    verify_pre_pm_snapshot_integrity,
    with_pre_pm_snapshot,
    write_pre_pm_snapshot,
)
from tradingagents.run_provenance import build_run_provenance

_NO_GIT = {"commit": None, "branch": None, "dirty": None}

TRADER_SENTINEL = "TRADER_SENTINEL"
AGGRESSIVE_SENTINEL = "AGGRESSIVE_SENTINEL"
CONSERVATIVE_SENTINEL = "CONSERVATIVE_SENTINEL"
NEUTRAL_SENTINEL = "NEUTRAL_SENTINEL"
RISK_REVIEW_SENTINEL = "RISK_REVIEW_SENTINEL"


class _PMLLM:
    """Same stub pattern as test_portfolio_manager_disposition.py."""

    def __init__(self, decision=None):
        self.decision = decision or PortfolioDecision(
            rating=PortfolioRating.HOLD, disposition="KEEP",
            executive_summary="s", investment_thesis="t",
        )
        self.prompts: list[str] = []

    def with_structured_output(self, _schema):
        return self

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return self.decision


def _state_with_sentinels():
    state = Propagator().create_initial_state("NVDA", "2026-01-07")
    state["investment_plan"] = "RESEARCH_MANAGER_SENTINEL plan"
    state["trader_investment_plan"] = TRADER_SENTINEL
    state["risk_debate_state"].update({
        "aggressive_history": AGGRESSIVE_SENTINEL,
        "conservative_history": CONSERVATIVE_SENTINEL,
        "neutral_history": NEUTRAL_SENTINEL,
        "history": (
            f"{AGGRESSIVE_SENTINEL}\n{CONSERVATIVE_SENTINEL}\n{NEUTRAL_SENTINEL}\n{RISK_REVIEW_SENTINEL}"
        ),
    })
    return state


def _run_provenance(**overrides):
    config = {
        "llm_provider": "ollama",
        "generation_llm": "gen-model", "generation_reasoning_effort": "low",
        "review_llm": "review-model", "review_reasoning_effort": "max",
        "validator_llm": "validator-model", "validator_reasoning_effort": "medium",
        "validation_enabled": False,
        "decision_memory_enabled": False, "decision_memory_root": None,
        "output_language": "English", "max_risk_discuss_rounds": 3,
        **overrides,
    }
    return build_run_provenance(
        selected_analysts=["market"], config=config,
        configured_seed=0, base_seed=184736291, trace_llm=False,
        git_info=_NO_GIT, package_version="test", prompt_hashes={},
    )


# --- Snapshot: build / fields / hash ----------------------------------------


@pytest.mark.unit
def test_build_snapshot_contains_required_fields():
    snapshot = build_pre_pm_snapshot(state=_state_with_sentinels(), run_provenance=_run_provenance())
    assert snapshot.schema_version == "1.0"
    assert snapshot.snapshot_id
    assert snapshot.created_at
    assert snapshot.content_hash
    assert snapshot.run_fingerprint
    assert snapshot.run.ticker == "NVDA"
    assert snapshot.run.analysis_date == "2026-01-07"
    assert snapshot.run_provenance["seed"]["base_used"] == 184736291
    assert snapshot.state["trader_investment_plan"] == TRADER_SENTINEL


@pytest.mark.unit
def test_snapshot_excludes_messages():
    state = _state_with_sentinels()
    assert "messages" in state  # sanity: the real state does carry it
    snapshot = build_pre_pm_snapshot(state=state, run_provenance=_run_provenance())
    assert "messages" not in snapshot.state


@pytest.mark.unit
def test_snapshot_records_decision_memory_mode():
    snapshot = build_pre_pm_snapshot(
        state=_state_with_sentinels(),
        run_provenance=_run_provenance(decision_memory_enabled=False, decision_memory_root=None),
    )
    assert snapshot.run_provenance["execution"]["decision_memory_enabled"] is False
    assert snapshot.run_provenance["execution"]["decision_memory_root"] is None


@pytest.mark.unit
def test_missing_ticker_or_date_raises():
    state = _state_with_sentinels()
    state["company_of_interest"] = ""
    with pytest.raises(IncompletePrePMSnapshotError):
        build_pre_pm_snapshot(state=state, run_provenance=_run_provenance())


@pytest.mark.unit
def test_content_hash_is_stable_for_identical_input():
    state = _state_with_sentinels()
    provenance = _run_provenance()
    a = build_pre_pm_snapshot(state=state, run_provenance=provenance)
    b = build_pre_pm_snapshot(state=state, run_provenance=provenance)
    # Everything except the random snapshot_id/created_at/content_hash
    # itself must hash identically for identical state+provenance.
    a_payload = a.model_dump(mode="json")
    b_payload = b.model_dump(mode="json")
    for key in ("snapshot_id", "created_at", "content_hash"):
        del a_payload[key], b_payload[key]
    assert a_payload == b_payload


@pytest.mark.unit
def test_tampered_content_fails_integrity_check(tmp_path):
    snapshot = build_pre_pm_snapshot(state=_state_with_sentinels(), run_provenance=_run_provenance())
    path = write_pre_pm_snapshot(snapshot, tmp_path / "ledger")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["state"]["trader_investment_plan"] = "TAMPERED"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert verify_pre_pm_snapshot_integrity(path) is False
    with pytest.raises(PrePMSnapshotIntegrityError):
        load_pre_pm_snapshot(path)


# --- Snapshot: storage -------------------------------------------------------


@pytest.mark.unit
def test_write_then_load_round_trips(tmp_path):
    snapshot = build_pre_pm_snapshot(state=_state_with_sentinels(), run_provenance=_run_provenance())
    path = write_pre_pm_snapshot(snapshot, tmp_path / "ledger")
    assert path.exists()
    assert path.parent.name == "NVDA"
    assert path.parent.parent.name == "pre_pm"
    loaded = load_pre_pm_snapshot(path)
    assert loaded.snapshot_id == snapshot.snapshot_id
    assert loaded.content_hash == snapshot.content_hash
    assert loaded.state["trader_investment_plan"] == TRADER_SENTINEL


@pytest.mark.unit
def test_identical_rewrite_is_a_no_op(tmp_path):
    snapshot = build_pre_pm_snapshot(state=_state_with_sentinels(), run_provenance=_run_provenance())
    ledger_root = tmp_path / "ledger"
    path_a = write_pre_pm_snapshot(snapshot, ledger_root)
    path_b = write_pre_pm_snapshot(snapshot, ledger_root)  # same object, same content
    assert path_a == path_b


@pytest.mark.unit
def test_differing_content_at_same_path_raises(tmp_path, monkeypatch):
    import tradingagents.pre_pm_snapshot as module

    snapshot = build_pre_pm_snapshot(state=_state_with_sentinels(), run_provenance=_run_provenance())
    ledger_root = tmp_path / "ledger"
    write_pre_pm_snapshot(snapshot, ledger_root)

    # Force a UUID collision to exercise the "never silently overwrite" path.
    # A genuinely different state guarantees different content regardless
    # of created_at (which can coincide at low clock resolution).
    monkeypatch.setattr(module.uuid, "uuid4", lambda: uuid.UUID(snapshot.snapshot_id))
    different_state = _state_with_sentinels()
    different_state["trader_investment_plan"] = "DIFFERENT_TRADER_PLAN"
    other = build_pre_pm_snapshot(state=different_state, run_provenance=_run_provenance())
    assert other.snapshot_id == snapshot.snapshot_id
    with pytest.raises(FileExistsError):
        write_pre_pm_snapshot(other, ledger_root)


@pytest.mark.unit
def test_resolve_pre_pm_root_is_a_sibling_of_recommendations(tmp_path):
    root = resolve_pre_pm_root({"ledger_root": str(tmp_path / "ledger")})
    assert root == tmp_path / "ledger" / "pre_pm"


# --- Automatic creation: with_pre_pm_snapshot wrapper ------------------------


@pytest.mark.unit
def test_wrapper_writes_snapshot_before_calling_the_node(tmp_path):
    calls = []

    def fake_node(state):
        calls.append(dict(state))
        return {"final_trade_decision": "Rating: Hold"}

    wrapped = with_pre_pm_snapshot(
        fake_node, run_provenance=_run_provenance(), ledger_root=tmp_path / "ledger",
    )
    state = _state_with_sentinels()
    result = wrapped(state)

    assert len(calls) == 1
    assert calls[0]["trader_investment_plan"] == TRADER_SENTINEL
    assert result["final_trade_decision"] == "Rating: Hold"
    written = list((tmp_path / "ledger" / "pre_pm" / "NVDA").glob("*.json"))
    assert len(written) == 1


@pytest.mark.unit
def test_wrapper_merges_snapshot_ref_into_node_result(tmp_path):
    wrapped = with_pre_pm_snapshot(
        lambda state: {"final_trade_decision": "x"},
        run_provenance=_run_provenance(), ledger_root=tmp_path / "ledger",
    )
    result = wrapped(_state_with_sentinels())
    ref = result["pre_pm_snapshot_ref"]
    assert set(ref) == {"snapshot_id", "content_hash", "path"}
    assert Path(ref["path"]).exists()


@pytest.mark.unit
def test_wrapper_never_breaks_the_node_call_on_write_failure(tmp_path, monkeypatch):
    import tradingagents.pre_pm_snapshot as module

    def _boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(module, "write_pre_pm_snapshot", _boom)
    calls = []
    wrapped = with_pre_pm_snapshot(
        lambda state: calls.append(1) or {"final_trade_decision": "ok"},
        run_provenance=_run_provenance(), ledger_root=tmp_path / "ledger",
    )
    result = wrapped(_state_with_sentinels())
    assert calls == [1]
    assert result["final_trade_decision"] == "ok"
    assert "pre_pm_snapshot_ref" not in result


# --- Equivalence: original PM input == replayed PM input --------------------


@pytest.mark.unit
def test_original_and_reloaded_snapshot_state_produce_identical_pm_prompts(tmp_path):
    """The core determinism requirement: the Portfolio Manager's prompt
    built from the live state and the one built from a snapshot reloaded
    from disk must be byte-identical, and both must carry every sentinel."""
    state = _state_with_sentinels()
    snapshot = build_pre_pm_snapshot(state=state, run_provenance=_run_provenance())
    path = write_pre_pm_snapshot(snapshot, tmp_path / "ledger")
    reloaded_state = load_pre_pm_snapshot(path).state

    llm_original = _PMLLM()
    llm_replayed = _PMLLM()
    create_portfolio_manager(llm_original)(state)
    create_portfolio_manager(llm_replayed)(reloaded_state)

    assert len(llm_original.prompts) == 1
    assert len(llm_replayed.prompts) == 1
    assert llm_original.prompts[0] == llm_replayed.prompts[0]

    for sentinel in (
        TRADER_SENTINEL, AGGRESSIVE_SENTINEL, CONSERVATIVE_SENTINEL,
        NEUTRAL_SENTINEL, RISK_REVIEW_SENTINEL,
    ):
        assert sentinel in llm_original.prompts[0]
        assert sentinel in llm_replayed.prompts[0]


@pytest.mark.unit
def test_reload_does_not_mutate_the_original_state_object():
    state = _state_with_sentinels()
    original_before = dict(state)
    snapshot = build_pre_pm_snapshot(state=state, run_provenance=_run_provenance())
    assert state == original_before  # build_pre_pm_snapshot never mutates its input
    assert snapshot.state is not state
