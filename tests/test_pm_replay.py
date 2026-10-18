"""PM replay (see fork README: "PrePMSnapshot"): re-run ONLY the Portfolio
Manager against a frozen PrePMSnapshot, producing an independent
PMReplayRecord -- never the original snapshot or RecommendationRecord,
never market data, never Decision Memory, never analysts/debate/trader/risk.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

import tradingagents.pm_replay as pm_replay
from tradingagents.agents.schemas import PortfolioDecision, PortfolioRating
from tradingagents.graph.propagation import Propagator
from tradingagents.pm_replay import (
    list_pm_replays,
    resolve_pm_replay_root,
    run_pm_replay,
)
from tradingagents.pre_pm_snapshot import (
    build_pre_pm_snapshot,
    load_pre_pm_snapshot,
    write_pre_pm_snapshot,
)
from tradingagents.run_provenance import build_run_provenance

_NO_GIT = {"commit": None, "branch": None, "dirty": None}


def _imported_modules(path: Path) -> set[str]:
    """Same AST-based check as test_recommendation_record_persistence.py's
    own "does not import the ledger/graph" tests."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    names = {
        alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names
    }
    names |= {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module
    }
    return names


class _Client:
    def __init__(self, model):
        self.model = model

    def get_llm(self):
        return self.model


class _PMLLM:
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


def _state():
    state = Propagator().create_initial_state("NVDA", "2026-01-07")
    state["investment_plan"] = "RESEARCH_MANAGER_SENTINEL"
    state["trader_investment_plan"] = "TRADER_SENTINEL"
    state["risk_debate_state"]["history"] = "AGGRESSIVE_SENTINEL CONSERVATIVE_SENTINEL NEUTRAL_SENTINEL"
    return state


def _provenance(**overrides):
    config = {
        "llm_provider": "ollama",
        "generation_llm": "snapshot-gen-model", "generation_reasoning_effort": "low",
        "validator_llm": "snapshot-validator-model", "validator_reasoning_effort": "medium",
        "validation_enabled": False,
        "decision_memory_enabled": False, "decision_memory_root": None,
        "output_language": "English", "max_risk_discuss_rounds": 3,
        **overrides,
    }
    return build_run_provenance(
        selected_analysts=["market"], config=config, configured_seed=0, base_seed=1,
        git_info=_NO_GIT, package_version="test", prompt_hashes={},
    )


def _write_snapshot(tmp_path, **provenance_overrides) -> Path:
    snapshot = build_pre_pm_snapshot(state=_state(), run_provenance=_provenance(**provenance_overrides))
    return write_pre_pm_snapshot(snapshot, tmp_path / "ledger")


# --- Loading --------------------------------------------------------------


@pytest.mark.unit
def test_loads_a_valid_snapshot_and_replays(tmp_path, monkeypatch):
    path = _write_snapshot(tmp_path)
    monkeypatch.setattr(pm_replay, "create_llm_client", lambda **k: _Client(_PMLLM()))

    record, written_path = run_pm_replay(snapshot_path=path, ledger_root=tmp_path / "ledger")

    assert written_path.exists()
    assert record.rating == "Hold"
    assert record.disposition == "KEEP"
    assert record.source_snapshot_hash == load_pre_pm_snapshot(path).content_hash


@pytest.mark.unit
def test_rejects_a_corrupted_snapshot(tmp_path, monkeypatch):
    path = _write_snapshot(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["state"]["trader_investment_plan"] = "TAMPERED"
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(pm_replay, "create_llm_client", lambda **k: _Client(_PMLLM()))

    with pytest.raises(Exception, match="content_hash"):
        run_pm_replay(snapshot_path=path, ledger_root=tmp_path / "ledger")


# --- Structural isolation: no analysts/debate/manager/trader/risk/graph ----


@pytest.mark.unit
def test_module_never_imports_the_graph():
    """A replay cannot run Analysts/debate/Research Manager/Trader/Risk
    because this module never constructs or imports the compiled graph at
    all -- not a runtime flag, a structural guarantee."""
    modules = _imported_modules(Path(pm_replay.__file__))
    assert not any("trading_graph" in m or "graph.setup" in m or "graph.propagation" in m for m in modules)


@pytest.mark.unit
def test_module_never_imports_decision_memory():
    """A replay cannot consult the live Decision Memory store because this
    module (and the Portfolio Manager node it calls) never imports
    TradingMemoryLog at all."""
    modules = _imported_modules(Path(pm_replay.__file__))
    assert not any("decision_log" in m for m in modules)

    import tradingagents.agents.managers.portfolio_manager as pm_module
    pm_modules = _imported_modules(Path(pm_module.__file__))
    assert not any("decision_log" in m for m in pm_modules)


@pytest.mark.unit
def test_replay_never_calls_a_market_data_vendor(tmp_path, monkeypatch):
    from tradingagents.dataflows import router

    def _boom(*a, **k):
        raise AssertionError("a replay must never call a market-data vendor")

    for vendors in router.VENDOR_METHODS.values():
        for vendor in list(vendors):
            monkeypatch.setitem(vendors, vendor, _boom)

    path = _write_snapshot(tmp_path)
    monkeypatch.setattr(pm_replay, "create_llm_client", lambda **k: _Client(_PMLLM()))
    record, _ = run_pm_replay(snapshot_path=path, ledger_root=tmp_path / "ledger")
    assert record.rating == "Hold"


# --- Never modifies the snapshot or any RecommendationRecord --------------


@pytest.mark.unit
def test_replay_does_not_modify_the_snapshot_file(tmp_path, monkeypatch):
    path = _write_snapshot(tmp_path)
    before = path.read_bytes()
    monkeypatch.setattr(pm_replay, "create_llm_client", lambda **k: _Client(_PMLLM()))

    run_pm_replay(snapshot_path=path, ledger_root=tmp_path / "ledger")

    assert path.read_bytes() == before


@pytest.mark.unit
def test_replay_does_not_touch_the_recommendations_ledger(tmp_path, monkeypatch):
    path = _write_snapshot(tmp_path)
    monkeypatch.setattr(pm_replay, "create_llm_client", lambda **k: _Client(_PMLLM()))

    run_pm_replay(snapshot_path=path, ledger_root=tmp_path / "ledger")

    assert not (tmp_path / "ledger" / "recommendations").exists()


# --- Multiple replays coexist ------------------------------------------------


@pytest.mark.unit
def test_multiple_replays_of_the_same_snapshot_coexist(tmp_path, monkeypatch):
    path = _write_snapshot(tmp_path)
    decisions = [
        PortfolioDecision(rating=PortfolioRating.HOLD, disposition="KEEP",
                           executive_summary="s", investment_thesis="t"),
        PortfolioDecision(rating=PortfolioRating.UNDERWEIGHT, disposition="REDUCE_RISK",
                           executive_summary="s2", investment_thesis="t2"),
        PortfolioDecision(rating=PortfolioRating.HOLD, disposition="KEEP",
                           executive_summary="s3", investment_thesis="t3"),
    ]
    records = []
    for decision in decisions:
        monkeypatch.setattr(pm_replay, "create_llm_client", lambda d=decision, **k: _Client(_PMLLM(d)))
        record, _ = run_pm_replay(snapshot_path=path, ledger_root=tmp_path / "ledger")
        records.append(record)

    snapshot_id = load_pre_pm_snapshot(path).snapshot_id
    replay_files = list_pm_replays(tmp_path / "ledger", snapshot_id)
    assert len(replay_files) == 3
    assert len({r.replay_id for r in records}) == 3  # distinct, never overwritten
    dispositions = {json.loads(f.read_text(encoding="utf-8"))["disposition"] for f in replay_files}
    assert dispositions == {"KEEP", "REDUCE_RISK"}


@pytest.mark.unit
def test_resolve_pm_replay_root_is_a_sibling_of_pre_pm(tmp_path):
    root = resolve_pm_replay_root({"ledger_root": str(tmp_path / "ledger")})
    assert root == tmp_path / "ledger" / "pm_replays"


@pytest.mark.unit
def test_list_pm_replays_empty_when_none_exist(tmp_path):
    assert list_pm_replays(tmp_path / "ledger", "no-such-snapshot") == []


# --- Model resolution: snapshot default vs. explicit override --------------


@pytest.mark.unit
def test_defaults_to_the_snapshot_s_own_generation_model(tmp_path, monkeypatch):
    path = _write_snapshot(tmp_path)
    seen_models = []

    def _capture(provider, model, **kwargs):
        seen_models.append(model)
        return _Client(_PMLLM())

    monkeypatch.setattr(pm_replay, "create_llm_client", _capture)
    record, _ = run_pm_replay(snapshot_path=path, ledger_root=tmp_path / "ledger")

    assert record.generation.name == "snapshot-gen-model"
    assert seen_models[0] == "snapshot-gen-model"


@pytest.mark.unit
def test_explicit_generation_model_override_wins(tmp_path, monkeypatch):
    path = _write_snapshot(tmp_path)
    monkeypatch.setattr(pm_replay, "create_llm_client", lambda **k: _Client(_PMLLM()))

    record, _ = run_pm_replay(
        snapshot_path=path, generation_model="override-model", ledger_root=tmp_path / "ledger",
    )

    assert record.generation.name == "override-model"


@pytest.mark.unit
def test_validation_enabled_without_any_validator_model_raises(tmp_path, monkeypatch):
    path = _write_snapshot(tmp_path, validator_llm=None)
    monkeypatch.setattr(pm_replay, "create_llm_client", lambda **k: _Client(_PMLLM()))

    with pytest.raises(ValueError, match="[Vv]alidat"):
        run_pm_replay(snapshot_path=path, validation_enabled=True, ledger_root=tmp_path / "ledger")


@pytest.mark.unit
def test_no_generation_model_anywhere_raises(tmp_path, monkeypatch):
    path = _write_snapshot(tmp_path, generation_llm=None)
    monkeypatch.setattr(pm_replay, "create_llm_client", lambda **k: _Client(_PMLLM()))

    with pytest.raises(ValueError, match="generation model"):
        run_pm_replay(snapshot_path=path, ledger_root=tmp_path / "ledger")
