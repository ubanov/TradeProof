"""Run provenance: an unconditional, JSON-safe snapshot of the metadata
needed to reproduce/compare one run (version, git, configured models/
reasoning-efforts/seeds, execution parameters, prompt-material hashes) --
the future metadata source for RecommendationRecord (see fork README: "Run
provenance"). Built once per run regardless of --trace-llm; trace mode's
own run_manifest.json reuses the SAME git/version/prompt-hash values so the
two can never diverge on the fields they share.

Numbered per the task's own "TESTS" list (1-20).
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import cli.run as cli_run
from cli.headless import run_single_analysis
from cli.stats_handler import StatsCallbackHandler
from tests.test_llm_trace_integration import _FakeTradingAgentsGraph
from tradingagents.llm_trace import hash_prompt_files, list_prompt_files
from tradingagents.run_provenance import build_run_provenance


@pytest.fixture
def fake_graph(monkeypatch):
    """Same fake graph as test_llm_trace_integration.py, reused by name here
    too (rather than imported) so pytest's fixture-by-parameter-name lookup
    doesn't collide with ruff's unused-import/redefinition checks."""
    monkeypatch.setattr(cli_run, "TradingAgentsGraph", _FakeTradingAgentsGraph)
    monkeypatch.setattr(
        cli_run, "build_analyst_execution_plan", lambda analysts: SimpleNamespace(specs=[])
    )
    return _FakeTradingAgentsGraph


def _run(tmp_path, *, trace_llm=False, **config_overrides):
    stats_handler = StatsCallbackHandler()
    config = {
        "max_risk_discuss_rounds": 3, "deep_think_llm": "A", "quick_think_llm": "B",
        "llm_provider": "ollama", "output_language": "Spanish",
    }
    config.update(config_overrides)
    return cli_run.execute_analysis(
        ticker="NVDA", analysis_date="2026-09-30", selected_analysts=["market", "social"],
        asset_type="stock", config=config,
        output_dir=tmp_path, stats_handler=stats_handler, trace_llm=trace_llm,
    )


# --- 1/2: provenance exists with trace on and off ---------------------------


@pytest.mark.unit
def test_1_provenance_exists_with_trace_disabled(fake_graph, tmp_path):
    execution = _run(tmp_path, trace_llm=False)
    assert execution.final_state["run_provenance"] is not None
    assert not (tmp_path / "debug").exists()


@pytest.mark.unit
def test_2_provenance_exists_with_trace_enabled(fake_graph, tmp_path):
    execution = _run(tmp_path, trace_llm=True)
    assert execution.final_state["run_provenance"] is not None
    assert (tmp_path / "debug" / "run_manifest.json").exists()


# --- 3: version is correct ---------------------------------------------------


@pytest.mark.unit
def test_3_version_is_correct(fake_graph, tmp_path):
    from tradingagents.llm_trace import _package_version

    execution = _run(tmp_path)
    assert execution.final_state["run_provenance"]["tradeproof_version"] == _package_version()


# --- 4: git commit/branch correct, or explicit UNKNOWN (null), never -------
# --- fabricated --------------------------------------------------------------


@pytest.mark.unit
def test_4_git_commit_branch_are_recorded_when_available():
    provenance = build_run_provenance(
        selected_analysts=["market"], config={},
        git_info={"commit": "abc123", "branch": "main", "dirty": False},
        package_version="9.9.9",
    )
    assert provenance["git"] == {"commit": "abc123", "branch": "main", "dirty": False}


@pytest.mark.unit
def test_4b_git_failure_yields_explicit_none_never_fabricated(monkeypatch):
    import tradingagents.run_provenance as rp

    monkeypatch.setattr(rp, "_git_info", lambda: {"commit": None, "branch": None, "dirty": None})
    provenance = build_run_provenance(selected_analysts=["market"], config={}, package_version="9.9.9")
    assert provenance["git"] == {"commit": None, "branch": None, "dirty": None}


# --- 5/6/7: generation/review/validator model + reasoning effort -----------


@pytest.mark.unit
def test_5_6_7_generation_review_validator_model_and_effort():
    config = {
        "llm_provider": "ollama",
        "generation_llm": "gen-model", "generation_reasoning_effort": "low",
        "review_llm": "review-model", "review_reasoning_effort": "max",
        "validator_llm": "validator-model", "validator_reasoning_effort": "medium",
        "validation_enabled": True,
    }
    provenance = build_run_provenance(
        selected_analysts=["market"], config=config,
        git_info={"commit": None, "branch": None, "dirty": None}, package_version="9.9.9",
    )
    models = provenance["models"]
    assert models["generation"] == {"name": "gen-model", "reasoning_effort": "low", "provider": "ollama"}
    assert models["review"] == {"name": "review-model", "reasoning_effort": "max", "provider": "ollama"}
    assert models["validator"]["name"] == "validator-model"
    assert models["validator"]["reasoning_effort"] == "medium"
    assert models["validator"]["provider"] == "ollama"


# --- 8: validation enabled/disabled is distinguishable, independent of -----
# --- whether a validator is configured --------------------------------------


@pytest.mark.unit
def test_8_validation_enabled_disabled_distinguishable_from_configured():
    base_config = {"llm_provider": "ollama", "validator_llm": "validator-model"}
    enabled = build_run_provenance(
        selected_analysts=["market"], config={**base_config, "validation_enabled": True},
        git_info={"commit": None, "branch": None, "dirty": None}, package_version="v",
    )
    disabled = build_run_provenance(
        selected_analysts=["market"], config={**base_config, "validation_enabled": False},
        git_info={"commit": None, "branch": None, "dirty": None}, package_version="v",
    )
    assert enabled["models"]["validator"]["validation_enabled"] is True
    assert disabled["models"]["validator"]["validation_enabled"] is False
    # Configured validator_llm survives even when validation is OFF -- never
    # collapsed into null just because it isn't being used this run.
    assert disabled["models"]["validator"]["name"] == "validator-model"


# --- 9: configured seed and base seed used are kept distinct ---------------


@pytest.mark.unit
def test_9_configured_seed_and_base_seed_are_differentiated():
    provenance = build_run_provenance(
        selected_analysts=["market"], config={},
        configured_seed=0, base_seed=184736291,
        git_info={"commit": None, "branch": None, "dirty": None}, package_version="v",
    )
    assert provenance["seed"]["configured"] == 0
    assert provenance["seed"]["base_used"] == 184736291


# --- 10/11/12: language, depth, selected analysts ---------------------------


@pytest.mark.unit
def test_10_11_12_language_depth_and_selected_analysts():
    provenance = build_run_provenance(
        selected_analysts=["market", "social", "news"],
        config={"output_language": "Spanish", "max_risk_discuss_rounds": 5},
        git_info={"commit": None, "branch": None, "dirty": None}, package_version="v",
    )
    assert provenance["execution"]["language"] == "Spanish"
    assert provenance["execution"]["depth"] == 5
    assert provenance["execution"]["selected_analysts"] == ["market", "social", "news"]


# --- 13: prompt hashes present without trace --------------------------------


@pytest.mark.unit
def test_13_prompt_hashes_present_without_trace(fake_graph, tmp_path):
    execution = _run(tmp_path, trace_llm=False)
    hashes = execution.final_state["run_provenance"]["prompts"]["hashes"]
    assert hashes  # non-empty
    assert "global_policy.txt" in hashes or any("global_policy" in k for k in hashes)


# --- 14: hashes are identical for identical prompt content -----------------


@pytest.mark.unit
def test_14_hashes_are_identical_for_the_same_prompt_content():
    first = hash_prompt_files()
    second = hash_prompt_files()
    assert first == second
    assert len(first) == len(list_prompt_files())


# --- 15: hash changes if the prompt content changes -------------------------


@pytest.mark.unit
def test_15_hash_changes_if_prompt_content_changes():
    import hashlib

    original = "You are a helpful assistant."
    changed = "You are a helpful assistant!!"  # one character different
    assert hashlib.sha256(original.encode("utf-8")).hexdigest() != hashlib.sha256(changed.encode("utf-8")).hexdigest()
    # And confirm hash_prompt_files genuinely hashes CONTENT, not just a path:
    # the same real file hashed by the function matches an independent,
    # direct hash of that file's own content.
    from importlib import resources

    hashes = hash_prompt_files()
    rel = next(iter(hashes))
    content = resources.files("tradingagents.prompts").joinpath(rel).read_text(encoding="utf-8")
    assert hashes[rel] == hashlib.sha256(content.encode("utf-8")).hexdigest()


# --- 16: result.json contains run_provenance --------------------------------


@pytest.mark.unit
def test_16_result_json_contains_run_provenance(fake_graph, tmp_path):
    result = run_single_analysis(
        ticker="NVDA", analysis_date="2026-09-30", analysts=["market"], depth_value=3,
        language="English", output_dir=tmp_path,
    )
    assert result["run_provenance"] is not None
    assert "models" in result["run_provenance"]
    json.dumps(result)  # JSON-safe end to end


# --- 17: a failed/partial run still carries provenance when possible -------


@pytest.mark.unit
def test_17_partial_run_still_carries_provenance(fake_graph, monkeypatch, tmp_path):
    class _BoomGraphCore:
        def stream(self, state, **kwargs):
            yield {"market_report": "partial market report"}
            raise RuntimeError("boom mid-run")

    class _BoomGraph(_FakeTradingAgentsGraph):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.graph = _BoomGraphCore()

    monkeypatch.setattr(cli_run, "TradingAgentsGraph", _BoomGraph)
    result = run_single_analysis(
        ticker="NVDA", analysis_date="2026-09-30", analysts=["market"], depth_value=3,
        language="English", output_dir=tmp_path,
    )
    assert result["status"] == "error"
    assert result["run_provenance"] is not None
    assert result["run_provenance"]["tradeproof_version"]


# --- 18: run_manifest and result provenance never diverge on shared fields -


@pytest.mark.unit
def test_18_run_manifest_and_result_provenance_agree_on_shared_fields(fake_graph, tmp_path):
    result = run_single_analysis(
        ticker="NVDA", analysis_date="2026-09-30", analysts=["market"], depth_value=3,
        language="English", output_dir=tmp_path, trace_llm=True,
    )
    manifest = json.loads((tmp_path / "debug" / "run_manifest.json").read_text())
    provenance = result["run_provenance"]
    assert manifest["package_version"] == provenance["tradeproof_version"]
    assert manifest["git"]["commit"] == provenance["git"]["commit"]
    assert manifest["git"]["branch"] == provenance["git"]["branch"]
    assert manifest["prompt_file_hashes"] == provenance["prompts"]["hashes"]
    assert provenance["trace"]["enabled"] is True
    assert provenance["trace"]["manifest_path"] is not None


# --- 19: structured state is unaffected -------------------------------------


@pytest.mark.unit
def test_19_structured_state_section_is_unaffected(fake_graph, tmp_path):
    result = run_single_analysis(
        ticker="NVDA", analysis_date="2026-09-30", analysts=["market"], depth_value=3,
        language="English", output_dir=tmp_path,
    )
    assert "structured" in result
    assert "source_coverage" in result
    assert "run_provenance" in result  # additive, coexists with both


# --- 20: the pipeline's final decision is unchanged -------------------------


@pytest.mark.unit
def test_20_final_decision_is_unchanged_by_provenance(fake_graph, tmp_path):
    result = run_single_analysis(
        ticker="NVDA", analysis_date="2026-09-30", analysts=["market"], depth_value=3,
        language="English", output_dir=tmp_path,
    )
    assert result["raw_final_decision"] == "**Rating**: Hold"
    assert result["rating_parse_status"] == "parsed"
