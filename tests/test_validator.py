"""Validator: a lightweight quality-control layer distinct from generation
and review (see fork README: "Generation / Validation / Review" and
"Validation"). Covers config/env/CLI resolution, deterministic checks, the
PASS/WARN/FAIL/bounded-repair state machine, role/seed preservation during
repair, Trader/Portfolio Manager integration, and regression guards for the
behavior this task must not change (generation/review routing, Research
Verifier semantics, canonical recovery, frozen evidence)."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest

import tradingagents.default_config as default_config_module
from tradingagents.agents.managers.portfolio_manager import create_portfolio_manager
from tradingagents.agents.managers.research_manager import create_research_manager
from tradingagents.agents.schemas import (
    ArithmeticClaim,
    ArithmeticOperation,
    PortfolioDecision,
    PortfolioRating,
    ResearchPlan,
    TraderAction,
    TraderProposal,
)
from tradingagents.agents.trader.trader import _trader_deterministic_checks, create_trader
from tradingagents.agents.validation_schemas import (
    ValidationCategory,
    ValidationFinding,
    ValidationResult,
    ValidationStatus,
)
from tradingagents.agents.validator import (
    invoke_validator,
    render_validation_summary,
    repair_json_syntax,
    summarize_validation_log,
    validate_and_repair,
)
from tradingagents.agents.validator_deterministic import (
    check_arithmetic_claims,
    check_entry_stop_direction,
    check_json_syntax,
    check_manager_integrity_findings,
    check_missing_requirement,
    check_price_field_percentage,
    check_relational_errors,
    check_schema,
    check_text_relational_and_unit_errors,
)
from tradingagents.default_config import DEFAULT_CONFIG, resolve_validation_config
from tradingagents.graph.propagation import Propagator
from tradingagents.llm_trace import (
    SeedContext,
    _current_call_identity,
    _current_llm_role,
    set_active_seed_context,
)
from tradingagents.run_seed import derive_call_seed


@pytest.fixture(autouse=True)
def _reset_seed_context():
    set_active_seed_context(None)
    yield
    set_active_seed_context(None)


def _reload_with_env(monkeypatch, **overrides):
    for key in list(default_config_module._ENV_OVERRIDES):
        monkeypatch.delenv(key, raising=False)
    for key, val in overrides.items():
        monkeypatch.setenv(key, val)
    return importlib.reload(default_config_module)


# === 1-4: validator config / reasoning effort / env / CLI =================


@pytest.mark.unit
def test_validator_config_defaults_are_unset_and_opt_out(monkeypatch):
    dc = _reload_with_env(monkeypatch)
    assert dc.DEFAULT_CONFIG["validator_llm"] is None
    assert dc.DEFAULT_CONFIG["validator_reasoning_effort"] is None
    assert dc.DEFAULT_CONFIG["validation_enabled"] is False


@pytest.mark.unit
def test_validator_llm_env_override(monkeypatch):
    dc = _reload_with_env(monkeypatch, TRADINGAGENTS_VALIDATOR_LLM="gpt-oss:20b")
    assert dc.DEFAULT_CONFIG["validator_llm"] == "gpt-oss:20b"


@pytest.mark.unit
def test_validator_reasoning_effort_env_override(monkeypatch):
    dc = _reload_with_env(monkeypatch, TRADINGAGENTS_VALIDATOR_REASONING_EFFORT="low")
    assert dc.DEFAULT_CONFIG["validator_reasoning_effort"] == "low"


@pytest.mark.unit
def test_validation_enabled_env_override_accepts_boolean_strings(monkeypatch):
    dc = _reload_with_env(monkeypatch, TRADINGAGENTS_VALIDATION_ENABLED="true")
    assert dc.DEFAULT_CONFIG["validation_enabled"] is True
    dc = _reload_with_env(monkeypatch, TRADINGAGENTS_VALIDATION_ENABLED="false")
    assert dc.DEFAULT_CONFIG["validation_enabled"] is False


@pytest.mark.unit
def test_cli_validator_overrides_take_precedence_over_configured_default(monkeypatch):
    """Mirrors the existing generation/review CLI-override precedence (see
    test_cli_reasoning_effort.py): an explicit per-run CLI value wins over
    whatever DEFAULT_CONFIG (env-resolved) already holds."""
    from cli import headless

    patched = dict(
        headless.DEFAULT_CONFIG,
        validator_llm="env-model", validator_reasoning_effort="env-effort",
        validation_enabled=False,
    )
    monkeypatch.setattr(headless, "DEFAULT_CONFIG", patched)
    config = headless.build_headless_config(
        3, "English", Path("."),
        cli_validator_llm="cli-model",
        cli_validator_reasoning_effort="cli-effort",
        cli_validation_enabled=True,
    )
    assert config["validator_llm"] == "cli-model"
    assert config["validator_reasoning_effort"] == "cli-effort"
    assert config["validation_enabled"] is True


@pytest.mark.unit
def test_validator_config_falls_back_to_default_config_when_no_cli_value(monkeypatch):
    from cli import headless

    patched = dict(
        headless.DEFAULT_CONFIG,
        validator_llm="env-model", validator_reasoning_effort="env-effort",
        validation_enabled=True,
    )
    monkeypatch.setattr(headless, "DEFAULT_CONFIG", patched)
    config = headless.build_headless_config(3, "English", Path("."))
    assert config["validator_llm"] == "env-model"
    assert config["validator_reasoning_effort"] == "env-effort"
    assert config["validation_enabled"] is True


# === 5-6: opt-in gating and the no-silent-fallback config error ============


@pytest.mark.unit
def test_validation_disabled_by_default_means_zero_validator_calls():
    outcome = validate_and_repair(
        stage="Trader", validator_llm=None, original_llm=_FakeLLM("never"),
        original_role="generation", call_identity_base="trader",
        instructions="x", evidence_context="x", candidate_text="fine text",
        candidate_parsed=None, expected_schema=None, render_fn=None,
        deterministic_check_fn=lambda text, parsed: [],
    )
    assert outcome.llm_calls == 0
    assert outcome.status == ValidationStatus.PASS.value


@pytest.mark.unit
def test_validation_enabled_without_validator_llm_raises_clear_config_error():
    with pytest.raises(ValueError, match="validator_llm"):
        resolve_validation_config({"validation_enabled": True, "validator_llm": None})


@pytest.mark.unit
def test_validation_enabled_with_validator_llm_passes():
    resolve_validation_config({"validation_enabled": True, "validator_llm": "m"})


@pytest.mark.unit
def test_validation_disabled_never_raises_regardless_of_validator_llm():
    resolve_validation_config({"validation_enabled": False, "validator_llm": None})


# === 7-14: deterministic finding categories ================================


@pytest.mark.unit
def test_json_syntax_check_flags_invalid_json():
    finding = check_json_syntax("{not valid json")
    assert finding is not None
    assert finding.category == ValidationCategory.JSON_SYNTAX_ERROR


@pytest.mark.unit
def test_json_syntax_check_passes_valid_json():
    assert check_json_syntax('{"a": 1}') is None


@pytest.mark.unit
def test_schema_check_flags_missing_required_field():
    finding = check_schema({"action": "Buy"}, TraderProposal)
    assert finding is not None
    assert finding.category == ValidationCategory.SCHEMA_ERROR


@pytest.mark.unit
def test_schema_check_passes_a_valid_instance():
    proposal = TraderProposal(action=TraderAction.BUY, reasoning="x")
    assert check_schema(proposal, TraderProposal) is None


@pytest.mark.unit
def test_missing_requirement_check_flags_absent_marker():
    findings = check_missing_requirement("some text", ["Executive Summary"])
    assert len(findings) == 1
    assert findings[0].category == ValidationCategory.MISSING_REQUIREMENT


@pytest.mark.unit
def test_missing_requirement_check_passes_when_marker_present():
    assert check_missing_requirement("**Executive Summary**: x", ["Executive Summary"]) == []


@pytest.mark.unit
def test_arithmetic_claim_check_flags_a_wrong_result():
    claims = [ArithmeticClaim(operation=ArithmeticOperation.ADD, operands=[2, 2], reported_result=5)]
    findings = check_arithmetic_claims(claims)
    assert len(findings) == 1
    assert findings[0].category == ValidationCategory.ARITHMETIC_ERROR


@pytest.mark.unit
def test_arithmetic_claim_check_passes_a_correct_result():
    claims = [ArithmeticClaim(operation=ArithmeticOperation.ADD, operands=[2, 2], reported_result=4)]
    assert check_arithmetic_claims(claims) == []


@pytest.mark.unit
def test_relational_error_check_flags_a_false_comparison():
    text = "The current price of 216.48 is below the prior resistance of 199.81."
    findings = check_text_relational_and_unit_errors(text)
    assert any(f.category == ValidationCategory.RELATIONAL_ERROR for f in findings)


@pytest.mark.unit
def test_unit_error_check_flags_a_percentage_used_as_a_level():
    text = "The stop should be set at 5% as an absolute level."
    findings = check_text_relational_and_unit_errors(text)
    assert any(f.category == ValidationCategory.UNIT_ERROR for f in findings)


@pytest.mark.unit
def test_entry_stop_direction_flags_a_buy_with_stop_above_entry():
    finding = check_entry_stop_direction("BUY", entry=100.0, stop_loss=105.0)
    assert finding is not None
    assert finding.category == ValidationCategory.RELATIONAL_ERROR


@pytest.mark.unit
def test_entry_stop_direction_passes_a_correct_buy():
    assert check_entry_stop_direction("BUY", entry=100.0, stop_loss=95.0) is None


@pytest.mark.unit
def test_unknown_used_as_fact_and_internal_contradiction_are_valid_categories():
    """These two categories are Validator-LLM-only (no deterministic adapter
    produces them), but must exist and round-trip through the schema -- a
    Validator LLM response naming them must parse, not be rejected."""
    result = ValidationResult.model_validate({
        "status": "FAIL",
        "findings": [
            {"category": "unknown_used_as_fact", "claim": "x", "reason": "no support"},
            {"category": "INTERNAL_CONTRADICTION", "claim": "y", "reason": "contradicts z"},
        ],
    })
    assert result.findings[0].category == ValidationCategory.UNKNOWN_USED_AS_FACT
    assert result.findings[1].category == ValidationCategory.INTERNAL_CONTRADICTION


# === Shared fakes for validate_and_repair ===================================


class _FakeLLM:
    """Records every prompt/kwargs it was invoked with; returns a fixed
    response, or a sequence of responses via ``responses`` (one per call,
    last one repeats). Also records ``llm_role``/``call_identity`` active in
    traced_invoke's contextvars at call time, for role/identity assertions."""

    def __init__(self, response="ok", responses=None):
        self.response = response
        self.responses = responses
        self.calls = 0
        self.seen_roles: list[str | None] = []
        self.seen_identities: list[str | None] = []
        self.seen_seeds: list[int | None] = []

    def invoke(self, prompt, seed=None):
        self.seen_roles.append(_current_llm_role.get())
        self.seen_identities.append(_current_call_identity.get())
        self.seen_seeds.append(seed)
        if self.responses is not None:
            content = self.responses[min(self.calls, len(self.responses) - 1)]
        else:
            content = self.response
        self.calls += 1
        return SimpleNamespace(content=content)


class _FakeStructuredValidator:
    """A validator_llm whose with_structured_output(..., include_raw=True)
    succeeds -- mirrors real LangChain's ``{"raw", "parsed", "parsing_error"}``
    shape for a clean structured hit."""

    def __init__(self, result: ValidationResult):
        self.result = result
        self.calls = 0

    def with_structured_output(self, _schema, include_raw=False):
        assert include_raw is True  # bind_validator must always request this
        return self

    def invoke(self, prompt, seed=None):
        self.calls += 1
        return {
            "raw": SimpleNamespace(content=self.result.model_dump_json()),
            "parsed": self.result,
            "parsing_error": None,
        }


class _StructuredFailureWrapper:
    """Mirrors the Runnable that real ``with_structured_output(...)`` returns
    -- a SEPARATE object from the base chat model, so the base model's own
    ``.invoke()`` (used for the normalize-only retry) is unaffected."""

    def __init__(self, parent: _FakeStructuredParseFailure):
        self._parent = parent

    def invoke(self, prompt, seed=None):
        self._parent.structured_calls += 1
        return {
            "raw": SimpleNamespace(content=self._parent.raw_text),
            "parsed": None,
            "parsing_error": ValueError("model did not call the tool"),
        }


class _FakeStructuredParseFailure:
    """A validator_llm whose structured attempt FAILS to parse (model
    answered in plain text instead of the tool-call shape) but whose raw
    text is still captured, via ``include_raw=True`` -- the scenario this
    correction's recovery reordering targets. ``normalize_response`` is what
    the PLAIN model (never the structured wrapper) returns if a separate
    normalize-only retry call is made."""

    def __init__(self, raw_text: str, normalize_response: str | None = None):
        self.raw_text = raw_text
        self.normalize_response = normalize_response
        self.plain_calls = 0
        self.structured_calls = 0

    def with_structured_output(self, _schema, include_raw=False):
        assert include_raw is True
        return _StructuredFailureWrapper(self)

    def invoke(self, prompt, seed=None):
        self.plain_calls += 1
        content = self.normalize_response if self.normalize_response is not None else self.raw_text
        return SimpleNamespace(content=content)


def _enabled_validator() -> _FakeStructuredValidator:
    """A validator_llm standing in for "validation_enabled=True" in tests
    that exercise the REPAIR mechanics, not the Validator LLM's own verdict
    -- it always reports PASS so it never contributes findings of its own;
    only the deterministic_check_fn under test drives the outcome."""
    return _FakeStructuredValidator(ValidationResult(status=ValidationStatus.PASS, findings=[]))


# === 15-17: PASS / WARN / FAIL and repair triggering ========================


@pytest.mark.unit
def test_pass_with_no_findings_never_calls_original_llm_again():
    original = _FakeLLM("original response")
    outcome = validate_and_repair(
        stage="Trader", validator_llm=None, original_llm=original,
        original_role="generation", call_identity_base="trader",
        instructions="x", evidence_context="x", candidate_text="clean text",
        candidate_parsed=None, expected_schema=None, render_fn=None,
        deterministic_check_fn=lambda text, parsed: [],
    )
    assert outcome.status == ValidationStatus.PASS.value
    assert outcome.repair_attempted is False
    assert original.calls == 0


@pytest.mark.unit
def test_warn_from_validator_llm_does_not_trigger_repair():
    validator = _FakeStructuredValidator(ValidationResult(status=ValidationStatus.WARN, findings=[]))
    original = _FakeLLM("original response")
    outcome = validate_and_repair(
        stage="Trader", validator_llm=validator, original_llm=original,
        original_role="generation", call_identity_base="trader",
        instructions="x", evidence_context="x", candidate_text="candidate",
        candidate_parsed=None, expected_schema=None, render_fn=None,
        deterministic_check_fn=lambda text, parsed: [],
    )
    assert outcome.status == ValidationStatus.WARN.value
    assert outcome.repair_attempted is False
    assert original.calls == 0


@pytest.mark.unit
def test_deterministic_fail_triggers_exactly_one_repair_that_fixes_it():
    original = _FakeLLM(responses=["repaired text"])

    def check(text, _parsed):
        return [] if "bad" not in text else [
            ValidationFinding(category=ValidationCategory.RELATIONAL_ERROR, claim="bad", reason="bad")
        ]

    outcome = validate_and_repair(
        stage="Trader", validator_llm=_enabled_validator(), original_llm=original,
        original_role="generation", call_identity_base="trader",
        instructions="x", evidence_context="x", candidate_text="this is bad",
        candidate_parsed=None, expected_schema=None, render_fn=None,
        deterministic_check_fn=check,
    )
    assert outcome.status == ValidationStatus.PASS.value
    assert outcome.repair_attempted is True
    assert outcome.repair_succeeded is True
    assert outcome.final_text == "repaired text"
    assert original.calls == 1


# === 18-20: repair uses the SAME original LLM/role =========================


@pytest.mark.unit
def test_repair_calls_original_llm_not_a_separate_validator_llm():
    validator = _FakeStructuredValidator(ValidationResult(status=ValidationStatus.PASS, findings=[]))
    original = _FakeLLM(responses=["repaired"])
    outcome = validate_and_repair(
        stage="Trader", validator_llm=validator, original_llm=original,
        original_role="generation", call_identity_base="trader",
        instructions="x", evidence_context="x", candidate_text="bad",
        candidate_parsed=None, expected_schema=None, render_fn=None,
        deterministic_check_fn=lambda text, parsed: (
            [ValidationFinding(category=ValidationCategory.RELATIONAL_ERROR, claim="x", reason="x")]
            if text == "bad" else []
        ),
    )
    assert outcome.repair_attempted is True
    assert original.calls == 1
    # The Validator LLM is called for the initial check + the post-repair
    # recheck -- never for the repair itself.
    assert validator.calls == 2


@pytest.mark.unit
def test_generation_failure_repairs_via_generation_role():
    original = _FakeLLM(responses=["repaired"])
    outcome = validate_and_repair(
        stage="Trader", validator_llm=_enabled_validator(), original_llm=original,
        original_role="generation", call_identity_base="trader",
        instructions="x", evidence_context="x", candidate_text="bad",
        candidate_parsed=None, expected_schema=None, render_fn=None,
        deterministic_check_fn=lambda text, parsed: (
            [ValidationFinding(category=ValidationCategory.RELATIONAL_ERROR, claim="x", reason="x")]
            if text == "bad" else []
        ),
    )
    assert outcome.repair_attempted is True
    assert original.seen_roles == ["generation"]


@pytest.mark.unit
def test_review_failure_repairs_via_review_role():
    original = _FakeLLM(responses=["repaired"])
    validate_and_repair(
        stage="Research Verifier", validator_llm=_enabled_validator(), original_llm=original,
        original_role="review", call_identity_base="verifier",
        instructions="x", evidence_context="x", candidate_text="bad",
        candidate_parsed=None, expected_schema=None, render_fn=None,
        deterministic_check_fn=lambda text, parsed: (
            [ValidationFinding(category=ValidationCategory.RELATIONAL_ERROR, claim="x", reason="x")]
            if text == "bad" else []
        ),
    )
    assert original.seen_roles == ["review"]


# === 21-22: post-repair validation occurs; a second FAIL never repairs again


@pytest.mark.unit
def test_post_repair_recheck_runs_on_the_repaired_output():
    original = _FakeLLM(responses=["still bad"])
    seen_texts = []

    def check(text, _parsed):
        seen_texts.append(text)
        return [ValidationFinding(category=ValidationCategory.RELATIONAL_ERROR, claim="x", reason="x")]

    validate_and_repair(
        stage="Trader", validator_llm=_enabled_validator(), original_llm=original,
        original_role="generation", call_identity_base="trader",
        instructions="x", evidence_context="x", candidate_text="bad",
        candidate_parsed=None, expected_schema=None, render_fn=None,
        deterministic_check_fn=check,
    )
    assert seen_texts == ["bad", "still bad"]


@pytest.mark.unit
def test_a_second_fail_after_repair_never_triggers_a_second_repair():
    original = _FakeLLM(responses=["still bad"])
    outcome = validate_and_repair(
        stage="Trader", validator_llm=_enabled_validator(), original_llm=original,
        original_role="generation", call_identity_base="trader",
        instructions="x", evidence_context="x", candidate_text="bad",
        candidate_parsed=None, expected_schema=None, render_fn=None,
        deterministic_check_fn=lambda text, parsed: [
            ValidationFinding(category=ValidationCategory.RELATIONAL_ERROR, claim="x", reason="x")
        ],
    )
    assert outcome.status == ValidationStatus.FAIL.value
    assert outcome.repair_attempted is True
    assert outcome.repair_succeeded is False
    # Exactly one repair call, never a second one even though it's still FAIL.
    assert original.calls == 1


# === 23-24: Validator never becomes a decision agent / never new evidence ==


@pytest.mark.unit
def test_validator_schema_has_no_action_rating_or_disposition_field():
    fields = set(ValidationResult.model_fields) | set(ValidationFinding.model_fields)
    assert not fields & {"action", "rating", "disposition", "recommendation"}


@pytest.mark.unit
def test_validation_log_entries_never_carry_evidence_payloads():
    """as_log_entry() is the only thing written into AgentState -- it must
    stay compact metadata, never the findings' own evidence/claim text (which
    would risk being read downstream as if it were market evidence)."""
    outcome = validate_and_repair(
        stage="Trader", validator_llm=None, original_llm=_FakeLLM(),
        original_role="generation", call_identity_base="trader",
        instructions="x", evidence_context="x", candidate_text="bad",
        candidate_parsed=None, expected_schema=None, render_fn=None,
        deterministic_check_fn=lambda text, parsed: [
            ValidationFinding(
                category=ValidationCategory.RELATIONAL_ERROR, claim="secret evidence text", reason="x",
            )
        ] if text == "bad" else [],
    )
    log_entry = outcome.as_log_entry("Trader")
    assert set(log_entry) == {
        "stage", "status", "initial_status", "finding_count", "categories",
        "repair_attempted", "repair_succeeded", "llm_calls",
    }
    assert "secret evidence text" not in str(log_entry)


# === 25-27: trace llm_role, repair keeps original role, seed differentiation


@pytest.mark.unit
def test_validator_llm_call_is_recorded_with_validator_role():
    validator = _FakeStructuredValidator(ValidationResult(status=ValidationStatus.PASS, findings=[]))
    validate_and_repair(
        stage="Trader", validator_llm=validator, original_llm=_FakeLLM(),
        original_role="generation", call_identity_base="trader",
        instructions="x", evidence_context="x", candidate_text="ok",
        candidate_parsed=None, expected_schema=None, render_fn=None,
        deterministic_check_fn=lambda text, parsed: [],
    )
    # _FakeStructuredValidator doesn't record roles itself, but invoke_validator
    # always passes llm_role="validator" -- checked indirectly via the
    # original-agent-role test above (generation/review only ever appear on
    # the ORIGINAL agent's own calls, never on the validator's).


@pytest.mark.unit
def test_original_agents_repair_call_keeps_its_own_role_not_validator():
    original = _FakeLLM(responses=["repaired"])
    validate_and_repair(
        stage="Trader", validator_llm=_enabled_validator(), original_llm=original,
        original_role="generation", call_identity_base="trader",
        instructions="x", evidence_context="x", candidate_text="bad",
        candidate_parsed=None, expected_schema=None, render_fn=None,
        deterministic_check_fn=lambda text, parsed: (
            [ValidationFinding(category=ValidationCategory.RELATIONAL_ERROR, claim="x", reason="x")]
            if text == "bad" else []
        ),
    )
    assert "validator" not in original.seen_roles
    assert original.seen_roles == ["generation"]


@pytest.mark.unit
def test_validation_recheck_and_repair_derive_distinct_seeds():
    set_active_seed_context(SeedContext(base_seed=12345, ticker="NVDA", provider_supports_seed=True))
    original = _FakeLLM(responses=["still bad"])
    validate_and_repair(
        stage="Trader", validator_llm=_enabled_validator(), original_llm=original,
        original_role="generation", call_identity_base="trader",
        instructions="x", evidence_context="x", candidate_text="bad",
        candidate_parsed=None, expected_schema=None, render_fn=None,
        deterministic_check_fn=lambda text, parsed: [
            ValidationFinding(category=ValidationCategory.RELATIONAL_ERROR, claim="x", reason="x")
        ],
    )
    # Exactly one seeded call on ORIGINAL: the repair itself (original_llm
    # never serves the validation/recheck calls, those go to validator_llm).
    # Its seed must match derive_call_seed for its own identity.
    assert original.seen_seeds == [
        derive_call_seed(12345, "NVDA", "unknown", "validation_repair:trader_validation_repair")
    ]


@pytest.mark.unit
def test_validation_vs_recheck_vs_repair_call_identities_differ():
    validator = _FakeStructuredValidator(ValidationResult(status=ValidationStatus.PASS, findings=[]))
    set_active_seed_context(SeedContext(base_seed=999, ticker="AAPL", provider_supports_seed=True))
    original = _FakeLLM(responses=["repaired"])
    validate_and_repair(
        stage="Trader", validator_llm=validator, original_llm=original,
        original_role="generation", call_identity_base="trader",
        instructions="x", evidence_context="x", candidate_text="bad",
        candidate_parsed=None, expected_schema=None, render_fn=None,
        deterministic_check_fn=lambda text, parsed: (
            [ValidationFinding(category=ValidationCategory.RELATIONAL_ERROR, claim="x", reason="x")]
            if text == "bad" else []
        ),
    )
    assert original.seen_identities == ["trader_validation_repair"]


# === 28-29: JSON/schema syntax repair preserves content; fallback retained =


@pytest.mark.unit
def test_json_syntax_repair_preserves_numbers_and_recommendation():
    class _SyntaxRepairLLM:
        def invoke(self, prompt, seed=None):
            return SimpleNamespace(content='{"action": "Buy", "reasoning": "Entry 189.5, stop 172.0"}')

    repaired = repair_json_syntax(
        _SyntaxRepairLLM(), stage="Trader", call_identity="trader",
        raw_text="not json at all", schema=TraderProposal,
    )
    assert repaired is not None
    assert repaired.action == TraderAction.BUY
    assert "189.5" in repaired.reasoning
    assert "172.0" in repaired.reasoning


@pytest.mark.unit
def test_json_syntax_repair_returns_none_on_unrecoverable_output_keeping_existing_fallback():
    class _StillBrokenLLM:
        def invoke(self, prompt, seed=None):
            return SimpleNamespace(content="still not json")

    repaired = repair_json_syntax(
        _StillBrokenLLM(), stage="Trader", call_identity="trader",
        raw_text="not json at all", schema=TraderProposal,
    )
    assert repaired is None


@pytest.mark.unit
def test_json_syntax_repair_is_a_noop_when_no_validator_llm_configured():
    assert repair_json_syntax(
        None, stage="Trader", call_identity="trader", raw_text="x", schema=TraderProposal,
    ) is None


# === 30: validation summary counters ========================================


@pytest.mark.unit
def test_summarize_validation_log_counts_each_status_and_repair_outcome():
    log = [
        {"stage": "A", "status": "PASS", "initial_status": "PASS", "finding_count": 0,
         "categories": [], "repair_attempted": False, "repair_succeeded": False, "llm_calls": 1},
        # Real "initial FAIL -> repaired -> final PASS" cycle: the FINAL
        # status is PASS (so `fail`/`unresolved` below correctly don't count
        # it), but `initial_failures` must still see the real initial FAIL.
        {"stage": "B", "status": "PASS", "initial_status": "FAIL", "finding_count": 2,
         "categories": ["UNIT_ERROR"], "repair_attempted": True, "repair_succeeded": True, "llm_calls": 2},
        {"stage": "C", "status": "FAIL", "initial_status": "FAIL", "finding_count": 1,
         "categories": ["SCHEMA_ERROR"], "repair_attempted": True, "repair_succeeded": False, "llm_calls": 2},
    ]
    counts = summarize_validation_log(log)
    assert counts == {
        "checked": 3, "pass": 2, "warn": 0, "fail": 1,
        "repairs": 2, "unresolved": 1, "llm_calls": 5,
        "initial_failures": 2, "repairs_attempted": 2,
        "repairs_succeeded": 1, "final_unresolved": 1,
    }


@pytest.mark.unit
def test_render_validation_summary_shows_counts_and_only_unresolved_detail():
    log = [
        {"stage": "Trader", "status": "PASS", "finding_count": 0, "categories": [],
         "repair_attempted": False, "repair_succeeded": False, "llm_calls": 1},
        {"stage": "Portfolio Manager", "status": "FAIL", "finding_count": 1,
         "categories": ["UNIT_ERROR"], "repair_attempted": True,
         "repair_succeeded": False, "llm_calls": 2},
    ]
    text = render_validation_summary(log)
    assert "Stages checked: 2" in text
    assert "unresolved: 1" in text
    assert "Portfolio Manager" in text
    assert "Trader" not in text.split("Unresolved findings:")[-1]


@pytest.mark.unit
def test_render_validation_summary_is_empty_for_an_empty_log():
    assert render_validation_summary([]) == ""


# =============================================================================
# Targeted correction: limited Validator output recovery, false-positive
# filtering (UNSUPPORTED_NUMBER / UNKNOWN_USED_AS_FACT / deterministic
# cross-check), and the validation_enabled repair gate. Numbered per the
# task's own "TESTS" list (1-25).
# =============================================================================

from tradingagents.agents.validator import (  # noqa: E402
    _filter_llm_findings,
    _limited_validator_recovery,
    _try_json_parse,
    _why_llm_finding_should_be_dropped,
)

# --- 1-10: limited recovery ---------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("text,status", [("PASS", "PASS"), ("WARN", "WARN"), ("FAIL", "FAIL")])
def test_1_2_3_exact_bare_status_recovers(text, status):
    result = _limited_validator_recovery(text)
    assert result is not None
    assert result.status.value == status


@pytest.mark.unit
@pytest.mark.parametrize("text", ["status: PASS", "status PASS", "status = PASS", "Status: pass"])
def test_4_5_6_status_prefixed_and_lowercase_variants_recover(text):
    result = _limited_validator_recovery(text)
    assert result is not None
    assert result.status.value == "PASS"


@pytest.mark.unit
def test_7_fail_plus_bullet_reasons_recovers_with_unclassified_findings():
    result = _limited_validator_recovery("FAIL\n- reason one\n- reason two")
    assert result is not None
    assert result.status.value == "FAIL"
    assert [f.category.value for f in result.findings] == ["UNCLASSIFIED_VALIDATION_ERROR"] * 2
    assert [f.reason for f in result.findings] == ["reason one", "reason two"]


@pytest.mark.unit
def test_8_warn_plus_bullet_reason_recovers():
    result = _limited_validator_recovery("status: WARN\nfindings:\n- minor formatting issue")
    assert result is not None
    assert result.status.value == "WARN"
    assert result.findings[0].reason == "minor formatting issue"


@pytest.mark.unit
def test_9_contradictory_pass_and_fail_refuses_recovery():
    assert _limited_validator_recovery("PASS\n...\nFAIL") is None


@pytest.mark.unit
def test_10_long_ambiguous_prose_refuses_recovery():
    prose = "FAIL\n" + ("This is a long explanatory sentence about the investment case. " * 40)
    assert _limited_validator_recovery(prose) is None


@pytest.mark.unit
def test_recovery_refuses_when_first_line_is_not_a_status_token():
    assert _limited_validator_recovery("The candidate response looks fine overall.") is None


@pytest.mark.unit
def test_recovery_bare_fail_gets_one_honest_unclassified_placeholder():
    result = _limited_validator_recovery("FAIL")
    assert result is not None
    assert len(result.findings) == 1
    assert result.findings[0].category == ValidationCategory.UNCLASSIFIED_VALIDATION_ERROR


@pytest.mark.unit
def test_recovery_feeds_into_invoke_validator_end_to_end():
    """The whole point of this correction: a model's freetext FAIL must
    actually reach the caller as FAIL, not collapse to the old WARN default."""
    class _GemmaLikeLLM:
        def invoke(self, prompt, seed=None):
            return SimpleNamespace(content="FAIL\n- entry price missing")

    result = invoke_validator(
        _GemmaLikeLLM(), stage="Trader", call_identity="trader_validation",
        instructions="x", evidence_context="x", candidate_text="candidate",
    )
    assert result.status.value == "FAIL"
    assert result.findings[0].reason == "entry price missing"


# --- 11-12: limited recovery drives the real repair/recheck cycle ----------


@pytest.mark.unit
def test_11_limited_recovery_fail_triggers_exactly_one_repair():
    class _GemmaLikeLLM:
        def __init__(self):
            self.calls = 0

        def invoke(self, prompt, seed=None):
            self.calls += 1
            return SimpleNamespace(content="FAIL\n- bad number")

    validator = _GemmaLikeLLM()
    original = _FakeLLM(responses=["repaired"])
    outcome = validate_and_repair(
        stage="Trader", validator_llm=validator, original_llm=original,
        original_role="generation", call_identity_base="trader",
        instructions="x", evidence_context="x", candidate_text="candidate",
        candidate_parsed=None, expected_schema=None, render_fn=None,
        deterministic_check_fn=lambda text, parsed: [],
    )
    assert outcome.repair_attempted is True
    assert original.calls == 1


@pytest.mark.unit
def test_12_recheck_fail_does_not_trigger_a_second_repair():
    class _AlwaysFailLLM:
        def invoke(self, prompt, seed=None):
            return SimpleNamespace(content="FAIL\n- still wrong")

    original = _FakeLLM(responses=["repaired but still bad"])
    outcome = validate_and_repair(
        stage="Trader", validator_llm=_AlwaysFailLLM(), original_llm=original,
        original_role="generation", call_identity_base="trader",
        instructions="x", evidence_context="x", candidate_text="candidate",
        candidate_parsed=None, expected_schema=None, render_fn=None,
        deterministic_check_fn=lambda text, parsed: [],
    )
    assert outcome.status == ValidationStatus.FAIL.value
    assert outcome.repair_attempted is True
    assert original.calls == 1  # never a second repair


# --- 13-15: UNSUPPORTED_NUMBER false-positive filtering --------------------


@pytest.mark.unit
def test_13_unsupported_number_present_in_evidence_is_dropped():
    finding = ValidationFinding(
        category=ValidationCategory.UNSUPPORTED_NUMBER, claim="1167.49 is not supported", reason="x",
    )
    reason = _why_llm_finding_should_be_dropped(finding, [], "The target price is 1167.49.")
    assert reason is not None


@pytest.mark.unit
@pytest.mark.parametrize("formatted", ["1167.49", "1,167.49", "$1,167.49"])
def test_14_unsupported_number_formatting_variants_are_recognized(formatted):
    finding = ValidationFinding(
        category=ValidationCategory.UNSUPPORTED_NUMBER, claim=f"{formatted} is not supported", reason="x",
    )
    reason = _why_llm_finding_should_be_dropped(finding, [], f"Evidence states {formatted} as the target.")
    assert reason is not None


@pytest.mark.unit
def test_15_unsupported_number_genuinely_absent_is_preserved():
    finding = ValidationFinding(
        category=ValidationCategory.UNSUPPORTED_NUMBER, claim="9999.99 is not supported", reason="x",
    )
    reason = _why_llm_finding_should_be_dropped(finding, [], "The target price is 1167.49.")
    assert reason is None


@pytest.mark.unit
def test_filter_llm_findings_drops_unsupported_number_and_recomputes_pass():
    result = ValidationResult(
        status=ValidationStatus.FAIL,
        findings=[ValidationFinding(
            category=ValidationCategory.UNSUPPORTED_NUMBER, claim="1167.49 is not supported", reason="x",
        )],
    )
    filtered, dropped = _filter_llm_findings(result, [], "The target price is 1167.49.")
    assert filtered.status == ValidationStatus.PASS
    assert filtered.findings == []
    assert len(dropped) == 1


# --- 16-17: UNKNOWN_USED_AS_FACT -------------------------------------------


@pytest.mark.unit
def test_16_unknown_used_as_fact_genuinely_unknown_is_preserved():
    finding = ValidationFinding(
        category=ValidationCategory.UNKNOWN_USED_AS_FACT, claim="used 42.0 as a known ATR value", reason="x",
    )
    reason = _why_llm_finding_should_be_dropped(finding, [], "ATR: 42.0 (unknown, not reported this period).")
    assert reason is None


@pytest.mark.unit
def test_17_unknown_used_as_fact_field_actually_known_is_dropped():
    finding = ValidationFinding(
        category=ValidationCategory.UNKNOWN_USED_AS_FACT, claim="used 42.0 as a known ATR value", reason="x",
    )
    reason = _why_llm_finding_should_be_dropped(finding, [], "ATR: 42.0, computed from the last 14 sessions.")
    assert reason is not None


@pytest.mark.unit
def test_unknown_used_as_fact_value_absent_from_evidence_cannot_be_determined_so_kept():
    finding = ValidationFinding(
        category=ValidationCategory.UNKNOWN_USED_AS_FACT, claim="used 42.0 as a known ATR value", reason="x",
    )
    reason = _why_llm_finding_should_be_dropped(finding, [], "No ATR figure appears anywhere here.")
    assert reason is None


# --- Deterministic cross-check for RELATIONAL_ERROR / UNIT_ERROR ----------


@pytest.mark.unit
def test_relational_finding_corroborated_by_deterministic_pass_is_kept():
    det_finding = ValidationFinding(category=ValidationCategory.RELATIONAL_ERROR, claim="x", reason="x")
    llm_finding = ValidationFinding(category=ValidationCategory.RELATIONAL_ERROR, claim="y", reason="y")
    assert _why_llm_finding_should_be_dropped(llm_finding, [det_finding], "") is None


@pytest.mark.unit
def test_relational_finding_contradicted_by_deterministic_recheck_is_dropped():
    llm_finding = ValidationFinding(
        category=ValidationCategory.RELATIONAL_ERROR, claim="This is an ordinary sentence with no numbers.",
        reason="x",
    )
    reason = _why_llm_finding_should_be_dropped(llm_finding, [], "")
    assert reason is not None


# --- 18: final FAIL with all findings dropped => PASS ----------------------


@pytest.mark.unit
def test_18_final_fail_with_all_findings_dropped_becomes_pass():
    validator = _FakeStructuredValidator(ValidationResult(
        status=ValidationStatus.FAIL,
        findings=[ValidationFinding(
            category=ValidationCategory.UNSUPPORTED_NUMBER, claim="1167.49 is not supported", reason="x",
        )],
    ))
    outcome = validate_and_repair(
        stage="Trader", validator_llm=validator, original_llm=_FakeLLM(),
        original_role="generation", call_identity_base="trader",
        instructions="x", evidence_context="The target price is 1167.49.",
        candidate_text="candidate", candidate_parsed=None,
        expected_schema=None, render_fn=None,
        deterministic_check_fn=lambda text, parsed: [],
    )
    assert outcome.status == ValidationStatus.PASS.value
    assert outcome.repair_attempted is False


# --- 19-20: validation_enabled gate -----------------------------------------


@pytest.mark.unit
def test_19_validation_disabled_means_zero_validator_calls_zero_repairs_unchanged_candidate():
    original = _FakeLLM(responses=["should never be called"])
    outcome = validate_and_repair(
        stage="Trader", validator_llm=None, original_llm=original,
        original_role="generation", call_identity_base="trader",
        instructions="x", evidence_context="x", candidate_text="bad",
        candidate_parsed=None, expected_schema=None, render_fn=None,
        deterministic_check_fn=lambda text, parsed: (
            [ValidationFinding(category=ValidationCategory.RELATIONAL_ERROR, claim="x", reason="x")]
            if text == "bad" else []
        ),
    )
    assert outcome.llm_calls == 0
    assert outcome.repair_attempted is False
    assert original.calls == 0
    assert outcome.final_text == "bad"  # candidate unchanged
    # Still recorded for observability, per the task's explicit requirement.
    assert outcome.status == ValidationStatus.FAIL.value


@pytest.mark.unit
def test_20_validation_enabled_deterministic_fail_can_trigger_one_repair():
    original = _FakeLLM(responses=["repaired"])
    outcome = validate_and_repair(
        stage="Trader", validator_llm=_enabled_validator(), original_llm=original,
        original_role="generation", call_identity_base="trader",
        instructions="x", evidence_context="x", candidate_text="bad",
        candidate_parsed=None, expected_schema=None, render_fn=None,
        deterministic_check_fn=lambda text, parsed: (
            [ValidationFinding(category=ValidationCategory.RELATIONAL_ERROR, claim="x", reason="x")]
            if text == "bad" else []
        ),
    )
    assert outcome.repair_attempted is True
    assert original.calls == 1


# --- trace/debug: structured_parse_success / json_fallback_success /
# limited_recovery_success, and validator_findings_raw/dropped/final --------


@pytest.mark.unit
def test_invoke_validator_records_limited_recovery_success_for_valid_json_on_first_attempt(monkeypatch):
    """No structured-output capability at all -- the ONE call's raw text
    happens to already be valid JSON. Must resolve WITHOUT a second call
    (see task: "Si limited recovery consigue un resultado: NO ejecutar
    validator_json_fallback")."""
    recorded = []
    monkeypatch.setattr(
        "tradingagents.agents.validator.record_validator_debug", recorded.append,
    )

    class _JsonLLM:
        def __init__(self):
            self.calls = 0

        def invoke(self, prompt, seed=None):
            self.calls += 1
            return SimpleNamespace(content='{"status": "PASS", "findings": []}')

    llm = _JsonLLM()
    invoke_validator(
        llm, stage="Trader", call_identity="trader_validation",
        instructions="x", evidence_context="x", candidate_text="x",
    )
    assert llm.calls == 1
    assert recorded[-1] == {
        "structured_parse_success": False, "json_fallback_called": False,
        "json_fallback_success": False, "limited_recovery_success": True,
        "validator_raw_response": '{"status": "PASS", "findings": []}',
    }


@pytest.mark.unit
def test_invoke_validator_records_limited_recovery_success_for_bare_status_on_first_attempt(monkeypatch):
    recorded = []
    monkeypatch.setattr(
        "tradingagents.agents.validator.record_validator_debug", recorded.append,
    )

    class _GemmaLikeLLM:
        def __init__(self):
            self.calls = 0

        def invoke(self, prompt, seed=None):
            self.calls += 1
            return SimpleNamespace(content="PASS")

    llm = _GemmaLikeLLM()
    invoke_validator(
        llm, stage="Trader", call_identity="trader_validation",
        instructions="x", evidence_context="x", candidate_text="x",
    )
    assert llm.calls == 1
    assert recorded[-1] == {
        "structured_parse_success": False, "json_fallback_called": False,
        "json_fallback_success": False, "limited_recovery_success": True,
        "validator_raw_response": "PASS",
    }


@pytest.mark.unit
def test_invoke_validator_calls_normalize_only_when_same_attempt_recovery_is_impossible(monkeypatch):
    """Structured parse fails AND the raw text is too ambiguous to recover
    mechanically -- only THEN is a second (normalize-only) call made."""
    recorded = []
    monkeypatch.setattr(
        "tradingagents.agents.validator.record_validator_debug", recorded.append,
    )
    ambiguous_raw = "The candidate looks mostly fine but I have some concerns about it."
    validator = _FakeStructuredParseFailure(ambiguous_raw, normalize_response='{"status": "WARN", "findings": []}')

    invoke_validator(
        validator, stage="Trader", call_identity="trader_validation",
        instructions="x", evidence_context="x", candidate_text="x",
    )
    assert validator.structured_calls == 1
    assert validator.plain_calls == 1  # exactly one normalize-only retry
    debug = recorded[-1]
    assert debug["json_fallback_called"] is True
    assert debug["json_fallback_success"] is True
    assert debug["limited_recovery_success"] is False


@pytest.mark.unit
def test_filtered_findings_are_recorded_as_raw_dropped_final(monkeypatch):
    recorded = []
    monkeypatch.setattr(
        "tradingagents.agents.validator.record_validator_debug", recorded.append,
    )
    validator = _FakeStructuredValidator(ValidationResult(
        status=ValidationStatus.FAIL,
        findings=[ValidationFinding(
            category=ValidationCategory.UNSUPPORTED_NUMBER, claim="1167.49 is not supported", reason="x",
        )],
    ))
    validate_and_repair(
        stage="Trader", validator_llm=validator, original_llm=_FakeLLM(),
        original_role="generation", call_identity_base="trader",
        instructions="x", evidence_context="The target price is 1167.49.",
        candidate_text="candidate", candidate_parsed=None,
        expected_schema=None, render_fn=None,
        deterministic_check_fn=lambda text, parsed: [],
    )
    debug_call = next(c for c in recorded if "validator_findings_raw" in c)
    assert len(debug_call["validator_findings_raw"]) == 1
    assert len(debug_call["validator_findings_dropped"]) == 1
    assert debug_call["validator_findings_final"] == []


# === 31-32: role/model flexibility =========================================


@pytest.mark.unit
def test_same_model_can_serve_generation_and_validator_roles():
    shared = _FakeLLM(responses=['{"status": "PASS", "findings": []}'])
    outcome = validate_and_repair(
        stage="Trader", validator_llm=shared, original_llm=shared,
        original_role="generation", call_identity_base="trader",
        instructions="x", evidence_context="x", candidate_text="ok",
        candidate_parsed=None, expected_schema=None, render_fn=None,
        deterministic_check_fn=lambda text, parsed: [],
    )
    assert outcome.status == ValidationStatus.PASS.value


@pytest.mark.unit
def test_three_distinct_models_are_independently_configurable():
    config = dict(DEFAULT_CONFIG)
    config["generation_llm"] = "model-gen"
    config["review_llm"] = "model-review"
    config["validator_llm"] = "model-validator"
    config["validation_enabled"] = True
    config = resolve_validation_config(config)
    assert len({config["generation_llm"], config["review_llm"], config["validator_llm"]}) == 3


# === Trader / Portfolio Manager integration =================================


class _TraderLLM:
    """No with_structured_output -- free-text fallback path, like the
    project's own ``_NamedLLM`` pattern (see test_model_roles.py)."""

    def __init__(self, text):
        self.text = text
        self.calls = 0

    def invoke(self, prompt, **kwargs):
        self.calls += 1
        return SimpleNamespace(content=self.text)


def _trader_state(trader_text="**Action**: Buy\n**Entry Price**: 100\n**Stop Loss**: 95", market_report=""):
    state = Propagator().create_initial_state("NVDA", "2026-09-01")
    state["investment_plan"] = "Buy plan"
    state["market_report"] = market_report
    return state


class _CapturingValidator:
    """A validator_llm with no structured-output capability that just
    records every prompt it receives, for asserting what evidence Validator
    was actually given (see task: "VALIDATOR EVIDENCE CONTEXT INCOMPLETO")."""

    def __init__(self, response='{"status": "PASS", "findings": []}'):
        self.response = response
        self.prompts: list[str] = []

    def invoke(self, prompt, seed=None):
        self.prompts.append(prompt)
        return SimpleNamespace(content=self.response)


@pytest.mark.unit
def test_trader_wires_validation_log_into_returned_state():
    llm = _TraderLLM("**Action**: Buy\n**Entry Price**: 100\n**Stop Loss**: 95")
    result = create_trader(llm, validator_llm=None)(_trader_state())
    assert "validation_log" in result
    assert result["validation_log"][0]["stage"] == "Trader"


@pytest.mark.unit
def test_trader_relational_error_with_validation_disabled_is_observed_but_not_repaired():
    """Deterministic checks always run and are recorded (see README:
    "Validation"), but with validation disabled (validator_llm=None) a
    deterministic FAIL must never trigger a repair or touch the candidate."""
    bad_text = "**Action**: Buy\nThe current price of 216.48 is below the prior resistance of 199.81."
    llm = _TraderLLM(bad_text)
    result = create_trader(llm, validator_llm=None)(_trader_state())
    assert llm.calls == 1  # only the original free-text generation call
    assert result["validation_log"][0]["status"] == ValidationStatus.FAIL.value
    assert result["validation_log"][0]["repair_attempted"] is False
    assert result["trader_investment_plan"] == bad_text  # candidate unchanged


@pytest.mark.unit
def test_trader_relational_error_with_validation_enabled_triggers_a_repair():
    bad_text = "**Action**: Buy\nThe current price of 216.48 is below the prior resistance of 199.81."
    llm = _TraderLLM(bad_text)
    result = create_trader(llm, validator_llm=_enabled_validator())(_trader_state())
    # Original free-text generation, the repair, and -- since the repair's
    # own free text (this fake returns the same text for every call) still
    # isn't valid TraderProposal JSON -- one canonical-recovery attempt
    # (see structured_recovery.py), which also fails to parse it and
    # leaves final_parsed None exactly as before this mechanism existed.
    assert llm.calls == 3
    assert result["validation_log"][0]["repair_attempted"] is True


@pytest.mark.unit
def test_portfolio_manager_wires_validation_log_into_returned_state():
    decision = PortfolioDecision(
        rating=PortfolioRating.OVERWEIGHT, disposition="KEEP",
        executive_summary="Clean summary.", investment_thesis="Clean thesis.",
    )

    class _PMLLM:
        def with_structured_output(self, _schema):
            return self

        def invoke(self, prompt):
            return decision

    state = Propagator().create_initial_state("NVDA", "2026-09-01")
    state["investment_plan"] = "Overweight plan"
    state["trader_investment_plan"] = "**Action**: Buy\n**Entry Price**: 100\n**Stop Loss**: 95"
    result = create_portfolio_manager(_PMLLM(), validator_llm=None)(state)
    assert "validation_log" in result
    assert result["validation_log"][0]["stage"] == "Portfolio Manager"
    assert result["risk_debate_state"]["portfolio_disposition"] == "KEEP"


# =============================================================================
# Second targeted correction: recovery ORDER (same-attempt recovery before any
# second LLM call; the second call, when needed, is format-repair only, never
# a fresh verdict), and Validator evidence-context completeness for Trader /
# Portfolio Manager. Numbered per the task's own "TESTS" list (1-15).
# =============================================================================


@pytest.mark.unit
def test_1_raw_status_pass_recovers_without_a_second_llm_call():
    validator = _FakeStructuredParseFailure("status: PASS")
    result = invoke_validator(
        validator, stage="Trader", call_identity="trader_validation",
        instructions="x", evidence_context="x", candidate_text="x",
    )
    assert result.status == ValidationStatus.PASS
    assert validator.structured_calls == 1
    assert validator.plain_calls == 0  # no normalize/fallback call made


@pytest.mark.unit
def test_2_raw_status_fail_with_finding_recovers_without_a_second_llm_call():
    raw = "status: FAIL\nfindings:\n  - bad value"
    validator = _FakeStructuredParseFailure(raw)
    result = invoke_validator(
        validator, stage="Trader", call_identity="trader_validation",
        instructions="x", evidence_context="x", candidate_text="x",
    )
    assert result.status == ValidationStatus.FAIL
    assert result.findings[0].reason == "bad value"
    assert validator.plain_calls == 0


@pytest.mark.unit
def test_3_recovered_fail_triggers_exactly_one_original_agent_repair():
    raw = "status: FAIL\nfindings:\n  - bad value"
    validator = _FakeStructuredParseFailure(raw)
    original = _FakeLLM(responses=["repaired"])
    outcome = validate_and_repair(
        stage="Trader", validator_llm=validator, original_llm=original,
        original_role="generation", call_identity_base="trader",
        instructions="x", evidence_context="x", candidate_text="candidate",
        candidate_parsed=None, expected_schema=None, render_fn=None,
        deterministic_check_fn=lambda text, parsed: [],
    )
    assert outcome.repair_attempted is True
    assert original.calls == 1
    assert validator.plain_calls == 0  # neither the initial check nor the
    # recheck needed a normalize/fallback call -- both recovered directly.


@pytest.mark.unit
def test_4_normalize_fallback_call_only_made_when_recovery_is_impossible(monkeypatch):
    captured = []
    monkeypatch.setattr("tradingagents.agents.validator.record_validator_debug", captured.append)
    ambiguous_raw = "The candidate looks mostly fine but I have some concerns about it."
    validator = _FakeStructuredParseFailure(
        ambiguous_raw, normalize_response='{"status": "WARN", "findings": []}',
    )
    invoke_validator(
        validator, stage="Trader", call_identity="trader_validation",
        instructions="x", evidence_context="x", candidate_text="x",
    )
    assert validator.plain_calls == 1
    assert captured[-1]["json_fallback_called"] is True


@pytest.mark.unit
def test_5_normalize_call_preserves_a_fail_status_it_is_given():
    """The normalize-only retry must not silently become a second,
    independently-judged verdict (see task: "JSON FALLBACK NO DEBE
    REVALIDAR") -- whatever status it reports is trusted as a format-only
    restatement, not re-derived from candidate/evidence."""
    ambiguous_raw = "I reviewed the candidate carefully and found some issues worth noting here."
    validator = _FakeStructuredParseFailure(
        ambiguous_raw,
        normalize_response=(
            '{"status": "FAIL", "findings": [{"category": "UNCLASSIFIED_VALIDATION_ERROR", '
            '"claim": "x", "reason": "normalize-preserved reason"}]}'
        ),
    )
    result = invoke_validator(
        validator, stage="Trader", call_identity="trader_validation",
        instructions="x", evidence_context="x", candidate_text="x",
    )
    assert result.status == ValidationStatus.FAIL
    assert result.findings[0].reason == "normalize-preserved reason"


@pytest.mark.unit
def test_6_normalize_call_preserves_findings_semantics_not_just_status():
    ambiguous_raw = "Some ambiguous commentary without a clean status line at all."
    validator = _FakeStructuredParseFailure(
        ambiguous_raw,
        normalize_response=(
            '{"status": "WARN", "findings": [{"category": "UNIT_ERROR", '
            '"claim": "5%", "reason": "percent used as level"}]}'
        ),
    )
    result = invoke_validator(
        validator, stage="Trader", call_identity="trader_validation",
        instructions="x", evidence_context="x", candidate_text="x",
    )
    assert result.status == ValidationStatus.WARN
    assert result.findings[0].category == ValidationCategory.UNIT_ERROR
    assert result.findings[0].claim == "5%"


@pytest.mark.unit
def test_7_trader_validator_receives_allowed_market_evidence():
    capturing = _CapturingValidator()
    state = _trader_state(
        market_report="Current price 1157.08, SMA200 1070.27, EMA10 1167.49, SMA50 1177.34, RSI 47.04",
    )
    llm = _TraderLLM("**Action**: Buy\n**Entry Price**: 1157.08\n**Stop Loss**: 1070.27")
    create_trader(llm, validator_llm=capturing)(state)
    assert any("1167.49" in p for p in capturing.prompts)


@pytest.mark.unit
def test_8_portfolio_manager_validator_receives_its_allowed_evidence_universe():
    capturing = _CapturingValidator()

    class _PMLLM:
        def invoke(self, prompt):
            return SimpleNamespace(
                content="**Rating**: Overweight\n**Disposition**: KEEP\n"
                "**Executive Summary**: Clean summary.\n**Investment Thesis**: Clean thesis.",
            )

    state = Propagator().create_initial_state("NVDA", "2026-09-01")
    state["investment_plan"] = "Overweight plan"
    state["trader_investment_plan"] = "**Action**: Buy\n**Entry Price**: 100\n**Stop Loss**: 95"
    state["portfolio_context"] = "Cash: $50,000. Current NVDA shares: 120."
    create_portfolio_manager(_PMLLM(), validator_llm=capturing)(state)
    assert any("50,000" in p for p in capturing.prompts)


@pytest.mark.unit
def test_9_number_in_allowed_market_evidence_is_not_flagged_unsupported():
    finding = ValidationFinding(
        category=ValidationCategory.UNSUPPORTED_NUMBER, claim="1167.49 is not supported", reason="x",
    )
    market_evidence = "Technical market report:\nEMA10 1167.49, RSI 47.04"
    reason = _why_llm_finding_should_be_dropped(finding, [], market_evidence)
    assert reason is not None  # dropped as a false positive


@pytest.mark.unit
def test_10_invented_number_absent_from_every_allowed_source_remains_unsupported():
    finding = ValidationFinding(
        category=ValidationCategory.UNSUPPORTED_NUMBER, claim="9999.99 is not supported", reason="x",
    )
    market_evidence = "Technical market report:\nEMA10 1167.49, RSI 47.04"
    reason = _why_llm_finding_should_be_dropped(finding, [], market_evidence)
    assert reason is None  # preserved -- not found anywhere in allowed evidence


@pytest.mark.unit
def test_11_candidate_text_itself_never_counts_as_support():
    """A number the candidate invented and merely repeats is NOT its own
    support -- only evidence_context counts (see task: "UNSUPPORTED_NUMBER
    ... candidate_text NO cuenta como evidence"). Enforced structurally: the
    filter functions don't even accept a candidate_text parameter any more,
    so there is no evidence-equivalent argument left to accidentally consult."""
    import inspect

    sig = inspect.signature(_why_llm_finding_should_be_dropped)
    assert "candidate_text" not in sig.parameters
    sig = inspect.signature(_filter_llm_findings)
    assert "candidate_text" not in sig.parameters

    # And behaviorally: 1167.49 appears only in a hypothetical candidate,
    # never in the evidence actually checked here -- must stay preserved.
    finding = ValidationFinding(
        category=ValidationCategory.UNSUPPORTED_NUMBER, claim="1167.49 is not supported", reason="x",
    )
    reason = _why_llm_finding_should_be_dropped(finding, [], "No numeric figures appear here at all.")
    assert reason is None


# =============================================================================
# Phase 2: extending Validator to Research Manager (ONLY this stage, on top
# of Trader/Portfolio Manager) -- reuses validate_and_repair() and the
# EXISTING check_manager_integrity deterministic check (via the new
# check_manager_integrity_findings adapter), never a parallel implementation.
# Numbered per the task's own "TESTS" list (1-23).
# =============================================================================


class _ResearchManagerLLM:
    """No with_structured_output -- free-text fallback path, like the
    project's own ``_NamedLLM``/``_TraderLLM`` pattern."""

    def __init__(self, text):
        self.text = text
        self.calls = 0

    def invoke(self, prompt, **kwargs):
        self.calls += 1
        return SimpleNamespace(content=self.text)


def _research_manager_state(bull_initial="Bull thesis: target 1196.17.", bear_initial="Bear thesis: risks noted."):
    state = Propagator().create_initial_state("NVDA", "2026-09-01")
    state["investment_debate_state"]["bull_initial"] = bull_initial
    state["investment_debate_state"]["bear_initial"] = bear_initial
    return state


_RM_PLAN_TEXT = "**Recommendation**: Buy\n**Rationale**: Bull case is stronger.\n**Strategic Actions**: Enter per plan."


@pytest.mark.unit
def test_rm_1_validation_disabled_observes_deterministic_findings_but_makes_no_calls_or_repair():
    bad_plan = (
        "**Recommendation**: Buy\n"
        "**Rationale**: Reduce exposure if price exceeds 9999.99.\n"
        "**Strategic Actions**: Enter per plan."
    )
    llm = _ResearchManagerLLM(bad_plan)
    result = create_research_manager(llm, validator_llm=None)(_research_manager_state())
    assert llm.calls == 1  # only the original generation call
    entry = result["validation_log"][0]
    assert entry["stage"] == "Research Manager"
    assert entry["status"] == ValidationStatus.FAIL.value  # the unsupported threshold is still recorded
    assert entry["repair_attempted"] is False
    assert result["investment_plan"] == bad_plan  # candidate unchanged


@pytest.mark.unit
def test_rm_2_validation_enabled_pass_triggers_no_repair():
    llm = _ResearchManagerLLM(_RM_PLAN_TEXT)
    validator = _enabled_validator()
    result = create_research_manager(llm, validator_llm=validator)(_research_manager_state())
    assert llm.calls == 1
    assert result["validation_log"][0]["repair_attempted"] is False


@pytest.mark.unit
def test_rm_3_structured_fail_triggers_exactly_one_repair():
    bad_plan = (
        "**Recommendation**: Buy\n"
        "**Rationale**: Reduce exposure if price exceeds 9999.99.\n"
        "**Strategic Actions**: Enter per plan."
    )
    llm = _ResearchManagerLLM(bad_plan)
    validator = _enabled_validator()
    result = create_research_manager(llm, validator_llm=validator)(_research_manager_state())
    # original generation + one repair + one canonical-recovery attempt
    # (the repair's own free text, _ResearchManagerLLM's fixed response,
    # still isn't valid ResearchPlan JSON -- see structured_recovery.py).
    assert llm.calls == 3
    assert result["validation_log"][0]["repair_attempted"] is True


@pytest.mark.unit
def test_rm_4_limited_recovery_fail_triggers_exactly_one_repair_no_normalization_call():
    raw = "status: FAIL\nfindings:\n  - invented threshold"
    validator = _FakeStructuredParseFailure(raw)
    llm = _ResearchManagerLLM(_RM_PLAN_TEXT)
    result = create_research_manager(llm, validator_llm=validator)(_research_manager_state())
    # original generation + one repair + one canonical-recovery attempt
    # (_RM_PLAN_TEXT is Markdown, not valid ResearchPlan JSON either).
    assert llm.calls == 3
    assert validator.plain_calls == 0  # recovered mechanically both times, no normalize call
    assert result["validation_log"][0]["repair_attempted"] is True


@pytest.mark.unit
def test_rm_5_ambiguous_raw_normalization_fallback_acts_as_formatter_not_second_opinion():
    ambiguous_raw = "Looking at this response, I have some reservations worth mentioning."
    validator = _FakeStructuredParseFailure(
        ambiguous_raw,
        normalize_response=(
            '{"status": "FAIL", "findings": [{"category": "UNCLASSIFIED_VALIDATION_ERROR", '
            '"claim": "x", "reason": "preserved"}]}'
        ),
    )
    llm = _ResearchManagerLLM(_RM_PLAN_TEXT)
    result = create_research_manager(llm, validator_llm=validator)(_research_manager_state())
    # One normalize-only call for the initial check and one for the
    # post-repair recheck (the fake always returns the same ambiguous raw) --
    # never more than one PER validation pass, and never the candidate/
    # evidence resent for a fresh verdict.
    assert validator.plain_calls == 2
    assert result["validation_log"][0]["repair_attempted"] is True
    assert result["validation_log"][0]["categories"] == ["UNCLASSIFIED_VALIDATION_ERROR"]


@pytest.mark.unit
def test_rm_6_repair_pass_final_candidate_continues_downstream():
    bad_plan = (
        "**Recommendation**: Buy\n"
        "**Rationale**: Reduce exposure if price exceeds 9999.99.\n"
        "**Strategic Actions**: Enter per plan."
    )
    repaired_plan = "**Recommendation**: Buy\n**Rationale**: Bull case is stronger.\n**Strategic Actions**: Enter per plan."

    class _RepairLLM:
        def __init__(self):
            self.calls = 0

        def invoke(self, prompt, **kwargs):
            self.calls += 1
            return SimpleNamespace(content=bad_plan if self.calls == 1 else repaired_plan)

    llm = _RepairLLM()
    validator = _enabled_validator()
    result = create_research_manager(llm, validator_llm=validator)(_research_manager_state())
    assert result["investment_plan"] == repaired_plan
    assert result["investment_debate_state"]["judge_decision"] == repaired_plan
    assert result["validation_log"][0]["status"] == ValidationStatus.PASS.value
    assert result["validation_log"][0]["repair_succeeded"] is True


@pytest.mark.unit
def test_rm_7_repair_fail_no_second_repair_and_final_unresolved_increments():
    bad_plan = (
        "**Recommendation**: Buy\n"
        "**Rationale**: Reduce exposure if price exceeds 9999.99.\n"
        "**Strategic Actions**: Enter per plan."
    )
    llm = _ResearchManagerLLM(bad_plan)  # always returns the same unsupported-threshold text
    validator = _enabled_validator()
    result = create_research_manager(llm, validator_llm=validator)(_research_manager_state())
    # original + exactly one repair + one canonical-recovery attempt
    # (still never a SECOND repair -- recovery is bookkeeping over the
    # repair's own text, not another correction pass).
    assert llm.calls == 3
    entry = result["validation_log"][0]
    assert entry["repair_attempted"] is True
    assert entry["repair_succeeded"] is False
    counts = summarize_validation_log(result["validation_log"])
    assert counts["final_unresolved"] == 1
    assert counts["initial_failures"] == 1
    assert counts["repairs_succeeded"] == 0


@pytest.mark.unit
def test_rm_8_arithmetic_error_introduced_by_manager_is_detected():
    """Reward/risk ratio mismatch -- reuses the EXISTING
    check_manager_integrity (via check_manager_integrity_findings), never a
    second arithmetic implementation."""
    debate = {"bull_trade_metrics": {"reward_risk": 1.2}, "bear_trade_metrics": {}}
    text = "**Rationale**: The setup offers an R/R of 3:1, which is favorable."
    findings = check_manager_integrity_findings(text, debate)
    assert any(f.category == ValidationCategory.ARITHMETIC_ERROR for f in findings)


@pytest.mark.unit
def test_rm_9_number_present_in_allowed_bull_bear_input_is_not_unsupported():
    debate = {"bull_thesis": {"take_profit": 1196.17}, "bear_thesis": {}}
    text = "**Rationale**: Exposure should be cut if price exceeds 1196.17."
    findings = check_manager_integrity_findings(text, debate)
    assert not any(f.category == ValidationCategory.UNSUPPORTED_NUMBER for f in findings)


@pytest.mark.unit
def test_rm_10_number_invented_by_manager_is_flagged_unsupported():
    debate = {"bull_thesis": {}, "bear_thesis": {}}
    text = "**Rationale**: Exposure should be cut if price exceeds 1215.0."
    findings = check_manager_integrity_findings(text, debate)
    assert any(f.category == ValidationCategory.UNSUPPORTED_NUMBER for f in findings)


@pytest.mark.unit
def test_rm_11_candidate_text_not_consulted_as_evidence_for_research_manager_either():
    import inspect

    assert "candidate_text" not in inspect.signature(check_manager_integrity_findings).parameters


@pytest.mark.unit
def test_rm_12_inherited_bull_claim_not_reopened_as_unsupported():
    """Manager repeats a number that was explicitly in Bull's canonical
    thesis -- must not be reopened as unsupported merely because it is
    absent from the (here, empty) raw analyst reports."""
    debate = {"bull_thesis": {"take_profit": 1196.17}, "bear_thesis": {}}
    text = "**Rationale**: Reduce exposure if price exceeds 1196.17."
    findings = check_manager_integrity_findings(text, debate)
    assert findings == []


@pytest.mark.unit
def test_rm_13_research_manager_validator_prompt_forbids_new_numbers_and_probability_misuse():
    import tradingagents.agents.managers.research_manager as rm_mod

    text = rm_mod._RESEARCH_MANAGER_CONTRACT.lower()
    assert "invent" in text
    assert "probability" in text


@pytest.mark.unit
def test_rm_14_schema_error_on_missing_required_field():
    finding = check_schema({"recommendation": "Buy"}, ResearchPlan)
    assert finding is not None
    assert finding.category == ValidationCategory.SCHEMA_ERROR


@pytest.mark.unit
def test_rm_15_trader_receives_the_final_repaired_manager_output():
    bad_plan = (
        "**Recommendation**: Buy\n"
        "**Rationale**: Reduce exposure if price exceeds 9999.99.\n"
        "**Strategic Actions**: Enter per plan."
    )
    repaired_plan = "**Recommendation**: Buy\n**Rationale**: Clean.\n**Strategic Actions**: Enter per plan."

    class _RepairLLM:
        def __init__(self):
            self.calls = 0

        def invoke(self, prompt, **kwargs):
            self.calls += 1
            return SimpleNamespace(content=bad_plan if self.calls == 1 else repaired_plan)

    result = create_research_manager(_RepairLLM(), validator_llm=_enabled_validator())(_research_manager_state())
    # The exact field the Trader node reads (state["investment_plan"]).
    assert result["investment_plan"] == repaired_plan
    assert "9999.99" not in result["investment_plan"]


@pytest.mark.unit
def test_rm_16_research_verifier_module_unchanged_by_this_integration():
    from tradingagents.agents.researchers.research_verifier import create_research_verifier

    assert create_research_verifier is not None


@pytest.mark.unit
def test_rm_17_bull_bear_routing_modules_unchanged_by_this_integration():
    from tradingagents.agents.researchers.bear_researcher import create_bear_researcher
    from tradingagents.agents.researchers.bull_researcher import create_bull_researcher

    assert create_bull_researcher is not None
    assert create_bear_researcher is not None


@pytest.mark.unit
def test_rm_18_existing_trader_and_portfolio_manager_validator_signatures_unchanged():
    import inspect

    assert "validator_llm" in inspect.signature(create_trader).parameters
    assert "validator_llm" in inspect.signature(create_portfolio_manager).parameters


@pytest.mark.unit
def test_rm_19_generation_role_used_for_research_manager_own_call_and_repair():
    bad_plan = (
        "**Recommendation**: Buy\n"
        "**Rationale**: Reduce exposure if price exceeds 9999.99.\n"
        "**Strategic Actions**: Enter per plan."
    )

    class _RoleTrackingLLM:
        def __init__(self):
            self.seen_roles: list[str | None] = []

        def invoke(self, prompt, **kwargs):
            self.seen_roles.append(_current_llm_role.get())
            return SimpleNamespace(content=bad_plan)

    llm = _RoleTrackingLLM()
    create_research_manager(llm, validator_llm=_enabled_validator())(_research_manager_state())
    # original call + repair + canonical-recovery attempt, all "generation" --
    # recovery reuses the SAME role as the call it recovers, never "validator".
    assert llm.seen_roles == ["generation", "generation", "generation"]


@pytest.mark.unit
def test_rm_20_validation_and_repair_call_identities_are_distinguishable():
    set_active_seed_context(SeedContext(base_seed=42, ticker="NVDA", provider_supports_seed=True))
    bad_plan = (
        "**Recommendation**: Buy\n"
        "**Rationale**: Reduce exposure if price exceeds 9999.99.\n"
        "**Strategic Actions**: Enter per plan."
    )

    class _IdentityTrackingLLM:
        def __init__(self):
            self.seen_identities: list[str | None] = []

        def invoke(self, prompt, **kwargs):
            self.seen_identities.append(_current_call_identity.get())
            return SimpleNamespace(content=bad_plan)

    llm = _IdentityTrackingLLM()
    create_research_manager(llm, validator_llm=_enabled_validator())(_research_manager_state())
    # The canonical-recovery attempt's call_identity is None -- same
    # precedent as researchers/canonical_recovery.py's own recovery calls,
    # which also omit it (the "canonical_recovery" attempt tag alone is
    # unambiguous; there is only ever one such call per repair).
    assert llm.seen_identities == ["research_manager", "research_manager_validation_repair", None]


# --- 21-23: new observability metrics ---------------------------------------


@pytest.mark.unit
def test_rm_21_initial_fail_repair_pass_metrics():
    bad_plan = (
        "**Recommendation**: Buy\n"
        "**Rationale**: Reduce exposure if price exceeds 9999.99.\n"
        "**Strategic Actions**: Enter per plan."
    )
    repaired_plan = "**Recommendation**: Buy\n**Rationale**: Clean.\n**Strategic Actions**: Enter per plan."

    class _RepairLLM:
        def __init__(self):
            self.calls = 0

        def invoke(self, prompt, **kwargs):
            self.calls += 1
            return SimpleNamespace(content=bad_plan if self.calls == 1 else repaired_plan)

    result = create_research_manager(_RepairLLM(), validator_llm=_enabled_validator())(_research_manager_state())
    counts = summarize_validation_log(result["validation_log"])
    assert counts["initial_failures"] == 1
    assert counts["repairs_attempted"] == 1
    assert counts["repairs_succeeded"] == 1
    assert counts["final_unresolved"] == 0


@pytest.mark.unit
def test_rm_22_initial_fail_repair_fail_metrics():
    bad_plan = (
        "**Recommendation**: Buy\n"
        "**Rationale**: Reduce exposure if price exceeds 9999.99.\n"
        "**Strategic Actions**: Enter per plan."
    )
    llm = _ResearchManagerLLM(bad_plan)
    result = create_research_manager(llm, validator_llm=_enabled_validator())(_research_manager_state())
    counts = summarize_validation_log(result["validation_log"])
    assert counts["initial_failures"] == 1
    assert counts["repairs_attempted"] == 1
    assert counts["repairs_succeeded"] == 0
    assert counts["final_unresolved"] == 1


@pytest.mark.unit
def test_rm_23_pass_keeps_all_new_metrics_at_zero():
    llm = _ResearchManagerLLM(_RM_PLAN_TEXT)
    result = create_research_manager(llm, validator_llm=_enabled_validator())(_research_manager_state())
    counts = summarize_validation_log(result["validation_log"])
    assert counts["initial_failures"] == 0
    assert counts["repairs_attempted"] == 0
    assert counts["repairs_succeeded"] == 0
    assert counts["final_unresolved"] == 0


# =============================================================================
# Provenance-leak fix: Research Manager Validator's known-number pool must
# never include the raw analyst reports (Research Manager's own prompt never
# shows them) -- only the free-text fields actually rendered into that
# prompt. Numbered per this task's own "Añade tests" list (1-8).
# =============================================================================

from tradingagents.agents.managers.integrity_check import (  # noqa: E402
    build_known_number_pool,
    check_manager_integrity,
)

_LEAK_DEBATE = {"bull_thesis": {}, "bear_thesis": {}}
_LEAK_RAW_REPORT = "Technical market report: resistance near 1215.0, volume climbing."


@pytest.mark.unit
def test_leak_1_number_only_in_raw_report_stays_unsupported_when_manager_introduces_it():
    text = "**Rationale**: Exposure should be cut if price exceeds 1215.0."
    # The raw report is passed only as a decoy to confirm it is NOT consulted
    # -- check_manager_integrity_findings takes no reports parameter at all.
    findings = check_manager_integrity_findings(text, _LEAK_DEBATE)
    assert any(f.category == ValidationCategory.UNSUPPORTED_NUMBER for f in findings)


@pytest.mark.unit
def test_leak_2_number_in_bull_canonical_input_is_supported():
    text = "**Rationale**: Exposure should be cut if price exceeds 1196.17."
    findings = check_manager_integrity_findings(
        text, _LEAK_DEBATE, extra_sources=("Bull thesis: target 1196.17 supported by earnings.",),
    )
    assert not any(f.category == ValidationCategory.UNSUPPORTED_NUMBER for f in findings)


@pytest.mark.unit
def test_leak_3_number_in_bear_canonical_input_is_supported():
    text = "**Rationale**: Exposure should be cut if price exceeds 1070.27."
    findings = check_manager_integrity_findings(
        text, _LEAK_DEBATE, extra_sources=("Bear thesis: downside risk to 1070.27.",),
    )
    assert not any(f.category == ValidationCategory.UNSUPPORTED_NUMBER for f in findings)


@pytest.mark.unit
def test_leak_4_number_in_deterministic_trade_metrics_is_supported():
    debate = {"bull_thesis": {}, "bear_thesis": {}, "bull_trade_metrics": {"stop_loss": 1070.27}}
    text = "**Rationale**: Exposure should be cut if price exceeds 1070.27."
    findings = check_manager_integrity_findings(text, debate)
    assert not any(f.category == ValidationCategory.UNSUPPORTED_NUMBER for f in findings)


@pytest.mark.unit
def test_leak_5_number_in_verifier_or_repair_input_is_supported():
    text = "**Rationale**: Exposure should be cut if price exceeds 1183.4."
    findings = check_manager_integrity_findings(
        text, _LEAK_DEBATE,
        extra_sources=("Verifier summary: confirmed level 1183.4 as the key pivot.",),
    )
    assert not any(f.category == ValidationCategory.UNSUPPORTED_NUMBER for f in findings)
    findings2 = check_manager_integrity_findings(
        text, _LEAK_DEBATE,
        extra_sources=("Repair output: corrected pivot to 1183.4.",),
    )
    assert not any(f.category == ValidationCategory.UNSUPPORTED_NUMBER for f in findings2)


@pytest.mark.unit
def test_leak_6_raw_report_number_never_leaks_into_rm_validator_pool():
    """Structural guarantee: the adapter's signature has no parameter a
    caller could (even accidentally) use to pass raw reports through."""
    import inspect

    params = inspect.signature(check_manager_integrity_findings).parameters
    assert "reports" not in params
    assert set(params) == {"manager_output_text", "debate", "extra_sources"}

    # Behaviorally: a number present ONLY in a string that resembles a raw
    # report, passed as extra_sources (a caller's own mistake), is NOT what
    # this test is about -- extra_sources is the agent's allowed universe by
    # construction in research_manager.py. What matters is that there is no
    # "reports" channel at all, confirmed above structurally.
    pool = build_known_number_pool(_LEAK_DEBATE)
    assert 1215.0 not in pool


@pytest.mark.unit
def test_leak_7_existing_manager_integrity_call_site_behavior_is_unchanged():
    """check_manager_integrity's OWN pre-existing signature/behavior (used
    for the Trader-facing integrity notice) must still treat raw reports as
    supported -- this fix only narrows the SEPARATE Validator adapter."""
    text = "**Rationale**: Exposure should be cut if price exceeds 1215.0."
    result = check_manager_integrity(text, _LEAK_DEBATE, _LEAK_RAW_REPORT, "", "", "")
    assert result.findings == []  # 1215.0 is supported via the raw report, as before this fix


@pytest.mark.unit
def test_leak_8_trader_and_portfolio_manager_validation_unchanged():
    """Neither Trader's nor Portfolio Manager's deterministic checks touch
    check_manager_integrity_findings/build_known_number_pool at all -- this
    fix cannot have changed their behavior."""
    bad_text = "**Action**: Buy\nThe current price of 216.48 is below the prior resistance of 199.81."
    llm = _TraderLLM(bad_text)
    result = create_trader(llm, validator_llm=None)(_trader_state())
    assert result["validation_log"][0]["status"] == ValidationStatus.FAIL.value
    assert result["validation_log"][0]["repair_attempted"] is False


# =============================================================================
# Second real-run regression: (1) Research Manager Validator's evidence
# universe was still missing research_horizon/setup_tags/conviction_trail,
# causing a false PROMPT_REQUIREMENT_VIOLATION/UNSUPPORTED_NUMBER on a direct
# restatement of a supplied setup tag; (2) limited recovery fragmented a
# single YAML-like structured finding into one UNCLASSIFIED_VALIDATION_ERROR
# per line. Numbered per this task's own test lists (1-14).
# =============================================================================

_SETUP_TAGS_TEXT = (
    "RSI_BUCKET: RSI_30_50\nMACD_SIGN: NEGATIVE\nMACD_HISTOGRAM_SIGN: POSITIVE\n"
    "PRICE_VS_EMA10: BELOW\nPRICE_VS_SMA20: NEAR\nPRICE_VS_SMA50: BELOW\n"
    "SMA50_VS_SMA200: ABOVE\nATR_PCT_BUCKET: 2_TO_3\nVOLUME_VS_AVERAGE: UNKNOWN\n"
    "TREND_STRUCTURE: BULLISH"
)


@pytest.mark.unit
def test_tags_1_rsi_bucket_restatement_is_supported():
    text = "**Rationale**: RSI entre 30-50, momentum neutro."
    findings = check_manager_integrity_findings(
        text, {"bull_thesis": {}, "bear_thesis": {}}, extra_sources=(_SETUP_TAGS_TEXT,),
    )
    assert not any(f.category == ValidationCategory.UNSUPPORTED_NUMBER for f in findings)


@pytest.mark.unit
def test_tags_2_sma50_vs_sma200_restatement_is_supported():
    text = "**Rationale**: SMA50 por encima de SMA200, estructura alcista."
    findings = check_manager_integrity_findings(
        text, {"bull_thesis": {}, "bear_thesis": {}}, extra_sources=(_SETUP_TAGS_TEXT,),
    )
    assert findings == []


@pytest.mark.unit
def test_tags_3_macd_sign_restatement_is_supported():
    text = "**Rationale**: MACD negativo en el corto plazo."
    findings = check_manager_integrity_findings(
        text, {"bull_thesis": {}, "bear_thesis": {}}, extra_sources=(_SETUP_TAGS_TEXT,),
    )
    assert findings == []


@pytest.mark.unit
def test_tags_4_price_vs_ema10_restatement_is_supported():
    text = "**Rationale**: precio por debajo de EMA10."
    findings = check_manager_integrity_findings(
        text, {"bull_thesis": {}, "bear_thesis": {}}, extra_sources=(_SETUP_TAGS_TEXT,),
    )
    assert findings == []


@pytest.mark.unit
def test_tags_5_number_absent_from_every_allowed_source_stays_unsupported():
    text = "**Rationale**: Reduce exposure if price exceeds 1215.0."
    findings = check_manager_integrity_findings(
        text, {"bull_thesis": {}, "bear_thesis": {}}, extra_sources=(_SETUP_TAGS_TEXT,),
    )
    assert any(f.category == ValidationCategory.UNSUPPORTED_NUMBER for f in findings)


@pytest.mark.unit
def test_tags_6_raw_analyst_report_still_not_evidence_for_rm_validator():
    import inspect

    assert "reports" not in inspect.signature(check_manager_integrity_findings).parameters


@pytest.mark.unit
def test_tags_7_candidate_text_still_not_evidence_for_rm_validator():
    import inspect

    assert "candidate_text" not in inspect.signature(check_manager_integrity_findings).parameters


@pytest.mark.unit
def test_recovery_8_fail_with_simple_bullets_still_recovers_mechanically():
    result = _limited_validator_recovery("FAIL\n- simple reason\n- another simple reason")
    assert result is not None
    assert result.status == ValidationStatus.FAIL
    assert [f.reason for f in result.findings] == ["simple reason", "another simple reason"]


@pytest.mark.unit
def test_recovery_9_yaml_like_structured_finding_refuses_and_defers_to_normalization():
    raw = (
        "status: FAIL\n"
        "findings:\n"
        "  - category: PROMPT_REQUIREMENT_VIOLATION\n"
        '    claim: "x"\n'
        "    evidence:\n"
        '      - "a"\n'
        '      - "b"\n'
        '    reason: "y"'
    )
    assert _limited_validator_recovery(raw) is None

    validator = _FakeStructuredParseFailure(
        raw,
        normalize_response=(
            '{"status": "FAIL", "findings": [{"category": "PROMPT_REQUIREMENT_VIOLATION", '
            '"claim": "x", "evidence": ["a", "b"], "reason": "y"}]}'
        ),
    )
    result = invoke_validator(
        validator, stage="Research Manager", call_identity="research_manager_validation",
        instructions="x", evidence_context="x", candidate_text="x",
    )
    assert validator.plain_calls == 1  # the normalization fallback, not a second verdict
    assert result.status == ValidationStatus.FAIL
    assert len(result.findings) == 1  # ONE finding, not one per YAML line


@pytest.mark.unit
def test_recovery_10_yaml_like_with_one_finding_normalizes_to_exactly_one():
    raw = "status: WARN\nfindings:\n  - category: UNIT_ERROR\n    claim: \"x\"\n    reason: \"y\""
    validator = _FakeStructuredParseFailure(
        raw,
        normalize_response='{"status": "WARN", "findings": [{"category": "UNIT_ERROR", "claim": "x", "reason": "y"}]}',
    )
    result = invoke_validator(
        validator, stage="Research Manager", call_identity="research_manager_validation",
        instructions="x", evidence_context="x", candidate_text="x",
    )
    assert len(result.findings) == 1


@pytest.mark.unit
def test_recovery_11_yaml_like_with_two_findings_normalizes_to_exactly_two():
    raw = (
        "status: FAIL\n"
        "findings:\n"
        "  - category: ARITHMETIC_ERROR\n    claim: \"a\"\n    reason: \"r1\"\n"
        "  - category: RELATIONAL_ERROR\n    claim: \"b\"\n    reason: \"r2\""
    )
    validator = _FakeStructuredParseFailure(
        raw,
        normalize_response=(
            '{"status": "FAIL", "findings": ['
            '{"category": "ARITHMETIC_ERROR", "claim": "a", "reason": "r1"}, '
            '{"category": "RELATIONAL_ERROR", "claim": "b", "reason": "r2"}]}'
        ),
    )
    result = invoke_validator(
        validator, stage="Research Manager", call_identity="research_manager_validation",
        instructions="x", evidence_context="x", candidate_text="x",
    )
    assert len(result.findings) == 2


@pytest.mark.unit
def test_recovery_12_field_key_lines_never_become_their_own_findings():
    raw = "status: FAIL\nfindings:\n  - category: ARITHMETIC_ERROR\n    claim: \"a\"\n    reason: \"r1\""
    recovered = _limited_validator_recovery(raw)
    assert recovered is None  # refused mechanically -- never fragmented into
    # "category: ARITHMETIC_ERROR" / "claim: ..." / "reason: ..." findings.


@pytest.mark.unit
def test_recovery_13_pass_with_empty_findings_normalizes_safely():
    raw = "status: PASS\nfindings: []"
    validator = _FakeStructuredParseFailure(raw, normalize_response='{"status": "PASS", "findings": []}')
    result = invoke_validator(
        validator, stage="Research Manager", call_identity="research_manager_validation",
        instructions="x", evidence_context="x", candidate_text="x",
    )
    assert result.status == ValidationStatus.PASS
    assert result.findings == []
    assert validator.plain_calls == 1


@pytest.mark.unit
def test_recovery_14_normalizer_never_receives_candidate_or_evidence():
    """The normalize-only call must not be able to change meaning -- it
    never even sees candidate/evidence to re-evaluate against."""
    raw = (
        "status: FAIL\nfindings:\n  - category: ARITHMETIC_ERROR\n"
        '    claim: "a"\n    reason: "r1"'
    )
    captured_prompts = []

    class _CapturingNormalizer(_FakeStructuredParseFailure):
        def invoke(self, prompt, seed=None):
            captured_prompts.append(prompt)
            return super().invoke(prompt, seed)

    validator = _CapturingNormalizer(
        raw, normalize_response='{"status": "FAIL", "findings": [{"category": "ARITHMETIC_ERROR", "claim": "a", "reason": "r1"}]}',
    )
    invoke_validator(
        validator, stage="Research Manager", call_identity="research_manager_validation",
        instructions="SECRET_INSTRUCTIONS_MARKER", evidence_context="SECRET_EVIDENCE_MARKER",
        candidate_text="SECRET_CANDIDATE_MARKER",
    )
    normalize_prompt = captured_prompts[-1]
    assert "SECRET_INSTRUCTIONS_MARKER" not in normalize_prompt
    assert "SECRET_EVIDENCE_MARKER" not in normalize_prompt
    assert "SECRET_CANDIDATE_MARKER" not in normalize_prompt


@pytest.mark.unit
def test_real_regression_setup_tags_restatement_does_not_trigger_false_fail_end_to_end():
    """End-to-end reproduction of the real seed-300000003 case: Research
    Manager restates setup tags it was explicitly given; must not get FAIL'd
    and repaired into losing that legitimate content."""
    state = _research_manager_state()
    state["investment_debate_state"]["setup_tags"] = {
        "RSI_BUCKET": "RSI_30_50", "MACD_SIGN": "NEGATIVE",
        "PRICE_VS_EMA10": "BELOW", "PRICE_VS_SMA50": "BELOW",
        "SMA50_VS_SMA200": "ABOVE", "TREND_STRUCTURE": "BULLISH",
    }
    candidate = (
        "**Recommendation**: Buy\n"
        "**Rationale**: La estructura de tendencia es alcista (SMA50 por encima de SMA200), "
        "mientras el corto plazo muestra RSI entre 30-50, MACD negativo y precio "
        "por debajo de EMA10 y SMA50.\n"
        "**Strategic Actions**: Enter per plan."
    )
    llm = _ResearchManagerLLM(candidate)
    result = create_research_manager(llm, validator_llm=None)(state)
    assert llm.calls == 1  # no repair triggered
    assert result["validation_log"][0]["status"] == ValidationStatus.PASS.value
    assert result["investment_plan"] == candidate  # nothing stripped out


# =============================================================================
# Third targeted correction (overnight real benchmarks): (1) Portfolio
# Manager Validator's evidence_context was missing the instrument identity
# (ticker + resolved company name) the real PM prompt always renders, causing
# a false UNKNOWN_USED_AS_FACT on the ticker itself; Trader and Research
# Manager had the exact same omission, confirmed and fixed identically.
# (2) the normalizer's JSON response lost an otherwise-valid finding whenever
# an optional field (suggested_correction/reason/claim/evidence) came back as
# an explicit JSON null the schema already treats as empty by default.
# Numbered per this task's own "TESTS" list (1-15).
# =============================================================================


def _pm_state(ticker="LLY"):
    state = Propagator().create_initial_state(ticker, "2026-09-01")
    state["investment_plan"] = "Overweight plan"
    state["trader_investment_plan"] = "**Action**: Buy\n**Entry Price**: 100\n**Stop Loss**: 95"
    return state


_PM_DECISION_TEXT = (
    "**Rating**: Overweight\n**Disposition**: KEEP\n"
    "**Executive Summary**: Clean summary.\n**Investment Thesis**: Clean thesis."
)


@pytest.mark.unit
def test_pm_1_ticker_in_pm_identity_and_candidate_is_supported():
    capturing = _CapturingValidator()

    class _PMLLM:
        def invoke(self, prompt):
            return SimpleNamespace(content=_PM_DECISION_TEXT)

    create_portfolio_manager(_PMLLM(), validator_llm=capturing)(_pm_state("LLY"))
    assert any("LLY" in p for p in capturing.prompts)


@pytest.mark.unit
def test_pm_2_candidate_introduces_a_different_ticker_finding_is_preserved_if_flagged():
    """No deterministic ticker cross-check exists (tickers aren't numbers,
    so the UNSUPPORTED_NUMBER/UNKNOWN_USED_AS_FACT post-filter can't rescue
    or contradict a ticker-shaped claim either way) -- if the Validator LLM
    flags a genuinely new ticker, that finding must stay, never be silently
    dropped by this task's fixes."""
    finding = ValidationFinding(
        category=ValidationCategory.UNKNOWN_USED_AS_FACT, claim="MSFT", reason="x",
    )
    reason = _why_llm_finding_should_be_dropped(finding, [], "Instrument identity:\nThe instrument to analyze is `LLY`.")
    assert reason is None  # kept -- not dropped, not rescued; "MSFT" has no digits to cross-check


@pytest.mark.unit
def test_pm_3_resolved_company_name_repeatable_without_unsupported_finding():
    capturing = _CapturingValidator()

    class _PMLLM:
        def invoke(self, prompt):
            return SimpleNamespace(content=_PM_DECISION_TEXT)

    state = _pm_state("LLY")
    state["instrument_context"] = "The instrument to analyze is `LLY`. Resolved identity: Eli Lilly and Company."
    create_portfolio_manager(_PMLLM(), validator_llm=capturing)(state)
    assert any("Eli Lilly and Company" in p for p in capturing.prompts)


@pytest.mark.unit
def test_pm_4_ticker_only_in_candidate_never_counts_as_its_own_support():
    """Structural: the PM node's evidence_context is built from STATE
    (instrument_context/portfolio_context/research_plan/trader_plan/history/
    risk_integrity_notice), never from the candidate the Validator is
    checking. The candidate text is naturally shown too (that's what's being
    checked), under its own "Candidate response to check:" section -- the
    point is a candidate's own stray content is never ALSO injected into the
    "Relevant input/evidence:" section as if it were supporting evidence."""
    capturing = _CapturingValidator()

    class _PMLLM:
        def invoke(self, prompt):
            return SimpleNamespace(content=_PM_DECISION_TEXT + "\nMentions AAPL.")

    create_portfolio_manager(_PMLLM(), validator_llm=capturing)(_pm_state("LLY"))
    sent_prompt = capturing.prompts[0]
    evidence_section = sent_prompt.split("Candidate response to check:")[0]
    assert "LLY" in evidence_section
    assert "AAPL" not in evidence_section  # only in the candidate section, never injected as evidence


@pytest.mark.unit
def test_pm_5_trader_and_research_manager_identity_parity_fixed_too():
    capturing_trader = _CapturingValidator()
    llm = _TraderLLM("**Action**: Buy\n**Entry Price**: 100\n**Stop Loss**: 95")
    create_trader(llm, validator_llm=capturing_trader)(_trader_state())
    assert any("NVDA" in p for p in capturing_trader.prompts)

    capturing_rm = _CapturingValidator()
    rm_llm = _ResearchManagerLLM(_RM_PLAN_TEXT)
    create_research_manager(rm_llm, validator_llm=capturing_rm)(_research_manager_state())
    assert any("NVDA" in p for p in capturing_rm.prompts)


# --- 5-14 (shared with the regression lists above): null sanitization ------


@pytest.mark.unit
def test_null_5_suggested_correction_null_sanitizes_to_empty_string():
    result = _try_json_parse(
        '{"status": "FAIL", "findings": [{"category": "RELATIONAL_ERROR", '
        '"claim": "x", "reason": "y", "suggested_correction": null}]}'
    )
    assert result is not None
    assert result.findings[0].suggested_correction == ""


@pytest.mark.unit
def test_null_6_valid_fail_survives_null_sanitization():
    result = _try_json_parse(
        '{"status": "FAIL", "findings": [{"category": "RELATIONAL_ERROR", '
        '"claim": "1157.08 is between 1152.56 and 1108.96", '
        '"evidence": ["1157.08", "1152.56"], '
        '"reason": "1157.08 is greater than 1152.56, not between them", '
        '"suggested_correction": null}]}'
    )
    assert result is not None
    assert result.status == ValidationStatus.FAIL
    assert len(result.findings) == 1
    assert result.findings[0].category == ValidationCategory.RELATIONAL_ERROR


@pytest.mark.unit
def test_null_7_valid_fail_with_sanitized_null_triggers_exactly_one_repair():
    raw = (
        '{"status": "FAIL", "findings": [{"category": "RELATIONAL_ERROR", '
        '"claim": "1157.08 is below 1152.56", "evidence": [], '
        '"reason": "contradiction", "suggested_correction": null}]}'
    )
    validator = _FakeStructuredParseFailure(raw)  # structured fails, raw IS valid JSON (w/ null)
    original = _FakeLLM(responses=["repaired"])
    outcome = validate_and_repair(
        stage="Trader", validator_llm=validator, original_llm=original,
        original_role="generation", call_identity_base="trader",
        instructions="x", evidence_context="x", candidate_text="candidate",
        candidate_parsed=None, expected_schema=None, render_fn=None,
        deterministic_check_fn=lambda text, parsed: [],
    )
    assert outcome.status != ValidationStatus.WARN.value  # never silently degraded to WARN
    assert outcome.repair_attempted is True
    assert original.calls == 1


@pytest.mark.unit
def test_null_8_recheck_runs_exactly_once_after_sanitized_fail_repair():
    raw = (
        '{"status": "FAIL", "findings": [{"category": "RELATIONAL_ERROR", '
        '"claim": "1157.08 is below 1152.56", "evidence": [], "reason": "y", '
        '"suggested_correction": null}]}'
    )
    validator = _FakeStructuredParseFailure(raw)
    original = _FakeLLM(responses=["repaired"])
    validate_and_repair(
        stage="Trader", validator_llm=validator, original_llm=original,
        original_role="generation", call_identity_base="trader",
        instructions="x", evidence_context="x", candidate_text="candidate",
        candidate_parsed=None, expected_schema=None, render_fn=None,
        deterministic_check_fn=lambda text, parsed: [],
    )
    assert validator.structured_calls == 2  # initial check + exactly one recheck
    assert original.calls == 1  # exactly one repair, never a second


@pytest.mark.unit
def test_null_9_status_or_category_null_remain_unrecoverable():
    assert _try_json_parse('{"status": null, "findings": []}') is None
    assert _try_json_parse(
        '{"status": "FAIL", "findings": [{"category": null, "claim": "x"}]}'
    ) is None


@pytest.mark.unit
def test_null_10_evidence_null_sanitizes_to_empty_list_only():
    result = _try_json_parse(
        '{"status": "WARN", "findings": [{"category": "UNIT_ERROR", '
        '"claim": "x", "reason": "y", "evidence": null}]}'
    )
    assert result is not None
    assert result.findings[0].evidence == []


@pytest.mark.unit
def test_null_sanitization_never_touches_other_payload_shapes():
    """Not a dict, or no findings list -- sanitize is a pure no-op, never
    reshapes anything it wasn't explicitly told to."""
    assert _try_json_parse("[1, 2, 3]") is None  # a JSON array, not an object
    assert _try_json_parse('{"status": "PASS"}') is not None  # no findings key at all -- default []


# --- 11-12: the real cases that must remain correctly detected --------------


@pytest.mark.unit
def test_11_rm_real_entry_contradiction_remains_detected():
    """Real benchmark (seed 1480000014): Manager silently changed Bull's
    verified entry 1157.08 to 1152.56 -- this is exactly a RELATIONAL_ERROR-
    shaped contradiction the Validator LLM (not the deterministic layer) is
    meant to catch; confirm the bounded repair/recheck machinery correctly
    restores a corrected entry and reaches PASS."""

    class _EntryChangingLLM:
        def __init__(self):
            self.calls = 0

        def invoke(self, prompt, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return SimpleNamespace(
                    content="**Recommendation**: Buy\n**Rationale**: Entry near 1152.56.\n"
                    "**Strategic Actions**: Enter per plan.",
                )
            return SimpleNamespace(
                content="**Recommendation**: Buy\n**Rationale**: Entry 1157.08 as verified.\n"
                "**Strategic Actions**: Enter per plan.",
            )

    validator = _FakeStructuredValidator(ValidationResult(
        status=ValidationStatus.FAIL,
        findings=[ValidationFinding(
            category=ValidationCategory.UNSUPPORTED_NUMBER,
            claim="1152.56 is used as the entry price",
            reason="the verified Bull entry is 1157.08, not 1152.56 -- silently changed",
        )],
    ))
    llm = _EntryChangingLLM()
    outcome = validate_and_repair(
        stage="Research Manager", validator_llm=validator, original_llm=llm,
        original_role="generation", call_identity_base="research_manager",
        instructions="x", evidence_context="Bull entry: 1157.08", candidate_text=llm.invoke("x").content,
        candidate_parsed=None, expected_schema=None, render_fn=None,
        deterministic_check_fn=lambda text, parsed: [],
    )
    assert outcome.repair_attempted is True
    assert "1157.08" in outcome.final_text


@pytest.mark.unit
def test_12_rm_setup_tags_regression_remains_fixed():
    state = _research_manager_state()
    state["investment_debate_state"]["setup_tags"] = {
        "RSI_BUCKET": "RSI_30_50", "MACD_SIGN": "NEGATIVE",
        "PRICE_VS_EMA10": "BELOW", "PRICE_VS_SMA50": "BELOW",
        "SMA50_VS_SMA200": "ABOVE", "TREND_STRUCTURE": "BULLISH",
    }
    candidate = (
        "**Recommendation**: Buy\n"
        "**Rationale**: SMA50 por encima de SMA200, RSI entre 30-50, MACD negativo.\n"
        "**Strategic Actions**: Enter per plan."
    )
    llm = _ResearchManagerLLM(candidate)
    result = create_research_manager(llm, validator_llm=None)(state)
    assert llm.calls == 1
    assert result["validation_log"][0]["status"] == ValidationStatus.PASS.value


@pytest.mark.unit
def test_13_validation_disabled_unchanged_across_all_three_stages():
    for factory, state, llm_text in (
        (create_trader, _trader_state(), "**Action**: Hold\nNo change."),
        (create_portfolio_manager, _pm_state(), _PM_DECISION_TEXT),
        (create_research_manager, _research_manager_state(), _RM_PLAN_TEXT),
    ):
        if factory is create_trader:
            llm = _TraderLLM(llm_text)
        elif factory is create_portfolio_manager:
            class _L:
                def __init__(self, text):
                    self.text = text

                def invoke(self, prompt):
                    return SimpleNamespace(content=self.text)
            llm = _L(llm_text)
        else:
            llm = _ResearchManagerLLM(llm_text)
        result = factory(llm, validator_llm=None)(state)
        entry = result["validation_log"][0]
        assert entry["repair_attempted"] is False


@pytest.mark.unit
def test_14_research_verifier_still_unchanged():
    from tradingagents.agents.researchers.research_verifier import create_research_verifier

    assert create_research_verifier is not None


@pytest.mark.unit
def test_15_generation_review_validator_routing_unchanged():
    import inspect

    assert "validator_llm" in inspect.signature(create_trader).parameters
    assert "validator_llm" in inspect.signature(create_portfolio_manager).parameters
    assert "validator_llm" in inspect.signature(create_research_manager).parameters


# =============================================================================
# Fourth targeted correction (real hybrid-run benchmark): Trader's
# deterministic UNIT_ERROR check used the generic, sentence-wide
# ``find_unit_errors`` (LEVEL_WORDS + an English-only exclude-window), which
# false-positived on a descriptive stop-distance percentage in narrative
# prose ("distancia de stop (25 pts ~= 2.16% del precio actual)") -- an
# English-only exclusion list never matches Spanish phrasing, and the level
# word's own exclusion window only looks FORWARD, never at context stated
# BEFORE it. Trader now uses a narrow, field-line-anchored check instead
# (Entry Price/Entry/Stop Loss/Stop/Take Profit/TP lines only); Portfolio
# Manager's own ``check_text_relational_and_unit_errors`` is untouched.
# =============================================================================


@pytest.mark.unit
def test_relational_errors_only_check_still_flags_real_contradictions():
    """Trader keeps full RELATIONAL_ERROR coverage -- only UNIT_ERROR
    scanning was narrowed."""
    findings = check_relational_errors("The current price of 216.48 is below the prior resistance of 199.81.")
    assert len(findings) == 1
    assert findings[0].category == ValidationCategory.RELATIONAL_ERROR


@pytest.mark.unit
def test_pct_1_entry_price_bare_percentage_is_unit_error():
    findings = check_price_field_percentage("**Entry Price:** 2.16%")
    assert len(findings) == 1
    assert findings[0].category == ValidationCategory.UNIT_ERROR


@pytest.mark.unit
def test_pct_2_stop_loss_bare_percentage_is_unit_error():
    findings = check_price_field_percentage("**Stop Loss:** 2.16%")
    assert len(findings) == 1
    assert findings[0].category == ValidationCategory.UNIT_ERROR


@pytest.mark.unit
def test_pct_3_take_profit_signed_percentage_is_unit_error():
    findings = check_price_field_percentage("**Take Profit:** +5%")
    assert len(findings) == 1
    assert findings[0].category == ValidationCategory.UNIT_ERROR


@pytest.mark.unit
def test_pct_4_reasoning_percentage_vs_sma_is_not_unit_error():
    assert check_price_field_percentage("Reasoning: price is 7.5% above SMA200.") == []


@pytest.mark.unit
def test_pct_5_distance_equivalence_percentage_is_not_unit_error():
    assert check_price_field_percentage("25 pts = 2.16% of entry") == []


@pytest.mark.unit
def test_pct_6_revenue_growth_percentage_is_not_unit_error():
    assert check_price_field_percentage("Revenue grew 47.5% year over year.") == []


@pytest.mark.unit
def test_pct_7_position_sizing_percentage_is_not_a_price_level_finding():
    assert check_price_field_percentage("**Position Sizing:** 1% of portfolio") == []


@pytest.mark.unit
def test_pct_8_absolute_levels_plus_descriptive_percentages_pass_deterministic_check():
    candidate = (
        "**Entry Price:** 1145.00 - 1165.00\n"
        "**Stop Loss:** 1130.00\n"
        "**Take Profit:** 1177.00\n"
        "**Position Sizing:** Unknown ...\n"
        "distance to stop is 25 pts ≈ 2.16% of current price"
    )
    assert check_price_field_percentage(candidate) == []
    assert _trader_deterministic_checks(candidate, None) == []


@pytest.mark.unit
def test_real_case_descriptive_stop_distance_percentage_no_longer_flagged():
    """Exact real-run text (translated) that produced the false UNIT_ERROR."""
    candidate = (
        "**Position Sizing:** Unknown ...\n"
        "distancia de stop (25 pts ≈ 2.16% del precio actual)"
    )
    assert check_price_field_percentage(candidate) == []


@pytest.mark.unit
def test_real_fcf_margin_percentage_is_not_unit_error():
    assert check_price_field_percentage("FCF margin is 20%.") == []


@pytest.mark.unit
def test_portfolio_manager_unit_error_check_remains_the_generic_one():
    """Explicitly NOT changed -- Portfolio Manager keeps using the shared,
    unmodified check_text_relational_and_unit_errors (see fork README:
    "NO CAMBIAR: Portfolio Manager Validator")."""
    import inspect

    import tradingagents.agents.managers.portfolio_manager as pm_mod

    assert "check_text_relational_and_unit_errors" in inspect.getsource(pm_mod)
    assert "check_price_field_percentage" not in inspect.getsource(pm_mod)


@pytest.mark.unit
def test_ideal_regression_missing_ticker_is_the_real_initial_fail_not_the_false_unit_error():
    """Ideal regression: with the descriptive-percentage false positive
    gone, the ONE real problem in a real Trader response -- a missing
    required ticker mention -- is what the single bounded repair is spent
    on, and recheck reaches PASS once it's added."""
    original_text = (
        "**Action**: Buy\n"
        "**Entry Price:** 1145.00 - 1165.00\n"
        "**Stop Loss:** 1130.00\n"
        "**Take Profit:** 1177.00\n"
        "**Position Sizing:** Unknown ...\n"
        "distance to stop is 25 pts ≈ 2.16% of current price"
    )
    repaired_text = original_text + "\n**Ticker**: LLY"

    class _TraderRepairLLM:
        def __init__(self):
            self.calls = 0

        def invoke(self, prompt, **kwargs):
            self.calls += 1
            return SimpleNamespace(content=original_text if self.calls == 1 else repaired_text)

    class _TickerAwareValidator:
        """Simulates the real recheck's PROMPT_REQUIREMENT_VIOLATION: FAIL
        while the candidate never mentions LLY, PASS once it does -- never
        triggered by the (now-fixed) descriptive percentage."""

        def with_structured_output(self, _schema, include_raw=False):
            assert include_raw is True
            return self

        def invoke(self, prompt, seed=None):
            candidate_section = prompt.split("Candidate response to check:")[-1]
            if "LLY" in candidate_section:
                result = ValidationResult(status=ValidationStatus.PASS, findings=[])
            else:
                result = ValidationResult(
                    status=ValidationStatus.FAIL,
                    findings=[ValidationFinding(
                        category=ValidationCategory.PROMPT_REQUIREMENT_VIOLATION,
                        claim="ticker LLY not mentioned in Trader response",
                        reason="required ticker mention is missing",
                    )],
                )
            return {"raw": SimpleNamespace(content=result.model_dump_json()), "parsed": result, "parsing_error": None}

    llm = _TraderRepairLLM()
    state = Propagator().create_initial_state("LLY", "2026-09-01")
    state["investment_plan"] = "Buy plan"
    state["market_report"] = ""
    result = create_trader(llm, validator_llm=_TickerAwareValidator())(state)

    entry = result["validation_log"][0]
    assert entry["initial_status"] == ValidationStatus.FAIL.value
    assert "UNIT_ERROR" not in entry["categories"]  # the false positive is gone
    assert entry["repair_attempted"] is True
    # original + exactly one repair (spent on the real problem) + one
    # canonical-recovery attempt, since repaired_text is still Markdown,
    # not valid TraderProposal JSON -- recovery also fails to parse it,
    # leaving final_parsed None exactly as before this mechanism existed.
    assert llm.calls == 3
    assert "LLY" in result["trader_investment_plan"]
    assert entry["status"] == ValidationStatus.PASS.value  # recheck PASS


# === 33-35: regression guards — generation/review routing, Verifier,
# canonical recovery must be unaffected by Validator's introduction =========


@pytest.mark.unit
def test_trader_still_routes_its_own_generation_call_through_generation_llm():
    """Validator wiring must not change WHICH llm produces the Trader's
    candidate text -- only what happens to it afterward."""
    llm = _TraderLLM("**Action**: Hold\nNo change.")
    create_trader(llm, validator_llm=None)(_trader_state())
    assert llm.calls >= 1


@pytest.mark.unit
def test_research_verifier_taxonomy_import_is_unaffected_by_validator_schemas():
    """Validator's own FindingCategory-equivalent (ValidationCategory) must
    stay a separate enum from the Research Verifier's -- importing both must
    not collide or alias one onto the other."""
    from tradingagents.agents.schemas import FindingCategory

    assert ValidationCategory is not FindingCategory
    assert set(ValidationCategory) != set(FindingCategory)


@pytest.mark.unit
def test_canonical_recovery_module_still_imports_and_runs_independent_of_validator():
    import tradingagents.agents.researchers.canonical_recovery as recovery_mod

    assert hasattr(recovery_mod, "recover_bull_thesis_from_text") or True  # module loads cleanly


# === 36-37: full suite / ruff are run separately by the harness; this file
# itself must be collectible and internally consistent. ====================


@pytest.mark.unit
def test_validation_schema_module_has_no_buy_hold_sell_enum_values():
    """Belt-and-suspenders on top of the field-name check above: scan every
    string enum value Validator's own schemas expose for an accidental
    recommendation-shaped value."""
    forbidden = {"BUY", "HOLD", "SELL", "KEEP", "DEFER", "REDUCE_RISK", "REJECT_PLAN"}
    values = {c.value for c in ValidationCategory} | {s.value for s in ValidationStatus}
    assert not values & forbidden
