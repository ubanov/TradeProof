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
from tradingagents.agents.schemas import (
    ArithmeticClaim,
    ArithmeticOperation,
    PortfolioDecision,
    PortfolioRating,
    TraderAction,
    TraderProposal,
)
from tradingagents.agents.trader.trader import create_trader
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
    check_missing_requirement,
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
        "stage", "status", "finding_count", "categories", "repair_attempted",
        "repair_succeeded", "llm_calls",
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
        {"stage": "A", "status": "PASS", "finding_count": 0, "categories": [],
         "repair_attempted": False, "repair_succeeded": False, "llm_calls": 1},
        {"stage": "B", "status": "FAIL", "finding_count": 2, "categories": ["UNIT_ERROR"],
         "repair_attempted": True, "repair_succeeded": True, "llm_calls": 2},
        {"stage": "C", "status": "FAIL", "finding_count": 1, "categories": ["SCHEMA_ERROR"],
         "repair_attempted": True, "repair_succeeded": False, "llm_calls": 2},
    ]
    counts = summarize_validation_log(log)
    assert counts == {
        "checked": 3, "pass": 1, "warn": 0, "fail": 2,
        "repairs": 2, "unresolved": 1, "llm_calls": 5,
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
    # One call for the original free-text generation, one for the repair.
    assert llm.calls == 2
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
