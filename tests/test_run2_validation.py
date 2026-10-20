"""Generic structured-call-with-one-repair orchestration (fork README:
"run2": "si una salida estructurada falla: un unico intento de repair; si
sigue fallando, degradar"). Uses ``BarrierSignal`` as the schema under test
-- it has exactly one numeric field (``confidence``), which keeps the
anti-hallucination check in
``structured_recovery.recover_structured_from_text`` (every number the
repair introduces must already appear in the free text) easy to satisfy
honestly in a fixture.
"""

from __future__ import annotations

import json

import pytest

from tradingagents.run2.schemas import BarrierSignal
from tradingagents.run2.validation import (
    STATUS_FAILED,
    STATUS_OK,
    STATUS_REPAIRED,
    run_structured_stage,
)


class _Resp:
    def __init__(self, content: str):
        self.content = content


class _FakeChatModel:
    """``structured_result``/``structured_raises`` control
    ``with_structured_output(schema).invoke(...)``; ``plain_responses`` is a
    FIFO queue of ``.content`` strings returned by plain ``.invoke(...)`` --
    used for both the freetext fallback (1st call) and, when this object is
    also passed as ``repair_llm``, the one repair call (2nd call)."""

    def __init__(self, *, structured_result=None, structured_raises=False, plain_responses=None):
        self.structured_result = structured_result
        self.structured_raises = structured_raises
        self.plain_responses = list(plain_responses or [])
        self.structured_calls = 0
        self.plain_calls = 0

    def with_structured_output(self, schema):
        outer = self

        class _Bound:
            def invoke(self, prompt):
                outer.structured_calls += 1
                if outer.structured_raises:
                    raise ValueError("structured output failed")
                return outer.structured_result

        return _Bound()

    def invoke(self, prompt):
        self.plain_calls += 1
        content = self.plain_responses.pop(0) if self.plain_responses else ""
        return _Resp(content)


def _signal_json(**overrides):
    base = {
        "direction": "LONG", "confidence": 0.55, "thesis": "t",
        "evidence_for": [], "evidence_against": [], "invalidation_conditions": [],
    }
    base.update(overrides)
    return json.dumps(base)


@pytest.mark.unit
def test_1_structured_call_succeeds_on_first_try_no_repair_needed():
    signal = BarrierSignal(direction="LONG", confidence=0.55, thesis="t")
    llm = _FakeChatModel(structured_result=signal)
    parsed, meta = run_structured_stage(llm, BarrierSignal, "prompt", "First Barrier Signal Analyst", llm_role="generation")
    assert parsed is signal
    assert meta["status"] == STATUS_OK
    assert meta["repaired"] is False
    assert llm.plain_calls == 0


@pytest.mark.unit
def test_2_malformed_structured_output_triggers_exactly_one_repair_that_succeeds():
    free_text = "The confidence is 0.55 and the direction is LONG."
    repair_json = _signal_json(direction="LONG", confidence=0.55)
    llm = _FakeChatModel(structured_raises=True, plain_responses=[free_text, repair_json])

    parsed, meta = run_structured_stage(llm, BarrierSignal, "prompt", "First Barrier Signal Analyst", llm_role="generation")

    assert parsed is not None
    assert parsed.direction == "LONG"
    assert meta["status"] == STATUS_REPAIRED
    assert meta["repair_attempts"] == 1
    # Exactly one freetext-fallback call + exactly one repair call -- never more.
    assert llm.plain_calls == 2


@pytest.mark.unit
def test_3_persistent_failure_after_the_one_repair_attempt_returns_none():
    free_text = "The model refused to answer in the requested shape."
    llm = _FakeChatModel(structured_raises=True, plain_responses=[free_text, "not valid json at all"])

    parsed, meta = run_structured_stage(llm, BarrierSignal, "prompt", "First Barrier Signal Analyst", llm_role="generation")

    assert parsed is None
    assert meta["status"] == STATUS_FAILED
    assert meta["repair_attempts"] == 1
    assert llm.plain_calls == 2  # still exactly one repair attempt, not a retry loop


@pytest.mark.unit
def test_4_repair_rejects_a_hallucinated_number_not_present_in_the_free_text():
    free_text = "The confidence is 0.55 and the direction is LONG."
    # 0.99 never appears in free_text -- recover_structured_from_text must reject this.
    repair_json = _signal_json(direction="LONG", confidence=0.99)
    llm = _FakeChatModel(structured_raises=True, plain_responses=[free_text, repair_json])

    parsed, meta = run_structured_stage(llm, BarrierSignal, "prompt", "First Barrier Signal Analyst", llm_role="generation")

    assert parsed is None
    assert meta["status"] == STATUS_FAILED


@pytest.mark.unit
def test_5_an_independent_repair_llm_is_used_instead_of_the_original():
    free_text = "The confidence is 0.4 and the direction is SHORT."
    repair_json = _signal_json(direction="SHORT", confidence=0.4)
    original = _FakeChatModel(structured_raises=True, plain_responses=[free_text])
    repair_llm = _FakeChatModel(plain_responses=[repair_json])

    parsed, meta = run_structured_stage(
        original, BarrierSignal, "prompt", "First Barrier Signal Analyst", llm_role="generation", repair_llm=repair_llm,
    )

    assert parsed is not None
    assert meta["status"] == STATUS_REPAIRED
    # The original model only ever produced the freetext fallback; the repair
    # model (not the original) made the one repair call.
    assert original.plain_calls == 1
    assert repair_llm.plain_calls == 1
