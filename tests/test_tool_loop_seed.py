"""Best-effort deterministic replay: Market/News/Fundamentals Analyst seed fix.

These three analysts build ``prompt | llm.bind_tools(TOOLS)`` and invoke the
*chain*. A ``RunnableSequence`` only forwards ``invoke()``'s extra kwargs to
its FIRST step -- here, the prompt template, which silently drops a ``seed``
kwarg instead of passing it to the LLM. The fix binds tools, pre-formats the
prompt, and invokes the *bound LLM directly* (mirroring the already-correct
Sentiment Analyst pattern), so ``traced_invoke``'s derived seed actually
reaches the model. These tests assert the seed on the fake model's
``_generate`` kwargs -- not just that it was passed to some earlier layer --
so a regression back to the chain-piped pattern would fail them.
"""
from __future__ import annotations

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from tradingagents.agents.analysts.fundamentals_analyst import create_fundamentals_analyst
from tradingagents.agents.analysts.market_analyst import create_market_analyst
from tradingagents.agents.analysts.news_analyst import create_news_analyst
from tradingagents.agents.analysts.sentiment_analyst import create_sentiment_analyst
from tradingagents.llm_trace import SeedContext, set_active_seed_context
from tradingagents.run_seed import derive_call_seed


class RecordingToolModel(BaseChatModel):
    """A real BaseChatModel (not a mock) so LangChain's own invoke() ->
    _generate() kwarg-forwarding plumbing is exercised for real, the same way
    ChatOpenAI's is. Answers with no tool call so the analyst's tool-loop
    (``_tools_or_clear``) ends after exactly one round."""

    calls: list = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "recording-tool-model"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.calls.append(kwargs)
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content="done", tool_calls=[]))]
        )


def _state():
    return {
        "trade_date": "2026-01-01",
        "company_of_interest": "NVDA",
        "instrument_context": "NVDA - NVIDIA Corporation",
        "asset_type": "stock",
        "messages": [HumanMessage(content="Analyze.")],
    }


@pytest.fixture(autouse=True)
def _reset_seed_context():
    set_active_seed_context(None)
    yield
    set_active_seed_context(None)


# --- 1, 2, 3: Market/News/Fundamentals tool-loop receives the seed ----------


@pytest.mark.unit
def test_market_analyst_tool_loop_receives_seed():
    llm = RecordingToolModel()
    set_active_seed_context(SeedContext(base_seed=12345, ticker="NVDA", provider_supports_seed=True))

    create_market_analyst(llm)(_state())

    assert len(llm.calls) == 1
    expected = derive_call_seed(12345, "NVDA", "unknown", "normal:tool_round_1")
    assert llm.calls[0].get("seed") == expected


@pytest.mark.unit
def test_news_analyst_tool_loop_receives_seed():
    llm = RecordingToolModel()
    set_active_seed_context(SeedContext(base_seed=12345, ticker="NVDA", provider_supports_seed=True))

    create_news_analyst(llm)(_state())

    assert len(llm.calls) == 1
    expected = derive_call_seed(12345, "NVDA", "unknown", "normal:tool_round_1")
    assert llm.calls[0].get("seed") == expected


@pytest.mark.unit
def test_fundamentals_analyst_tool_loop_receives_seed():
    llm = RecordingToolModel()
    set_active_seed_context(SeedContext(base_seed=12345, ticker="NVDA", provider_supports_seed=True))

    create_fundamentals_analyst(llm)(_state())

    assert len(llm.calls) == 1
    expected = derive_call_seed(12345, "NVDA", "unknown", "normal:tool_round_1")
    assert llm.calls[0].get("seed") == expected


@pytest.mark.unit
def test_market_analyst_seed_changes_across_tool_rounds():
    """A second tool round (longer message history) must derive a different
    seed -- call_identity is keyed on the running message count."""
    llm = RecordingToolModel()
    set_active_seed_context(SeedContext(base_seed=12345, ticker="NVDA", provider_supports_seed=True))

    state = _state()
    create_market_analyst(llm)(state)
    state["messages"] = state["messages"] + [HumanMessage(content="more")]
    create_market_analyst(llm)(state)

    assert len(llm.calls) == 2
    assert llm.calls[0]["seed"] != llm.calls[1]["seed"]


@pytest.mark.unit
def test_no_seed_context_means_no_seed_kwarg_for_tool_loop():
    llm = RecordingToolModel()
    create_market_analyst(llm)(_state())
    assert "seed" not in llm.calls[0]


# --- 4: the fix does not change Sentiment (or other) nodes ------------------


class RecordingStructuredModel(BaseChatModel):
    """Mirrors Sentiment Analyst's no-tools, with_structured_output-less
    (free-text fallback) path -- bind_structured raises NotImplementedError
    so invoke_structured_or_freetext falls straight to the plain-LLM path,
    which is where the seed must still arrive, exactly as before this fix."""

    calls: list = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "recording-structured-model"

    def with_structured_output(self, schema, **kwargs):
        raise NotImplementedError

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.calls.append(kwargs)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="report"))])


@pytest.mark.unit
def test_sentiment_analyst_still_receives_seed_unchanged(monkeypatch):
    """Sentiment Analyst never used the buggy chain-piped pattern -- this
    pins that the tool-loop fix did not regress it."""
    import tradingagents.agents.analysts.sentiment_analyst as sentiment_mod

    monkeypatch.setattr(sentiment_mod, "get_news", type("T", (), {"func": staticmethod(lambda *a, **k: "")}))
    monkeypatch.setattr(sentiment_mod, "fetch_stocktwits_messages", lambda *a, **k: "")
    monkeypatch.setattr(sentiment_mod, "fetch_reddit_posts", lambda *a, **k: "")
    monkeypatch.setattr(sentiment_mod, "jev_screen", lambda *a, **k: None)

    llm = RecordingStructuredModel()
    set_active_seed_context(SeedContext(base_seed=12345, ticker="NVDA", provider_supports_seed=True))

    create_sentiment_analyst(llm)(_state())

    assert len(llm.calls) == 1
    assert "seed" in llm.calls[0]
