"""Spanish Trader action/recommendation labels must parse like their English
equivalents -- a real report containing "**Acción: Buy**" used to come back
as action: null in result.json because the English-only label parser did not
recognize "Acción" (see fork README: "Parse Spanish Trader action")."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from cli.headless import result_document


def _execution(trader_plan, research_plan="", final_decision="", rating="REVIEW"):
    state = {
        "trader_investment_plan": trader_plan,
        "investment_plan": research_plan,
        "final_trade_decision": final_decision,
    }
    return SimpleNamespace(
        final_state=state,
        ticker="NVDA",
        analysis_date="2026-09-01",
        rating=rating,
        report_path=None,
        duration_seconds=1.0,
        stats={"llm_calls": 1, "tool_calls": 0, "tokens_in": 1, "tokens_out": 1},
        fallback_warnings=0,
        base_seed=None,
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "trader_plan, expected",
    [
        ("**Acción: Buy**\n\nEnter now.", "Buy"),
        ("**Acción:** **Sell**\n\nExit now.", "Sell"),
        ("Acción: Hold\n\nNo change.", "Hold"),
    ],
)
def test_spanish_accion_label_parses_the_trader_action(trader_plan, expected, tmp_path):
    result = result_document(_execution(trader_plan), tmp_path)
    assert result["action"] == expected


@pytest.mark.unit
def test_accion_without_accent_mark_also_parses():
    result = result_document(_execution("Accion: Hold\n\nNo change."), None)
    assert result["action"] == "Hold"


@pytest.mark.unit
@pytest.mark.parametrize(
    "trader_plan, expected",
    [
        ("**Action**: Buy\n\nEnter now.", "Buy"),
        ("**Action:** **Sell**\n\nExit now.", "Sell"),
        ("Action: Hold\n\nNo change.", "Hold"),
    ],
)
def test_existing_english_action_cases_still_work(trader_plan, expected):
    result = result_document(_execution(trader_plan), None)
    assert result["action"] == expected


@pytest.mark.unit
def test_spanish_recomendacion_label_parses_the_research_recommendation():
    result = result_document(
        _execution("**Action**: Buy", research_plan="**Recomendación: Overweight**\n\nAdd gradually."),
        None,
    )
    assert result["research_recommendation"] == "Overweight"


@pytest.mark.unit
def test_unparseable_action_stays_null_not_guessed():
    result = result_document(
        _execution("The team debated Buy versus Sell at length without a clear call."), None
    )
    assert result["action"] is None


@pytest.mark.unit
def test_trader_action_research_recommendation_and_rating_stay_separate_fields():
    """Concept separation: Trader action, Research Manager recommendation,
    Portfolio Manager rating, and Portfolio Manager disposition must never
    overwrite each other even when all four are present at once."""
    result = result_document(
        _execution(
            "**Acción: Buy**\n\nEnter now.",
            research_plan="**Recomendación: Overweight**\n\nAdd gradually.",
            final_decision="**Rating**: Hold\n\n**Disposition**: DEFER\n\n**Executive Summary**: x",
            rating="Hold",
        ),
        None,
    )
    assert result["action"] == "Buy"
    assert result["research_recommendation"] == "Overweight"
    assert result["rating"] == "Hold"
