"""SEC EDGAR company events (Form 8-K) for the News Analyst -- see fork
README: "SEC EDGAR company events" and the audit/evaluation that preceded
this. Reuses sec_edgar.py's CIK resolution/HTTP/cache plumbing; adds the
submissions endpoint, point-in-time filtering gated strictly on filingDate,
deterministic Item-code categorization, and source_coverage. First scope:
Form 8-K only -- no LLM involved anywhere in extraction.

Numbered per the task's own "TESTS" list (1-20).
"""

from __future__ import annotations

from unittest import mock

import pytest

from tradingagents.dataflows import source_coverage as sc
from tradingagents.dataflows.vendors import sec_edgar, sec_edgar_events as see

CIK_PADDED = "0000789019"
TICKER_MAP = {
    "0": {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft Corporation"},
    "1": {"cik_str": 59478, "ticker": "LLY", "title": "Eli Lilly and Company"},
}


def _submissions(forms, filing_dates, accessions=None, primary_docs=None, items=None, acceptance=None):
    n = len(forms)
    return {
        "cik": "789019",
        "filings": {
            "recent": {
                "form": forms,
                "filingDate": filing_dates,
                "accessionNumber": accessions or [f"0000789019-26-{i:06d}" for i in range(n)],
                "primaryDocument": primary_docs or ["doc.htm"] * n,
                "items": items or [""] * n,
                "acceptanceDateTime": acceptance or [f"{d}T16:05:00.000Z" for d in filing_dates],
            }
        },
    }


# A realistic MSFT-like window: mostly Form 4 insider filings, one real 8-K
# with two Item codes -- mirrors the real submissions response inspected
# during the audit for MSFT 2026-09-30, 30-day window.
MSFT_LIKE = _submissions(
    forms=["4", "4", "8-K", "4", "144"],
    filing_dates=["2026-09-17", "2026-09-11", "2026-09-02", "2026-09-01", "2026-09-01"],
    accessions=["0000789019-26-000224", "0000789019-26-000208", "0001193125-26-380280",
                "0000789019-26-000159", "0001950047-26-008927"],
    primary_docs=["form4.xml", "form4.xml", "d291965d8k.htm", "form4.xml", "primary_doc.xml"],
    items=["", "", "7.01,9.01", "", ""],
)

# A realistic LLY-like window: only Form 4/3-A insider filings, zero 8-K --
# mirrors the real submissions response inspected for LLY 2026-09-30,
# 30-day window (0 of 8 filings were 8-K).
LLY_LIKE = _submissions(
    forms=["4", "4", "4", "4", "4", "3/A", "4", "4"],
    filing_dates=["2026-09-22", "2026-09-22", "2026-09-22", "2026-09-22",
                   "2026-09-11", "2026-09-04", "2026-09-04", "2026-09-04"],
)


@pytest.fixture(autouse=True)
def _clean(monkeypatch, tmp_path):
    monkeypatch.setattr(sec_edgar, "get_config", lambda: {"data_cache_dir": str(tmp_path)})
    sc.set_active_source_coverage_collector(None)
    yield
    sc.set_active_source_coverage_collector(None)


def _mock_fetch(monkeypatch, submissions_payload, call_log=None):
    def fake_fetch_json(url):
        if call_log is not None:
            call_log.append(url)
        if "company_tickers" in url:
            return TICKER_MAP
        return submissions_payload
    monkeypatch.setattr(sec_edgar, "_fetch_json", fake_fetch_json)


# --- 1: ticker -> CIK is reused -----------------------------------------


@pytest.mark.unit
def test_1_ticker_to_cik_resolution_is_reused(monkeypatch):
    _mock_fetch(monkeypatch, MSFT_LIKE)
    spy = mock.Mock(side_effect=sec_edgar.cik_for)
    monkeypatch.setattr(see, "cik_for", spy)
    see.fetch_company_events("MSFT", "2026-09-30", 30)
    spy.assert_called_once_with("MSFT")


# --- 2: submissions fetch/cache ------------------------------------------


@pytest.mark.unit
def test_2_submissions_are_fetched_and_cached(monkeypatch):
    calls = []
    _mock_fetch(monkeypatch, MSFT_LIKE, call_log=calls)
    see.fetch_company_events("MSFT", "2026-09-30", 30)
    submissions_calls = [c for c in calls if "submissions" in c]
    assert len(submissions_calls) == 1
    assert CIK_PADDED in submissions_calls[0]


# --- 3: 8-K inside the window passes -------------------------------------


@pytest.mark.unit
def test_3_8k_inside_window_passes(monkeypatch):
    _mock_fetch(monkeypatch, MSFT_LIKE)
    monkeypatch.setattr(see, "_extract_excerpt", lambda url, items: (None, "WRAPPER_ONLY"))
    events, _ = see.fetch_company_events("MSFT", "2026-09-30", 30)
    assert len(events) == 1
    assert events[0].form == "8-K"
    assert events[0].filed_at == "2026-09-02"


# --- 4: future 8-K is excluded --------------------------------------------


@pytest.mark.unit
def test_4_future_8k_is_excluded(monkeypatch):
    future = _submissions(forms=["8-K"], filing_dates=["2026-10-01"], items=["8.01"])
    _mock_fetch(monkeypatch, future)
    events, raw_count = see.fetch_company_events("MSFT", "2026-09-30", 30)
    assert events == []
    assert raw_count == 0


# --- 5: filing before start_date is excluded ------------------------------


@pytest.mark.unit
def test_5_filing_before_window_start_is_excluded(monkeypatch):
    too_old = _submissions(forms=["8-K"], filing_dates=["2026-07-01"], items=["8.01"])
    _mock_fetch(monkeypatch, too_old)
    events, raw_count = see.fetch_company_events("MSFT", "2026-09-30", 30)
    assert events == []
    assert raw_count == 0


# --- 6: other forms are never returned as company events ------------------


@pytest.mark.unit
def test_6_non_8k_forms_are_not_returned_as_events_but_count_as_raw(monkeypatch):
    _mock_fetch(monkeypatch, MSFT_LIKE)
    monkeypatch.setattr(see, "_extract_excerpt", lambda url, items: (None, "WRAPPER_ONLY"))
    events, raw_count = see.fetch_company_events("MSFT", "2026-09-30", 30)
    assert all(e.form == "8-K" for e in events)
    assert raw_count == 5  # every filing in the window, any form
    assert len(events) == 1  # only the one 8-K


# --- 7: item codes are parsed ---------------------------------------------


@pytest.mark.unit
def test_7_item_codes_are_parsed():
    assert see._parse_items("7.01,9.01") == ("7.01", "9.01")
    assert see._parse_items("") == ()
    assert see._parse_items("2.02") == ("2.02",)


# --- 8: categorization is correct ------------------------------------------


@pytest.mark.unit
def test_8_categorization_is_correct():
    assert see._categories_for(("2.02",)) == ("earnings/results",)
    assert see._categories_for(("5.02",)) == ("management/director change",)
    assert see._categories_for(("9.99",)) == ()  # unknown item -> no invented category


# --- 9: multiple item codes ------------------------------------------------


@pytest.mark.unit
def test_9_multiple_item_codes_produce_multiple_categories(monkeypatch):
    _mock_fetch(monkeypatch, MSFT_LIKE)
    monkeypatch.setattr(see, "_extract_excerpt", lambda url, items: (None, "WRAPPER_ONLY"))
    events, _ = see.fetch_company_events("MSFT", "2026-09-30", 30)
    assert events[0].item_numbers == ("7.01", "9.01")
    assert events[0].categories == ("regulation FD disclosure", "financial statements/exhibits")


# --- 10: URL/accession are correct -----------------------------------------


@pytest.mark.unit
def test_10_url_and_accession_are_correct(monkeypatch):
    _mock_fetch(monkeypatch, MSFT_LIKE)
    monkeypatch.setattr(see, "_extract_excerpt", lambda url, items: (None, "WRAPPER_ONLY"))
    events, _ = see.fetch_company_events("MSFT", "2026-09-30", 30)
    event = events[0]
    assert event.accession_number == "0001193125-26-380280"
    assert event.url == (
        "https://www.sec.gov/Archives/edgar/data/789019/000119312526380280/d291965d8k.htm"
    )


# --- 11: source_coverage SUCCESS_WITH_DATA ---------------------------------


@pytest.mark.unit
def test_11_source_coverage_success_with_data(monkeypatch):
    from tradingagents.agents.tools import get_company_events

    _mock_fetch(monkeypatch, MSFT_LIKE)
    monkeypatch.setattr(see, "_extract_excerpt", lambda url, items: (None, "WRAPPER_ONLY"))
    collector = sc.SourceCoverageCollector()
    sc.set_active_source_coverage_collector(collector)
    get_company_events.func("MSFT", "2026-09-30", 30)
    entry = collector.snapshot()["news"]["company_events"]
    assert entry["status"] == "SUCCESS_WITH_DATA"
    assert entry["raw_count"] == 5
    assert entry["usable_count"] == 1


# --- 12: source_coverage SUCCESS_EMPTY --------------------------------------


@pytest.mark.unit
def test_12_source_coverage_success_empty(monkeypatch):
    from tradingagents.agents.tools import get_company_events

    _mock_fetch(monkeypatch, LLY_LIKE)
    collector = sc.SourceCoverageCollector()
    sc.set_active_source_coverage_collector(collector)
    get_company_events.func("LLY", "2026-09-30", 30)
    entry = collector.snapshot()["news"]["company_events"]
    assert entry["status"] == "SUCCESS_EMPTY"
    assert entry["raw_count"] == 8
    assert entry["usable_count"] == 0


# --- 13: cache avoids a duplicate fetch -------------------------------------


@pytest.mark.unit
def test_13_cache_avoids_duplicate_fetch(monkeypatch):
    calls = []
    _mock_fetch(monkeypatch, MSFT_LIKE, call_log=calls)
    monkeypatch.setattr(see, "_extract_excerpt", lambda url, items: (None, "WRAPPER_ONLY"))
    see.fetch_company_events("MSFT", "2026-09-30", 30)
    see.fetch_company_events("MSFT", "2026-09-30", 30)
    submissions_calls = [c for c in calls if "submissions" in c]
    assert len(submissions_calls) == 1  # second call served from the file cache


# --- 14: News gets a separate block -----------------------------------------


@pytest.mark.unit
def test_14_news_gets_a_separate_company_events_block(monkeypatch):
    from tradingagents.agents.tools import get_company_events

    _mock_fetch(monkeypatch, MSFT_LIKE)
    monkeypatch.setattr(see, "_extract_excerpt", lambda url, items: ("Example excerpt text.", "EXCERPT_AVAILABLE"))
    result = get_company_events.func("MSFT", "2026-09-30", 30)
    assert result.startswith("## SEC COMPANY EVENTS")
    assert "Items 7.01, 9.01" in result
    assert "Categories: regulation FD disclosure; financial statements/exhibits" in result
    assert "Source: SEC EDGAR" in result


# --- 15: Fundamentals is unaffected ------------------------------------------


@pytest.mark.unit
def test_15_fundamentals_module_is_untouched_by_the_new_module():
    # The new module only imports from sec_edgar (CIK/HTTP/cache plumbing);
    # it defines no statement logic and never touches _STATEMENTS/_as_of.
    import inspect

    source = inspect.getsource(see)
    assert "_STATEMENTS" not in source
    assert "_as_of" not in source
    assert "get_balance_sheet" not in source


# --- 16: Sentiment is unaffected ---------------------------------------------


@pytest.mark.unit
def test_16_sentiment_module_does_not_import_sec_edgar_events():
    import inspect

    import tradingagents.agents.analysts.sentiment_analyst as sentiment_mod

    source = inspect.getsource(sentiment_mod)
    assert "sec_edgar" not in source


# --- 17: no LLM is used for extraction ---------------------------------------


@pytest.mark.unit
def test_17_no_llm_is_used_anywhere_in_extraction():
    import inspect

    source = inspect.getsource(see)
    for forbidden in ("ChatOpenAI", "generation_llm", "invoke_structured", "bind_tools", "ChatAnthropic"):
        assert forbidden not in source


# --- 18: LLY-like empty case --------------------------------------------------


@pytest.mark.unit
def test_18_lly_like_case_is_success_empty(monkeypatch):
    _mock_fetch(monkeypatch, LLY_LIKE)
    events, raw_count = see.fetch_company_events("LLY", "2026-09-30", 30)
    assert events == []
    assert raw_count == 8
    rendered = see.render_company_events(events, "LLY", "2026-08-31", "2026-09-30")
    assert "No SEC Form 8-K company events found" in rendered
    assert "not an absence of news" in rendered


# --- 19: MSFT-like 8-K case ---------------------------------------------------


@pytest.mark.unit
def test_19_msft_like_case_is_success_with_data(monkeypatch):
    _mock_fetch(monkeypatch, MSFT_LIKE)
    monkeypatch.setattr(see, "_extract_excerpt", lambda url, items: (None, "WRAPPER_ONLY"))
    events, raw_count = see.fetch_company_events("MSFT", "2026-09-30", 30)
    assert raw_count == 5
    assert len(events) == 1
    event = events[0]
    assert event.filed_at == "2026-09-02"
    assert event.item_numbers == ("7.01", "9.01")
    assert event.point_in_time_valid is True


# --- 20: no future leakage, exact boundary ------------------------------------


@pytest.mark.unit
def test_20_no_future_leakage_at_the_exact_boundary(monkeypatch):
    boundary = _submissions(
        forms=["8-K", "8-K", "8-K"],
        filing_dates=["2026-09-30", "2026-10-01", "2026-08-31"],
        items=["8.01", "8.01", "8.01"],
    )
    _mock_fetch(monkeypatch, boundary)
    monkeypatch.setattr(see, "_extract_excerpt", lambda url, items: (None, "WRAPPER_ONLY"))
    events, raw_count = see.fetch_company_events("MSFT", "2026-09-30", 30)
    filed_dates = {e.filed_at for e in events}
    assert "2026-09-30" in filed_dates   # on the analysis date itself -- included
    assert "2026-10-01" not in filed_dates   # one day after -- excluded
    assert "2026-08-31" in filed_dates   # within the 30-day window start -- included
    assert raw_count == 2  # only the two in-window filings count; the future one never does
