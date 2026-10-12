"""Tests for deterministic instrument-identity resolution (#814) and the
context-anchored message placeholder (#888)."""

import unittest
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage

from tradingagents.agents.context import (
    build_instrument_context,
    create_msg_delete,
    get_instrument_context_from_state,
    resolve_instrument_identity,
)
from tradingagents.dataflows.date_window import get_current_date

_TODAY = get_current_date()
_FUTURE_DATE = (datetime.strptime(_TODAY, "%Y-%m-%d") + timedelta(days=30)).strftime("%Y-%m-%d")
_PAST_DATE = (datetime.strptime(_TODAY, "%Y-%m-%d") - timedelta(days=30)).strftime("%Y-%m-%d")

_FULL_IDENTITY = {
    "company_name": "Microsoft Corporation",
    "sector": "Technology",
    "industry": "Software - Infrastructure",
    "exchange": "NMS",
    "quote_type": "EQUITY",
}


@pytest.mark.unit
class ResolveInstrumentIdentityTests(unittest.TestCase):
    def setUp(self):
        resolve_instrument_identity.cache_clear()

    def test_resolves_company_metadata_from_yfinance(self):
        with patch("tradingagents.dataflows.vendors.yahoo.market.yf.Ticker") as mock:
            mock.return_value.info = {
                "longName": "TOTO LTD.",
                "shortName": "TOTO",
                "sector": "Industrials",
                "industry": "Building Products & Equipment",
                "exchange": "PNK",
                "quoteType": "EQUITY",
            }
            identity = resolve_instrument_identity("totdy")
        mock.assert_called_once_with("TOTDY")
        self.assertEqual(identity["company_name"], "TOTO LTD.")
        self.assertEqual(identity["sector"], "Industrials")
        self.assertEqual(identity["industry"], "Building Products & Equipment")
        self.assertEqual(identity["exchange"], "PNK")

    def test_falls_back_to_short_name(self):
        with patch("tradingagents.dataflows.vendors.yahoo.market.yf.Ticker") as mock:
            mock.return_value.info = {"shortName": "TOTO", "sector": "Industrials"}
            identity = resolve_instrument_identity("TOTDY")
        self.assertEqual(identity["company_name"], "TOTO")

    def test_skips_placeholder_values(self):
        with patch("tradingagents.dataflows.vendors.yahoo.market.yf.Ticker") as mock:
            mock.return_value.info = {"longName": "  ", "sector": "None", "industry": "n/a"}
            identity = resolve_instrument_identity("TOTDY")
        self.assertEqual(identity, {})

    def test_fails_open_on_exception(self):
        with patch(
            "tradingagents.dataflows.vendors.yahoo.market.yf.Ticker",
            side_effect=RuntimeError("rate limited"),
        ):
            self.assertEqual(resolve_instrument_identity("TOTDY"), {})

    def test_result_is_cached(self):
        with patch("tradingagents.dataflows.vendors.yahoo.market.yf.Ticker") as mock:
            mock.return_value.info = {"longName": "TOTO LTD."}
            first = resolve_instrument_identity("TOTDY")
            second = resolve_instrument_identity("TOTDY")
        mock.assert_called_once()  # second call served from cache
        self.assertEqual(first, second)


@pytest.mark.unit
class BuildInstrumentContextTests(unittest.TestCase):
    def test_mentions_exact_symbol_without_identity(self):
        context = build_instrument_context("7203.T")
        self.assertIn("7203.T", context)
        self.assertIn("exchange suffix", context)
        self.assertNotIn("Resolved identity", context)

    def test_injects_resolved_identity(self):
        context = build_instrument_context(
            "TOTDY", "stock",
            {
                "company_name": "TOTO LTD.",
                "sector": "Industrials",
                "industry": "Building Products & Equipment",
                "exchange": "PNK",
            },
        )
        self.assertIn("Company: TOTO LTD.", context)
        self.assertIn("Industrials / Building Products & Equipment", context)
        self.assertIn("Exchange: PNK", context)
        self.assertIn("Do not substitute a different company", context)

    def test_crypto_uses_name_label_and_keeps_hint(self):
        context = build_instrument_context(
            "BTC-USD", "crypto", {"company_name": "Bitcoin USD"}
        )
        self.assertIn("Name: Bitcoin USD", context)
        self.assertIn("crypto asset rather than a company", context)

    # --- 1: live run keeps today's full profile, unchanged ------------------

    def test_live_run_shows_full_identity_no_curr_date(self):
        context = build_instrument_context("MSFT", "stock", _FULL_IDENTITY)
        self.assertIn("Company: Microsoft Corporation", context)
        self.assertIn("Technology / Software - Infrastructure", context)
        self.assertIn("Exchange: NMS", context)
        self.assertNotIn("withheld", context)
        self.assertNotIn("Historical instrument identity anchor", context)

    def test_live_run_shows_full_identity_curr_date_today_or_future(self):
        for curr_date in (_TODAY, _FUTURE_DATE):
            with self.subTest(curr_date=curr_date):
                context = build_instrument_context("MSFT", "stock", _FULL_IDENTITY, curr_date)
                self.assertIn("Company: Microsoft Corporation", context)
                self.assertIn("Technology / Software - Infrastructure", context)
                self.assertIn("Exchange: NMS", context)
                self.assertNotIn("withheld", context)

    # --- 2: historical run withholds sector/industry/exchange ---------------

    def test_historical_run_keeps_only_name_and_quote_type(self):
        context = build_instrument_context("MSFT", "stock", _FULL_IDENTITY, _PAST_DATE)
        self.assertIn("Ticker: MSFT", context)
        self.assertIn("Current vendor-reported company name: Microsoft Corporation", context)
        self.assertIn("Quote type: EQUITY", context)
        self.assertIn(_PAST_DATE, context)
        self.assertIn("not point-in-time evidence", context)
        self.assertIn("intentionally withheld", context)
        self.assertIn("Historical sector, industry and exchange metadata are unavailable", context)

    def test_historical_run_never_leaks_sector_industry_exchange_values(self):
        # --- 3: no accidental leak in some other phrase, anywhere in the text.
        context = build_instrument_context("MSFT", "stock", _FULL_IDENTITY, _PAST_DATE)
        self.assertNotIn("Technology", context)
        self.assertNotIn("Software - Infrastructure", context)
        self.assertNotIn("NMS", context)
        self.assertNotIn("Business classification", context)
        self.assertNotIn("Sector:", context)
        self.assertNotIn("Industry:", context)
        self.assertNotIn("Exchange:", context)

    def test_historical_run_crypto_uses_asset_wording(self):
        context = build_instrument_context(
            "BTC-USD", "crypto", {"company_name": "Bitcoin USD", "quote_type": "CRYPTOCURRENCY"}, _PAST_DATE,
        )
        self.assertIn("Current vendor-reported name: Bitcoin USD", context)
        self.assertIn("a different asset", context)
        self.assertNotIn("a different company", context)

    def test_historical_run_without_identity_adds_no_anchor_or_withhold_text(self):
        # A failed/empty lookup must not fabricate an anchor or a withhold
        # notice about data that was never fetched in the first place.
        context = build_instrument_context("MSFT", "stock", {}, _PAST_DATE)
        self.assertNotIn("Historical instrument identity anchor", context)
        self.assertNotIn("withheld", context)

    # --- 4: the SAME cached identity renders differently live vs historical -

    def test_same_identity_dict_renders_differently_live_vs_historical(self):
        live = build_instrument_context("MSFT", "stock", _FULL_IDENTITY, _FUTURE_DATE)
        historical = build_instrument_context("MSFT", "stock", _FULL_IDENTITY, _PAST_DATE)
        self.assertIn("Exchange: NMS", live)
        self.assertNotIn("Exchange: NMS", historical)
        self.assertIn("Current vendor-reported company name", historical)
        self.assertNotIn("Current vendor-reported company name", live)

    # --- 6: instrument_context (built once) still reaches state/agents ------

    def test_historical_context_flows_through_get_instrument_context_from_state(self):
        precomputed = build_instrument_context("MSFT", "stock", _FULL_IDENTITY, _PAST_DATE)
        state = {"company_of_interest": "MSFT", "instrument_context": precomputed}
        self.assertEqual(get_instrument_context_from_state(state), precomputed)
        self.assertIn("intentionally withheld", get_instrument_context_from_state(state))


@pytest.mark.unit
class GetInstrumentContextFromStateTests(unittest.TestCase):
    def test_prefers_precomputed_context(self):
        state = {"company_of_interest": "TOTDY", "instrument_context": "PRECOMPUTED"}
        self.assertEqual(get_instrument_context_from_state(state), "PRECOMPUTED")

    def test_fallback_is_network_free_ticker_only(self):
        # No instrument_context and no yfinance call — must not hit the network.
        with patch("tradingagents.dataflows.vendors.yahoo.market.yf.Ticker") as mock:
            context = get_instrument_context_from_state(
                {"company_of_interest": "NVDA", "asset_type": "stock"}
            )
        mock.assert_not_called()
        self.assertIn("NVDA", context)

    def test_fallback_respects_asset_type(self):
        context = get_instrument_context_from_state(
            {"company_of_interest": "BTC-USD", "asset_type": "crypto"}
        )
        self.assertIn("crypto asset", context)


@pytest.mark.unit
class ContextAnchoredPlaceholderTests(unittest.TestCase):
    """#888 — the message-clear placeholder must not be a bare 'Continue'."""

    def _run(self, state_extra):
        state = {
            "messages": [
                HumanMessage(content="old", id="h1"),
                AIMessage(content="reply", id="a1"),
            ],
            **state_extra,
        }
        return create_msg_delete()(state)

    def test_placeholder_is_not_bare_continue(self):
        result = self._run(
            {"company_of_interest": "EC", "asset_type": "stock", "trade_date": "2026-05-28"}
        )
        placeholder = result["messages"][-1]
        self.assertIsInstance(placeholder, HumanMessage)
        self.assertNotEqual(placeholder.content.strip(), "Continue")

    def test_placeholder_carries_resolved_identity(self):
        result = self._run(
            {
                "company_of_interest": "EC",
                "instrument_context": "The instrument to analyze is `EC`. Resolved identity: Company: Ecopetrol.",
                "trade_date": "2026-05-28",
            }
        )
        content = result["messages"][-1].content
        self.assertIn("Ecopetrol", content)
        self.assertIn("2026-05-28", content)

    def test_old_messages_are_removed(self):
        result = self._run({"company_of_interest": "EC", "trade_date": "2026-05-28"})
        removals = [m for m in result["messages"] if isinstance(m, RemoveMessage)]
        humans = [m for m in result["messages"] if isinstance(m, HumanMessage)]
        self.assertEqual(len(removals), 2)
        self.assertEqual(len(humans), 1)

    def test_safe_defaults_when_state_minimal(self):
        result = create_msg_delete()({"messages": [], "company_of_interest": "EC"})
        placeholder = result["messages"][-1]
        self.assertNotEqual(placeholder.content.strip(), "Continue")
        self.assertIn("EC", placeholder.content)


@pytest.mark.unit
class BenchmarkRegionalUnaffectedTests(unittest.TestCase):
    """5: resolve_benchmark keys only off the ticker's own suffix string --
    confirms the instrument-identity point-in-time fix (sector/industry/
    exchange withholding) cannot change regional benchmark selection, since
    that function never reads sector/industry/exchange/company_name at all."""

    def test_resolve_benchmark_does_not_import_instrument_identity(self):
        import inspect

        from tradingagents.graph import settlement

        source = inspect.getsource(settlement.resolve_benchmark)
        for forbidden in ("resolve_instrument_identity", "build_instrument_context", "identity["):
            self.assertNotIn(forbidden, source)

    def test_resolve_benchmark_suffix_mapping_is_unchanged(self):
        from tradingagents.graph.settlement import resolve_benchmark

        config = {"benchmark_map": {".T": "^N225", "": "SPY"}}
        self.assertEqual(resolve_benchmark("7203.T", config), "^N225")
        self.assertEqual(resolve_benchmark("MSFT", config), "SPY")


if __name__ == "__main__":
    unittest.main()
