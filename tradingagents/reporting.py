"""Reusable report-tree writer shared by the CLI and the programmatic API.

Writes a run's per-section markdown (analysts, research, trading, risk,
portfolio) plus a consolidated ``complete_report.md`` under ``save_path``. The
CLI and ``TradingAgentsGraph.save_reports`` both call this, so a headless / API
run produces the same on-disk report tree a CLI run does.
"""

from datetime import datetime
from pathlib import Path

from tradingagents.agents.managers.integrity_check import render_manager_integrity_report
from tradingagents.agents.researchers.research_summary import render_research_context_summary
from tradingagents.agents.researchers.verification import render_verification_history
from tradingagents.agents.risk_mgmt.integrity_check import render_risk_integrity_report
from tradingagents.agents.schemas import render_risk_stance_summary
from tradingagents.agents.validator import render_validation_summary


def write_report_tree(
    final_state: dict,
    ticker: str,
    save_path,
    *,
    error_banner: str | None = None,
    output_filename: str = "complete_report.md",
) -> Path:
    """Save a run's reports to ``save_path``; return the combined-report path.

    Every section below is already independently conditional on the matching
    piece of ``final_state`` being present, so this degrades gracefully to a
    partial report for a run that failed partway through -- pass
    ``error_banner`` to prepend an explicit, unmissable notice that the run
    did not reach a valid final decision (never silently present a partial
    result as if it were a successful one), and ``output_filename`` to save
    it under a distinct name (e.g. ``partial_report.md``) instead of
    overwriting a prior successful ``complete_report.md`` for the same ticker.
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    sections = []

    # 1. Analysts
    analysts_dir = save_path / "1_analysts"
    analyst_parts = []
    if final_state.get("market_report"):
        analysts_dir.mkdir(exist_ok=True)
        (analysts_dir / "market.md").write_text(final_state["market_report"], encoding="utf-8")
        analyst_parts.append(("Market Analyst", final_state["market_report"]))
    if final_state.get("sentiment_report"):
        analysts_dir.mkdir(exist_ok=True)
        (analysts_dir / "sentiment.md").write_text(final_state["sentiment_report"], encoding="utf-8")
        analyst_parts.append(("Sentiment Analyst", final_state["sentiment_report"]))
    if final_state.get("news_report"):
        analysts_dir.mkdir(exist_ok=True)
        (analysts_dir / "news.md").write_text(final_state["news_report"], encoding="utf-8")
        analyst_parts.append(("News Analyst", final_state["news_report"]))
    if final_state.get("fundamentals_report"):
        analysts_dir.mkdir(exist_ok=True)
        (analysts_dir / "fundamentals.md").write_text(final_state["fundamentals_report"], encoding="utf-8")
        analyst_parts.append(("Fundamentals Analyst", final_state["fundamentals_report"]))
    if analyst_parts:
        content = "\n\n".join(f"### {name}\n{text}" for name, text in analyst_parts)
        sections.append(f"## I. Analyst Team Reports\n\n{content}")

    # 2. Research
    if final_state.get("investment_debate_state"):
        research_dir = save_path / "2_research"
        debate = final_state["investment_debate_state"]
        research_parts = []
        if debate.get("bull_thesis") or debate.get("bear_thesis") or debate.get("setup_tags"):
            research_dir.mkdir(exist_ok=True)
            context_summary = render_research_context_summary(debate)
            (research_dir / "context.md").write_text(context_summary, encoding="utf-8")
            research_parts.append(("Research Context", context_summary))
        if debate.get("bull_history"):
            research_dir.mkdir(exist_ok=True)
            (research_dir / "bull.md").write_text(debate["bull_history"], encoding="utf-8")
            research_parts.append(("Bull Researcher", debate["bull_history"]))
        if debate.get("bear_history"):
            research_dir.mkdir(exist_ok=True)
            (research_dir / "bear.md").write_text(debate["bear_history"], encoding="utf-8")
            research_parts.append(("Bear Researcher", debate["bear_history"]))
        if debate.get("verification_history"):
            research_dir.mkdir(exist_ok=True)
            verification_detail = render_verification_history(
                debate["verification_history"],
                repaired_agents=debate.get("repaired_agents", []),
                include_details=True,
            )
            (research_dir / "verification.md").write_text(
                verification_detail, encoding="utf-8"
            )
            verification_summary = render_verification_history(
                debate["verification_history"],
                repaired_agents=debate.get("repaired_agents", []),
                include_details=False,
            )
            research_parts.append(("Research Verification", verification_summary))
        if debate.get("judge_decision"):
            research_dir.mkdir(exist_ok=True)
            (research_dir / "manager.md").write_text(debate["judge_decision"], encoding="utf-8")
            research_parts.append(("Research Manager", debate["judge_decision"]))
        if debate.get("manager_integrity_status"):
            integrity_report = render_manager_integrity_report(
                debate["manager_integrity_status"], debate.get("manager_integrity_findings", [])
            )
            research_dir.mkdir(exist_ok=True)
            (research_dir / "manager_integrity.md").write_text(integrity_report, encoding="utf-8")
            research_parts.append(("Research Manager Integrity", integrity_report))
        if research_parts:
            content = "\n\n".join(f"### {name}\n{text}" for name, text in research_parts)
            sections.append(f"## II. Research Team Decision\n\n{content}")

    # 3. Trading
    if final_state.get("trader_investment_plan"):
        trading_dir = save_path / "3_trading"
        trading_dir.mkdir(exist_ok=True)
        (trading_dir / "trader.md").write_text(final_state["trader_investment_plan"], encoding="utf-8")
        sections.append(f"## III. Trading Team Plan\n\n### Trader\n{final_state['trader_investment_plan']}")

    # 4. Risk Management
    if final_state.get("risk_debate_state"):
        risk_dir = save_path / "4_risk"
        risk = final_state["risk_debate_state"]
        risk_parts = []
        if any(risk.get(k) for k in ("aggressive_risk_level", "conservative_risk_level", "neutral_risk_level")):
            risk_parts.append(("Risk Stance Summary", render_risk_stance_summary(risk)))
        if risk.get("risk_integrity_status"):
            risk_parts.append((
                "Risk Integrity Check",
                render_risk_integrity_report(
                    risk["risk_integrity_status"], risk.get("risk_integrity_findings", [])
                ),
            ))
        if risk.get("aggressive_history"):
            risk_dir.mkdir(exist_ok=True)
            (risk_dir / "aggressive.md").write_text(risk["aggressive_history"], encoding="utf-8")
            risk_parts.append(("Aggressive Analyst", risk["aggressive_history"]))
        if risk.get("conservative_history"):
            risk_dir.mkdir(exist_ok=True)
            (risk_dir / "conservative.md").write_text(risk["conservative_history"], encoding="utf-8")
            risk_parts.append(("Conservative Analyst", risk["conservative_history"]))
        if risk.get("neutral_history"):
            risk_dir.mkdir(exist_ok=True)
            (risk_dir / "neutral.md").write_text(risk["neutral_history"], encoding="utf-8")
            risk_parts.append(("Neutral Analyst", risk["neutral_history"]))
        if risk_parts:
            content = "\n\n".join(f"### {name}\n{text}" for name, text in risk_parts)
            sections.append(f"## IV. Risk Management Team Decision\n\n{content}")

        # 5. Portfolio Manager
        if risk.get("judge_decision"):
            portfolio_dir = save_path / "5_portfolio"
            portfolio_dir.mkdir(exist_ok=True)
            (portfolio_dir / "decision.md").write_text(risk["judge_decision"], encoding="utf-8")
            decision_text = risk["judge_decision"]
            # The real structured/parsed disposition -- never derived from the
            # rating (see fork README: "real PM disposition, not inferred").
            disposition = risk.get("portfolio_disposition") or "not recorded"
            sections.append(
                f"## V. Portfolio Manager Decision\n\n### Portfolio Manager\n{decision_text}"
                f"\n\n(Disposition: {disposition})"
            )

    # 6. Validation (see fork README: "Generation / Validation / Review") --
    # compact counts plus only unresolved findings, never a per-stage dump.
    validation_summary = render_validation_summary(final_state.get("validation_log", []))
    if validation_summary:
        sections.append(f"## VI. Validation\n\n{validation_summary}")

    # Write consolidated report
    header = f"# Trading Analysis Report: {ticker}\n\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    if error_banner:
        header += (
            "> **INCOMPLETE RUN -- NOT A FINAL RECOMMENDATION**\n"
            f"> {error_banner}\n\n"
        )
    (save_path / output_filename).write_text(header + "\n\n".join(sections), encoding="utf-8")
    return save_path / output_filename
