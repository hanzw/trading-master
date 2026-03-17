"""Debate moderator and final synthesizer."""

from __future__ import annotations

import logging

from ..models import (
    Action,
    AnalysisState,
    AnalystReport,
    Recommendation,
    Signal,
)
from .llm import get_llm
from .structured_output import async_llm_call_with_retry

logger = logging.getLogger(__name__)

# ── Debate ────────────────────────────────────────────────────────────

_DEBATE_SYSTEM_PROMPT = """\
You are a {analyst_type} analyst participating in an investment debate. \
You have already submitted your initial report. Now you are reviewing the other \
analysts' perspectives. Considering their arguments, revise your assessment if \
warranted or reinforce your original view with additional reasoning.

You MUST respond with valid JSON only, no other text. Use this exact schema:
{{
  "signal": "STRONG_BUY | BUY | HOLD | SELL | STRONG_SELL",
  "confidence": <int 0-100>,
  "summary": "<revised analysis incorporating debate points>",
  "bull_case": "<updated bull case>",
  "bear_case": "<updated bear case>",
  "price_target": <float or null>,
  "key_factors": ["<factor1>", "<factor2>", ...],
  "revision_notes": "<what changed from your initial view and why>"
}}"""


def _build_debate_prompt(
    own_report: AnalystReport,
    other_reports: list[AnalystReport],
    ticker: str,
) -> str:
    parts: list[str] = [f"Ticker: {ticker}\n"]

    parts.append(
        f"Your Initial Report ({own_report.analyst}):\n"
        f"  Signal: {own_report.signal.value}  Confidence: {own_report.confidence:.0f}%\n"
        f"  Summary: {own_report.summary}\n"
        f"  Bull case: {own_report.bull_case}\n"
        f"  Bear case: {own_report.bear_case}\n"
    )

    parts.append("Other Analysts' Reports:\n")
    for r in other_reports:
        parts.append(
            f"  [{r.analyst}] Signal: {r.signal.value}  "
            f"Confidence: {r.confidence:.0f}%\n"
            f"    Summary: {r.summary}\n"
            f"    Bull: {r.bull_case}\n"
            f"    Bear: {r.bear_case}\n"
        )

    parts.append(
        "Review the other perspectives and provide your revised assessment as JSON. "
        "Adjust your signal and confidence if their arguments are compelling, "
        "or strengthen your original view if you disagree."
    )
    return "\n".join(parts)


async def run_debate(state: AnalysisState) -> list[AnalystReport]:
    """Run a debate round: each analyst revises their view after seeing others.

    Returns a list of revised AnalystReports.
    """
    llm = get_llm()
    reports = state.analyst_reports
    revised: list[AnalystReport] = []

    for report in reports:
        others = [r for r in reports if r.analyst != report.analyst]
        system = _DEBATE_SYSTEM_PROMPT.format(analyst_type=report.analyst)
        user_prompt = _build_debate_prompt(report, others, state.ticker)

        try:
            data, tokens, cost = await async_llm_call_with_retry(
                llm, system, user_prompt, max_retries=2,
            )
            state.total_tokens += tokens
            state.total_cost += cost

            revised.append(AnalystReport(
                analyst=report.analyst,
                signal=Signal(data.get("signal", report.signal.value)),
                confidence=float(data.get("confidence", report.confidence)),
                summary=data.get("summary", report.summary),
                bull_case=data.get("bull_case", report.bull_case),
                bear_case=data.get("bear_case", report.bear_case),
                price_target=data.get("price_target", report.price_target),
                key_factors=data.get("key_factors", report.key_factors),
                revised=True,
                revision_notes=data.get("revision_notes", ""),
            ))
        except Exception as exc:
            logger.error("Debate round failed for %s: %s", report.analyst, exc)
            state.errors.append(f"debate_{report.analyst}: {exc}")
            # Keep original report if debate fails
            revised.append(report)

    return revised


# ── Synthesis ─────────────────────────────────────────────────────────

_SYNTHESIS_SYSTEM_PROMPT = """\
You are the chief investment strategist. You have received analyst reports \
(after a debate round) and a risk assessment. Synthesize everything into a \
final investment recommendation.

You MUST respond with valid JSON only, no other text. Use this exact schema:
{
  "action": "BUY | SELL | HOLD",
  "confidence": <int 0-100>,
  "summary": "<comprehensive rationale for the decision>",
  "debate_notes": "<key points of agreement/disagreement among analysts>"
}"""


def _build_synthesis_prompt(state: AnalysisState) -> str:
    parts: list[str] = [f"Ticker: {state.ticker}\n"]

    reports = state.debate_reports or state.analyst_reports
    if reports:
        parts.append("Analyst Reports (post-debate):\n")
        for r in reports:
            parts.append(
                f"  [{r.analyst}] Signal: {r.signal.value}  "
                f"Confidence: {r.confidence:.0f}%\n"
                f"    Summary: {r.summary}\n"
                f"    Bull: {r.bull_case}\n"
                f"    Bear: {r.bear_case}\n"
                f"    Price Target: {r.price_target}\n"
            )
            if r.revised and r.revision_notes:
                parts.append(f"    Revision notes: {r.revision_notes}\n")

    if state.risk_assessment:
        ra = state.risk_assessment
        parts.append(
            f"Risk Assessment:\n"
            f"  Risk Score: {ra.risk_score:.0f}/100\n"
            f"  Max Position: {ra.max_position_size} shares\n"
            f"  Stop-Loss: {ra.suggested_stop_loss}\n"
            f"  Impact: {ra.portfolio_impact}\n"
            f"  Approved: {ra.approved}\n"
            f"  Warnings: {', '.join(ra.warnings) if ra.warnings else 'none'}\n"
        )

    if state.market_data:
        parts.append(f"Current Price: ${state.market_data.current_price:.2f}\n")

    parts.append(
        "Produce your final recommendation as JSON. "
        "If risk is not approved, lean towards HOLD."
    )
    return "\n".join(parts)


async def synthesize(state: AnalysisState) -> Recommendation:
    """Synthesize all reports and risk into a final Recommendation."""
    llm = get_llm()
    user_prompt = _build_synthesis_prompt(state)

    try:
        data, tokens, cost = await async_llm_call_with_retry(
            llm, _SYNTHESIS_SYSTEM_PROMPT, user_prompt, max_retries=2,
        )
        state.total_tokens += tokens
        state.total_cost += cost

        reports = state.debate_reports or state.analyst_reports

        return Recommendation(
            ticker=state.ticker,
            action=Action(data.get("action", "HOLD")),
            confidence=float(data.get("confidence", 50)),
            summary=data.get("summary", ""),
            analyst_reports=reports,
            risk_assessment=state.risk_assessment,
            debate_notes=data.get("debate_notes", ""),
            llm_tokens_used=state.total_tokens,
            llm_cost_usd=state.total_cost,
        )
    except Exception as exc:
        logger.error("Synthesis failed: %s", exc)
        state.errors.append(f"synthesis: {exc}")
        return Recommendation(
            ticker=state.ticker,
            action=Action.HOLD,
            confidence=0.0,
            summary=f"Synthesis failed: {exc}. Defaulting to HOLD.",
            analyst_reports=state.debate_reports or state.analyst_reports,
            risk_assessment=state.risk_assessment,
            llm_tokens_used=state.total_tokens,
            llm_cost_usd=state.total_cost,
        )
