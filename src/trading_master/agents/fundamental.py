"""Fundamental analyst agent."""

from __future__ import annotations

import logging

from ..models import AnalysisState, AnalystReport, Signal
from .llm import get_llm
from .structured_output import async_llm_call_with_retry

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are an expert fundamental equity analyst. Analyze the provided financial data \
and produce a rigorous investment assessment. Focus on valuation metrics, financial \
health, growth trajectory, and competitive positioning.

You MUST respond with valid JSON only, no other text. Use this exact schema:
{
  "signal": "STRONG_BUY | BUY | HOLD | SELL | STRONG_SELL",
  "confidence": <int 0-100>,
  "summary": "<concise analysis>",
  "bull_case": "<optimistic scenario>",
  "bear_case": "<pessimistic scenario>",
  "price_target": <float or null>,
  "key_factors": ["<factor1>", "<factor2>", ...]
}"""


def _build_user_prompt(state: AnalysisState) -> str:
    parts: list[str] = [f"Ticker: {state.ticker}\n"]

    if state.market_data:
        md = state.market_data
        parts.append(
            f"Market Data:\n"
            f"  Price: ${md.current_price:.2f}\n"
            f"  Market Cap: ${md.market_cap:,.0f}\n"
            f"  P/E: {md.pe_ratio}  Forward P/E: {md.forward_pe}\n"
            f"  Beta: {md.beta}  Dividend Yield: {md.dividend_yield}\n"
            f"  52-week range: ${md.fifty_two_week_low:.2f} - ${md.fifty_two_week_high:.2f}\n"
            f"  Sector: {md.sector}  Industry: {md.industry}\n"
        )

    if state.fundamental_data:
        fd = state.fundamental_data
        parts.append(
            f"Fundamental Data:\n"
            f"  Revenue: {fd.revenue}  Revenue Growth: {fd.revenue_growth}\n"
            f"  Net Income: {fd.net_income}  EPS: {fd.eps}\n"
            f"  P/E: {fd.pe_ratio}  Forward P/E: {fd.forward_pe}  PEG: {fd.peg_ratio}\n"
            f"  P/B: {fd.price_to_book}  D/E: {fd.debt_to_equity}\n"
            f"  FCF: {fd.free_cash_flow}  Profit Margin: {fd.profit_margin}\n"
            f"  ROE: {fd.roe}  Current Ratio: {fd.current_ratio}\n"
            f"  Summary: {fd.summary}\n"
        )

    parts.append(
        "Provide your fundamental analysis as JSON."
    )
    return "\n".join(parts)


async def analyze_fundamental(state: AnalysisState) -> AnalystReport:
    """Run fundamental analysis via LLM and return an AnalystReport."""
    llm = get_llm()
    user_prompt = _build_user_prompt(state)

    try:
        data, tokens, cost = await async_llm_call_with_retry(
            llm, _SYSTEM_PROMPT, user_prompt, max_retries=2,
        )
        state.total_tokens += tokens
        state.total_cost += cost

        return AnalystReport(
            analyst="fundamental",
            signal=Signal(data.get("signal", "HOLD")),
            confidence=float(data.get("confidence", 50)),
            summary=data.get("summary", ""),
            bull_case=data.get("bull_case", ""),
            bear_case=data.get("bear_case", ""),
            price_target=data.get("price_target"),
            key_factors=data.get("key_factors", []),
        )
    except Exception as exc:
        logger.error("Fundamental analysis failed: %s", exc)
        state.errors.append(f"fundamental: {exc}")
        return AnalystReport(
            analyst="fundamental",
            signal=Signal.HOLD,
            confidence=0.0,
            summary=f"Analysis failed: {exc}",
        )
