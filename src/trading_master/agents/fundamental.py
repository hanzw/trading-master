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

IMPORTANT: Do NOT rely only on current-period snapshots. You will receive multi-year \
revenue, margin, and cash flow trends. Factor these into your analysis:
- Revenue CAGR tells you whether the business is growing or shrinking over 3+ years.
- Margin trend (expanding/compressing/stable) reveals operating leverage and pricing power.
- Earnings quality (cash flow backing) distinguishes real earnings from accounting tricks.
- Accruals ratio flags aggressive accounting when high (>0.10 is a red flag).

Weight long-term trends heavily in your signal and confidence.

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
        )

        # Multi-year trends
        if fd.revenue_history:
            rev_strs = " -> ".join(f"${v/1e9:.1f}B" if abs(v) >= 1e9 else f"${v/1e6:.0f}M" for v in fd.revenue_history)
            cagr_str = f" (CAGR: {fd.revenue_cagr_3yr*100:+.1f}%)" if fd.revenue_cagr_3yr is not None else ""
            parts.append(f"  Revenue History (multi-year): {rev_strs}{cagr_str}\n")
        if fd.margin_history:
            m_strs = " -> ".join(f"{m*100:.1f}%" for m in fd.margin_history)
            parts.append(f"  Margin Trend: {fd.margin_trend} ({m_strs})\n")
        if fd.earnings_quality is not None:
            quality_label = "strong" if fd.earnings_quality >= 80 else "moderate" if fd.earnings_quality >= 50 else "weak"
            parts.append(f"  Earnings Quality: {fd.earnings_quality:.0f}/100 ({quality_label} cash flow backing)\n")
        if fd.accruals_ratio is not None:
            health = "low, healthy" if abs(fd.accruals_ratio) < 0.05 else "moderate" if abs(fd.accruals_ratio) < 0.10 else "high, red flag"
            parts.append(f"  Accruals Ratio: {fd.accruals_ratio:.2f} ({health})\n")

        parts.append(f"  Summary: {fd.summary}\n")

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
