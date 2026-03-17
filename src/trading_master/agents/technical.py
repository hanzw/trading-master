"""Technical analyst agent."""

from __future__ import annotations

import logging

from ..models import AnalysisState, AnalystReport, Signal
from .llm import get_llm
from .structured_output import async_llm_call_with_retry

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are an expert technical analyst. Analyze the provided technical indicators \
and price data to assess trend direction, momentum, and optimal entry/exit points. \
Focus on chart patterns, moving averages, RSI, MACD, and volume analysis.

You MUST respond with valid JSON only, no other text. Use this exact schema:
{
  "signal": "STRONG_BUY | BUY | HOLD | SELL | STRONG_SELL",
  "confidence": <int 0-100>,
  "summary": "<concise trend analysis>",
  "bull_case": "<bullish technical scenario>",
  "bear_case": "<bearish technical scenario>",
  "price_target": <float or null>,
  "key_factors": ["<factor1>", "<factor2>", ...]
}"""


def _build_user_prompt(state: AnalysisState) -> str:
    parts: list[str] = [f"Ticker: {state.ticker}\n"]

    if state.market_data:
        md = state.market_data
        parts.append(
            f"Price Data:\n"
            f"  Current: ${md.current_price:.2f}  Open: ${md.open:.2f}\n"
            f"  High: ${md.high:.2f}  Low: ${md.low:.2f}\n"
            f"  Volume: {md.volume:,}  Avg Volume: {md.avg_volume:,}\n"
            f"  52-week High: ${md.fifty_two_week_high:.2f}  "
            f"52-week Low: ${md.fifty_two_week_low:.2f}\n"
        )

    if state.technical_data:
        td = state.technical_data
        parts.append(
            f"Technical Indicators:\n"
            f"  RSI(14): {td.rsi_14}\n"
            f"  MACD: {td.macd}  Signal: {td.macd_signal}  Histogram: {td.macd_histogram}\n"
            f"  SMA(20): {td.sma_20}  SMA(50): {td.sma_50}  SMA(200): {td.sma_200}\n"
            f"  Bollinger Upper: {td.bollinger_upper}  Lower: {td.bollinger_lower}\n"
            f"  ATR(14): {td.atr_14}\n"
            f"  Volume SMA(20): {td.volume_sma_20}\n"
            f"  Overall Trend: {td.trend}\n"
            f"  Active Signals: {', '.join(td.signals) if td.signals else 'none'}\n"
        )

    parts.append(
        "Provide your technical analysis as JSON."
    )
    return "\n".join(parts)


async def analyze_technical(state: AnalysisState) -> AnalystReport:
    """Run technical analysis via LLM and return an AnalystReport."""
    llm = get_llm()
    user_prompt = _build_user_prompt(state)

    try:
        data, tokens, cost = await async_llm_call_with_retry(
            llm, _SYSTEM_PROMPT, user_prompt, max_retries=2,
        )
        state.total_tokens += tokens
        state.total_cost += cost

        return AnalystReport(
            analyst="technical",
            signal=Signal(data.get("signal", "HOLD")),
            confidence=float(data.get("confidence", 50)),
            summary=data.get("summary", ""),
            bull_case=data.get("bull_case", ""),
            bear_case=data.get("bear_case", ""),
            price_target=data.get("price_target"),
            key_factors=data.get("key_factors", []),
        )
    except Exception as exc:
        logger.error("Technical analysis failed: %s", exc)
        state.errors.append(f"technical: {exc}")
        return AnalystReport(
            analyst="technical",
            signal=Signal.HOLD,
            confidence=0.0,
            summary=f"Analysis failed: {exc}",
        )
