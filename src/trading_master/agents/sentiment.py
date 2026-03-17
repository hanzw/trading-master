"""Sentiment analyst agent."""

from __future__ import annotations

import logging

from ..models import AnalysisState, AnalystReport, Signal
from .llm import get_llm
from .structured_output import async_llm_call_with_retry

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are an expert market sentiment analyst. Analyze the provided sentiment data \
including news headlines, social media activity, and sentiment scores. Assess \
the overall market mood, identify dominant narratives, and determine whether \
sentiment supports a bullish or bearish outlook.

You MUST respond with valid JSON only, no other text. Use this exact schema:
{
  "signal": "STRONG_BUY | BUY | HOLD | SELL | STRONG_SELL",
  "confidence": <int 0-100>,
  "summary": "<market mood assessment and narrative analysis>",
  "bull_case": "<bullish sentiment scenario>",
  "bear_case": "<bearish sentiment scenario>",
  "price_target": <float or null>,
  "key_factors": ["<factor1>", "<factor2>", ...]
}"""


def _build_user_prompt(state: AnalysisState) -> str:
    parts: list[str] = [f"Ticker: {state.ticker}\n"]

    if state.sentiment_data:
        sd = state.sentiment_data
        parts.append(
            f"Sentiment Scores:\n"
            f"  Overall: {sd.overall_score:.2f} (range: -1 to +1)\n"
            f"  News Score: {sd.news_score:.2f}\n"
            f"  Reddit Score: {sd.reddit_score:.2f}\n"
        )
        if sd.key_themes:
            parts.append(f"  Key Themes: {', '.join(sd.key_themes)}\n")
        if sd.news_headlines:
            parts.append("  Recent Headlines:\n")
            for headline in sd.news_headlines[:10]:
                parts.append(f"    - {headline}\n")
        if sd.reddit_posts:
            parts.append("  Reddit Highlights:\n")
            for post in sd.reddit_posts[:5]:
                parts.append(f"    - {post}\n")

    parts.append(
        "Provide your sentiment analysis as JSON."
    )
    return "\n".join(parts)


async def analyze_sentiment(state: AnalysisState) -> AnalystReport:
    """Run sentiment analysis via LLM and return an AnalystReport."""
    llm = get_llm()
    user_prompt = _build_user_prompt(state)

    try:
        data, tokens, cost = await async_llm_call_with_retry(
            llm, _SYSTEM_PROMPT, user_prompt, max_retries=2,
        )
        state.total_tokens += tokens
        state.total_cost += cost

        return AnalystReport(
            analyst="sentiment",
            signal=Signal(data.get("signal", "HOLD")),
            confidence=float(data.get("confidence", 50)),
            summary=data.get("summary", ""),
            bull_case=data.get("bull_case", ""),
            bear_case=data.get("bear_case", ""),
            price_target=data.get("price_target"),
            key_factors=data.get("key_factors", []),
        )
    except Exception as exc:
        logger.error("Sentiment analysis failed: %s", exc)
        state.errors.append(f"sentiment: {exc}")
        return AnalystReport(
            analyst="sentiment",
            signal=Signal.HOLD,
            confidence=0.0,
            summary=f"Analysis failed: {exc}",
        )
