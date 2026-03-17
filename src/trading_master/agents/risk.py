"""Risk manager agent."""

from __future__ import annotations

import logging

from ..config import get_config
from ..models import AnalysisState, MarketRegime, RiskAssessment
from .llm import get_llm
from .structured_output import async_llm_call_with_retry

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a conservative risk manager for an equity portfolio. Evaluate the proposed \
trade based on analyst reports, current portfolio composition, and market conditions. \
Your primary mandate is capital preservation.

You MUST respond with valid JSON only, no other text. Use this exact schema:
{
  "risk_score": <int 0-100, higher = riskier>,
  "max_position_size": <float, max shares to buy/sell>,
  "suggested_stop_loss": <float price or null>,
  "portfolio_impact": "<description of portfolio impact>",
  "warnings": ["<warning1>", "<warning2>", ...],
  "approved": <true or false>
}"""


def _build_user_prompt(state: AnalysisState) -> str:
    cfg = get_config().risk
    parts: list[str] = [f"Ticker: {state.ticker}\n"]

    parts.append(
        f"Risk Limits:\n"
        f"  Max position size: {cfg.max_position_pct}% of portfolio\n"
        f"  Max sector exposure: {cfg.max_sector_pct}% of portfolio\n"
        f"  Default stop-loss: {cfg.stop_loss_pct}% below entry\n"
    )

    if state.market_data:
        md = state.market_data
        parts.append(
            f"Market Data:\n"
            f"  Price: ${md.current_price:.2f}  Sector: {md.sector}\n"
            f"  Beta: {md.beta}  Volume: {md.volume:,}\n"
        )

    # Use debate reports if available, otherwise analyst reports
    reports = state.debate_reports or state.analyst_reports
    if reports:
        parts.append("Analyst Reports:\n")
        for r in reports:
            parts.append(
                f"  [{r.analyst}] Signal: {r.signal.value}  "
                f"Confidence: {r.confidence:.0f}%\n"
                f"    Summary: {r.summary}\n"
            )

    if state.portfolio_state:
        ps = state.portfolio_state
        parts.append(
            f"Portfolio State:\n"
            f"  Cash: ${ps.cash:,.2f}  Total Value: ${ps.total_value:,.2f}\n"
            f"  Positions: {len(ps.positions)}\n"
        )
        if state.ticker in ps.positions:
            pos = ps.positions[state.ticker]
            parts.append(
                f"  Existing {state.ticker} position: {pos.quantity} shares "
                f"@ ${pos.avg_cost:.2f}  P&L: {pos.pnl_pct:.1f}%\n"
            )
        # Sector exposure
        sector = state.market_data.sector if state.market_data else ""
        if sector and ps.total_value > 0:
            sector_value = sum(
                p.market_value for p in ps.positions.values() if p.sector == sector
            )
            sector_pct = (sector_value / ps.total_value) * 100
            parts.append(f"  Current {sector} exposure: {sector_pct:.1f}%\n")

    if state.macro_data:
        m = state.macro_data
        parts.append("Macro Environment:\n")
        if m.vix is not None:
            parts.append(f"  VIX: {m.vix:.1f} ({m.vix_regime})\n")
        if m.us_10yr_yield is not None:
            parts.append(f"  10-Year Treasury Yield: {m.us_10yr_yield:.2f}%\n")
        if m.yield_curve_spread is not None:
            status = "INVERTED" if m.yield_curve_inverted else "normal"
            parts.append(f"  Yield Curve Spread (10yr-2yr): {m.yield_curve_spread:.2f}% ({status})\n")
        if m.sp500_price is not None:
            above_below = "above" if m.sp500_above_sma200 else "BELOW"
            parts.append(f"  S&P 500: ${m.sp500_price:.2f} ({above_below} 200-day SMA)\n")
        parts.append(f"  Market Regime: {m.regime.value.upper()}\n")
        if m.regime == MarketRegime.CRISIS:
            parts.append(
                "  ** CRISIS regime detected — strongly consider reducing equity exposure **\n"
            )

    parts.append("Evaluate risk and provide your assessment as JSON.")
    return "\n".join(parts)


async def assess_risk(state: AnalysisState) -> RiskAssessment:
    """Run risk assessment via LLM and return a RiskAssessment."""
    llm = get_llm()
    user_prompt = _build_user_prompt(state)

    try:
        data, tokens, cost = await async_llm_call_with_retry(
            llm, _SYSTEM_PROMPT, user_prompt, max_retries=2,
        )
        state.total_tokens += tokens
        state.total_cost += cost

        warnings = data.get("warnings", [])

        # Inject automatic crisis warning if macro regime is CRISIS
        if state.macro_data and state.macro_data.regime == MarketRegime.CRISIS:
            warnings.insert(
                0,
                "CRISIS regime detected — reduce equity exposure",
            )

        return RiskAssessment(
            risk_score=float(data.get("risk_score", 50)),
            max_position_size=float(data.get("max_position_size", 0)),
            suggested_stop_loss=data.get("suggested_stop_loss"),
            portfolio_impact=data.get("portfolio_impact", ""),
            warnings=warnings,
            approved=bool(data.get("approved", True)),
        )
    except Exception as exc:
        logger.error("Risk assessment failed: %s", exc)
        state.errors.append(f"risk: {exc}")
        return RiskAssessment(
            risk_score=100.0,
            portfolio_impact=f"Risk assessment failed: {exc}",
            warnings=[f"Risk engine error: {exc}"],
            approved=False,
        )
