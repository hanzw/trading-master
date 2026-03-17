"""LangGraph StateGraph orchestration for the analysis pipeline."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from ..models import (
    AnalysisState,
    AnalystReport,
    FundamentalData,
    MacroData,
    MarketData,
    PortfolioState,
    Recommendation,
    RiskAssessment,
    SentimentData,
    TechnicalData,
)
from .fundamental import analyze_fundamental
from .moderator import run_debate, synthesize
from .risk import assess_risk
from .sentiment import analyze_sentiment
from .technical import analyze_technical
from ..portfolio.sizing import compute_position_size
from ..portfolio.correlation import check_correlation_ok, fetch_returns
from ..portfolio import risk_metrics

logger = logging.getLogger(__name__)


# ── LangGraph TypedDict state ────────────────────────────────────────

class GraphState(TypedDict, total=False):
    ticker: str
    market_data: dict[str, Any] | None
    fundamental_data: dict[str, Any] | None
    technical_data: dict[str, Any] | None
    sentiment_data: dict[str, Any] | None
    macro_data: dict[str, Any] | None
    analyst_reports: list[dict[str, Any]]
    debate_reports: list[dict[str, Any]]
    risk_assessment: dict[str, Any] | None
    recommendation: dict[str, Any] | None
    portfolio_state: dict[str, Any] | None
    quantitative_risk: dict[str, Any] | None
    errors: list[str]
    total_tokens: int
    total_cost: float


def _state_to_analysis(gs: GraphState) -> AnalysisState:
    """Convert LangGraph dict state to Pydantic AnalysisState."""
    return AnalysisState(
        ticker=gs.get("ticker", ""),
        market_data=MarketData(**gs["market_data"]) if gs.get("market_data") else None,
        fundamental_data=FundamentalData(**gs["fundamental_data"]) if gs.get("fundamental_data") else None,
        technical_data=TechnicalData(**gs["technical_data"]) if gs.get("technical_data") else None,
        sentiment_data=SentimentData(**gs["sentiment_data"]) if gs.get("sentiment_data") else None,
        macro_data=MacroData(**gs["macro_data"]) if gs.get("macro_data") else None,
        analyst_reports=[AnalystReport(**r) for r in gs.get("analyst_reports", [])],
        debate_reports=[AnalystReport(**r) for r in gs.get("debate_reports", [])],
        risk_assessment=RiskAssessment(**gs["risk_assessment"]) if gs.get("risk_assessment") else None,
        portfolio_state=PortfolioState(**gs["portfolio_state"]) if gs.get("portfolio_state") else None,
        errors=gs.get("errors", []),
        total_tokens=gs.get("total_tokens", 0),
        total_cost=gs.get("total_cost", 0.0),
    )


def _sync_counters(analysis: AnalysisState, gs: GraphState) -> None:
    """Copy token/cost/error counters back to graph state."""
    gs["total_tokens"] = analysis.total_tokens
    gs["total_cost"] = analysis.total_cost
    gs["errors"] = analysis.errors


# ── Node functions (async for LangGraph async invocation) ────────────

async def collect_data(gs: GraphState) -> GraphState:
    """Fetch market, fundamental, technical, sentiment, and macro data in parallel.

    Data fetchers are synchronous (yfinance), so we use ``asyncio.to_thread``
    to run them concurrently without blocking the event loop.
    """
    ticker = gs["ticker"]
    errors: list[str] = gs.get("errors", [])

    # Import fetchers up-front so ImportErrors surface before the gather.
    fetchers: dict[str, Any] = {}
    try:
        from ..data.market import fetch_market_data
        fetchers["market"] = fetch_market_data
    except ImportError:
        logger.warning("market data fetcher not available")
    try:
        from ..data.fundamentals import fetch_fundamentals
        fetchers["fundamental"] = fetch_fundamentals
    except ImportError:
        logger.warning("fundamentals data fetcher not available")
    try:
        from ..data.technical import fetch_technicals
        fetchers["technical"] = fetch_technicals
    except ImportError:
        logger.warning("technical data fetcher not available")
    try:
        from ..data.sentiment import fetch_sentiment
        fetchers["sentiment"] = fetch_sentiment
    except ImportError:
        logger.warning("sentiment data fetcher not available")
    try:
        from ..data.macro import fetch_macro_data
        fetchers["macro"] = fetch_macro_data
    except ImportError:
        logger.warning("macro data fetcher not available")

    # Build coroutines -- macro takes no args, the rest take ticker
    coros: dict[str, Any] = {}
    for name, fn in fetchers.items():
        if name == "macro":
            coros[name] = asyncio.to_thread(fn)
        else:
            coros[name] = asyncio.to_thread(fn, ticker)

    # Run all fetchers concurrently
    keys = list(coros.keys())
    results = await asyncio.gather(*coros.values(), return_exceptions=True)
    fetched = dict(zip(keys, results))

    # Process results with fallbacks
    def _handle(name: str, default_factory):
        val = fetched.get(name)
        if val is None or isinstance(val, Exception):
            if isinstance(val, Exception):
                logger.error("Failed to fetch %s data: %s", name, val)
                errors.append(f"{name}_data: {val}")
            return default_factory()
        return val

    md = _handle("market", lambda: MarketData(ticker=ticker))
    gs["market_data"] = md.model_dump() if not isinstance(md, dict) else md

    fd = _handle("fundamental", lambda: FundamentalData(ticker=ticker))
    gs["fundamental_data"] = fd.model_dump() if not isinstance(fd, dict) else fd

    td = _handle("technical", lambda: TechnicalData(ticker=ticker))
    gs["technical_data"] = td.model_dump() if not isinstance(td, dict) else td

    sd = _handle("sentiment", lambda: SentimentData(ticker=ticker))
    gs["sentiment_data"] = sd.model_dump() if not isinstance(sd, dict) else sd

    macro = fetched.get("macro")
    if macro is None or isinstance(macro, Exception):
        if isinstance(macro, Exception):
            logger.error("Failed to fetch macro data: %s", macro)
            errors.append(f"macro_data: {macro}")
        gs["macro_data"] = None
    else:
        gs["macro_data"] = macro.model_dump(mode="json")

    gs["errors"] = errors
    return gs


async def run_analysts(gs: GraphState) -> GraphState:
    """Run all three analyst agents in parallel."""
    analysis = _state_to_analysis(gs)

    results = await asyncio.gather(
        analyze_fundamental(analysis),
        analyze_technical(analysis),
        analyze_sentiment(analysis),
        return_exceptions=True,
    )
    reports: list[AnalystReport] = []
    for r in results:
        if isinstance(r, Exception):
            logger.error("Analyst failed: %s", r)
            analysis.errors.append(f"analyst: {r}")
        else:
            reports.append(r)

    gs["analyst_reports"] = [r.model_dump() for r in reports]
    _sync_counters(analysis, gs)
    return gs


async def debate_node(gs: GraphState) -> GraphState:
    """Run a debate round among analysts."""
    analysis = _state_to_analysis(gs)
    revised = await run_debate(analysis)
    gs["debate_reports"] = [r.model_dump() for r in revised]
    _sync_counters(analysis, gs)
    return gs


async def risk_node(gs: GraphState) -> GraphState:
    """Run risk assessment."""
    analysis = _state_to_analysis(gs)
    assessment = await assess_risk(analysis)
    gs["risk_assessment"] = assessment.model_dump()
    _sync_counters(analysis, gs)
    return gs


async def quantitative_risk_node(gs: GraphState) -> GraphState:
    """Run quantitative risk checks: position sizing + correlation + regime + CVaR."""
    ticker = gs.get("ticker", "")
    errors: list[str] = gs.get("errors", [])

    try:
        from ..config import get_config
        cfg = get_config()
        max_position_pct = cfg.risk.max_position_pct
        holding_days = cfg.risk.holding_days

        # 1. Extract ATR and price
        td = gs.get("technical_data") or {}
        md = gs.get("market_data") or {}
        atr_14 = td.get("atr_14") or 0.0
        price = md.get("current_price") or 0.0

        # 2. Portfolio value
        ps = gs.get("portfolio_state") or {}
        portfolio_value = ps.get("total_value") or 10000.0

        # 3. Extract regime from macro_data
        macro = gs.get("macro_data") or {}
        regime = macro.get("regime", None)
        # regime may be a string like "bull" or enum value
        if regime is not None:
            regime = str(regime).lower()

        # 4. Compute quantitative position size (regime-aware, holding-period scaled)
        sizing_result = compute_position_size(
            price=price,
            atr_14=atr_14,
            portfolio_value=portfolio_value,
            max_position_pct=max_position_pct,
            regime=regime,
            holding_days=holding_days,
        )

        # 5. Correlation check against existing holdings
        existing_tickers = []
        positions = ps.get("positions", {})
        if isinstance(positions, dict):
            existing_tickers = [t for t, p in positions.items() if t != ticker]

        correlation_ok = True
        avg_correlation = 0.0
        correlation_details: dict[str, Any] = {}

        if existing_tickers:
            try:
                correlation_ok, avg_correlation, correlation_details = (
                    check_correlation_ok(ticker, existing_tickers)
                )
                # Apply correlation haircut to sizing
                if avg_correlation > 0:
                    sizing_result = compute_position_size(
                        price=price,
                        atr_14=atr_14,
                        portfolio_value=portfolio_value,
                        max_position_pct=max_position_pct,
                        existing_correlation=avg_correlation,
                        regime=regime,
                        holding_days=holding_days,
                    )
            except Exception as exc:
                logger.warning("Correlation check failed (non-fatal): %s", exc)
                correlation_details = {"error": str(exc)}

        # 6. Override LLM's hallucinated max_position_size in risk_assessment
        ra = gs.get("risk_assessment") or {}
        ra["max_position_size"] = float(sizing_result["shares"])
        warnings = ra.get("warnings", [])

        # 7. Regime-based approval / warnings
        if regime == "crisis":
            ra["approved"] = False
            warnings.append(
                "CRISIS regime \u2014 new equity positions not recommended"
            )
        elif regime == "bear":
            warnings.append(
                "BEAR regime \u2014 reduced position sizes applied"
            )

        # 8. If correlation check fails, reject the trade
        if not correlation_ok:
            ra["approved"] = False
            warnings.append(
                f"QUANT: High correlation with existing holdings "
                f"(avg={avg_correlation:.2f}). Trade rejected."
            )

        ra["warnings"] = warnings
        gs["risk_assessment"] = ra

        # 9. Portfolio-level CVaR check (wrapped in try/except — non-blocking)
        portfolio_cvar = None
        new_portfolio_cvar = None
        cvar_warning = False
        try:
            if existing_tickers and positions:
                import numpy as np

                # Fetch returns for existing tickers + new ticker
                all_tickers = existing_tickers + [ticker]
                returns_array, valid_tickers = fetch_returns(all_tickers)

                if returns_array is not None and len(valid_tickers) >= 1:
                    # Build current weights from positions
                    current_values = {}
                    for t in valid_tickers:
                        if t == ticker:
                            continue
                        pos = positions.get(t, {})
                        if isinstance(pos, dict):
                            qty = pos.get("quantity", 0)
                            cp = pos.get("current_price", pos.get("avg_cost", 0))
                        else:
                            qty = 0
                            cp = 0
                        current_values[t] = qty * cp

                    total_current = sum(current_values.values())

                    if total_current > 0:
                        # Current portfolio CVaR (existing positions only)
                        existing_valid = [t for t in valid_tickers if t != ticker and t in current_values]
                        if existing_valid:
                            existing_indices = [valid_tickers.index(t) for t in existing_valid]
                            existing_returns = returns_array[:, existing_indices]
                            existing_weights = np.array(
                                [current_values[t] / total_current for t in existing_valid]
                            )
                            portfolio_cvar = risk_metrics.cvar(
                                existing_weights, existing_returns,
                                confidence=0.95, portfolio_value=portfolio_value,
                            )

                        # Simulate adding the new position
                        if ticker in valid_tickers:
                            new_dollar = sizing_result["shares"] * price
                            new_total = total_current + new_dollar

                            if new_total > 0:
                                new_valid = [t for t in valid_tickers if t in current_values or t == ticker]
                                new_indices = [valid_tickers.index(t) for t in new_valid]
                                new_returns = returns_array[:, new_indices]
                                new_weights_list = []
                                for t in new_valid:
                                    if t == ticker:
                                        new_weights_list.append(new_dollar / new_total)
                                    else:
                                        new_weights_list.append(current_values.get(t, 0) / new_total)
                                new_weights = np.array(new_weights_list)

                                new_portfolio_cvar = risk_metrics.cvar(
                                    new_weights, new_returns,
                                    confidence=0.95, portfolio_value=portfolio_value,
                                )

                                # Scale CVaR threshold by regime multiplier
                                from ..portfolio.sizing import _REGIME_MULTIPLIERS
                                regime_mult = _REGIME_MULTIPLIERS.get((regime or "bull").lower(), 1.0)
                                cvar_threshold = 0.05 * portfolio_value * regime_mult
                                if new_portfolio_cvar > cvar_threshold:
                                    warnings.append(
                                        f"BLOCKED: Trade would increase portfolio CVaR to "
                                        f"${new_portfolio_cvar:,.0f} (threshold: ${cvar_threshold:,.0f})"
                                    )
                                    ra["approved"] = False  # HARD GATE — block the trade
                                    cvar_warning = True
                                    ra["warnings"] = warnings
                                    gs["risk_assessment"] = ra

        except Exception as exc:
            logger.warning("Portfolio CVaR check failed (non-fatal): %s", exc)

        # 10. Store quantitative results in state
        gs["quantitative_risk"] = {
            "sizing": sizing_result,
            "regime": regime,
            "correlation_ok": correlation_ok,
            "avg_correlation": avg_correlation,
            "correlation_details": correlation_details,
            "portfolio_cvar": portfolio_cvar,
            "new_portfolio_cvar": new_portfolio_cvar,
            "cvar_warning": cvar_warning,
        }

    except Exception as exc:
        logger.error("Quantitative risk check failed: %s", exc)
        errors.append(f"quantitative_risk: {exc}")
        gs["quantitative_risk"] = {"error": str(exc)}

    gs["errors"] = errors
    return gs


async def synthesize_node(gs: GraphState) -> GraphState:
    """Produce final recommendation, filtered through the drawdown circuit breaker."""
    analysis = _state_to_analysis(gs)
    rec = await synthesize(analysis)

    # Apply circuit breaker filter
    try:
        from ..config import get_config
        from ..portfolio.circuit_breaker import DrawdownCircuitBreaker
        from ..portfolio.tracker import PortfolioTracker

        cfg = get_config()
        breaker = DrawdownCircuitBreaker(max_drawdown_pct=cfg.circuit_breaker.max_drawdown_pct)
        tracker = PortfolioTracker()
        state = tracker.get_state()
        original_action = rec.action
        rec.action = breaker.filter_recommendation(rec.action, state.total_value)
        if rec.action != original_action:
            rec.summary = (
                f"[CIRCUIT BREAKER: {original_action.value}→{rec.action.value}] {rec.summary}"
            )
    except Exception as exc:
        logger.warning("Circuit breaker check failed (non-fatal): %s", exc)

    gs["recommendation"] = rec.model_dump()
    _sync_counters(analysis, gs)
    return gs


# ── Graph construction (cached singleton) ─────────────────────────────

_compiled_graph = None


def build_graph():
    """Build and compile the analysis LangGraph (cached as module singleton)."""
    global _compiled_graph
    if _compiled_graph is not None:
        return _compiled_graph

    graph = StateGraph(GraphState)

    graph.add_node("collect_data", collect_data)
    graph.add_node("run_analysts", run_analysts)
    graph.add_node("run_debate", debate_node)
    graph.add_node("assess_risk", risk_node)
    graph.add_node("quantitative_risk", quantitative_risk_node)
    graph.add_node("synthesize", synthesize_node)

    graph.set_entry_point("collect_data")
    graph.add_edge("collect_data", "run_analysts")
    graph.add_edge("run_analysts", "run_debate")
    graph.add_edge("run_debate", "assess_risk")
    graph.add_edge("assess_risk", "quantitative_risk")
    graph.add_edge("quantitative_risk", "synthesize")
    graph.add_edge("synthesize", END)

    _compiled_graph = graph.compile()
    return _compiled_graph


async def run_analysis(
    ticker: str,
    portfolio_state: PortfolioState | None = None,
) -> Recommendation:
    """Run the full analysis pipeline for a ticker.

    Returns the final Recommendation.
    """
    ticker = ticker.upper()
    logger.info("Starting analysis for %s", ticker)

    graph = build_graph()

    initial_state: GraphState = {
        "ticker": ticker,
        "market_data": None,
        "fundamental_data": None,
        "technical_data": None,
        "sentiment_data": None,
        "macro_data": None,
        "analyst_reports": [],
        "debate_reports": [],
        "risk_assessment": None,
        "recommendation": None,
        "portfolio_state": portfolio_state.model_dump() if portfolio_state else None,
        "quantitative_risk": None,
        "errors": [],
        "total_tokens": 0,
        "total_cost": 0.0,
    }

    # Use ainvoke for async graph with async node functions
    final_state = await graph.ainvoke(initial_state)

    if final_state.get("recommendation"):
        rec = Recommendation(**final_state["recommendation"])
    else:
        from ..models import Action
        rec = Recommendation(
            ticker=ticker,
            action=Action.HOLD,
            confidence=0.0,
            summary="Analysis pipeline failed to produce a recommendation.",
            llm_tokens_used=final_state.get("total_tokens", 0),
            llm_cost_usd=final_state.get("total_cost", 0.0),
        )

    logger.info(
        "Analysis complete for %s: %s (confidence=%.0f%%, tokens=%d, cost=$%.4f)",
        ticker, rec.action.value, rec.confidence,
        rec.llm_tokens_used, rec.llm_cost_usd,
    )
    return rec
