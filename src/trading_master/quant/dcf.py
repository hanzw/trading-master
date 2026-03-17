"""Discounted Cash Flow valuation model."""

from __future__ import annotations


def dcf_valuation(
    fcf_current: float,
    growth_rate_5yr: float,
    terminal_growth: float = 0.03,
    discount_rate: float = 0.10,
    shares_outstanding: float = 1.0,
    margin_of_safety: float = 0.25,
    current_price: float | None = None,
) -> dict:
    """Simple 2-stage DCF model.

    Stage 1: 5 years of projected FCF at *growth_rate_5yr*.
    Stage 2: Terminal value = FCF_5 * (1 + terminal_growth) / (discount_rate - terminal_growth).

    Returns
    -------
    dict with keys:
        intrinsic_value        – per-share fair value
        with_margin_of_safety  – intrinsic * (1 - margin_of_safety)
        upside_pct             – vs current_price (None if price not given)
        fcf_projections        – list of 5 projected FCFs
        terminal_value         – undiscounted terminal value
        pv_fcf                 – present value of stage-1 cash flows
        pv_terminal            – present value of terminal value
    """
    if discount_rate <= terminal_growth:
        raise ValueError(
            f"discount_rate ({discount_rate}) must exceed terminal_growth ({terminal_growth})"
        )

    # Stage 1 – project FCFs for years 1-5
    fcf_projections: list[float] = []
    fcf = fcf_current
    for _ in range(5):
        fcf *= 1.0 + growth_rate_5yr
        fcf_projections.append(fcf)

    # PV of stage-1
    pv_fcf = sum(
        cf / (1.0 + discount_rate) ** (i + 1) for i, cf in enumerate(fcf_projections)
    )

    # Terminal value at end of year 5
    terminal_value = fcf_projections[-1] * (1.0 + terminal_growth) / (
        discount_rate - terminal_growth
    )
    pv_terminal = terminal_value / (1.0 + discount_rate) ** 5

    total_value = pv_fcf + pv_terminal
    intrinsic_value = total_value / shares_outstanding
    with_mos = intrinsic_value * (1.0 - margin_of_safety)

    upside_pct: float | None = None
    if current_price is not None and current_price > 0:
        upside_pct = (intrinsic_value - current_price) / current_price

    return {
        "intrinsic_value": intrinsic_value,
        "with_margin_of_safety": with_mos,
        "upside_pct": upside_pct,
        "fcf_projections": fcf_projections,
        "terminal_value": terminal_value,
        "pv_fcf": pv_fcf,
        "pv_terminal": pv_terminal,
    }


def gordon_growth_model(
    dividend_per_share: float,
    growth_rate: float,
    discount_rate: float,
) -> float:
    """Gordon Growth Model: P = D * (1 + g) / (r - g).

    Raises ValueError if discount_rate <= growth_rate.
    """
    if discount_rate <= growth_rate:
        raise ValueError(
            f"discount_rate ({discount_rate}) must exceed growth_rate ({growth_rate})"
        )
    return dividend_per_share * (1.0 + growth_rate) / (discount_rate - growth_rate)


def auto_dcf(ticker: str) -> dict:
    """Auto-compute DCF using yfinance data.

    Fetches FCF from cashflow, shares from info, estimates growth from
    historical FCF trajectory.

    Returns dcf_valuation() result dict augmented with:
        current_price, ticker, verdict ("undervalued" / "overvalued" / "fair").
    """
    import yfinance as yf  # lazy import

    stock = yf.Ticker(ticker)
    info = stock.info

    # --- Free cash flow ---
    cf = stock.cashflow
    if cf is None or cf.empty:
        raise ValueError(f"No cashflow data available for {ticker}")

    # yfinance cashflow row name varies; try common labels
    fcf_row = None
    for label in ("Free Cash Flow", "FreeCashFlow"):
        if label in cf.index:
            fcf_row = cf.loc[label]
            break
    if fcf_row is None:
        raise ValueError(f"Cannot find Free Cash Flow row for {ticker}")

    fcf_values = fcf_row.dropna().sort_index()
    fcf_current = float(fcf_values.iloc[-1])

    # Estimate 5-year growth from historical CAGR (use up to 4 years of data)
    if len(fcf_values) >= 2 and fcf_values.iloc[0] > 0 and fcf_current > 0:
        n_years = len(fcf_values) - 1
        cagr = (fcf_current / float(fcf_values.iloc[0])) ** (1.0 / n_years) - 1.0
        growth_rate_5yr = max(min(cagr, 0.30), -0.10)  # clamp
    else:
        growth_rate_5yr = 0.05  # fallback

    shares = float(info.get("sharesOutstanding", 1))
    current_price = float(info.get("currentPrice", info.get("previousClose", 0)))

    result = dcf_valuation(
        fcf_current=fcf_current,
        growth_rate_5yr=growth_rate_5yr,
        shares_outstanding=shares,
        current_price=current_price,
    )

    # Verdict
    upside = result["upside_pct"]
    if upside is not None:
        if upside > 0.15:
            verdict = "undervalued"
        elif upside < -0.15:
            verdict = "overvalued"
        else:
            verdict = "fair"
    else:
        verdict = "unknown"

    result["ticker"] = ticker.upper()
    result["current_price"] = current_price
    result["growth_rate_5yr"] = growth_rate_5yr
    result["verdict"] = verdict
    return result
