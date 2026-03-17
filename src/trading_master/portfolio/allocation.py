"""Asset allocation framework — classification, drift detection, rebalance suggestions."""

from __future__ import annotations

from ..models import AssetClass, AllocationModel, AllocationTarget, PortfolioState

# ── Ticker -> AssetClass mapping (covers common ETFs and stocks) ──────

TICKER_CLASS_MAP: dict[str, AssetClass] = {
    # US Equity
    "SPY": AssetClass.US_EQUITY, "VOO": AssetClass.US_EQUITY,
    "QQQ": AssetClass.US_EQUITY, "QQQM": AssetClass.US_EQUITY,
    "VIG": AssetClass.US_EQUITY, "JEPI": AssetClass.US_EQUITY,
    "WMT": AssetClass.US_EQUITY, "MSFT": AssetClass.US_EQUITY,
    "GOOGL": AssetClass.US_EQUITY, "BRK-B": AssetClass.US_EQUITY,
    "CHAT": AssetClass.US_EQUITY,
    # International equity
    "VXUS": AssetClass.INTL_EQUITY, "EFA": AssetClass.INTL_EQUITY,
    "VWO": AssetClass.INTL_EQUITY, "IEFA": AssetClass.INTL_EQUITY,
    # Short-term treasury
    "SGOV": AssetClass.SHORT_TERM_TREASURY, "SHV": AssetClass.SHORT_TERM_TREASURY,
    "BIL": AssetClass.SHORT_TERM_TREASURY,
    # Fixed income
    "TLT": AssetClass.FIXED_INCOME, "BND": AssetClass.FIXED_INCOME,
    "AGG": AssetClass.FIXED_INCOME, "VCIT": AssetClass.FIXED_INCOME,
    # REITs
    "VNQ": AssetClass.REITS, "SCHH": AssetClass.REITS,
    # Commodities
    "COPX": AssetClass.COMMODITIES, "GLD": AssetClass.COMMODITIES,
    "IAU": AssetClass.COMMODITIES, "SLV": AssetClass.COMMODITIES,
    # Alternatives
    "ARKVX": AssetClass.ALTERNATIVES, "RVI": AssetClass.ALTERNATIVES,
}

# ── Preset allocation models ─────────────────────────────────────────

PRESET_MODELS: dict[str, AllocationModel] = {
    "balanced": AllocationModel(name="balanced", targets=[
        AllocationTarget(asset_class=AssetClass.US_EQUITY, target_pct=50, min_pct=40, max_pct=60),
        AllocationTarget(asset_class=AssetClass.FIXED_INCOME, target_pct=20, min_pct=15, max_pct=30),
        AllocationTarget(asset_class=AssetClass.SHORT_TERM_TREASURY, target_pct=15, min_pct=5, max_pct=25),
        AllocationTarget(asset_class=AssetClass.COMMODITIES, target_pct=5, min_pct=0, max_pct=10),
        AllocationTarget(asset_class=AssetClass.ALTERNATIVES, target_pct=5, min_pct=0, max_pct=10),
        AllocationTarget(asset_class=AssetClass.CASH, target_pct=5, min_pct=2, max_pct=15),
    ]),
    "growth": AllocationModel(name="growth", targets=[
        AllocationTarget(asset_class=AssetClass.US_EQUITY, target_pct=70, min_pct=60, max_pct=80),
        AllocationTarget(asset_class=AssetClass.FIXED_INCOME, target_pct=10, min_pct=5, max_pct=20),
        AllocationTarget(asset_class=AssetClass.SHORT_TERM_TREASURY, target_pct=10, min_pct=5, max_pct=20),
        AllocationTarget(asset_class=AssetClass.COMMODITIES, target_pct=5, min_pct=0, max_pct=10),
        AllocationTarget(asset_class=AssetClass.ALTERNATIVES, target_pct=3, min_pct=0, max_pct=10),
        AllocationTarget(asset_class=AssetClass.CASH, target_pct=2, min_pct=1, max_pct=10),
    ]),
    "conservative": AllocationModel(name="conservative", targets=[
        AllocationTarget(asset_class=AssetClass.US_EQUITY, target_pct=30, min_pct=20, max_pct=40),
        AllocationTarget(asset_class=AssetClass.FIXED_INCOME, target_pct=30, min_pct=20, max_pct=40),
        AllocationTarget(asset_class=AssetClass.SHORT_TERM_TREASURY, target_pct=25, min_pct=15, max_pct=35),
        AllocationTarget(asset_class=AssetClass.COMMODITIES, target_pct=5, min_pct=0, max_pct=10),
        AllocationTarget(asset_class=AssetClass.ALTERNATIVES, target_pct=0, min_pct=0, max_pct=5),
        AllocationTarget(asset_class=AssetClass.CASH, target_pct=10, min_pct=5, max_pct=20),
    ]),
}

# ── Suggested tickers per asset class (for rebalance suggestions) ─────

SUGGESTED_TICKERS: dict[AssetClass, list[str]] = {
    AssetClass.US_EQUITY: ["VOO", "QQQ", "VIG"],
    AssetClass.INTL_EQUITY: ["VXUS", "EFA"],
    AssetClass.FIXED_INCOME: ["BND", "TLT"],
    AssetClass.SHORT_TERM_TREASURY: ["SGOV", "SHV"],
    AssetClass.REITS: ["VNQ"],
    AssetClass.COMMODITIES: ["GLD", "COPX"],
    AssetClass.ALTERNATIVES: ["RVI"],
    AssetClass.CASH: [],
}


def classify_ticker(ticker: str) -> AssetClass:
    """Map a ticker to its asset class. Default to US_EQUITY for unknown stocks."""
    return TICKER_CLASS_MAP.get(ticker.upper(), AssetClass.US_EQUITY)


def compute_current_allocation(state: PortfolioState) -> list[AllocationTarget]:
    """Compute current allocation by asset class from portfolio positions."""
    state.recalculate()
    total = state.total_value
    if total <= 0:
        return []

    # Accumulate market value per asset class
    class_values: dict[AssetClass, float] = {}
    for ticker, pos in state.positions.items():
        ac = classify_ticker(ticker)
        class_values[ac] = class_values.get(ac, 0.0) + pos.market_value

    # Always include cash
    class_values[AssetClass.CASH] = class_values.get(AssetClass.CASH, 0.0) + state.cash

    targets: list[AllocationTarget] = []
    for ac in AssetClass:
        value = class_values.get(ac, 0.0)
        pct = (value / total) * 100.0 if total > 0 else 0.0
        targets.append(AllocationTarget(
            asset_class=ac,
            target_pct=0.0,  # no target yet — just current snapshot
            min_pct=0.0,
            max_pct=100.0,
            current_pct=round(pct, 2),
        ))

    return targets


def compute_drift(
    current: list[AllocationTarget], model: AllocationModel
) -> list[AllocationTarget]:
    """Compare current allocation to target model, compute drift for each class."""
    # Build lookup from current allocation
    current_map: dict[AssetClass, float] = {
        t.asset_class: t.current_pct for t in current
    }

    result: list[AllocationTarget] = []
    for target in model.targets:
        cur_pct = current_map.get(target.asset_class, 0.0)
        drift = round(cur_pct - target.target_pct, 2)
        result.append(AllocationTarget(
            asset_class=target.asset_class,
            target_pct=target.target_pct,
            min_pct=target.min_pct,
            max_pct=target.max_pct,
            current_pct=cur_pct,
            drift_pct=drift,
        ))

    # Include asset classes that are in the portfolio but not in the model
    model_classes = {t.asset_class for t in model.targets}
    for t in current:
        if t.asset_class not in model_classes and t.current_pct > 0:
            result.append(AllocationTarget(
                asset_class=t.asset_class,
                target_pct=0.0,
                min_pct=0.0,
                max_pct=0.0,
                current_pct=t.current_pct,
                drift_pct=t.current_pct,  # 100% drift since target is 0
            ))

    return result


def needs_rebalance(
    drift_targets: list[AllocationTarget], threshold_pct: float = 5.0
) -> bool:
    """Returns True if any asset class has drifted beyond the threshold."""
    return any(abs(t.drift_pct) > threshold_pct for t in drift_targets)


def _estimate_tax_for_sell(
    state: PortfolioState, asset_class: AssetClass, amount_usd: float,
) -> dict:
    """Estimate capital gains tax for selling *amount_usd* of a given asset class.

    Uses avg_cost from positions (not individual lots) so the result is an
    approximation.  Without per-lot purchase dates we fall back to a heuristic:
    positions held for >365 days from the first recorded action are treated as
    long-term.  If no action history is available the holding period defaults to
    short-term to be conservative.
    """
    from datetime import datetime, timezone

    # Identify positions in this asset class
    class_positions = [
        (ticker, pos)
        for ticker, pos in state.positions.items()
        if classify_ticker(ticker) == asset_class
    ]
    if not class_positions:
        return {}

    # Aggregate values
    total_class_value = sum(pos.market_value for _, pos in class_positions)
    if total_class_value <= 0:
        return {}

    # Weighted-average gain calculation across all positions in this class
    total_gain = 0.0
    weighted_days = 0.0
    total_weight = 0.0

    for ticker, pos in class_positions:
        # What fraction of the sell comes from this position?
        frac = pos.market_value / total_class_value
        pos_sell_value = amount_usd * frac
        if pos.current_price <= 0 or pos.quantity <= 0:
            continue
        shares_to_sell = pos_sell_value / pos.current_price
        gain = (pos.current_price - pos.avg_cost) * shares_to_sell
        total_gain += gain

        # Estimate holding period from action history
        days_held = _estimate_days_held(ticker)
        weighted_days += days_held * frac
        total_weight += frac

    avg_days_held = weighted_days / total_weight if total_weight > 0 else 0
    holding_period = "long-term" if avg_days_held > 365 else "short-term"
    tax_rate = 0.15 if holding_period == "long-term" else 0.35
    estimated_tax = total_gain * tax_rate if total_gain > 0 else 0.0

    # Build human-readable warning
    label = asset_class.value.replace("_", " ").title()
    if total_gain > 0:
        tax_warning = (
            f"SELL {label} would realize ~${total_gain:,.0f} {holding_period} "
            f"capital gain (~${estimated_tax:,.0f} tax)"
        )
    else:
        tax_warning = (
            f"SELL {label} would realize ~${total_gain:,.0f} capital loss (no tax owed)"
        )

    return {
        "estimated_gain": round(total_gain, 2),
        "holding_period": holding_period,
        "estimated_tax": round(estimated_tax, 2),
        "tax_warning": tax_warning,
    }


def _estimate_days_held(ticker: str) -> float:
    """Estimate how many days a position has been held by looking at the earliest
    BUY action in the database.  Returns 0 if no history is found (conservative
    — will be classified as short-term).
    """
    from datetime import datetime, timezone
    try:
        from ..db import get_db
        db = get_db()
        row = db.conn.execute(
            "SELECT MIN(timestamp) FROM actions WHERE ticker = ? AND action = 'BUY'",
            (ticker.upper(),),
        ).fetchone()
        if row and row[0]:
            first_buy = datetime.fromisoformat(row[0])
            now = datetime.now(tz=first_buy.tzinfo)
            return max((now - first_buy).days, 0)
    except Exception:
        pass
    return 0


def suggest_rebalance(
    state: PortfolioState, model: AllocationModel
) -> list[dict]:
    """Suggest trades to rebalance.

    Returns list of dicts with keys:
        asset_class, direction, amount_usd, suggested_tickers,
        estimated_gain, holding_period, estimated_tax, tax_warning  (SELL only)
    """
    current = compute_current_allocation(state)
    drift_targets = compute_drift(current, model)

    state.recalculate()
    total = state.total_value

    suggestions: list[dict] = []
    for t in drift_targets:
        if abs(t.drift_pct) < 0.5:
            continue  # negligible drift

        amount_usd = abs(t.drift_pct) / 100.0 * total
        direction = "SELL" if t.drift_pct > 0 else "BUY"

        entry: dict = {
            "asset_class": t.asset_class.value,
            "direction": direction,
            "amount_usd": round(amount_usd, 2),
            "drift_pct": t.drift_pct,
            "current_pct": t.current_pct,
            "target_pct": t.target_pct,
            "suggested_tickers": SUGGESTED_TICKERS.get(t.asset_class, []),
        }

        # Add tax awareness for SELL suggestions
        if direction == "SELL":
            tax_info = _estimate_tax_for_sell(state, t.asset_class, amount_usd)
            entry.update({
                "estimated_gain": tax_info.get("estimated_gain", 0.0),
                "holding_period": tax_info.get("holding_period", "short-term"),
                "estimated_tax": tax_info.get("estimated_tax", 0.0),
                "tax_warning": tax_info.get("tax_warning", ""),
            })

        suggestions.append(entry)

    # Sort by absolute drift descending (biggest misalignments first)
    suggestions.sort(key=lambda s: abs(s["drift_pct"]), reverse=True)
    return suggestions


# ── Regime-conditional allocation adjustments ────────────────────────

REGIME_ADJUSTMENTS: dict[str, dict[AssetClass, float]] = {
    "bull": {},  # No adjustment — use preset as-is
    "sideways": {
        AssetClass.US_EQUITY: -5,
        AssetClass.SHORT_TERM_TREASURY: +3,
        AssetClass.CASH: +2,
    },
    "bear": {
        AssetClass.US_EQUITY: -15,
        AssetClass.FIXED_INCOME: +5,
        AssetClass.SHORT_TERM_TREASURY: +7,
        AssetClass.CASH: +3,
    },
    "crisis": {
        AssetClass.US_EQUITY: -25,
        AssetClass.FIXED_INCOME: +5,
        AssetClass.SHORT_TERM_TREASURY: +12,
        AssetClass.CASH: +8,
    },
}


def regime_adjusted_model(base_model: AllocationModel, regime: str) -> AllocationModel:
    """Apply regime adjustments to allocation targets. Adjustments shift target_pct
    while keeping min/max constraints valid. Returns a new AllocationModel."""
    adjustments = REGIME_ADJUSTMENTS.get(regime, {})
    if not adjustments:
        return base_model

    new_targets: list[AllocationTarget] = []
    for t in base_model.targets:
        adj = adjustments.get(t.asset_class, 0.0)
        new_target_pct = t.target_pct + adj
        # Clamp to 0-100 and ensure min <= target <= max still holds
        new_target_pct = max(0.0, min(100.0, new_target_pct))
        # Widen min/max if the adjustment pushes target outside the original range
        new_min = min(t.min_pct, new_target_pct)
        new_max = max(t.max_pct, new_target_pct)
        new_targets.append(AllocationTarget(
            asset_class=t.asset_class,
            target_pct=round(new_target_pct, 2),
            min_pct=round(new_min, 2),
            max_pct=round(new_max, 2),
        ))

    return AllocationModel(
        name=f"{base_model.name}+{regime}",
        targets=new_targets,
        rebalance_threshold_pct=base_model.rebalance_threshold_pct,
    )


def regime_allocation_alert(
    state: PortfolioState, model: AllocationModel, regime: str,
) -> list[str]:
    """Cross-reference current allocation against regime-adjusted model.
    Returns list of alert strings like:
    'BEAR regime: US equity at 76% exceeds adjusted target of 55% (range 40-60%)'
    """
    adjusted = regime_adjusted_model(model, regime)
    current = compute_current_allocation(state)
    current_map: dict[AssetClass, float] = {
        t.asset_class: t.current_pct for t in current
    }

    alerts: list[str] = []
    regime_upper = regime.upper()
    for t in adjusted.targets:
        cur_pct = current_map.get(t.asset_class, 0.0)
        label = t.asset_class.value.replace("_", " ").title()
        if cur_pct > t.max_pct:
            alerts.append(
                f"{regime_upper} regime: {label} at {cur_pct:.0f}% exceeds "
                f"adjusted target of {t.target_pct:.0f}% "
                f"(range {t.min_pct:.0f}-{t.max_pct:.0f}%)"
            )
        elif cur_pct < t.min_pct:
            alerts.append(
                f"{regime_upper} regime: {label} at {cur_pct:.0f}% below "
                f"adjusted target of {t.target_pct:.0f}% "
                f"(range {t.min_pct:.0f}-{t.max_pct:.0f}%)"
            )

    return alerts


# ── Overlap-aware concentration scoring ──────────────────────────────

# Sub-class grouping: tickers that track the same underlying
OVERLAP_GROUPS: dict[str, list[str]] = {
    "sp500_trackers": ["SPY", "VOO", "IVV", "SPLG"],
    "nasdaq100_trackers": ["QQQ", "QQQM", "ONEQ"],
    "total_market": ["VTI", "ITOT", "SPTM"],
    "long_treasury": ["TLT", "VGLT", "SPTL"],
    "short_treasury": ["SGOV", "SHV", "BIL"],
}

# Preferred ticker per group (lowest expense ratio)
_PREFERRED_TICKER: dict[str, str] = {
    "sp500_trackers": "VOO",
    "nasdaq100_trackers": "QQQM",
    "total_market": "VTI",
    "long_treasury": "VGLT",
    "short_treasury": "SGOV",
}


def detect_overlaps(positions: dict[str, float]) -> list[dict]:
    """Find overlapping positions. *positions* maps ticker -> market value.
    Returns list of:
    {group: "sp500_trackers", tickers: ["SPY", "VOO"], combined_pct: 23.3,
     suggestion: "Consider consolidating into VOO (lowest expense ratio)"}
    """
    total_value = sum(positions.values())
    if total_value <= 0:
        return []

    held_tickers = {t.upper() for t in positions}
    results: list[dict] = []

    for group_name, group_tickers in OVERLAP_GROUPS.items():
        found = [t for t in group_tickers if t in held_tickers]
        if len(found) < 2:
            continue
        combined_value = sum(positions.get(t, 0.0) for t in found)
        combined_pct = round((combined_value / total_value) * 100, 2)
        preferred = _PREFERRED_TICKER.get(group_name, found[0])
        results.append({
            "group": group_name,
            "tickers": found,
            "combined_pct": combined_pct,
            "suggestion": f"Consider consolidating into {preferred} (lowest expense ratio)",
        })

    return results


def effective_concentration(
    positions: dict[str, float], total_value: float,
) -> dict:
    """Compute concentration treating overlap groups as single positions.
    Returns {effective_positions: int, herfindahl: float, top3_pct: float,
             overlap_warnings: list}
    """
    if total_value <= 0:
        return {
            "effective_positions": 0,
            "herfindahl": 0.0,
            "top3_pct": 0.0,
            "overlap_warnings": [],
        }

    # Build effective position map: merge overlap groups into single entries
    effective: dict[str, float] = {}
    assigned: set[str] = set()

    for group_name, group_tickers in OVERLAP_GROUPS.items():
        group_upper = [t.upper() for t in group_tickers]
        found = [t for t in group_upper if t in positions]
        if found:
            combined = sum(positions[t] for t in found)
            effective[group_name] = combined
            assigned.update(found)

    # Add non-overlapping positions
    for ticker, value in positions.items():
        if ticker.upper() not in assigned:
            effective[ticker] = value

    # Compute metrics
    effective_positions = len(effective)
    weights = [(v / total_value) * 100 for v in effective.values()]
    weights.sort(reverse=True)

    herfindahl = sum(w ** 2 for w in weights)
    top3_pct = round(sum(weights[:3]), 2) if len(weights) >= 3 else round(sum(weights), 2)

    overlap_warnings = detect_overlaps(positions)

    return {
        "effective_positions": effective_positions,
        "herfindahl": round(herfindahl, 2),
        "top3_pct": top3_pct,
        "overlap_warnings": overlap_warnings,
    }
