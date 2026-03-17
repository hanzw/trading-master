"""Backtesting framework — track whether past recommendations were correct."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta

import numpy as np
import yfinance as yf

from ..db import get_db

logger = logging.getLogger(__name__)

# Signal directions: positive means "expecting price to go up"
_BULLISH_ACTIONS = {"BUY", "STRONG_BUY"}
_BEARISH_ACTIONS = {"SELL", "STRONG_SELL"}


def _fetch_price_at(ticker: str, date: datetime) -> float | None:
    """Fetch the closing price for *ticker* on or near *date* using yfinance."""
    try:
        start = date - timedelta(days=5)
        end = date + timedelta(days=5)
        hist = yf.Ticker(ticker).history(start=start.strftime("%Y-%m-%d"),
                                          end=end.strftime("%Y-%m-%d"))
        if hist.empty:
            return None
        # Find the closest date
        idx = hist.index.get_indexer([date], method="nearest")[0]
        return float(hist.iloc[idx]["Close"])
    except Exception as exc:
        logger.warning("Could not fetch price for %s near %s: %s", ticker, date, exc)
        return None


def _fetch_price_on_date(ticker: str, target: datetime) -> float | None:
    """Fetch closing price on a specific date (with a small window for weekends/holidays)."""
    return _fetch_price_at(ticker, target)


def _is_correct(action: str, return_pct: float) -> bool:
    """Determine if a recommendation was correct given the subsequent return."""
    if action in _BULLISH_ACTIONS:
        return return_pct > 0
    elif action in _BEARISH_ACTIONS:
        return return_pct < 0
    else:
        # HOLD — correct if the price didn't move more than +/-5%
        return abs(return_pct) <= 5.0


def track_recommendation_outcomes(
    horizons_days: list[int] | None = None,
    ticker: str | None = None,
) -> list[dict]:
    """For each past recommendation, fetch the price at recommendation time
    and the price at each horizon. Compute return and whether the signal was correct.

    Returns list of dicts: {
        id, ticker, action, confidence, rec_date, rec_price,
        analyst_reports, outcomes: {30: {price, return_pct, correct}, ...}
    }
    """
    if horizons_days is None:
        horizons_days = [30, 90, 180]

    db = get_db()
    recs = db.get_recommendations(ticker=ticker, limit=500)

    results: list[dict] = []
    now = datetime.now()

    for rec in recs:
        rec_date = datetime.fromisoformat(rec["timestamp"])
        action = rec["action"]
        rec_ticker = rec["ticker"]

        # Fetch price at recommendation time
        rec_price = _fetch_price_at(rec_ticker, rec_date)
        if rec_price is None or rec_price <= 0:
            continue

        # Parse analyst_reports
        try:
            analyst_reports = json.loads(rec.get("analyst_reports", "[]"))
        except (json.JSONDecodeError, TypeError):
            analyst_reports = []

        outcomes: dict[int, dict] = {}
        for h in horizons_days:
            target_date = rec_date + timedelta(days=h)
            if target_date > now:
                # Horizon hasn't elapsed yet
                continue
            future_price = _fetch_price_on_date(rec_ticker, target_date)
            if future_price is None:
                continue
            return_pct = (future_price - rec_price) / rec_price * 100
            outcomes[h] = {
                "price": future_price,
                "return_pct": round(return_pct, 2),
                "correct": _is_correct(action, return_pct),
            }

        results.append({
            "id": rec["id"],
            "ticker": rec_ticker,
            "action": action,
            "confidence": rec["confidence"],
            "rec_date": rec_date.isoformat(),
            "rec_price": rec_price,
            "analyst_reports": analyst_reports,
            "outcomes": outcomes,
        })

    return results


def compute_hit_rate(outcomes: list[dict], horizon: int = 90) -> dict:
    """Compute hit rate for a specific horizon.

    Returns {hit_rate, avg_return, n_total, n_correct, n_buy, n_sell}
    """
    eligible = [o for o in outcomes if horizon in o.get("outcomes", {})]
    if not eligible:
        return {
            "hit_rate": None,
            "avg_return": None,
            "n_total": 0,
            "n_correct": 0,
            "n_buy": 0,
            "n_sell": 0,
        }

    n_correct = sum(1 for o in eligible if o["outcomes"][horizon]["correct"])
    returns = [o["outcomes"][horizon]["return_pct"] for o in eligible]
    n_buy = sum(1 for o in eligible if o["action"] in _BULLISH_ACTIONS)
    n_sell = sum(1 for o in eligible if o["action"] in _BEARISH_ACTIONS)

    return {
        "hit_rate": round(n_correct / len(eligible) * 100, 2),
        "avg_return": round(float(np.mean(returns)), 2),
        "n_total": len(eligible),
        "n_correct": n_correct,
        "n_buy": n_buy,
        "n_sell": n_sell,
    }


def compute_agent_accuracy(outcomes: list[dict], horizon: int = 90) -> dict:
    """Per-agent (fundamental/technical/sentiment) accuracy breakdown.

    Parse analyst_reports from the recommendation JSON.
    Returns {fundamental: {hit_rate, avg_confidence, n}, technical: {...}, sentiment: {...}}
    """
    agents: dict[str, dict] = {}

    for o in outcomes:
        h_outcome = o.get("outcomes", {}).get(horizon)
        if h_outcome is None:
            continue

        actual_return = h_outcome["return_pct"]

        for report in o.get("analyst_reports", []):
            analyst = report.get("analyst", "unknown")
            signal = report.get("signal", "HOLD")
            confidence = report.get("confidence", 50.0)

            if analyst not in agents:
                agents[analyst] = {"correct": 0, "total": 0, "confidences": []}

            # Determine if this agent's individual signal was correct
            correct = _is_correct(signal, actual_return)
            agents[analyst]["total"] += 1
            if correct:
                agents[analyst]["correct"] += 1
            agents[analyst]["confidences"].append(confidence)

    result: dict[str, dict] = {}
    for analyst, data in agents.items():
        total = data["total"]
        result[analyst] = {
            "hit_rate": round(data["correct"] / total * 100, 2) if total > 0 else None,
            "avg_confidence": round(float(np.mean(data["confidences"])), 2) if data["confidences"] else None,
            "n": total,
        }

    return result


def compute_calibration(outcomes: list[dict], horizon: int = 90) -> list[dict]:
    """Calibration check: for confidence buckets (0-25, 25-50, 50-75, 75-100),
    what was the actual hit rate?

    Good calibration means a 75% confidence bucket has ~75% accuracy.
    Returns list of {bucket, count, hit_rate, avg_confidence}
    """
    buckets = [
        {"label": "0-25", "lo": 0, "hi": 25},
        {"label": "25-50", "lo": 25, "hi": 50},
        {"label": "50-75", "lo": 50, "hi": 75},
        {"label": "75-100", "lo": 75, "hi": 100},
    ]

    results: list[dict] = []

    for bucket in buckets:
        eligible = []
        for o in outcomes:
            if horizon not in o.get("outcomes", {}):
                continue
            conf = o["confidence"]
            in_range = bucket["lo"] <= conf < bucket["hi"]
            is_boundary = (bucket["hi"] == 100 and conf == 100)
            if in_range or is_boundary:
                eligible.append(o)

        if not eligible:
            results.append({
                "bucket": bucket["label"],
                "count": 0,
                "hit_rate": None,
                "avg_confidence": None,
            })
            continue

        n_correct = sum(1 for o in eligible if o["outcomes"][horizon]["correct"])
        confidences = [o["confidence"] for o in eligible]

        results.append({
            "bucket": bucket["label"],
            "count": len(eligible),
            "hit_rate": round(n_correct / len(eligible) * 100, 2),
            "avg_confidence": round(float(np.mean(confidences)), 2),
        })

    return results


def backtest_summary(ticker: str | None = None) -> dict:
    """Master function: calls track_recommendation_outcomes, compute_hit_rate,
    compute_agent_accuracy, compute_calibration. Returns full summary dict."""
    outcomes = track_recommendation_outcomes(ticker=ticker)

    hit_30 = compute_hit_rate(outcomes, 30)
    hit_90 = compute_hit_rate(outcomes, 90)
    hit_180 = compute_hit_rate(outcomes, 180)

    agent_accuracy = compute_agent_accuracy(outcomes, 90)
    calibration = compute_calibration(outcomes, 90)

    return {
        "total_recommendations": len(outcomes),
        "hit_rate_30d": hit_30,
        "hit_rate_90d": hit_90,
        "hit_rate_180d": hit_180,
        "agent_accuracy": agent_accuracy,
        "calibration": calibration,
        "outcomes": outcomes,
        "last_updated": datetime.now().isoformat(),
    }
