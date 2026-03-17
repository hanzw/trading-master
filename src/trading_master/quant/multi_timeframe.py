"""Multi-Timeframe Technical Analysis — consensus signals across timeframes.

Computes RSI, MACD, and SMA trend on daily, weekly, and monthly bars,
then produces a weighted consensus signal. Professional technical analysts
check multiple timeframes to avoid false signals on a single timeframe.

Signal logic:
  - Each timeframe produces a score from -1 (bearish) to +1 (bullish)
  - Scores are weighted: monthly > weekly > daily (higher = more reliable)
  - Consensus score determines overall signal (STRONG_BUY to STRONG_SELL)

Timeframe alignment:
  - All timeframes bullish → high conviction (STRONG_BUY)
  - Mixed signals → caution (HOLD)
  - All bearish → STRONG_SELL
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TimeframeSignal:
    """Signal from a single timeframe."""
    timeframe: str            # "daily", "weekly", "monthly"
    rsi: float | None = None
    macd_histogram: float | None = None
    sma_trend: float = 0.0   # +1 above SMA, -1 below, 0 unknown
    score: float = 0.0       # -1 to +1 composite
    signal: str = "NEUTRAL"   # BULLISH, BEARISH, NEUTRAL


@dataclass
class MultiTimeframeResult:
    """Result of multi-timeframe analysis."""
    ticker: str
    timeframes: list[TimeframeSignal]
    consensus_score: float    # -1 to +1
    consensus_signal: str     # STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL
    alignment: str            # "aligned_bull", "aligned_bear", "mixed", "neutral"
    n_bullish: int
    n_bearish: int
    n_neutral: int

    @property
    def is_aligned(self) -> bool:
        """True if all timeframes agree on direction."""
        return self.alignment in ("aligned_bull", "aligned_bear")

    @property
    def signal_summary(self) -> dict[str, str]:
        """Timeframe → signal mapping."""
        return {tf.timeframe: tf.signal for tf in self.timeframes}


def _resample_to_weekly(prices: np.ndarray, n: int) -> np.ndarray:
    """Resample daily prices to weekly (every 5th day closing price)."""
    if len(prices) < 5:
        return prices
    # Take every 5th price (end of week), plus the last price
    weekly = prices[4::5]
    if len(prices) % 5 != 0:
        weekly = np.append(weekly, prices[-1])
    return weekly


def _resample_to_monthly(prices: np.ndarray) -> np.ndarray:
    """Resample daily prices to monthly (every 21st day closing price)."""
    if len(prices) < 21:
        return prices
    monthly = prices[20::21]
    if len(prices) % 21 != 0:
        monthly = np.append(monthly, prices[-1])
    return monthly


def _compute_rsi(prices: np.ndarray, period: int = 14) -> float | None:
    """Compute RSI from price series."""
    if len(prices) < period + 1:
        return None

    deltas = np.diff(prices)
    gains = np.maximum(deltas, 0)
    losses = np.abs(np.minimum(deltas, 0))

    avg_gain = gains[-period:].mean()
    avg_loss = losses[-period:].mean()

    if avg_loss < 1e-10:
        return 100.0

    rs = avg_gain / avg_loss
    return float(100.0 - (100.0 / (1.0 + rs)))


def _compute_macd_histogram(prices: np.ndarray) -> float | None:
    """Compute MACD histogram (MACD - Signal)."""
    if len(prices) < 26:
        return None

    def ema(data, span):
        alpha = 2.0 / (span + 1)
        result = np.zeros_like(data)
        result[0] = data[0]
        for i in range(1, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
        return result

    ema12 = ema(prices, 12)
    ema26 = ema(prices, 26)
    macd_line = ema12 - ema26

    if len(macd_line) < 9:
        return float(macd_line[-1])

    signal_line = ema(macd_line, 9)
    histogram = macd_line[-1] - signal_line[-1]
    return float(histogram)


def _compute_sma_trend(prices: np.ndarray, short: int = 20, long: int = 50) -> float:
    """Compute SMA trend: +1 if price > SMA_short > SMA_long, -1 if reversed, 0 mixed."""
    if len(prices) < long:
        if len(prices) >= short:
            sma_s = prices[-short:].mean()
            return 1.0 if prices[-1] > sma_s else -1.0
        return 0.0

    sma_short = prices[-short:].mean()
    sma_long = prices[-long:].mean()

    if prices[-1] > sma_short > sma_long:
        return 1.0
    elif prices[-1] < sma_short < sma_long:
        return -1.0
    else:
        return 0.0


def _score_timeframe(rsi: float | None, macd_hist: float | None, sma_trend: float) -> float:
    """Combine indicators into a single score from -1 to +1."""
    scores = []

    if rsi is not None:
        if rsi > 70:
            scores.append(-0.5)    # overbought → bearish signal
        elif rsi > 60:
            scores.append(0.3)
        elif rsi > 40:
            scores.append(0.0)
        elif rsi > 30:
            scores.append(-0.3)
        else:
            scores.append(0.5)     # oversold → bullish signal (mean reversion)

    if macd_hist is not None:
        if macd_hist > 0:
            scores.append(min(macd_hist * 10, 1.0))
        else:
            scores.append(max(macd_hist * 10, -1.0))

    scores.append(sma_trend)

    if not scores:
        return 0.0

    return float(np.clip(np.mean(scores), -1.0, 1.0))


def multi_timeframe_analysis(
    daily_prices: np.ndarray,
    ticker: str = "UNKNOWN",
    weights: tuple[float, float, float] = (0.25, 0.35, 0.40),
) -> MultiTimeframeResult:
    """Analyze a ticker across daily, weekly, and monthly timeframes.

    Parameters
    ----------
    daily_prices : (n,) daily closing prices (oldest first)
    ticker : asset identifier
    weights : (daily_weight, weekly_weight, monthly_weight) for consensus

    Returns
    -------
    MultiTimeframeResult with per-timeframe signals and consensus.
    """
    daily_prices = np.asarray(daily_prices, dtype=float)

    if len(daily_prices) < 30:
        raise ValueError(f"Need at least 30 daily prices, got {len(daily_prices)}.")

    # Resample
    weekly_prices = _resample_to_weekly(daily_prices, len(daily_prices))
    monthly_prices = _resample_to_monthly(daily_prices)

    timeframes: list[TimeframeSignal] = []

    for label, prices, sma_params in [
        ("daily", daily_prices, (20, 50)),
        ("weekly", weekly_prices, (10, 20)),
        ("monthly", monthly_prices, (5, 10)),
    ]:
        rsi = _compute_rsi(prices)
        macd_hist = _compute_macd_histogram(prices)
        sma_trend = _compute_sma_trend(prices, *sma_params)

        score = _score_timeframe(rsi, macd_hist, sma_trend)

        if score > 0.2:
            signal = "BULLISH"
        elif score < -0.2:
            signal = "BEARISH"
        else:
            signal = "NEUTRAL"

        timeframes.append(TimeframeSignal(
            timeframe=label,
            rsi=rsi,
            macd_histogram=macd_hist,
            sma_trend=sma_trend,
            score=score,
            signal=signal,
        ))

    # Consensus
    w_daily, w_weekly, w_monthly = weights
    total_w = w_daily + w_weekly + w_monthly
    consensus = (
        w_daily * timeframes[0].score
        + w_weekly * timeframes[1].score
        + w_monthly * timeframes[2].score
    ) / total_w

    # Classify consensus
    if consensus > 0.5:
        consensus_signal = "STRONG_BUY"
    elif consensus > 0.15:
        consensus_signal = "BUY"
    elif consensus > -0.15:
        consensus_signal = "HOLD"
    elif consensus > -0.5:
        consensus_signal = "SELL"
    else:
        consensus_signal = "STRONG_SELL"

    # Alignment
    n_bull = sum(1 for tf in timeframes if tf.signal == "BULLISH")
    n_bear = sum(1 for tf in timeframes if tf.signal == "BEARISH")
    n_neut = sum(1 for tf in timeframes if tf.signal == "NEUTRAL")

    if n_bull == 3:
        alignment = "aligned_bull"
    elif n_bear == 3:
        alignment = "aligned_bear"
    elif n_neut == 3:
        alignment = "neutral"
    else:
        alignment = "mixed"

    return MultiTimeframeResult(
        ticker=ticker,
        timeframes=timeframes,
        consensus_score=float(consensus),
        consensus_signal=consensus_signal,
        alignment=alignment,
        n_bullish=n_bull,
        n_bearish=n_bear,
        n_neutral=n_neut,
    )
