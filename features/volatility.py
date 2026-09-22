"""
Volatility features (spec Section 3, STRATEGY_SPECIFICATION.md Section 3).

Wilder smoothing throughout for ATR (and reused by RSI/ADX in momentum.py /
regime/detector.py) rather than a plain EMA, because Wilder's recursive form
is what the spec's formulas actually name and it's what most charting
platforms (including Pine's built-ins) compute -- using a different
smoothing here would make the Python and Pine implementations diverge on
values even when the code "matches" structurally.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


def true_range(high: list[float], low: list[float], close: list[float]) -> list[float]:
    """close[i-1] undefined for i=0 -- first TR is just high[0]-low[0]."""
    n = len(high)
    tr = [0.0] * n
    if n == 0:
        return tr
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                   abs(high[i] - close[i - 1]),
                   abs(low[i] - close[i - 1]))
    return tr


def wilder_smooth(values: list[float], period: int) -> list[float]:
    """Wilder's recursive smoothing: seed with a simple average of the first
    `period` values, then smooth forward. Returns a list the same length as
    `values`, with the first `period-1` entries equal to the seed (there is
    no earlier data to smooth from) -- callers needing "not yet available"
    semantics should check length against `period`, not read these as
    meaningful.
    """
    n = len(values)
    if n == 0:
        return []
    if n < period:
        seed = sum(values) / n
        return [seed] * n
    out = [0.0] * n
    seed = sum(values[:period]) / period
    for i in range(period):
        out[i] = seed
    prev = seed
    for i in range(period, n):
        prev = prev + (values[i] - prev) / period
        out[i] = prev
    return out


@dataclass(frozen=True)
class VolatilitySnapshot:
    atr: float
    atr_pct: float
    volatility_regime: str    # HIGH_VOLATILITY | LOW_VOLATILITY | NORMAL | UNKNOWN


def _percentile_rank(values: list[float], x: float) -> float:
    if not values:
        return 0.5
    below = sum(1 for v in values if v < x)
    return below / len(values)


def compute_volatility(high: list[float], low: list[float], close: list[float],
                       *, atr_period: int = 14, vol_lookback: int = 200,
                       vol_high_percentile: float = 80.0,
                       vol_low_percentile: float = 20.0) -> VolatilitySnapshot:
    if len(close) < atr_period + 1:
        return VolatilitySnapshot(atr=0.0, atr_pct=0.0, volatility_regime="UNKNOWN")

    tr = true_range(high, low, close)
    atr_series = wilder_smooth(tr, atr_period)
    atr = atr_series[-1]
    price = close[-1] if close[-1] != 0 else 1e-9
    atr_pct = atr / price

    window = atr_series[-vol_lookback:] if len(atr_series) >= vol_lookback else atr_series
    price_window = close[-len(window):]
    atr_pct_series = [a / p if p != 0 else 0.0 for a, p in zip(window, price_window)]

    if len(atr_pct_series) < 20:
        regime = "UNKNOWN"
    else:
        rank = _percentile_rank(atr_pct_series, atr_pct) * 100
        if rank >= vol_high_percentile:
            regime = "HIGH_VOLATILITY"
        elif rank <= vol_low_percentile:
            regime = "LOW_VOLATILITY"
        else:
            regime = "NORMAL"

    return VolatilitySnapshot(atr=atr, atr_pct=atr_pct, volatility_regime=regime)


def ema(values: list[float], period: int) -> list[float]:
    """Standard EMA (NOT Wilder) -- used for the fast EMA, MACD, and stretch
    features, matching how those are conventionally defined and how Pine's
    ta.ema() computes them."""
    n = len(values)
    if n == 0:
        return []
    k = 2.0 / (period + 1)
    out = [values[0]]
    for i in range(1, n):
        out.append(values[i] * k + out[-1] * (1 - k))
    return out


def stdev(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    m = sum(values) / n
    return math.sqrt(sum((v - m) ** 2 for v in values) / (n - 1))
