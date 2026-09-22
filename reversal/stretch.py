"""Statistical stretch (STRATEGY_SPECIFICATION.md Section 2)."""
from __future__ import annotations

from dataclasses import dataclass

from features.volatility import ema, stdev


def clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


@dataclass(frozen=True)
class StretchSnapshot:
    zscore: float
    ema_distance: float       # ATR-normalized
    bollinger_pct_b: float
    percentile_rank: float
    stretch_score: float      # signed: + = oversold (bullish evidence), - = overbought


def _percentile_rank(values: list[float], x: float) -> float:
    if not values:
        return 0.5
    below = sum(1 for v in values if v < x)
    return below / len(values)


def compute_stretch(close: list[float], *, atr: float, lookback: int = 20,
                    ema_fast_period: int = 9, bollinger_k: float = 2.0,
                    percentile_lookback: int = 100, z_cap: float = 3.0,
                    atr_cap: float = 2.5) -> StretchSnapshot:
    if len(close) < lookback + 1:
        return StretchSnapshot(0.0, 0.0, 0.5, 0.5, 0.0)

    window = close[-lookback:]
    sma = sum(window) / len(window)
    sd = stdev(window)
    zscore = (close[-1] - sma) / sd if sd > 0 else 0.0

    ema_series = ema(close, ema_fast_period)
    ema_distance = (close[-1] - ema_series[-1]) / atr if atr > 0 else 0.0

    upper = sma + bollinger_k * sd
    lower = sma - bollinger_k * sd
    band_width = upper - lower
    pct_b = (close[-1] - lower) / band_width if band_width > 0 else 0.5

    p_window = close[-percentile_lookback:] if len(close) >= percentile_lookback else close
    p_rank = _percentile_rank(p_window, close[-1])

    stretch_score = sum((
        clamp(-zscore / z_cap),
        clamp(-ema_distance / atr_cap),
        clamp((0.5 - pct_b) * 2),
        clamp((0.5 - p_rank) * 2),
    )) / 4.0

    return StretchSnapshot(zscore=zscore, ema_distance=ema_distance,
                           bollinger_pct_b=pct_b, percentile_rank=p_rank,
                           stretch_score=stretch_score)
