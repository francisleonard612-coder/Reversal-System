"""
Regime engine (spec Section 15, STRATEGY_SPECIFICATION.md Section 8).

Wilder's ADX for trend strength, layered with the volatility regime from
features/volatility.py (reported separately -- a market can be TREND_UP and
HIGH_VOLATILITY at once; cramming both into one enum would lose that).

The regime's effect on the decision is a THRESHOLD MULTIPLIER, not a
trade/no-trade switch by itself (Section 15's explicit requirement): a
reversal against a strong trend needs a higher score than one inside a
range, but a strong enough score can still clear a trending regime's higher
bar. UNKNOWN is the one exception -- insufficient history to classify at
all is a hard gate, not a soft multiplier, since there's no basis to price
the trade's risk.
"""
from __future__ import annotations

from dataclasses import dataclass

from features.volatility import true_range, wilder_smooth


@dataclass(frozen=True)
class RegimeSnapshot:
    regime: str               # TREND_UP | TREND_DOWN | RANGE | TRANSITION | UNKNOWN
    adx: float
    plus_di: float
    minus_di: float


def _directional_movement(high: list[float], low: list[float]) -> tuple[list[float], list[float]]:
    n = len(high)
    plus_dm = [0.0] * n
    minus_dm = [0.0] * n
    for i in range(1, n):
        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]
        plus_dm[i] = up_move if (up_move > down_move and up_move > 0) else 0.0
        minus_dm[i] = down_move if (down_move > up_move and down_move > 0) else 0.0
    return plus_dm, minus_dm


def compute_adx(high: list[float], low: list[float], close: list[float],
                *, period: int = 14) -> tuple[float, float, float]:
    n = len(close)
    if n < period * 2:
        return 0.0, 0.0, 0.0
    plus_dm, minus_dm = _directional_movement(high, low)
    tr = true_range(high, low, close)
    atr = wilder_smooth(tr, period)
    plus_di_series = wilder_smooth(plus_dm, period)
    minus_di_series = wilder_smooth(minus_dm, period)

    plus_di = [100 * p / a if a > 0 else 0.0 for p, a in zip(plus_di_series, atr)]
    minus_di = [100 * m / a if a > 0 else 0.0 for m, a in zip(minus_di_series, atr)]
    dx = [100 * abs(p - m) / (p + m) if (p + m) > 0 else 0.0
          for p, m in zip(plus_di, minus_di)]
    adx_series = wilder_smooth(dx, period)
    return adx_series[-1], plus_di[-1], minus_di[-1]


def compute_regime(high: list[float], low: list[float], close: list[float],
                   *, ema_slope_value: float, adx_period: int = 14,
                   trend_slope_min: float = 0.001, adx_trend_min: float = 25.0,
                   adx_range_max: float = 20.0, min_bars: int = 50
                   ) -> RegimeSnapshot:
    if len(close) < min_bars:
        return RegimeSnapshot("UNKNOWN", 0.0, 0.0, 0.0)

    adx, plus_di, minus_di = compute_adx(high, low, close, period=adx_period)

    if adx >= adx_trend_min and ema_slope_value > trend_slope_min:
        regime = "TREND_UP"
    elif adx >= adx_trend_min and ema_slope_value < -trend_slope_min:
        regime = "TREND_DOWN"
    elif adx <= adx_range_max:
        regime = "RANGE"
    else:
        regime = "TRANSITION"

    return RegimeSnapshot(regime=regime, adx=adx, plus_di=plus_di, minus_di=minus_di)
