"""
Momentum features (spec Section 11, STRATEGY_SPECIFICATION.md Section 4).
"""
from __future__ import annotations

from dataclasses import dataclass

from features.volatility import ema, wilder_smooth


def rsi(close: list[float], period: int = 14) -> list[float]:
    """Wilder's RSI. Returns a series the same length as `close`; entries
    before `period` are computed from a padded seed (see wilder_smooth) and
    should not be treated as meaningful -- callers check len(close) >= period
    before trusting rsi[-1]."""
    n = len(close)
    if n < 2:
        return [50.0] * n
    gains = [max(close[i] - close[i - 1], 0.0) for i in range(1, n)]
    losses = [max(close[i - 1] - close[i], 0.0) for i in range(1, n)]
    avg_gain = wilder_smooth(gains, period)
    avg_loss = wilder_smooth(losses, period)
    out = [50.0]  # no prior bar for close[0]
    for g, l in zip(avg_gain, avg_loss):
        if l == 0:
            out.append(100.0 if g > 0 else 50.0)
        else:
            rs = g / l
            out.append(100.0 - 100.0 / (1.0 + rs))
    return out


@dataclass(frozen=True)
class MacdSnapshot:
    macd: list[float]
    signal: list[float]
    hist: list[float]


def macd(close: list[float], *, fast: int = 12, slow: int = 26,
         signal: int = 9) -> MacdSnapshot:
    if len(close) < slow:
        return MacdSnapshot(macd=[0.0] * len(close), signal=[0.0] * len(close),
                            hist=[0.0] * len(close))
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = [f - s for f, s in zip(ema_fast, ema_slow)]
    signal_line = ema(macd_line, signal)
    hist = [m - s for m, s in zip(macd_line, signal_line)]
    return MacdSnapshot(macd=macd_line, signal=signal_line, hist=hist)


def roc(close: list[float], period: int = 10) -> float:
    if len(close) <= period or close[-period - 1] == 0:
        return 0.0
    return (close[-1] - close[-period - 1]) / close[-period - 1]


def ema_slope(ema_series: list[float], lookback: int = 5) -> float:
    if len(ema_series) <= lookback or ema_series[-lookback - 1] == 0:
        return 0.0
    return (ema_series[-1] - ema_series[-lookback - 1]) / ema_series[-lookback - 1]


@dataclass(frozen=True)
class MomentumSnapshot:
    rsi: float
    rsi_slope: float
    macd_hist: float
    macd_hist_prev: float
    roc: float
    ema_fast: float
    ema_slope: float


def compute_momentum(close: list[float], *, rsi_period: int = 14,
                     rsi_slope_lookback: int = 3, macd_fast: int = 12,
                     macd_slow: int = 26, macd_signal: int = 9,
                     roc_period: int = 10, ema_fast_period: int = 9,
                     ema_slope_lookback: int = 5) -> MomentumSnapshot:
    rsi_series = rsi(close, rsi_period)
    m = macd(close, fast=macd_fast, slow=macd_slow, signal=macd_signal)
    ema_fast_series = ema(close, ema_fast_period)

    r_slope = (rsi_series[-1] - rsi_series[-rsi_slope_lookback - 1]
              if len(rsi_series) > rsi_slope_lookback else 0.0)
    hist_prev = m.hist[-2] if len(m.hist) >= 2 else m.hist[-1] if m.hist else 0.0

    return MomentumSnapshot(
        rsi=rsi_series[-1] if rsi_series else 50.0,
        rsi_slope=r_slope,
        macd_hist=m.hist[-1] if m.hist else 0.0,
        macd_hist_prev=hist_prev,
        roc=roc(close, roc_period),
        ema_fast=ema_fast_series[-1] if ema_fast_series else (close[-1] if close else 0.0),
        ema_slope=ema_slope(ema_fast_series, ema_slope_lookback),
    )
