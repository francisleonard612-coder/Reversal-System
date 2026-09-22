"""
Price action reversal features (spec Section 12, STRATEGY_SPECIFICATION.md
Section 5). No single pattern is sufficient alone -- this module only
detects; the scoring blend in reversal/scoring.py is what enforces that no
one pattern can carry a decision by itself.
"""
from __future__ import annotations

from dataclasses import dataclass

from features.structure import SwingPoint, find_swings


@dataclass(frozen=True)
class PriceActionSnapshot:
    bullish_rejection: bool
    bearish_rejection: bool
    bullish_engulfing: bool
    bearish_engulfing: bool
    failed_breakdown: bool     # broke a swing low then reclaimed it -- bullish
    failed_breakout: bool      # broke a swing high then lost it -- bearish
    higher_low: bool
    lower_high: bool

    def bullish_count(self) -> int:
        return sum((self.bullish_rejection, self.bullish_engulfing,
                   self.failed_breakdown, self.higher_low))

    def bearish_count(self) -> int:
        return sum((self.bearish_rejection, self.bearish_engulfing,
                   self.failed_breakout, self.lower_high))


_PATTERN_TYPES = 4   # bullish_count()/bearish_count() check exactly this many


def _rejection(open_: float, high: float, low: float, close: float,
              *, wick_ratio: float, range_min: float) -> tuple[bool, bool]:
    body = abs(close - open_)
    rng = high - low
    if rng <= 0:
        return False, False
    upper_wick = high - max(open_, close)
    lower_wick = min(open_, close) - low
    bullish = (lower_wick > wick_ratio * max(body, 1e-9)
              and lower_wick > range_min * rng)
    bearish = (upper_wick > wick_ratio * max(body, 1e-9)
              and upper_wick > range_min * rng)
    return bullish, bearish


def _engulfing(open_prev: float, close_prev: float, open_: float, close: float
              ) -> tuple[bool, bool]:
    body_prev = abs(close_prev - open_prev)
    body = abs(close - open_)
    prior_bearish = close_prev < open_prev
    prior_bullish = close_prev > open_prev
    bullish = (prior_bearish and close > open_prev and open_ < close_prev
              and body > body_prev)
    bearish = (prior_bullish and close < open_prev and open_ > close_prev
              and body > body_prev)
    return bullish, bearish


def _failed_break(high: list[float], low: list[float], close: list[float],
                  swings: list[SwingPoint], *, lookback: int
                  ) -> tuple[bool, bool]:
    n = len(close)
    recent_highs = [s for s in swings if s.kind == "high" and s.index < n - 1]
    recent_lows = [s for s in swings if s.kind == "low" and s.index < n - 1]
    swing_high = max((s.price for s in recent_highs), default=None)
    swing_low = min((s.price for s in recent_lows), default=None)

    window = slice(max(0, n - 1 - lookback), n - 1)
    failed_breakout = bool(
        swing_high is not None and any(h > swing_high for h in high[window])
        and close[-1] < swing_high)
    failed_breakdown = bool(
        swing_low is not None and any(l < swing_low for l in low[window])
        and close[-1] > swing_low)
    return failed_breakdown, failed_breakout


def compute_price_action(open_: list[float], high: list[float], low: list[float],
                         close: list[float], *, swing_window: int = 3,
                         rejection_wick_ratio: float = 2.0,
                         rejection_range_min: float = 0.5,
                         failure_lookback: int = 5) -> PriceActionSnapshot:
    n = len(close)
    if n < 2:
        return PriceActionSnapshot(False, False, False, False, False, False, False, False)

    bull_rej, bear_rej = _rejection(open_[-1], high[-1], low[-1], close[-1],
                                    wick_ratio=rejection_wick_ratio,
                                    range_min=rejection_range_min)
    bull_eng, bear_eng = _engulfing(open_[-2], close[-2], open_[-1], close[-1])

    swings = find_swings(high, low, swing_window=swing_window)
    failed_breakdown, failed_breakout = _failed_break(
        high, low, close, swings, lookback=failure_lookback)

    lows = [s for s in swings if s.kind == "low"]
    highs = [s for s in swings if s.kind == "high"]
    higher_low = len(lows) >= 2 and lows[-1].price > lows[-2].price
    lower_high = len(highs) >= 2 and highs[-1].price < highs[-2].price

    return PriceActionSnapshot(
        bullish_rejection=bull_rej, bearish_rejection=bear_rej,
        bullish_engulfing=bull_eng, bearish_engulfing=bear_eng,
        failed_breakdown=failed_breakdown, failed_breakout=failed_breakout,
        higher_low=higher_low, lower_high=lower_high)
