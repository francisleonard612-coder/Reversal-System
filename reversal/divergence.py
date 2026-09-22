"""
Divergence (spec Section 14, STRATEGY_SPECIFICATION.md Section 7).

Optional evidence: contributes 0 (not a penalty) when fewer than two
confirmed swing points of the relevant kind exist to compare, per Section
14's "divergence must not be mandatory if data quality or structure does
not support it."
"""
from __future__ import annotations

from dataclasses import dataclass

from features.structure import SwingPoint, find_swings


@dataclass(frozen=True)
class DivergenceSnapshot:
    bullish_divergence: bool
    bearish_divergence: bool
    evaluable: bool     # False if too few swings existed to compare at all


def compute_divergence(high: list[float], low: list[float],
                       rsi_series: list[float], macd_hist_series: list[float],
                       *, swing_window: int = 3) -> DivergenceSnapshot:
    swings = find_swings(high, low, swing_window=swing_window)
    lows = [s for s in swings if s.kind == "low"]
    highs = [s for s in swings if s.kind == "high"]

    bullish = False
    if len(lows) >= 2:
        a, b = lows[-2], lows[-1]
        if b.index < len(rsi_series) and a.index < len(rsi_series):
            price_lower_low = b.price < a.price
            rsi_higher_low = rsi_series[b.index] > rsi_series[a.index]
            macd_higher_low = (macd_hist_series[b.index] > macd_hist_series[a.index]
                               if b.index < len(macd_hist_series) and a.index < len(macd_hist_series)
                               else False)
            bullish = price_lower_low and (rsi_higher_low or macd_higher_low)

    bearish = False
    if len(highs) >= 2:
        a, b = highs[-2], highs[-1]
        if b.index < len(rsi_series) and a.index < len(rsi_series):
            price_higher_high = b.price > a.price
            rsi_lower_high = rsi_series[b.index] < rsi_series[a.index]
            macd_lower_high = (macd_hist_series[b.index] < macd_hist_series[a.index]
                               if b.index < len(macd_hist_series) and a.index < len(macd_hist_series)
                               else False)
            bearish = price_higher_high and (rsi_lower_high or macd_lower_high)

    evaluable = len(lows) >= 2 or len(highs) >= 2
    return DivergenceSnapshot(bullish_divergence=bullish, bearish_divergence=bearish,
                              evaluable=evaluable)
