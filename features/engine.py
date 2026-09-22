"""
Feature engine (spec Section 8/47 pipeline step). Single entry point that
takes CLOSED candle history and produces every input the regime detector,
reversal scorer, and decision engine need -- computed exactly once per
closed candle, not recomputed piecemeal by each consumer.

Reads only candles already marked `is_closed` (data/candles.py) -- the
anti-repainting rule enforced at the one place all features originate.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from data.candles import Candle
from features.momentum import MomentumSnapshot, compute_momentum, macd, rsi
from features.structure import StructureSnapshot, compute_structure
from features.volatility import VolatilitySnapshot, compute_volatility, ema


@dataclass(frozen=True)
class FeatureSnapshot:
    symbol: str
    candle_index: int          # index of the candle this snapshot was computed FROM (the last closed one)
    close: float
    high: list[float] = field(repr=False)
    low: list[float] = field(repr=False)
    close_series: list[float] = field(repr=False)
    open_series: list[float] = field(repr=False)
    volatility: VolatilitySnapshot
    momentum: MomentumSnapshot
    structure: StructureSnapshot
    rsi_series: list[float] = field(repr=False)
    macd_hist_series: list[float] = field(repr=False)
    ema_fast_series: list[float] = field(repr=False)
    sufficient_data: bool


def build_features(candles: list[Candle], cfg: dict) -> FeatureSnapshot | None:
    """`candles` must already be CLOSED-only (data/candles.py guarantees
    this by construction). Returns None if there aren't even enough bars to
    build a well-formed snapshot -- callers treat None as
    NO_TRADE_MINIMUM_DATA, computing nothing further.
    """
    if not candles:
        return None
    symbol = candles[0].symbol
    high = [c.high for c in candles]
    low = [c.low for c in candles]
    close = [c.close for c in candles]
    open_ = [c.open for c in candles]

    min_bars = max(cfg["stretch"]["percentile_lookback"],
                   cfg["volatility"]["vol_lookback"],
                   cfg["regime"]["min_bars"]) + cfg["structure"]["swing_window"]
    sufficient = len(close) >= min_bars

    vol = compute_volatility(
        high, low, close, atr_period=cfg["volatility"]["atr_period"],
        vol_lookback=cfg["volatility"]["vol_lookback"],
        vol_high_percentile=cfg["volatility"]["vol_high_percentile"],
        vol_low_percentile=cfg["volatility"]["vol_low_percentile"])

    mom = compute_momentum(
        close, rsi_period=cfg["momentum"]["rsi_period"],
        rsi_slope_lookback=cfg["momentum"]["rsi_slope_lookback"],
        macd_fast=cfg["momentum"]["macd_fast"], macd_slow=cfg["momentum"]["macd_slow"],
        macd_signal=cfg["momentum"]["macd_signal"], roc_period=cfg["momentum"]["roc_period"],
        ema_fast_period=cfg["stretch"]["ema_fast_period"],
        ema_slope_lookback=cfg["momentum"]["ema_slope_lookback"])

    struct = compute_structure(
        high, low, close, atr=vol.atr, swing_window=cfg["structure"]["swing_window"],
        zone_merge_pct=cfg["structure"]["zone_merge_pct"],
        zone_decay_half_life=cfg["structure"]["zone_decay_half_life"],
        reclaim_lookback=cfg["structure"]["reclaim_lookback"])

    rsi_series = rsi(close, cfg["momentum"]["rsi_period"])
    macd_hist_series = macd(close, fast=cfg["momentum"]["macd_fast"],
                            slow=cfg["momentum"]["macd_slow"],
                            signal=cfg["momentum"]["macd_signal"]).hist
    ema_fast_series = ema(close, cfg["stretch"]["ema_fast_period"])

    return FeatureSnapshot(
        symbol=symbol, candle_index=len(candles) - 1, close=close[-1],
        high=high, low=low, close_series=close, open_series=open_,
        volatility=vol, momentum=mom, structure=struct,
        rsi_series=rsi_series, macd_hist_series=macd_hist_series,
        ema_fast_series=ema_fast_series, sufficient_data=sufficient)
