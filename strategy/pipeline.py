"""
Per-symbol Level 1 pipeline (spec Section 47's research flow, as a live
state machine).

evaluate_on_close() is called once per newly CLOSED candle (never on a tick
-- Level 1 makes decisions on closed-bar boundaries, matching
STRATEGY_SPECIFICATION.md throughout). It is a pure orchestration layer:
every actual computation lives in features/, regime/, reversal/, and
strategy/decision_engine.py; this module only wires them together in the
right order and tracks the one piece of state those modules can't hold
themselves -- the pending setup awaiting confirmation.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from features.engine import FeatureSnapshot, build_features
from regime.detector import compute_regime
from reversal.confirmation import PendingSetup, check_confirmation
from reversal.divergence import compute_divergence
from reversal.exhaustion import compute_exhaustion
from reversal.price_action import compute_price_action
from reversal.scoring import ReversalWeights, score_reversal
from reversal.stretch import compute_stretch
from reversal.support_resistance import sr_bearish_evidence, sr_bullish_evidence
from strategy.decision_engine import Decision, evaluate_signal


class SymbolPipeline:
    def __init__(self, symbol: str, cfg: dict):
        self.symbol = symbol
        self.cfg = cfg
        self.weights = ReversalWeights(**cfg["reversal"]["weights"])
        self.weights.validate()
        self._pending_setup: PendingSetup | None = None
        self.last_features: FeatureSnapshot | None = None
        self.last_decision: Decision | None = None

    def evaluate_on_close(self, candles: list) -> Decision:
        """`candles` is the full CLOSED-only history for this symbol
        (data/candles.py's `.history()`). Returns a Decision; the caller is
        responsible for fetching a real proposal and calling
        strategy.decision_engine.apply_economics_and_risk before treating
        a TRADE_* result as executable.
        """
        cfg = self.cfg
        now = time.time()
        # Bounded window (see config/settings.yaml's max_lookback_bars):
        # nothing downstream needs more history than its largest lookback,
        # and handing build_features the full ever-growing history made
        # every closed-candle evaluation O(n) in total candles seen -- an
        # O(n^2) backtest. `candles` itself (the caller's full history) is
        # untouched; only the slice passed downstream is bounded.
        max_lookback = cfg["candles"].get("max_lookback_bars", 400)
        window = candles[-max_lookback:] if len(candles) > max_lookback else candles
        feat = build_features(window, cfg)
        self.last_features = feat

        if feat is None or not feat.sufficient_data:
            d = evaluate_signal(
                symbol=self.symbol, timestamp=now, sufficient_data=False,
                regime="UNKNOWN", volatility_regime="UNKNOWN",
                bullish_score=0.0, bearish_score=0.0,
                setup_threshold=cfg["reversal"]["setup_threshold"],
                regime_multiplier=1.0,
                disable_on_high_vol=cfg["regime"]["disable_on_high_vol"],
                confirmed_direction=None, confirmation_reason="")
            self.last_decision = d
            return d

        regime_snap = compute_regime(
            feat.high, feat.low, feat.close_series,
            ema_slope_value=feat.momentum.ema_slope,
            adx_period=cfg["regime"]["adx_period"],
            trend_slope_min=cfg["regime"]["trend_slope_min"],
            adx_trend_min=cfg["regime"]["adx_trend_min"],
            adx_range_max=cfg["regime"]["adx_range_max"],
            min_bars=cfg["regime"]["min_bars"])

        stretch = compute_stretch(
            feat.close_series, atr=feat.volatility.atr,
            lookback=cfg["stretch"]["lookback"],
            ema_fast_period=cfg["stretch"]["ema_fast_period"],
            bollinger_k=cfg["stretch"]["bollinger_k"],
            percentile_lookback=cfg["stretch"]["percentile_lookback"],
            z_cap=cfg["stretch"]["z_cap"], atr_cap=cfg["stretch"]["atr_cap"])
        exhaustion = compute_exhaustion(
            feat.momentum, rsi_oversold=cfg["momentum"]["rsi_oversold"],
            rsi_overbought=cfg["momentum"]["rsi_overbought"])
        pa = compute_price_action(
            feat.open_series, feat.high, feat.low, feat.close_series,
            swing_window=cfg["structure"]["swing_window"],
            rejection_wick_ratio=cfg["price_action"]["rejection_wick_ratio"],
            rejection_range_min=cfg["price_action"]["rejection_range_min"],
            failure_lookback=cfg["price_action"]["failure_lookback"])
        srb = sr_bullish_evidence(
            feat.structure, proximity_atr=cfg["structure"]["sr_proximity_atr"],
            min_strength=cfg["structure"]["sr_min_strength"])
        srs = sr_bearish_evidence(
            feat.structure, proximity_atr=cfg["structure"]["sr_proximity_atr"],
            min_strength=cfg["structure"]["sr_min_strength"])
        div = compute_divergence(
            feat.high, feat.low, feat.rsi_series, feat.macd_hist_series,
            swing_window=cfg["structure"]["swing_window"])

        score = score_reversal(stretch=stretch, exhaustion=exhaustion,
                               price_action=pa, sr_bullish=srb, sr_bearish=srs,
                               divergence=div, weights=self.weights)

        multiplier = cfg["regime"]["threshold_multiplier"].get(regime_snap.regime, 1.0)
        threshold = cfg["reversal"]["setup_threshold"]
        bar_index = feat.candle_index   # always len(window)-1: last position, for array access only

        # Update or create the pending setup for whichever direction (if
        # any) newly qualifies. Only one setup is tracked at a time --
        # Section 17 describes setup -> confirmation as a single-threaded
        # progression, and the conflicting-signal case is a hard NO_TRADE
        # in decision_engine.py, not something this layer needs to arbitrate.
        bull_qualifies = score.bullish >= threshold * multiplier
        bear_qualifies = score.bearish >= threshold * multiplier

        if self._pending_setup is None and bull_qualifies and not bear_qualifies:
            self._pending_setup = PendingSetup(
                direction="bullish", setup_high=candles[-1].high,
                setup_low=candles[-1].low, score_at_setup=score.bullish)
        elif self._pending_setup is None and bear_qualifies and not bull_qualifies:
            self._pending_setup = PendingSetup(
                direction="bearish", setup_high=candles[-1].high,
                setup_low=candles[-1].low, score_at_setup=score.bearish)
        elif self._pending_setup is not None:
            # A setup already exists and this bar did not just create it --
            # age it by exactly one closed candle. bars_waited (not array
            # index subtraction) is the source of truth for age; see
            # reversal/confirmation.py's module docstring for why.
            self._pending_setup.bars_waited += 1

        confirmed_direction = None
        confirmation_reason = ""
        if self._pending_setup is not None:
            result = check_confirmation(
                self._pending_setup, age=self._pending_setup.bars_waited,
                bar_index=bar_index,
                high=feat.high, low=feat.low, close=feat.close_series,
                ema_fast=feat.ema_fast_series, macd_hist=feat.macd_hist_series,
                confirmation_window=cfg["reversal"]["confirmation_window"])
            confirmation_reason = result.reason
            if result.confirmed:
                confirmed_direction = self._pending_setup.direction
                self._pending_setup = None    # consumed
            elif "discarded" in result.reason:
                self._pending_setup = None    # expired, drop it

        decision = evaluate_signal(
            symbol=self.symbol, timestamp=now, sufficient_data=True,
            regime=regime_snap.regime, volatility_regime=feat.volatility.volatility_regime,
            bullish_score=score.bullish, bearish_score=score.bearish,
            setup_threshold=threshold, regime_multiplier=multiplier,
            disable_on_high_vol=cfg["regime"]["disable_on_high_vol"],
            confirmed_direction=confirmed_direction,
            confirmation_reason=confirmation_reason)
        self.last_decision = decision
        return decision
