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

from features.engine import FeatureSnapshot
from models.dataset import BarEvidence, compute_evidence, features_for
from reversal.confirmation import PendingSetup, check_confirmation
from reversal.scoring import ReversalWeights
from strategy.decision_engine import Decision, evaluate_signal


class SymbolPipeline:
    def __init__(self, symbol: str, cfg: dict):
        self.symbol = symbol
        self.cfg = cfg
        self.weights = ReversalWeights(**cfg["reversal"]["weights"])
        self.weights.validate()
        self._pending_setup: PendingSetup | None = None
        self.last_features: FeatureSnapshot | None = None
        self.last_evidence: BarEvidence | None = None
        self.last_decision: Decision | None = None
        self.n_available = 0

    def features_for_direction(self, direction: str) -> dict | None:
        """Level 2 feature vector for a trade in `direction` on the bar just
        evaluated; None before enough history exists."""
        if self.last_evidence is None:
            return None
        return features_for(self.last_evidence, direction, self.n_available)

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
        # One shared computation with the Level 2 training replay
        # (models/dataset.compute_evidence) -- the features a model is served
        # live are, by construction, the ones it was trained on.
        ev = compute_evidence(window, cfg, self.weights)
        self.last_evidence = ev
        self.last_features = ev.feat if ev is not None else None
        self.n_available = len(candles)

        if ev is None:
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

        feat, regime_snap, score = ev.feat, ev.regime_snap, ev.score
        multiplier = ev.regime_multiplier
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
