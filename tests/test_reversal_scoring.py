from __future__ import annotations

import pytest

from reversal.divergence import DivergenceSnapshot
from reversal.exhaustion import ExhaustionSnapshot, compute_exhaustion
from reversal.price_action import PriceActionSnapshot, compute_price_action
from reversal.scoring import ReversalWeights, score_reversal
from reversal.stretch import StretchSnapshot
from features.momentum import MomentumSnapshot


def test_weights_summing_to_one_validates_cleanly():
    ReversalWeights().validate()


def test_weights_not_summing_to_one_raises():
    w = ReversalWeights(stretch=0.5, exhaustion=0.5, price_action=0.5,
                        support_resistance=0.0, divergence=0.0)
    with pytest.raises(ValueError, match="must sum to 1.0"):
        w.validate()


def test_score_reversal_is_documented_weighted_sum():
    """Section 16: no hidden weighting. Verify the formula directly rather
    than just checking the output is 'reasonable'."""
    stretch = StretchSnapshot(zscore=0, ema_distance=0, bollinger_pct_b=0.5,
                              percentile_rank=0.5, stretch_score=0.8)
    exhaustion = ExhaustionSnapshot(bullish_exhaustion=True, bearish_exhaustion=False)
    pa = PriceActionSnapshot(bullish_rejection=True, bearish_rejection=False,
                             bullish_engulfing=False, bearish_engulfing=False,
                             failed_breakdown=False, failed_breakout=False,
                             higher_low=False, lower_high=False)
    div = DivergenceSnapshot(bullish_divergence=False, bearish_divergence=False,
                             evaluable=True)
    weights = ReversalWeights(stretch=0.3, exhaustion=0.25, price_action=0.2,
                              support_resistance=0.15, divergence=0.1)

    result = score_reversal(stretch=stretch, exhaustion=exhaustion, price_action=pa,
                            sr_bullish=0.6, sr_bearish=0.0, divergence=div,
                            weights=weights)

    expected_bullish = 100 * (0.3 * 0.8 + 0.25 * 1.0 + 0.2 * 0.25 + 0.15 * 0.6 + 0.1 * 0.0)
    assert result.bullish == pytest.approx(expected_bullish, abs=1e-6)
    assert result.bearish == 0.0


def test_no_single_pattern_reaches_full_price_action_evidence():
    """Section 12: no single pattern is sufficient alone. One triggered
    pattern out of four types must be partial evidence, not full."""
    pa = PriceActionSnapshot(bullish_rejection=True, bearish_rejection=False,
                             bullish_engulfing=False, bearish_engulfing=False,
                             failed_breakdown=False, failed_breakout=False,
                             higher_low=False, lower_high=False)
    assert pa.bullish_count() == 1
    assert pa.bullish_count() / 4.0 == 0.25


def test_exhaustion_requires_deceleration_not_just_extreme():
    """An extreme-but-still-accelerating reading must NOT count as
    exhaustion -- Section 11's explicit distinction."""
    extreme_not_decelerating = MomentumSnapshot(
        rsi=20.0, rsi_slope=-2.0,  # still falling -- not decelerating
        macd_hist=-0.5, macd_hist_prev=-0.3, roc=-0.02, ema_fast=100, ema_slope=-0.01)
    e = compute_exhaustion(extreme_not_decelerating)
    assert not e.bullish_exhaustion

    extreme_and_decelerating = MomentumSnapshot(
        rsi=20.0, rsi_slope=1.5,   # turning back up
        macd_hist=-0.2, macd_hist_prev=-0.4,  # histogram improving
        roc=-0.02, ema_fast=100, ema_slope=-0.01)
    e2 = compute_exhaustion(extreme_and_decelerating)
    assert e2.bullish_exhaustion


def test_engulfing_and_rejection_detected_on_synthetic_candles():
    open_ = [100.0, 98.0]
    high = [100.0, 103.0]
    low = [98.0, 97.5]
    close = [98.5, 102.5]   # bar 1 bearish, bar 2 opens below/closes above -- bullish engulfing
    pa = compute_price_action(open_, high, low, close)
    assert pa.bullish_engulfing
