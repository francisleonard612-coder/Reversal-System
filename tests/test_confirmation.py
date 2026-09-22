from __future__ import annotations

from reversal.confirmation import PendingSetup, check_confirmation


def _setup(direction="bullish"):
    return PendingSetup(direction=direction, setup_high=105.0, setup_low=100.0,
                        score_at_setup=60.0)


def test_setup_just_formed_awaits_confirmation():
    s = _setup()
    r = check_confirmation(s, age=0, bar_index=10, high=[105]*11, low=[100]*11,
                           close=[102]*11, ema_fast=[102]*11, macd_hist=[0]*11)
    assert not r.confirmed
    assert "just formed" in r.reason


def test_follow_through_beyond_setup_high_confirms_bullish():
    s = _setup()
    close = [102.0] * 11 + [106.0]   # bar 11 closes beyond setup_high (105)
    r = check_confirmation(s, age=1, bar_index=11, high=[105.0]*12, low=[100.0]*12,
                           close=close, ema_fast=[102.0]*12, macd_hist=[0.0]*12)
    assert r.confirmed
    assert "follow-through" in r.reason


def test_ema_reclaim_confirms_bullish():
    s = _setup()
    close = [102.0] * 11 + [103.0]
    ema_fast = [104.0] * 11 + [102.0]   # close now above EMA
    r = check_confirmation(s, age=1, bar_index=11, high=[105.0]*12, low=[100.0]*12,
                           close=close, ema_fast=ema_fast, macd_hist=[0.0]*12)
    assert r.confirmed
    assert "EMA" in r.reason


def test_macd_zero_cross_confirms_bullish():
    s = _setup()
    close = [102.0] * 12
    macd_hist = [0.0] * 10 + [-0.1, 0.1]   # crossed from negative to positive
    r = check_confirmation(s, age=1, bar_index=11, high=[105.0]*12, low=[100.0]*12,
                           close=close, ema_fast=[102.0]*12, macd_hist=macd_hist)
    assert r.confirmed
    assert "MACD" in r.reason


def test_setup_discarded_after_confirmation_window_expires():
    s = _setup()
    close = [102.0] * 10
    r = check_confirmation(s, age=5, bar_index=5, high=[105.0]*10, low=[100.0]*10,
                           close=close, ema_fast=[102.0]*10, macd_hist=[0.0]*10,
                           confirmation_window=3)
    assert not r.confirmed
    assert "expired" in r.reason
    assert "discarded" in r.reason


def test_no_confirmation_when_nothing_triggers():
    s = _setup()
    close = [102.0] * 3
    r = check_confirmation(s, age=1, bar_index=1, high=[105.0]*3, low=[100.0]*3,
                           close=close, ema_fast=[104.0]*3, macd_hist=[0.0]*3,
                           confirmation_window=3)
    assert not r.confirmed
    assert "awaiting" in r.reason


def test_age_is_independent_of_bar_index_across_a_sliding_window():
    """The bug this API shape exists to prevent: once the feature window
    slides, bar_index is constant (always len(window)-1) across calls, so
    age must come from an explicit counter, never from index subtraction."""
    s = _setup()
    close = [102.0] * 400
    r_early = check_confirmation(s, age=0, bar_index=399, high=[105.0]*400,
                                 low=[100.0]*400, close=close,
                                 ema_fast=[102.0]*400, macd_hist=[0.0]*400)
    r_late = check_confirmation(s, age=4, bar_index=399, high=[105.0]*400,
                                low=[100.0]*400, close=close,
                                ema_fast=[102.0]*400, macd_hist=[0.0]*400,
                                confirmation_window=3)
    assert "just formed" in r_early.reason
    assert "expired" in r_late.reason
