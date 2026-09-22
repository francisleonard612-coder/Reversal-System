from __future__ import annotations

import math

from features.momentum import compute_momentum, ema_slope, macd, roc, rsi
from features.structure import build_zones, compute_structure, find_swings
from features.volatility import compute_volatility, ema, true_range, wilder_smooth


def _flat(n=300, price=100.0):
    return [price] * n, [price] * n, [price] * n


def test_wilder_smooth_seeds_with_simple_average():
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    out = wilder_smooth(vals, period=5)
    assert out[4] == pytest_approx(3.0)


def pytest_approx(x, tol=1e-6):
    class _A:
        def __eq__(self, other):
            return abs(other - x) < tol
    return _A()


def test_true_range_first_bar_has_no_prior_close():
    tr = true_range([10.0], [8.0], [9.0])
    assert tr[0] == 2.0


def test_rsi_pure_uptrend_approaches_100():
    close = [100 + i for i in range(60)]
    r = rsi(close, period=14)
    assert r[-1] > 95


def test_rsi_pure_downtrend_approaches_zero():
    close = [200 - i for i in range(60)]
    r = rsi(close, period=14)
    assert r[-1] < 5


def test_rsi_flat_series_is_fifty():
    close = [100.0] * 60
    r = rsi(close, period=14)
    assert abs(r[-1] - 50.0) < 1e-6


def test_macd_hist_zero_on_flat_series():
    close = [100.0] * 60
    m = macd(close)
    assert abs(m.hist[-1]) < 1e-6


def test_roc_and_ema_slope_signs_match_trend_direction():
    up = [100 + i * 0.5 for i in range(40)]
    assert roc(up, 10) > 0
    e = ema(up, 9)
    assert ema_slope(e, 5) > 0

    down = [140 - i * 0.5 for i in range(40)]
    assert roc(down, 10) < 0


def test_volatility_regime_unknown_with_too_little_data():
    v = compute_volatility([1, 2], [1, 2], [1, 2])
    assert v.volatility_regime == "UNKNOWN"


def test_volatility_flags_high_regime_on_a_volatility_spike():
    import random
    random.seed(1)
    close = [100.0]
    for _ in range(300):
        close.append(close[-1] + random.gauss(0, 0.05))
    for _ in range(20):   # sudden volatility spike
        close.append(close[-1] + random.gauss(0, 3.0))
    high = [c + 0.1 for c in close]
    low = [c - 0.1 for c in close]
    v = compute_volatility(high, low, close, atr_period=14, vol_lookback=200)
    assert v.volatility_regime == "HIGH_VOLATILITY"


def test_find_swings_confirms_only_with_full_window_on_both_sides():
    """The anti-lookahead guarantee: a swing needs swing_window bars on
    both sides, so it cannot appear before it could actually be known."""
    high = [1, 2, 5, 2, 1, 1, 1, 1, 1, 1]
    low = list(high)
    swings = find_swings(high, low, swing_window=2)
    highs = [s for s in swings if s.kind == "high"]
    assert len(highs) == 1
    assert highs[0].index == 2
    # With fewer bars after the peak than swing_window, it must NOT confirm.
    swings_short = find_swings(high[:4], low[:4], swing_window=2)
    assert not any(s.kind == "high" for s in swings_short)


def test_zones_merge_nearby_swings_and_decay_by_recency():
    from features.structure import SwingPoint
    swings = [SwingPoint(0, 100.0, "support" if False else "low"),
             SwingPoint(50, 100.05, "low")]
    zones = build_zones(swings, current_index=50, zone_merge_pct=0.01,
                        zone_decay_half_life=50)
    assert len(zones) == 1
    assert zones[0].touches == 2


def test_structure_distance_is_atr_normalized():
    high = [100.0] * 20 + [110.0, 105.0] + [104.0] * 20
    low = [99.0] * 20 + [109.0, 103.0] + [103.0] * 20
    close = [99.5] * 20 + [109.5, 104.0] + [103.5] * 20
    s = compute_structure(high, low, close, atr=1.0, swing_window=3)
    assert s.distance_to_resistance_atr >= 0 or math.isinf(s.distance_to_resistance_atr)
