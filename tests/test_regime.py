from __future__ import annotations

import random

from regime.detector import compute_adx, compute_regime


def _series(n, drift, noise=0.1, seed=1):
    random.seed(seed)
    close = [100.0]
    for _ in range(n):
        close.append(close[-1] + drift + random.gauss(0, noise))
    high = [c + noise for c in close]
    low = [c - noise for c in close]
    return high, low, close


def test_regime_unknown_with_insufficient_bars():
    high, low, close = _series(20, 0.1)
    r = compute_regime(high, low, close, ema_slope_value=0.01, min_bars=50)
    assert r.regime == "UNKNOWN"


def test_regime_trend_up_on_a_strong_uptrend():
    high, low, close = _series(200, 0.3, noise=0.05)
    r = compute_regime(high, low, close, ema_slope_value=0.01, min_bars=50,
                       trend_slope_min=0.001, adx_trend_min=25)
    assert r.regime == "TREND_UP"
    assert r.adx > 0


def test_regime_trend_down_on_a_strong_downtrend():
    high, low, close = _series(200, -0.3, noise=0.05)
    r = compute_regime(high, low, close, ema_slope_value=-0.01, min_bars=50,
                       trend_slope_min=0.001, adx_trend_min=25)
    assert r.regime == "TREND_DOWN"


def test_regime_range_on_a_directionless_series():
    high, low, close = _series(200, 0.0, noise=0.3)
    r = compute_regime(high, low, close, ema_slope_value=0.0, min_bars=50,
                       adx_range_max=20)
    assert r.regime in ("RANGE", "TRANSITION")   # noise-dependent, never a false TREND


def test_adx_zero_with_too_little_history():
    adx, plus_di, minus_di = compute_adx([1, 2], [1, 2], [1, 2], period=14)
    assert adx == 0.0
