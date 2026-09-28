"""MC-calibrated expiry: horizons, live outcome tracking, calibration,
per-trade expiry pricing, and the percentile setup threshold."""
from __future__ import annotations

import asyncio
import time

import numpy as np
import pytest
import yaml

from data.database import Database
from execution.executor import execute_trade
from strategy.decision_engine import TRADE_BULLISH, Decision
from strategy.expiry import (ExpiryCalibrator, Horizon, HorizonEstimate, OutcomeTracker,
                             SetupDeduper, offered_horizons, parse_horizons,
                             replay_setup_outcomes)

H = parse_horizons(["3t", "5t", "1m", "2m"])


def test_parse_and_units():
    h = Horizon.parse("7t")
    assert (h.value, h.unit, h.is_ticks, h.seconds) == (7, "t", True, None)
    assert Horizon.parse("2m").seconds == 120
    with pytest.raises(ValueError):
        Horizon.parse("5x")
    with pytest.raises(ValueError):
        parse_horizons(["5t", "5t"])


def test_only_offered_expiries_kept():
    hs = parse_horizons(["3t", "15t", "10s", "1m", "90m"])
    assert [h.label for h in offered_horizons(hs, [("1t", "10t"), ("15s", "1h")])] == ["3t", "1m"]
    assert offered_horizons(hs, []) == hs


def test_tracker_emits_tick_and_time_outcomes():
    rows = []
    tr = OutcomeTracker(H, rows.append)
    tr.start(symbol="R", direction="bullish", confirmed=True, setup_epoch=100, score=60, regime="RANGE")
    prices = [10.0, 10.1, 10.2, 9.9, 9.8, 10.3]            # entry = first tick after setup
    for i, p in enumerate(prices):
        tr.on_tick("R", 101 + i, p)
    got = {r["horizon"]: r for r in rows}
    assert got["3t"]["exit_price"] == 9.9 and got["3t"]["won"] is False   # 3 ticks after 10.0
    assert got["5t"]["exit_price"] == 10.3 and got["5t"]["won"] is True
    for i in range(130):                                  # 1m: last tick at/before entry+60s
        tr.on_tick("R", 107 + i, 10.5 if i == 50 else 11.0)
    got = {r["horizon"]: r for r in rows}
    assert got["1m"]["exit_price"] == 11.0 and got["2m"]["won"] is True
    assert tr.pending["R"] == []                          # finished setups are dropped


def test_deduper_counts_one_setup_once():
    d = SetupDeduper(window_seconds=180)
    assert d.accept("R", "bullish", False, 0)
    assert not d.accept("R", "bullish", False, 60)        # same setup, still unconfirmed
    assert d.accept("R", "bullish", True, 120)            # confirmation always counts
    assert d.accept("R", "bearish", False, 130)
    assert d.accept("R", "bullish", False, 400)           # a new setup later


def _cal(rows, **kw):
    c = ExpiryCalibrator(H, mc_draws=4000, **{"min_samples": 100, **kw})
    c.load_counts(rows)
    return c


def test_calibrator_prefers_better_horizon_and_needs_samples():
    c = _cal([("R", "bullish", "1m", True, 70, 120), ("R", "bullish", "2m", True, 55, 120),
              ("R", "bullish", "3t", True, 40, 60)])       # 3t: too few samples
    est = c.estimates("R", "bullish")
    assert [e.horizon.label for e in est] == ["1m", "2m"]
    assert est[0].p_lower < est[0].p_mean < 70 / 120 + 0.01
    assert c.estimates("R", "bearish") == []


def test_calibrator_confirmed_first_then_qualified_then_pooled():
    rows = [("A", "bullish", "1m", True, 30, 60), ("A", "bullish", "1m", False, 40, 60),
            ("B", "bullish", "2m", True, 70, 120)]
    c = _cal(rows)
    by = {e.horizon.label: e for e in c.estimates("A", "bullish")}
    assert by["1m"].population == "qualified" and by["1m"].n == 120
    assert by["2m"].population == "pooled-confirmed"
    assert _cal(rows, pool_symbols=False).estimates("A", "bullish")[0].horizon.label == "1m"


def test_bonferroni_makes_bound_stricter_with_more_horizons():
    one = _cal([("R", "bullish", "1m", True, 70, 120)]).estimates("R", "bullish")[0]
    rows = [("R", "bullish", h.label, True, 70, 120) for h in H]
    many = _cal(rows).estimates("R", "bullish")[0]
    assert many.p_lower < one.p_lower


class _Client:
    def __init__(self, payouts):
        self.payouts, self.bought = payouts, None

    async def proposal(self, *, symbol, contract_type, amount, currency, duration, duration_unit):
        m = self.payouts[f"{duration}{duration_unit}"]
        return {"id": f"p{duration}{duration_unit}", "ask_price": amount, "payout": amount * m}

    async def buy(self, pid, price, *, idempotency_key):
        self.bought = pid
        return {"contract_id": 1, "buy_price": price}


class _Risk:
    def can_trade(self, stake):
        return type("R", (), {"allowed": True, "reason": ""})()


def _est(label, p, lo, n=300):
    return HorizonEstimate(Horizon.parse(label), int(p * n), n, "confirmed", p, lo)


def _run(options, payouts, tmp_path, require_edge=True):
    db = Database(str(tmp_path / "e.db"))
    client = _Client(payouts)
    d = Decision(symbol="R_100", timestamp=time.time(), decision=TRADE_BULLISH,
                 reason_code=TRADE_BULLISH, confirmed=True)
    d, edge, resp = asyncio.run(execute_trade(
        client, db, decision=d, signal_id=None, symbol="R_100", currency="USD", duration=5,
        duration_unit="m", stake=1.0, min_payout_multiple=1.2, risk_manager=_Risk(),
        expiry_options=options, expiry_require_edge=require_edge))
    return d, edge, resp, client, db


def test_expiry_chosen_by_worst_case_ev_at_real_payout(tmp_path):
    # 5t has the better probability but a poor tick payout; 2m wins on EV
    opts = [_est("5t", 0.62, 0.58), _est("2m", 0.60, 0.56)]
    d, edge, resp, client, db = _run(opts, {"5t": 1.70, "2m": 1.95}, tmp_path)
    assert client.bought == "p2m" and (resp["_duration"], resp["_duration_unit"]) == (2, "m")
    row = db.conn.execute("SELECT duration, duration_unit FROM trades").fetchone()
    assert tuple(row) == (2, "m")
    pred = db.conn.execute("SELECT model_id, model_version FROM predictions").fetchone()
    assert pred["model_id"] == "expiry_mc" and pred["model_version"].startswith("2m")


def test_expiry_gate_refuses_without_worst_case_edge(tmp_path):
    opts = [_est("1m", 0.52, 0.49)]
    d, edge, resp, client, _ = _run(opts, {"1m": 1.95}, tmp_path)
    assert resp is None and d.reason_code == "NO_TRADE_POOR_ECONOMICS" and client.bought is None
    d, _, resp, _, _ = _run(opts, {"1m": 1.95}, tmp_path, require_edge=False)
    assert resp is not None                      # demo can trade the best expiry without proof


def test_replay_backfills_minute_outcomes_only():
    cfg = yaml.safe_load(open("config/settings.yaml"))
    from data.candles import Candle
    rng = np.random.default_rng(3)
    x, candles = 0.0, []
    for i in range(1500):
        ticks = []
        for _ in range(20):
            x += -0.08 * x + rng.normal(0, 0.35)
            ticks.append(1000 + x)
        candles.append(Candle("R", i * 60, i * 60 + 59, ticks[0], max(ticks), min(ticks),
                              ticks[-1], 20, True, False, 60))
    rows = replay_setup_outcomes(candles, cfg, parse_horizons(["5t", "1m", "3m"]))
    assert rows and {r["horizon"] for r in rows} == {"1m", "3m"}
    assert np.mean([r["won"] for r in rows]) > 0.6         # planted reversion is found


def test_percentile_threshold_uses_only_past_scores_and_finds_setups():
    from strategy.pipeline import SymbolPipeline
    cfg = yaml.safe_load(open("config/settings.yaml"))
    from data.candles import Candle
    rng = np.random.default_rng(5)
    p, candles = 1000.0, []
    for i in range(1200):
        t = [p := p + rng.normal(0, 0.3) for _ in range(20)]
        candles.append(Candle("R", i * 60, i * 60 + 59, t[0], max(t), min(t), t[-1], 20, True, False, 60))

    def setups(mode):
        c = yaml.safe_load(yaml.safe_dump(cfg))
        c["reversal"]["threshold_mode"] = mode
        pl = SymbolPipeline("R", c)
        n = 0
        for i in range(len(candles)):
            d = pl.evaluate_on_close(candles[max(0, i - 399): i + 1])
            n += d.reason_code in ("NO_TRADE_UNCONFIRMED", "TRADE_BULLISH", "TRADE_BEARISH")
        return n, pl
    n_pct, pl = setups("percentile")
    n_abs, _ = setups("absolute")
    assert n_pct > n_abs
    assert pl.threshold_source.startswith("p")
    thr, _ = pl.effective_threshold("RANGE", 1.0)
    assert thr >= cfg["reversal"]["min_absolute_score"]


def test_setup_outcomes_storage_is_idempotent(tmp_path):
    db = Database(str(tmp_path / "s.db"))
    row = {"symbol": "R", "setup_epoch": 60, "direction": "bullish", "confirmed": True,
           "horizon": "1m", "won": True, "source": "replay"}
    db.record_setup_outcomes([row, {**row, "horizon": "2m", "won": False}])
    db.record_setup_outcomes([{**row, "source": "live"}])       # same setup+horizon: ignored
    assert sorted(db.setup_outcome_counts()) == [("R", "bullish", "1m", 1, 1, 1),
                                                 ("R", "bullish", "2m", 1, 0, 1)]
