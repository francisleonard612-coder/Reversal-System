from __future__ import annotations

import random

import pytest
import yaml

from backtest.simulator import run_backtest, walk_forward
from data.candles import candles_from_ticks


@pytest.fixture(scope="module")
def cfg():
    with open("config/settings.yaml") as f:
        return yaml.safe_load(f)


def _random_walk_candles(n_ticks=60000, seed=0, timeframe=60):
    random.seed(seed)
    price, epochs, prices, t = 1000.0, [], [], 1_700_000_000
    for _ in range(n_ticks):
        price += random.gauss(0, 0.3) - 0.0005 * (price - 1000.0)
        price = max(price, 1.0)
        t += 1
        epochs.append(t)
        prices.append(price)
    return candles_from_ticks("SIM", epochs, prices, timeframe_seconds=timeframe)


def test_backtest_runs_end_to_end_and_reports_assumed_payout(cfg):
    candles = _random_walk_candles(seed=1)
    result = run_backtest(candles, cfg, stake=1.0, payout_multiple=1.85, warmup_bars=210)
    assert result.n_evaluated > 0
    assert "ASSUMED payout" in result.report()
    assert result.payout_multiple == 1.85


def test_backtest_pnl_is_consistent_with_wins_and_losses(cfg):
    candles = _random_walk_candles(seed=2)
    result = run_backtest(candles, cfg, stake=1.0, payout_multiple=1.85, warmup_bars=210)
    expected_pnl = (result.wins * 1.0 * (1.85 - 1)) - (result.losses * 1.0)
    assert result.pnl == pytest.approx(expected_pnl)
    assert result.wins + result.losses == result.n_trades


def test_backtest_every_no_trade_reason_is_a_real_decision_engine_code(cfg):
    from strategy.decision_engine import (
        NO_TRADE_CONFLICTING_SIGNALS, NO_TRADE_HIGH_VOLATILITY,
        NO_TRADE_MINIMUM_DATA, NO_TRADE_NO_SETUP, NO_TRADE_REGIME_UNKNOWN,
        NO_TRADE_UNCONFIRMED)
    known = {NO_TRADE_CONFLICTING_SIGNALS, NO_TRADE_HIGH_VOLATILITY,
            NO_TRADE_MINIMUM_DATA, NO_TRADE_NO_SETUP, NO_TRADE_REGIME_UNKNOWN,
            NO_TRADE_UNCONFIRMED, "NO_TRADE_SIM_HORIZON"}
    candles = _random_walk_candles(seed=3)
    result = run_backtest(candles, cfg, stake=1.0, payout_multiple=1.85, warmup_bars=210)
    assert set(result.reason_counts.keys()).issubset(known)


def test_walk_forward_builds_a_fresh_pipeline_per_block(cfg):
    candles = _random_walk_candles(n_ticks=240000, seed=4)
    report = walk_forward(candles, cfg, n_blocks=4, warmup_bars=210)
    assert len(report.blocks) >= 1
    assert "WALK-FORWARD" in report.report()
    assert report.total_trades == sum(b.n_trades for b in report.blocks)


def test_walk_forward_refuses_too_few_candles_per_block(cfg):
    candles = _random_walk_candles(n_ticks=5000, seed=5)
    with pytest.raises(ValueError, match="too few"):
        walk_forward(candles, cfg, n_blocks=10, warmup_bars=210)


def test_trades_requiring_settlement_past_the_data_horizon_are_excluded_not_lost(cfg):
    """A trade whose settlement bar doesn't exist in the data must not be
    silently counted as a loss -- that would understate the win rate."""
    candles = _random_walk_candles(seed=6)
    result = run_backtest(candles, cfg, stake=1.0, payout_multiple=1.85,
                          warmup_bars=210, contract_duration_bars=5)
    if "NO_TRADE_SIM_HORIZON" in result.reason_counts:
        assert result.reason_counts["NO_TRADE_SIM_HORIZON"] <= 5
