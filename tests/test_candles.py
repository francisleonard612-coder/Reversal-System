from __future__ import annotations

import pytest

from data.candles import CandleBuilder, candles_from_ticks
from data.validation import TickValidator


def test_candle_builder_aggregates_ohlc_correctly():
    b = CandleBuilder("R_100", timeframe_seconds=60)
    for e, p in [(0, 100), (10, 105), (20, 95), (59, 102)]:
        assert b.add_tick(e, p) is None
    closed = b.add_tick(60, 110)
    assert closed is not None
    assert closed.open == 100
    assert closed.high == 105
    assert closed.low == 95
    assert closed.close == 102
    assert closed.is_closed
    assert closed.n_ticks == 4


def test_no_fabricated_volume():
    b = CandleBuilder("R_100", 60)
    b.add_tick(0, 100)
    c = b.add_tick(60, 101)
    assert c.has_volume is False


def test_forming_candle_never_appears_in_history():
    b = CandleBuilder("R_100", 60)
    b.add_tick(0, 100)
    b.add_tick(30, 101)
    assert b.history() == []
    assert b.current_forming is not None
    b.add_tick(60, 102)
    assert len(b.history()) == 1
    assert b.history()[0].is_closed


def test_out_of_order_tick_rejected():
    b = CandleBuilder("R_100", 60)
    b.add_tick(100, 1.0)
    with pytest.raises(ValueError):
        b.add_tick(50, 1.0)


def test_candles_from_ticks_drops_the_final_forming_bar():
    """A backtest must never evaluate against a bar that wasn't actually
    closed at that point in history."""
    epochs = [0, 30, 60, 90, 120, 130]   # last bar (120-179) never closes
    prices = [1, 2, 3, 4, 5, 6]
    candles = candles_from_ticks("R_100", epochs, prices, timeframe_seconds=60)
    assert len(candles) == 2
    assert all(c.is_closed for c in candles)


def test_tick_validator_rejects_nan_and_nonpositive():
    v = TickValidator()
    assert not v.validate(0, float("nan")).valid
    assert not v.validate(0, -1.0).valid
    assert not v.validate(0, 0.0).valid


def test_tick_validator_rejects_out_of_order_and_duplicates():
    v = TickValidator()
    assert v.validate(100, 1.0).valid
    assert not v.validate(50, 1.0).valid
    assert not v.validate(100, 1.0).valid   # duplicate epoch


def test_tick_validator_flags_gap_without_rejecting():
    v = TickValidator(max_gap_seconds=10.0)
    v.validate(0, 1.0)
    result = v.validate(100, 1.0)
    assert result.valid
    assert "gap" in result.reason
