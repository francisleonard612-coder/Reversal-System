from __future__ import annotations

import time

import pytest

from data.database import Database
from strategy.decision_engine import Decision, TRADE_BULLISH


@pytest.fixture
def db(tmp_path):
    return Database(str(tmp_path / "test.db"))


def test_schema_creates_all_required_tables(db):
    tables = {r["name"] for r in db.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    required = {"ticks", "candles", "features", "signals", "rejected_signals",
               "trades", "outcomes", "predictions", "model_versions",
               "model_performance", "regime_history", "learning_events",
               "system_events", "parameter_changes"}
    assert required.issubset(tables)


def test_record_no_trade_signal_also_writes_a_rejected_signal_row(db):
    d = Decision(symbol="R_100", timestamp=time.time(), decision="NO_TRADE",
                reason_code="NO_TRADE_NO_SETUP", explanation="test",
                bullish_score=10.0, bearish_score=5.0)
    signal_id = db.record_signal(d)
    row = db.conn.execute("SELECT * FROM rejected_signals WHERE signal_id=?",
                          (signal_id,)).fetchone()
    assert row is not None
    assert row["would_be_direction"] == "bullish"   # higher of the two scores
    assert row["outcome_evaluated"] == 0


def test_record_trade_signal_does_not_write_a_rejected_row(db):
    d = Decision(symbol="R_100", timestamp=time.time(), decision=TRADE_BULLISH,
                reason_code=TRADE_BULLISH, explanation="test",
                bullish_score=80.0, bearish_score=5.0, confirmed=True)
    signal_id = db.record_signal(d)
    row = db.conn.execute("SELECT * FROM rejected_signals WHERE signal_id=?",
                          (signal_id,)).fetchone()
    assert row is None


def test_reason_histogram_counts_correctly(db):
    for code in ["NO_TRADE_NO_SETUP", "NO_TRADE_NO_SETUP", "NO_TRADE_HIGH_VOLATILITY"]:
        d = Decision(symbol="R_100", timestamp=time.time(), decision="NO_TRADE",
                    reason_code=code, explanation="")
        db.record_signal(d)
    hist = db.reason_histogram()
    assert hist["NO_TRADE_NO_SETUP"] == 2
    assert hist["NO_TRADE_HIGH_VOLATILITY"] == 1


def test_trade_lifecycle_open_then_settle(db):
    signal_id = db.record_signal(Decision(
        symbol="R_100", timestamp=time.time(), decision=TRADE_BULLISH,
        reason_code=TRADE_BULLISH, explanation="", confirmed=True))
    trade_id = db.record_trade_open(
        signal_id=signal_id, symbol="R_100", contract_id=12345,
        idempotency_key="k1", contract_type="CALL", stake=1.0, payout=1.9,
        buy_price=1.0, entry_spot=100.0)
    assert trade_id > 0
    db.record_trade_result(12345, won=True, pnl=0.9, exit_spot=101.0)
    row = db.conn.execute("SELECT * FROM trades WHERE contract_id=?", (12345,)).fetchone()
    assert row["won"] == 1
    assert row["pnl"] == pytest.approx(0.9)
    summary = db.trade_summary()
    assert summary["trades"] == 1
    assert summary["wins"] == 1


def test_idempotency_key_is_unique(db):
    signal_id = db.record_signal(Decision(
        symbol="R_100", timestamp=time.time(), decision=TRADE_BULLISH,
        reason_code=TRADE_BULLISH, explanation="", confirmed=True))
    db.record_trade_open(signal_id=signal_id, symbol="R_100", contract_id=1,
                         idempotency_key="dup", contract_type="CALL", stake=1.0,
                         payout=1.9, buy_price=1.0, entry_spot=100.0)
    with pytest.raises(Exception):
        db.record_trade_open(signal_id=signal_id, symbol="R_100", contract_id=2,
                             idempotency_key="dup", contract_type="CALL", stake=1.0,
                             payout=1.9, buy_price=1.0, entry_spot=100.0)


def test_is_healthy_reports_true_on_a_working_connection(db):
    assert db.is_healthy()


def test_log_event_never_raises_even_after_close(db):
    db.close()
    db.log_event("ERROR", "test", "should not raise")  # must not throw


# --- Section 28/29 counterfactual reconciliation --------------------------

def _decision(direction: str, ts: float, symbol="R_100") -> Decision:
    bullish, bearish = (80.0, 5.0) if direction == "bullish" else (5.0, 80.0)
    return Decision(symbol=symbol, timestamp=ts, decision="NO_TRADE",
                    reason_code="NO_TRADE_UNCONFIRMED", explanation="",
                    bullish_score=bullish, bearish_score=bearish)


def _candle(symbol, close_epoch, close, open_=None):
    from data.candles import Candle
    open_ = open_ if open_ is not None else close
    return Candle(symbol=symbol, open_epoch=close_epoch - 60, close_epoch=close_epoch,
                 open=open_, high=max(open_, close), low=min(open_, close),
                 close=close, n_ticks=5, is_closed=True, timeframe_seconds=60)


def test_reconcile_leaves_unevaluated_when_settlement_candle_does_not_exist_yet(db):
    signal_id = db.record_signal(Decision(symbol="R_100", timestamp=100.0,
                                          decision="NO_TRADE",
                                          reason_code="NO_TRADE_UNCONFIRMED",
                                          explanation="", bullish_score=80.0,
                                          bearish_score=5.0))
    db.record_candle(_candle("R_100", 100, 1000.0))
    n = db.reconcile_rejected_signals(duration_bars=5)
    assert n == 0
    row = db.conn.execute("SELECT * FROM rejected_signals").fetchone()
    assert row["outcome_evaluated"] == 0


def test_reconcile_marks_bullish_would_have_won_when_price_rose(db):
    db.record_signal(_decision("bullish", ts=100.0))
    db.record_candle(_candle("R_100", 100, 1000.0))
    for i in range(1, 6):
        db.record_candle(_candle("R_100", 100 + i * 60, 1000.0 + i))
    n = db.reconcile_rejected_signals(duration_bars=5)
    assert n == 1
    row = db.conn.execute("SELECT * FROM rejected_signals").fetchone()
    assert row["outcome_evaluated"] == 1
    assert row["would_have_won"] == 1
    assert row["entry_price"] == 1000.0
    assert row["outcome_price"] == 1005.0


def test_reconcile_marks_bullish_would_have_lost_when_price_fell(db):
    db.record_signal(_decision("bullish", ts=100.0))
    db.record_candle(_candle("R_100", 100, 1000.0))
    for i in range(1, 6):
        db.record_candle(_candle("R_100", 100 + i * 60, 1000.0 - i))
    db.reconcile_rejected_signals(duration_bars=5)
    row = db.conn.execute("SELECT * FROM rejected_signals").fetchone()
    assert row["would_have_won"] == 0


def test_reconcile_bearish_direction_is_the_mirror(db):
    db.record_signal(_decision("bearish", ts=100.0))
    db.record_candle(_candle("R_100", 100, 1000.0))
    for i in range(1, 6):
        db.record_candle(_candle("R_100", 100 + i * 60, 1000.0 - i))  # price fell
    db.reconcile_rejected_signals(duration_bars=5)
    row = db.conn.execute("SELECT * FROM rejected_signals").fetchone()
    assert row["would_have_won"] == 1   # PUT wins when price falls


def test_reconcile_is_idempotent_and_safe_to_rerun(db):
    db.record_signal(_decision("bullish", ts=100.0))
    db.record_candle(_candle("R_100", 100, 1000.0))
    for i in range(1, 6):
        db.record_candle(_candle("R_100", 100 + i * 60, 1000.0 + i))
    first = db.reconcile_rejected_signals(duration_bars=5)
    second = db.reconcile_rejected_signals(duration_bars=5)
    assert first == 1
    assert second == 0   # already-evaluated rows are not touched again


def test_reconcile_settlement_candle_is_strictly_after_entry_not_equal():
    """The anti-lookahead guarantee applied to reconciliation itself: if the
    entry candle and a same-epoch candle were both eligible as 'settlement',
    reconciliation would be trivially right or wrong by construction. The
    query uses close_epoch > entry, never >=."""
    from data.database import Database
    import tempfile, os
    path = tempfile.mktemp(suffix=".db")
    db = Database(path)
    try:
        db.record_signal(_decision("bullish", ts=100.0))
        entry = _candle("R_100", 100, 1000.0)
        db.record_candle(entry)
        # No candle after entry at all -- must NOT settle against entry itself.
        n = db.reconcile_rejected_signals(duration_bars=1)
        assert n == 0
        row = db.conn.execute("SELECT * FROM rejected_signals").fetchone()
        assert row["outcome_evaluated"] == 0
    finally:
        db.close()
        os.remove(path)


def test_traded_signals_are_never_written_to_rejected_signals_even_after_reconcile(db):
    """A taken trade's real outcome comes from trades/executor.py, not this
    counterfactual path -- reconcile must not touch it."""
    signal_id = db.record_signal(Decision(
        symbol="R_100", timestamp=100.0, decision=TRADE_BULLISH,
        reason_code=TRADE_BULLISH, explanation="", confirmed=True,
        bullish_score=90.0))
    db.record_candle(_candle("R_100", 100, 1000.0))
    for i in range(1, 6):
        db.record_candle(_candle("R_100", 100 + i * 60, 1000.0 + i))
    db.reconcile_rejected_signals(duration_bars=5)
    assert db.conn.execute("SELECT COUNT(*) c FROM rejected_signals").fetchone()["c"] == 0

