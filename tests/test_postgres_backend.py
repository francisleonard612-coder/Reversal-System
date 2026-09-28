"""The new persistence methods against a real Postgres, schema applied from
supabase/schema.sql. Skipped unless TEST_DATABASE_URL points at a scratch
database (never at production)."""
from __future__ import annotations

import os
import time

import pytest

DSN = os.getenv("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not DSN, reason="set TEST_DATABASE_URL to a scratch Postgres")


@pytest.fixture
def pg():
    import psycopg
    from data.postgres import PostgresDatabase
    with psycopg.connect(DSN, autocommit=True) as c:
        c.execute("drop schema public cascade; create schema public;")
        c.execute("do $$ begin if not exists (select from pg_roles where rolname='authenticated') "
                  "then create role authenticated; end if; end $$;")
        c.execute(open("supabase/schema.sql").read())
    db = PostgresDatabase(DSN)
    yield db
    db.close()


def _trade(db, cid, sym="R_100"):
    db.record_trade_open(signal_id=None, symbol=sym, contract_id=cid, idempotency_key=str(cid),
                         contract_type="CALL", stake=1.0, payout=1.95, buy_price=1.0, entry_spot=1.0)


def test_settlement_idempotent_and_risk_state(pg):
    _trade(pg, 1)
    _trade(pg, 2)
    assert pg.record_trade_result(1, won=False, pnl=-1.0) is True
    assert pg.record_trade_result(1, won=True, pnl=5.0) is False
    assert [t["contract_id"] for t in pg.open_trades()] == [2]
    s = pg.risk_state_since(time.time() - 3600)
    assert s == {"daily_pnl": -1.0, "trades_today": 2, "consecutive_losses": 1}
    time.sleep(0.01)
    pg.record_risk_reset("t")
    assert pg.risk_state_since(time.time() - 3600)["consecutive_losses"] == 0
    assert pg.known_contract_ids() == {1, 2}


def test_candles_predictions_models_dashboard(pg):
    from data.candles import Candle
    for i in range(3):
        pg.record_candle(Candle("R_100", i * 60, i * 60 + 59, 1, 2, 0.5, 1.5 + i, 10, True, False, 60))
    c = pg.load_candles("R_100")
    assert [x.close for x in c] == [1.5, 2.5, 3.5] and pg.candle_symbols() == ["R_100"]
    assert [x.close for x in pg.load_candles("R_100", limit=2)] == [2.5, 3.5]
    pg.record_prediction(signal_id=None, symbol="R_100", model_id="m", model_version="v",
                         probability=0.6, calibrated_probability=0.58, probability_lower=0.54,
                         payout_multiple=1.95, expected_value=0.131)
    pg.record_model_version(model_id="m", version="v", period_start=0, period_end=1,
                            features=["a"], parameters={"x": 1}, calibration_method="platt",
                            promotion_status="REJECTED")
    d = pg.dashboard(0)
    assert d["predictions"]["n"] == 1 and d["latest_model"]["promotion_status"] == "REJECTED"
    assert d["candles_stored"] == {"R_100": 3}


def test_setup_outcomes_and_trade_expiry(pg):
    row = {"symbol": "R", "setup_epoch": 60, "direction": "bullish", "confirmed": True,
           "horizon": "1m", "won": True, "source": "replay"}
    assert pg.record_setup_outcomes([row, {**row, "horizon": "5t", "won": False}]) == 2
    assert pg.record_setup_outcomes([row]) == 0
    assert sorted(pg.setup_outcome_counts()) == [("R", "bullish", "1m", True, 1, 1),
                                                 ("R", "bullish", "5t", True, 0, 1)]
    pg.record_trade_open(signal_id=None, symbol="R", contract_id=9, idempotency_key="9",
                         contract_type="CALL", stake=1, payout=1.9, buy_price=1, entry_spot=1,
                         duration=5, duration_unit="t")
    assert pg.open_trades()[0]["contract_id"] == 9
