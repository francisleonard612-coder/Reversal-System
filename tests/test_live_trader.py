"""Live-loop behavior against a fake Deriv client: restart recovery, settlement,
ambiguous buys, risk wiring, Level 2 EV gate, config horizon guard."""
from __future__ import annotations

import asyncio
import time

import pytest

from config.loader import ConfigError, Settings
from data.database import Database
from deriv.client import BuyAmbiguousError
from execution.executor import execute_trade, monitor_settlement
from execution.live import LiveTrader, build_risk_manager, local_day_start
from risk.staking import StakingEngine
from strategy.decision_engine import TRADE_BULLISH, Decision


class FakeClient:
    is_connected = True

    def __init__(self):
        self.poc_stream: dict = {}
        self.poll_results: list = []
        self.portfolio_contracts: list = []
        self.buy_error = None
        self.bought = 0
        self.subscribed: list = []

    async def proposal(self, **kw):
        return {"id": "p1", "ask_price": kw["amount"], "payout": kw["amount"] * 1.95}

    async def buy(self, proposal_id, price, *, idempotency_key):
        if self.buy_error:
            raise self.buy_error
        self.bought += 1
        return {"contract_id": 1000 + self.bought, "buy_price": price, "start_spot": 100.0}

    async def wait_for_settlement(self, contract_id, timeout=120.0):
        return self.poc_stream

    async def proposal_open_contract(self, contract_id):
        return self.poll_results.pop(0) if self.poll_results else {}

    async def portfolio(self):
        return self.portfolio_contracts

    async def subscribe_ticks(self, sym):
        self.subscribed.append(sym)
        return asyncio.Queue()


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "t.db"))
    monkeypatch.setenv("TRADING_MODE", "demo")
    return Settings()


@pytest.fixture
def db(tmp_path):
    return Database(str(tmp_path / "t.db"))


def _decision():
    return Decision(symbol="R_100", timestamp=time.time(), decision=TRADE_BULLISH,
                    reason_code=TRADE_BULLISH, confirmed=True, bullish_score=70.0)


def _open_trade(db, cid, symbol="R_100"):
    db.record_trade_open(signal_id=None, symbol=symbol, contract_id=cid, idempotency_key=str(cid),
                         contract_type="CALL", stake=1.0, payout=1.95, buy_price=1.0, entry_spot=1.0)


def test_live_contract_must_match_research_horizon(tmp_path, monkeypatch):
    monkeypatch.setenv("CONTRACT_DURATION", "3")
    monkeypatch.setenv("CONTRACT_DURATION_UNIT", "m")
    with pytest.raises(ConfigError, match="must match"):
        Settings()


def test_tick_contract_allowed_but_flagged(monkeypatch):
    monkeypatch.setenv("CONTRACT_DURATION_UNIT", "t")
    assert Settings().horizon_matches_research is False


def test_all_configured_risk_limits_reach_the_risk_manager(settings):
    settings.risk.update(max_trades_per_day=7, cooldown_seconds=12.0, max_concurrent_trades=2)
    rm = build_risk_manager(settings)
    assert (rm.max_trades_per_day, rm.cooldown_seconds, rm.max_concurrent_trades) == (7, 12.0, 2)


def test_buy_registers_open_so_trade_count_and_cooldown_apply(settings, db):
    rm = build_risk_manager(settings)
    rm.balance = rm.peak_balance = 100.0
    rm.cooldown_seconds = 60.0

    async def go():
        client = FakeClient()
        d, _, resp = await execute_trade(client, db, decision=_decision(), signal_id=None,
                                         symbol="R_100", currency="USD", duration=5,
                                         duration_unit="m", stake=1.0, min_payout_multiple=1.8,
                                         risk_manager=rm)
        assert resp is not None
        assert rm.trades_today == 1 and rm.open_trades == 1
        d2, _, resp2 = await execute_trade(client, db, decision=_decision(), signal_id=None,
                                           symbol="R_100", currency="USD", duration=5,
                                           duration_unit="m", stake=1.0,
                                           min_payout_multiple=1.8, risk_manager=rm)
        assert resp2 is None and d2.reason_code == "NO_TRADE_RISK"
    asyncio.run(go())


def test_settlement_silence_is_never_booked_as_a_loss(settings, db):
    _open_trade(db, 7)
    rm = build_risk_manager(settings)
    rm.balance = rm.peak_balance = 100.0
    rm.open_trades = 1
    st = StakingEngine()
    client = FakeClient()
    client.poc_stream = {}                               # stream went silent
    client.poll_results = [{}, {"is_sold": 1, "profit": 0.95, "status": "won", "exit_tick": 101}]

    async def go():
        import execution.executor as ex
        real_sleep = asyncio.sleep
        ex.asyncio.sleep = lambda s: real_sleep(0)
        try:
            return await monitor_settlement(client, db, rm, st, 7, expected_seconds=0,
                                            grace_seconds=0)
        finally:
            ex.asyncio.sleep = real_sleep
    poc = asyncio.run(go())
    assert poc["status"] == "won"
    row = db.conn.execute("SELECT won, pnl FROM trades WHERE contract_id=7").fetchone()
    assert (row["won"], row["pnl"]) == (1, 0.95)
    assert rm.consecutive_losses == 0 and rm.daily_pnl == 0.95 and rm.open_trades == 0


def test_settlement_booked_exactly_once(settings, db):
    _open_trade(db, 8)
    rm = build_risk_manager(settings)
    rm.balance = rm.peak_balance = 100.0
    client = FakeClient()
    client.poc_stream = {"is_sold": 1, "profit": -1.0, "status": "lost"}
    st = StakingEngine()
    asyncio.run(monitor_settlement(client, db, rm, st, 8))
    asyncio.run(monitor_settlement(client, db, rm, st, 8))   # duplicate observation
    assert rm.daily_pnl == -1.0 and rm.consecutive_losses == 1


def test_ambiguous_buy_that_went_through_is_recorded(settings, db):
    rm = build_risk_manager(settings)
    rm.balance = rm.peak_balance = 100.0
    client = FakeClient()
    client.buy_error = BuyAmbiguousError("BuyAmbiguous", "timeout")
    client.portfolio_contracts = [{"contract_id": 555, "symbol": "R_100", "contract_type": "CALL",
                                   "purchase_time": time.time(), "buy_price": 1.0}]
    d, _, resp = asyncio.run(execute_trade(
        client, db, decision=_decision(), signal_id=None, symbol="R_100", currency="USD",
        duration=5, duration_unit="m", stake=1.0, min_payout_multiple=1.8, risk_manager=rm))
    assert resp["contract_id"] == 555
    assert [t["contract_id"] for t in db.open_trades()] == [555]
    assert rm.open_trades == 1


def test_ambiguous_buy_not_found_releases(settings, db):
    rm = build_risk_manager(settings)
    rm.balance = rm.peak_balance = 100.0
    client = FakeClient()
    client.buy_error = BuyAmbiguousError("BuyAmbiguous", "timeout")
    d, _, resp = asyncio.run(execute_trade(
        client, db, decision=_decision(), signal_id=None, symbol="R_100", currency="USD",
        duration=5, duration_unit="m", stake=1.0, min_payout_multiple=1.8, risk_manager=rm))
    assert resp is None and d.reason_code == "NO_TRADE_BUY_AMBIGUOUS"


def test_restart_restores_limits_and_reattaches_open_contracts(settings, db):
    for i, pnl in enumerate([-10.0, -10.0, -6.0]):
        _open_trade(db, 100 + i)
        db.record_trade_result(100 + i, won=False, pnl=pnl)
    _open_trade(db, 200)                                   # still open at "restart"
    client = FakeClient()
    client.poc_stream = {"is_sold": 1, "profit": 0.9, "status": "won"}
    trader = LiveTrader(settings, client, db)

    async def go():
        trader.restore_state(balance=500.0)
        assert "R_100" in trader.open_symbols               # blocked until 200 settles
        assert trader.risk.consecutive_losses == 3
        assert trader.risk.emergency_stopped                # -26 today > 25 daily limit
        await asyncio.sleep(0.05)
    asyncio.run(go())
    assert db.open_trades() == []                          # reattached and settled


def test_operator_reset_clears_the_streak_but_not_todays_pnl(db):
    for i in range(3):
        _open_trade(db, 300 + i)
        db.record_trade_result(300 + i, won=False, pnl=-1.0)
    time.sleep(0.01)
    db.record_risk_reset("checked")
    s = db.risk_state_since(local_day_start())
    assert s["consecutive_losses"] == 0 and s["daily_pnl"] == -3.0


def test_watchdog_resubscribes_silent_symbol(settings, db):
    client = FakeClient()
    trader = LiveTrader(settings, client, db)
    trader.symbols = ["R_100"]
    trader.last_tick_at["R_100"] = time.time() - 999

    async def go():
        task = asyncio.create_task(trader.watchdog(interval=0.01))
        await asyncio.sleep(0.05)
        task.cancel()
    asyncio.run(go())
    assert client.subscribed == ["R_100"]


class _Model:
    model_id, version = "m", "v1"

    def __init__(self, p, lo):
        self.p, self.lo = p, lo

    def predict(self, features):
        return self.p, self.p, self.lo


@pytest.mark.parametrize("p,lo,trades", [(0.60, 0.56, True),     # clears break-even 0.513
                                         (0.60, 0.50, False),    # point estimate only
                                         (0.45, 0.40, False)])   # negative EV
def test_level2_ev_gate(settings, db, p, lo, trades):
    rm = build_risk_manager(settings)
    rm.balance = rm.peak_balance = 100.0
    d, edge, resp = asyncio.run(execute_trade(
        FakeClient(), db, decision=_decision(), signal_id=None, symbol="R_100", currency="USD",
        duration=5, duration_unit="m", stake=1.0, min_payout_multiple=1.8, risk_manager=rm,
        level2_model=_Model(p, lo), level2_features={}))
    assert (resp is not None) is trades
    assert edge.expected_value == pytest.approx(p * 1.95 - 1)
    row = db.conn.execute("SELECT calibrated_probability, expected_value FROM predictions").fetchone()
    assert row["calibrated_probability"] == p
    if not trades:
        assert d.reason_code == "NO_TRADE_POOR_ECONOMICS"


def test_full_live_run_smoke(settings, db):
    """Startup -> seed -> tick stream -> candles -> signals persisted, no crash."""
    import random
    from deriv.client import HistoricalCandle, TickMessage

    settings.symbols = ["R_100"]
    settings.raw["candles"]["seed_bars"] = 400   # the fake tick stream below starts right after 400 bars

    class StreamClient(FakeClient):
        def __init__(self):
            super().__init__()
            self.q = asyncio.Queue()

        async def balance(self):
            return 1000.0

        async def verify_rise_fall_available(self, sym, cur):
            return True, f"{sym}: CALL and PUT confirmed available"

        async def candle_history(self, sym, count=400, granularity=60, end="latest"):
            rnd, p, out = random.Random(1), 100.0, []
            for i in range(count):
                p += rnd.gauss(0, 0.2)
                out.append(HistoricalCandle(1_700_000_000 + i * 60, p, p + .1, p - .1, p))
            return out

        async def subscribe_ticks(self, sym):
            return self.q

        async def rise_fall_duration_limits(self, sym):
            return [("1t", "10t"), ("15s", "1d")]

    client = StreamClient()
    trader = LiveTrader(settings, client, db)

    async def go():
        task = asyncio.create_task(trader.run())
        rnd, p, t = random.Random(2), 100.0, 1_700_000_000 + 401 * 60
        for _ in range(60 * 30):                 # 30 minutes of 1-second ticks
            p += rnd.gauss(0, 0.05)
            t += 1
            await client.q.put(TickMessage("R_100", float(t), p, 2))
        for _ in range(100):                     # wait for warm-up + processing
            await asyncio.sleep(0.1)
            if db.conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0] >= 25:
                break
        task.cancel()
        for t_ in list(trader._tasks):
            t_.cancel()
    asyncio.run(go())
    n_signals = db.conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
    n_candles = db.conn.execute("SELECT COUNT(*) FROM candles").fetchone()[0]
    assert n_signals >= 25 and n_candles >= 420
    assert [h.label for h in trader.symbol_horizons["R_100"]] == \
        ["3t", "5t", "7t", "10t", "1m", "2m", "3m", "5m"]


def test_startup_survives_slow_database_and_dropped_subscribe(settings, db):
    """Regression: seeding persisted candles on the event loop froze it long
    enough for Deriv to drop the socket, and the failed subscribe then killed
    the process. Seeding must not block the loop, and a failed subscribe must
    retry."""
    import random
    from deriv.client import DerivAPIError, HistoricalCandle

    settings.symbols = ["R_100"]
    settings.raw["candles"]["seed_bars"] = 400
    real_record = db.record_candles
    db.record_candles = lambda cs: (time.sleep(0.6), real_record(cs))[1]   # slow Supabase

    class Client(FakeClient):
        fails = 1

        async def candle_history(self, sym, count=400, granularity=60, end="latest"):
            rnd, p = random.Random(1), 100.0
            return [HistoricalCandle(1_700_000_000 + i * 60, p, p + .1, p - .1,
                                     p := p + rnd.gauss(0, .2)) for i in range(count)]

        async def subscribe_ticks(self, sym):
            if Client.fails:
                Client.fails -= 1
                raise DerivAPIError("Disconnected", "socket closed")
            return asyncio.Queue()

        async def ensure_connected(self):
            pass

    trader = LiveTrader(settings, Client(), db)
    trader.symbols = ["R_100"]
    from data.candles import CandleBuilder
    from strategy.pipeline import SymbolPipeline
    trader.builders["R_100"] = CandleBuilder("R_100", 60)
    trader.pipelines["R_100"] = SymbolPipeline("R_100", settings.raw)

    async def go():
        stamps = []

        async def heartbeat():
            while True:
                stamps.append(time.monotonic())
                await asyncio.sleep(0.02)
        hb = asyncio.create_task(heartbeat())
        await trader.seed_candles()
        hb.cancel()
        return max(b - a for a, b in zip(stamps, stamps[1:]))
    longest_freeze = asyncio.run(go())
    assert "R_100" in trader.queues                        # subscribe retried and succeeded
    assert longest_freeze < 0.3                            # never frozen by the 0.6s write
    assert db.conn.execute("SELECT COUNT(*) FROM candles").fetchone()[0] >= 390
