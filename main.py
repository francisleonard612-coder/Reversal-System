"""
Main entry point and CLI (spec Section 51).

Supports: --backtest, --walk-forward, --diagnostics, --status, and live
operation in research/demo/live mode via config. --train, --evaluate,
--online-sim, and --dashboard are Level 2/3 commands and are not
implemented yet -- they print a clear "not implemented at Level 1" message
rather than silently doing nothing, per Section 62's ban on fake
implementations.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time

from config.loader import ConfigError, Settings
from data.candles import CandleBuilder, candles_from_history
from data.postgres import open_database
from data.validation import TickValidator
from deriv.client import DerivAPIError, DerivAuthError, DerivClient
from execution.executor import execute_trade, monitor_settlement
from risk.manager import RiskManager
from risk.staking import StakingEngine
from strategy.pipeline import SymbolPipeline

logger = logging.getLogger("reversal")

NOT_YET_IMPLEMENTED = {
    "train": "Level 2's probability models don't exist yet -- there is nothing to train.",
    "evaluate": "Level 2/3 benchmarking (Section 43) needs a probability model to compare against Level 1.",
    "online-sim": "Section 42's online simulator is a Level 3 feature (online learning).",
    "dashboard": "Section 52's dashboard is staged for the next phase.",
}


def _setup_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)-8s %(name)s %(message)s")


async def run_live(settings: Settings) -> None:
    # DB_BACKEND=postgres routes to Supabase; sqlite is fine locally but on
    # Railway the container filesystem is ephemeral, so the decision/signal
    # history dies on every redeploy without postgres.
    db = open_database(sqlite_path=settings.storage_path)
    d = settings.deriv
    client = DerivClient(
        app_id=d["app_id"], api_token=d["api_token"], ws_url=d["ws_url"],
        request_timeout=d["request_timeout"], api_base_url=d["api_base_url"],
        auth_mode=d["auth_mode"], account_id=d["account_id"] or None,
        use_real_account=settings.use_real_account,
        max_requests_per_minute=d["max_requests_per_minute"])

    try:
        await client.connect()
    except DerivAuthError as exc:
        logger.error("authentication failed: %s", exc)
        raise SystemExit(1)

    balance = await client.balance()
    logger.info("connected, mode=%s balance=%.2f %s symbols=%s",
               settings.mode, balance, settings.currency, settings.symbols)

    usable = []
    for symbol in settings.symbols:
        ok, why = await client.verify_rise_fall_available(symbol, settings.currency)
        logger.info("contract check %s", why)
        if ok:
            usable.append(symbol)
        else:
            db.log_event("WARNING", "discovery", why)
    if not usable:
        raise RuntimeError("no configured symbol offers CALL/PUT -- refusing to run")

    risk = RiskManager(base_stake=settings.risk["base_stake"],
                       max_stake=settings.risk["max_stake"],
                       max_daily_loss=settings.risk["max_daily_loss"],
                       max_drawdown=settings.risk["max_drawdown"],
                       max_consecutive_losses=settings.risk["max_consecutive_losses"])
    risk.balance = balance
    risk.peak_balance = balance
    staking = StakingEngine(method=settings.staking["method"],
                            base_stake=settings.risk["base_stake"],
                            max_stake=settings.risk["max_stake"],
                            martingale_enabled=settings.staking["martingale_enabled"])

    builders = {sym: CandleBuilder(sym, settings["candles"]["timeframe_seconds"])
               for sym in usable}
    validators = {sym: TickValidator() for sym in usable}
    pipelines = {sym: SymbolPipeline(sym, settings.raw) for sym in usable}
    open_contracts: set[str] = set()

    queues = {}
    timeframe = settings["candles"]["timeframe_seconds"]
    # Fetch enough candles to fill the same window features/engine.py's
    # build_features() ever looks at (max_lookback_bars) -- seeding more
    # than that is wasted bandwidth, seeding less just delays warm-up.
    seed_count = settings["candles"].get("max_lookback_bars", 400)
    for sym in usable:
        try:
            raw_hist = await client.candle_history(
                sym, count=seed_count, granularity=timeframe)
            seed_candles = candles_from_history(sym, raw_hist, timeframe)
            n = builders[sym].seed_closed(seed_candles)
        except Exception:
            # A cold-start hiccup on one symbol (bad response shape, a
            # transient Deriv API error, a validation failure in
            # seed_closed) must never take down the whole live loop --
            # fall back to the old raw-tick seed instead. Slower warm-up
            # beats a crash. This ONLY runs if seed_closed never
            # succeeded, so the builder is still untouched here -- a
            # raw-tick reseed is safe to attempt.
            logger.warning("%s: candle-history seed failed, falling back "
                          "to raw-tick seed", sym, exc_info=True)
            try:
                history = await client.tick_history(sym, count=5000)
                for t in history:
                    v = validators[sym].validate(t.epoch, t.price)
                    if v.valid:
                        newly_closed = builders[sym].add_tick(t.epoch, t.price)
                        if newly_closed is not None:
                            db.record_candle(newly_closed)
                logger.info("%s: seeded %d closed candles from raw-tick "
                           "fallback (persisted)", sym, len(builders[sym].closed))
            except Exception:
                logger.warning("%s: raw-tick fallback also failed -- "
                              "starting with zero seeded candles", sym,
                              exc_info=True)
        else:
            # seed_closed already succeeded -- the builder now correctly
            # holds `n` candles in memory. Persisting them to Supabase is
            # best-effort from here on and MUST NOT fall back to a raw-tick
            # reseed on failure: this builder's internal clock has already
            # advanced past every seeded candle's close_epoch, so a
            # raw-tick reseed here would immediately hit "ticks must
            # arrive in order" and wipe out the good in-memory seed just
            # persisted -- a DB hiccup and a bad in-memory seed are two
            # different failures and must not be handled as if they were
            # the same one.
            logger.info("%s: seeded %d closed candles from server-"
                       "aggregated history", sym, n)
            try:
                # Persisted the same way live candles are, via the same
                # ON CONFLICT (symbol, close_epoch) DO NOTHING path --
                # every restart re-seeds overlapping history, and this is
                # what makes that a no-op instead of a duplicate row (see
                # data/database.py's v3 migration for why that matters).
                for c in seed_candles:
                    db.record_candle(c)
            except Exception:
                logger.warning("%s: could not persist seeded history -- "
                              "will still trade from the %d candles held "
                              "in memory", sym, n, exc_info=True)
        queues[sym] = await client.subscribe_ticks(sym)

    logger.info("entering live loop (mode=%s)", settings.mode)


    async def handle_symbol(sym: str) -> None:
        while True:
            tick = await queues[sym].get()
            v = validators[sym].validate(tick.epoch, tick.price)
            if not v.valid:
                db.log_event("WARNING", "data_quality", v.reason, {"symbol": sym})
                continue
            newly_closed = builders[sym].add_tick(tick.epoch, tick.price)
            if newly_closed is None:
                continue

            # Persist the candle before anything else. This is the raw
            # material Level 2 retrains features from and the only way to
            # reconstruct what actually happened after a REJECTED signal
            # (rejected_signals has no outcome captured at decision time --
            # Section 29 forbids that -- so its outcome is reconstructed
            # later by joining on symbol+ts against this table). Individual
            # ticks are deliberately NOT persisted -- 60s candles are the
            # granularity everything downstream actually needs, and storing
            # every tick would burn through Supabase's free tier fast for no
            # corresponding benefit.
            db.record_candle(newly_closed)

            decision = pipelines[sym].evaluate_on_close(builders[sym].history())
            signal_id = db.record_signal(decision)

            if not decision.will_trade or sym in open_contracts:
                continue
            if not client.is_connected:
                continue

            open_contracts.add(sym)
            cfg_contract = settings["contract"]
            decision, edge, buy_resp = await execute_trade(
                client, db, decision=decision, signal_id=signal_id, symbol=sym,
                currency=settings.currency, duration=cfg_contract["duration"],
                duration_unit=cfg_contract["duration_unit"],
                stake=staking.stake_for()[0],
                min_payout_multiple=settings["edge"]["min_payout_multiple"],
                risk_manager=risk, research_mode=(settings.mode == "research"))

            # execute_trade() can downgrade the signal-level decision
            # (economics, risk, research mode, a bad proposal) AFTER the row
            # above was already written with the pre-downgrade TRADE_* value.
            # Correct it here so `signals` never claims a trade happened
            # when execute_trade refused it -- see update_signal_outcome's
            # docstring in data/database.py.
            db.update_signal_outcome(signal_id, decision)

            if buy_resp is not None:
                contract_id = buy_resp["contract_id"]
                logger.info("%s: bought %s contract %s", sym, decision.contract_type
                           if hasattr(decision, "contract_type") else decision.decision,
                           contract_id)
                asyncio.create_task(_settle_and_release(
                    client, db, risk, staking, contract_id, sym, open_contracts))
            else:
                open_contracts.discard(sym)
                logger.info("%s: %s (%s)", sym, decision.reason_code, decision.explanation)

    async def _settle_and_release(client, db, risk, staking, contract_id, sym, open_set):
        try:
            await monitor_settlement(client, db, risk, staking, contract_id)
        finally:
            open_set.discard(sym)

    await asyncio.gather(*(handle_symbol(sym) for sym in usable))


def cmd_backtest(settings: Settings, args) -> None:
    import random

    from backtest.simulator import run_backtest
    from data.candles import candles_from_ticks

    logger.info("no --data-file given: generating a synthetic tick stream "
               "for a smoke run. Pass --data-file for a real backtest against "
               "collected Deriv history.")
    random.seed(0)
    price, epochs, prices, t = 1000.0, [], [], 1_700_000_000
    for _ in range(400000):
        price += random.gauss(0, 0.3) - 0.0005 * (price - 1000.0)
        price = max(price, 1.0)
        t += 1
        epochs.append(t)
        prices.append(price)
    candles = candles_from_ticks("SIM", epochs, prices,
                                 timeframe_seconds=settings["candles"]["timeframe_seconds"])
    result = run_backtest(candles, settings.raw, stake=settings.risk["base_stake"],
                          payout_multiple=args.payout_multiple,
                          warmup_bars=settings["backtest"]["warmup_bars"])
    print(result.report())


def cmd_walk_forward(settings: Settings, args) -> None:
    import random

    from backtest.simulator import walk_forward
    from data.candles import candles_from_ticks

    random.seed(0)
    price, epochs, prices, t = 1000.0, [], [], 1_700_000_000
    for _ in range(400000):
        price += random.gauss(0, 0.3) - 0.0005 * (price - 1000.0)
        price = max(price, 1.0)
        t += 1
        epochs.append(t)
        prices.append(price)
    candles = candles_from_ticks("SIM", epochs, prices,
                                 timeframe_seconds=settings["candles"]["timeframe_seconds"])
    report = walk_forward(candles, settings.raw, n_blocks=settings["backtest"]["n_blocks"],
                          stake=settings.risk["base_stake"],
                          payout_multiple=args.payout_multiple,
                          warmup_bars=settings["backtest"]["warmup_bars"])
    print(report.report())


def cmd_reconcile(settings: Settings) -> None:
    """Section 28/29's counterfactual analysis, run as a batch job. Safe to
    call repeatedly -- e.g. from a Railway cron schedule or a periodic call
    alongside the live loop; a row whose settlement candle hasn't closed
    yet is simply left for the next call."""
    db = open_database(sqlite_path=settings.storage_path)
    n = db.reconcile_rejected_signals(
        duration_bars=settings["contract"]["duration_bars_approx"])
    print(f"reconciled {n} rejected signal(s)")


def cmd_diagnostics(settings: Settings) -> None:
    db = open_database(sqlite_path=settings.storage_path)
    print("Signal reason histogram:")
    for code, n in db.reason_histogram().items():
        print(f"  {code:<32}{n}")
    print("\nTrade summary:", db.trade_summary())


def cmd_status(settings: Settings) -> None:
    print(f"mode={settings.mode} symbols={settings.symbols} "
         f"martingale_enabled={settings.staking['martingale_enabled']} "
         f"db_backend={settings.db_backend}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Deriv Reversal Intelligence System (Level 1)")
    parser.add_argument("--mode", choices=["demo", "paper", "live"], default=None)
    parser.add_argument("--backtest", action="store_true")
    parser.add_argument("--walk-forward", action="store_true")
    parser.add_argument("--online-sim", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--dashboard", action="store_true")
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--reconcile", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--payout-multiple", type=float, default=1.85,
                        dest="payout_multiple")
    args = parser.parse_args()

    try:
        settings = Settings()
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        raise SystemExit(1)

    _setup_logging(settings.log_level)

    for flag, msg in NOT_YET_IMPLEMENTED.items():
        if getattr(args, flag.replace("-", "_")):
            print(f"--{flag}: {msg}")
            return

    if args.backtest:
        cmd_backtest(settings, args)
    elif args.walk_forward:
        cmd_walk_forward(settings, args)
    elif args.diagnostics:
        cmd_diagnostics(settings)
    elif args.reconcile:
        cmd_reconcile(settings)
    elif args.status:
        cmd_status(settings)
    else:
        asyncio.run(run_live(settings))


if __name__ == "__main__":
    main()
