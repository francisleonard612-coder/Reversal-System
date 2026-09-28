"""
Main entry point and CLI (spec Section 51).

    python main.py                      live loop, mode from TRADING_MODE
    python main.py --fetch-history --days 30
                                        download real Deriv candles into the DB
    python main.py --backtest [--symbol R_100 | --data-file f.csv]
    python main.py --walk-forward [--symbol R_100 | --data-file f.csv]
    python main.py --evaluate           Level 2 model vs Level 1, held-out, no saving
    python main.py --train              fit + save Level 2 model (promoted only if it earns it)
    python main.py --reconcile          fill counterfactual outcomes for rejected signals
    python main.py --dashboard          operating summary
    python main.py --reset-risk         operator reset of the loss-streak / daily-loss latch
    python main.py --diagnostics | --status

Backtests and training run on REAL stored candles when they exist (fill them
with --fetch-history). The synthetic random-walk stream is only a smoke test
and is labeled as such in its output. --online-sim (Level 3 online learning)
is not implemented and says so rather than doing something fake (Section 62).
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import sys
import time

from config.loader import ConfigError, Settings
from data.postgres import open_database
from deriv.client import DerivAuthError, DerivClient

logger = logging.getLogger("reversal")

NOT_YET_IMPLEMENTED = {
    "online-sim": "Section 42's online simulator is a Level 3 feature (online learning) "
                  "and is not implemented.",
}


def _setup_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)-8s %(name)s %(message)s")


def _make_client(settings: Settings) -> DerivClient:
    d = settings.deriv
    return DerivClient(
        app_id=d["app_id"], api_token=d["api_token"], ws_url=d["ws_url"],
        request_timeout=d["request_timeout"], api_base_url=d["api_base_url"],
        auth_mode=d["auth_mode"], account_id=d["account_id"] or None,
        use_real_account=settings.use_real_account,
        max_requests_per_minute=d["max_requests_per_minute"])


def _open_db(settings: Settings):
    # DB_BACKEND=postgres routes to Supabase; sqlite is fine locally but on
    # Railway the container filesystem is ephemeral.
    return open_database(sqlite_path=settings.storage_path)


# ---------------------------------------------------------------- live

async def run_live(settings: Settings) -> None:
    from execution.live import LiveTrader
    from models.level2 import load_for_live

    db = _open_db(settings)
    client = _make_client(settings)
    try:
        await client.connect()
    except DerivAuthError as exc:
        logger.error("authentication failed: %s", exc)
        raise SystemExit(1)

    model = None
    if settings.level2["enabled"]:
        model = load_for_live(settings.level2["model_path"], settings.raw, logger)
    trader = LiveTrader(settings, client, db, level2_model=model)
    logger.info("connected, mode=%s symbols=%s", settings.mode, settings.symbols)
    await trader.run()


# ------------------------------------------------------------ history

async def _fetch_history(settings: Settings, days: float) -> None:
    from data.candles import candles_from_history

    db = _open_db(settings)
    client = _make_client(settings)
    await client.connect()
    tf = settings["candles"]["timeframe_seconds"]
    total = int(days * 86400 / tf)
    try:
        for sym in settings.symbols:
            raw = await client.candle_history_paged(sym, total, granularity=tf)
            candles = candles_from_history(sym, raw, tf)
            for c in candles:
                db.record_candle(c)
            span = ((candles[-1].close_epoch - candles[0].close_epoch) / 86400) if candles else 0
            print(f"{sym}: stored {len(candles)} candles ({span:.1f} days)")
    finally:
        await client.close()


def _load_csv_candles(path: str, symbol: str, tf: int) -> list:
    """CSV with columns epoch|close_epoch|time, open, high, low, close."""
    from data.candles import Candle
    out = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            r = {k.strip().lower(): v for k, v in row.items()}
            ep = r.get("close_epoch") or r.get("epoch") or r.get("time")
            ep = int(float(ep))
            out.append(Candle(symbol=r.get("symbol") or symbol, open_epoch=ep - tf + 1,
                              close_epoch=ep, open=float(r["open"]), high=float(r["high"]),
                              low=float(r["low"]), close=float(r["close"]), n_ticks=0,
                              is_closed=True, timeframe_seconds=tf))
    out.sort(key=lambda c: c.close_epoch)
    return out


def _synthetic_candles(settings: Settings) -> list:
    import random
    from data.candles import candles_from_ticks
    random.seed(0)
    price, epochs, prices, t = 1000.0, [], [], 1_700_000_000
    for _ in range(400000):
        price += random.gauss(0, 0.3) - 0.0005 * (price - 1000.0)
        price = max(price, 1.0)
        t += 1
        epochs.append(t)
        prices.append(price)
    return candles_from_ticks("SIM", epochs, prices,
                              timeframe_seconds=settings["candles"]["timeframe_seconds"])


def _candles_for_research(settings: Settings, args) -> tuple[list, str]:
    tf = settings["candles"]["timeframe_seconds"]
    if args.data_file:
        return _load_csv_candles(args.data_file, args.symbol or "CSV", tf), f"CSV {args.data_file}"
    db = _open_db(settings)
    sym = args.symbol or (settings.symbols[0] if settings.symbols else None)
    candles = db.load_candles(sym) if sym else []
    if len(candles) >= settings["backtest"]["warmup_bars"] * 2:
        return candles, f"{len(candles)} stored {sym} candles"
    print(f"!! only {len(candles)} stored candles for {sym} -- falling back to a SYNTHETIC "
          f"random walk. These results say nothing about real markets. Run "
          f"`python main.py --fetch-history --days 30` first.\n")
    return _synthetic_candles(settings), "SYNTHETIC random walk (smoke test only)"


def cmd_backtest(settings: Settings, args) -> None:
    from backtest.simulator import run_backtest
    candles, source = _candles_for_research(settings, args)
    print(f"data: {source}")
    result = run_backtest(candles, settings.raw, stake=settings.risk["base_stake"],
                          payout_multiple=args.payout_multiple,
                          warmup_bars=settings["backtest"]["warmup_bars"])
    print(result.report())
    print(f"break-even win rate at {args.payout_multiple:.2f}x: {1 / args.payout_multiple:.4f}")


def cmd_walk_forward(settings: Settings, args) -> None:
    from backtest.simulator import walk_forward
    candles, source = _candles_for_research(settings, args)
    print(f"data: {source}")
    report = walk_forward(candles, settings.raw, n_blocks=settings["backtest"]["n_blocks"],
                          stake=settings.risk["base_stake"],
                          payout_multiple=args.payout_multiple,
                          warmup_bars=settings["backtest"]["warmup_bars"])
    print(report.report())


# ------------------------------------------------------------- level 2

def _level2_dataset(settings: Settings, args):
    from models.level2 import build_dataset
    db = _open_db(settings)
    symbols = [args.symbol] if args.symbol else settings.symbols
    by_sym = {s: db.load_candles(s) for s in symbols}
    by_sym = {s: c for s, c in by_sym.items() if c}
    if not by_sym:
        print("no stored candles -- run `python main.py --fetch-history --days 30` first")
        raise SystemExit(1)
    print("replaying stored candles through the live feature pipeline: "
          + ", ".join(f"{s}={len(c)}" for s, c in by_sym.items()))
    t0 = time.time()
    ds = build_dataset(by_sym, settings.raw)
    print(f"{len(ds.y)} labeled examples in {time.time() - t0:.0f}s, "
          f"base win rate {ds.y.mean():.4f}" if len(ds.y) else "no labeled examples")
    return ds, db


def _level2_report(settings: Settings, ds):
    from models.level2 import evaluate
    l2 = settings.level2
    try:
        return evaluate(ds, n_blocks=l2["n_blocks"], min_block_size=l2["min_block_size"],
                        corr_floor=l2["promotion_corr_floor"])
    except ValueError as exc:
        print(f"cannot evaluate: {exc}")
        raise SystemExit(1)


def cmd_evaluate(settings: Settings, args) -> None:
    ds, _ = _level2_dataset(settings, args)
    print(_level2_report(settings, ds).report())


def cmd_train(settings: Settings, args) -> None:
    from models.level2 import fit_final
    ds, db = _level2_dataset(settings, args)
    report = _level2_report(settings, ds)
    print(report.report())
    model = fit_final(ds, settings.raw, report)
    path = settings.level2["model_path"]
    model.save(path)
    db.record_model_version(
        model_id=model.model_id, version=model.version,
        period_start=model.trained_on["epoch_start"], period_end=model.trained_on["epoch_end"],
        features=model.feature_names,
        parameters={"trained_on": model.trained_on, "fingerprint": model.fingerprint,
                    "failures": model.promotion_failures},
        calibration_method="platt", promotion_status=model.promotion_status)
    print(f"\nsaved {model.promotion_status} model {model.version} -> {path}")
    if model.promoted:
        print("live use requires level2.enabled: true (or LEVEL2_ENABLED=true). Keep "
              "TRADING_MODE=research/demo and watch v_level2_calibration before going live.")
    else:
        print("the live bot will refuse this model; it's saved only for inspection.")


# ----------------------------------------------------------- operations

def cmd_reconcile(settings: Settings) -> None:
    db = _open_db(settings)
    n = db.reconcile_rejected_signals(duration_bars=settings["contract"]["duration_bars_approx"])
    print(f"reconciled {n} rejected signal(s)")


def cmd_dashboard(settings: Settings) -> None:
    from execution.live import local_day_start
    db = _open_db(settings)
    d = db.dashboard(local_day_start())
    t, today = d["trades"], d["today"]
    print(f"MODE {settings.mode}  symbols={settings.symbols}  "
          f"contract={settings['contract']['duration']}{settings['contract']['duration_unit']}  "
          f"level2={'on' if settings.level2['enabled'] else 'off'}")
    print(f"\nTRADES  settled={t['trades']} wins={t['wins']} win_rate={t['win_rate']:.4f} "
          f"pnl={t['pnl']:+.2f}  open={d['open_trades']}" if t["trades"] else
          f"\nTRADES  none settled yet  open={d['open_trades']}")
    print(f"TODAY   pnl={today['daily_pnl']:+.2f} trades={today['trades_today']} "
          f"losing_streak={today['consecutive_losses']}  "
          f"(limits: daily loss {settings.risk['max_daily_loss']}, "
          f"streak {settings.risk['max_consecutive_losses']}, "
          f"trades/day {settings.risk['max_trades_per_day']})")
    print("\nSIGNALS by reason:")
    for code, n in list(d["signals_by_reason"].items())[:12]:
        print(f"  {code:<34}{n}")
    cf = d["counterfactual"]
    rate = f"{cf['would_have_won_rate']:.4f}" if cf["would_have_won_rate"] is not None else "n/a"
    print(f"\nCOUNTERFACTUAL  rejected signals reconciled={cf['evaluated']} would_have_won={rate}")
    print(f"CANDLES stored  {d['candles_stored']}")
    lm = d["latest_model"]
    print(f"LEVEL 2 latest model: {lm['version']} {lm['promotion_status']}" if lm
          else "LEVEL 2 no model trained yet")
    if d["predictions"]["n"]:
        print(f"        predictions={d['predictions']['n']} "
              f"mean P(win)={d['predictions']['mean_probability']:.4f}")
    if d["recent_errors"]:
        print("\nRECENT WARNINGS/ERRORS:")
        for e in d["recent_errors"]:
            print(f"  {time.strftime('%m-%d %H:%M', time.localtime(e['ts']))} "
                  f"{e['level']:<8}{e['category']}: {e['message'][:90]}")


def cmd_reset_risk(settings: Settings, note: str) -> None:
    db = _open_db(settings)
    db.record_risk_reset(note)
    print("risk reset recorded: the losing streak restarts from zero on next start. "
          "Today's P/L still counts toward the daily loss limit.")


def cmd_diagnostics(settings: Settings) -> None:
    db = _open_db(settings)
    print("Signal reason histogram:")
    for code, n in db.reason_histogram().items():
        print(f"  {code:<32}{n}")
    print("\nTrade summary:", db.trade_summary())


def cmd_status(settings: Settings) -> None:
    print(f"mode={settings.mode} symbols={settings.symbols} "
          f"contract={settings['contract']['duration']}{settings['contract']['duration_unit']} "
          f"horizon_matches_research={settings.horizon_matches_research} "
          f"martingale_enabled={settings.staking['martingale_enabled']} "
          f"level2_enabled={settings.level2['enabled']} db_backend={settings.db_backend}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Deriv Reversal Intelligence System")
    parser.add_argument("--backtest", action="store_true")
    parser.add_argument("--walk-forward", action="store_true")
    parser.add_argument("--fetch-history", action="store_true")
    parser.add_argument("--days", type=float, default=30.0)
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--data-file", default=None, dest="data_file")
    parser.add_argument("--online-sim", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--dashboard", action="store_true")
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--reconcile", action="store_true")
    parser.add_argument("--reset-risk", action="store_true", dest="reset_risk")
    parser.add_argument("--note", default="")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--payout-multiple", type=float, default=1.85, dest="payout_multiple")
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

    if args.fetch_history:
        asyncio.run(_fetch_history(settings, args.days))
    elif args.backtest:
        cmd_backtest(settings, args)
    elif args.walk_forward:
        cmd_walk_forward(settings, args)
    elif args.evaluate:
        cmd_evaluate(settings, args)
    elif args.train:
        cmd_train(settings, args)
    elif args.dashboard:
        cmd_dashboard(settings)
    elif args.diagnostics:
        cmd_diagnostics(settings)
    elif args.reconcile:
        cmd_reconcile(settings)
    elif args.reset_risk:
        cmd_reset_risk(settings, args.note)
    elif args.status:
        cmd_status(settings)
    else:
        asyncio.run(run_live(settings))


if __name__ == "__main__":
    main()
