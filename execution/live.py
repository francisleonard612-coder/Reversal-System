"""
Live trading loop (spec Sections 7, 38, 48, 59) -- extracted from main.py
into a class so every piece of it can be exercised against a fake Deriv
client in tests.

What this adds over the previous inline loop in main.py:

- RESTART-SAFE RISK. Today's P/L, trade count and losing streak are rebuilt
  from the trade journal on startup (RiskManager.restore). Before, any
  crash or redeploy silently reset every "hard" limit to zero.
- NO ORPHANED CONTRACTS. Trades bought but not settled before a restart are
  re-attached to settlement tracking and keep their symbol blocked.
- EVERY CONFIGURED RISK CONTROL IS ACTUALLY WIRED. max_trades_per_day,
  cooldown_seconds, max_concurrent_trades, max_proposal_age_seconds and
  stale_tick_seconds were all in settings.yaml but never reached the code
  that enforces them; the staking parameters (martingale trigger/factor/
  steps, Kelly fraction) likewise.
- STALE-FEED WATCHDOG. A symbol whose tick stream goes silent is
  re-subscribed instead of the bot sitting idle indefinitely.
- REAL BALANCE, PERIODICALLY. Drawdown and stake-vs-balance checks use the
  account balance re-read from Deriv, not one that only drifts by booked P/L.
- COUNTERFACTUALS WHILE RUNNING. Rejected-signal outcomes are reconciled on
  a timer instead of only when someone remembers to run --reconcile.
- LEVEL 2 WHEN EARNED. If level2.enabled and a PROMOTED model matching the
  current config exists, trades additionally need positive expected value
  from its calibrated P(win) and the real quoted payout.
"""
from __future__ import annotations

import asyncio
import logging
import time

from data.candles import CandleBuilder, candles_from_history
from data.validation import TickValidator
from execution.executor import execute_trade, monitor_settlement
from risk.manager import RiskManager
from risk.staking import StakingEngine
from strategy.expiry import (ExpiryCalibrator, Horizon, OutcomeTracker, SetupDeduper,
                             format_table, offered_horizons, qualifying_direction)
from strategy.pipeline import SymbolPipeline

logger = logging.getLogger("reversal.live")

STATUS_LOG_INTERVAL_SECONDS = 15 * 60


def local_day_start(now: float | None = None) -> float:
    """Start of the current local day -- the same boundary RiskManager uses
    when it rolls its daily counters."""
    t = time.localtime(time.time() if now is None else now)
    return time.mktime((t.tm_year, t.tm_mon, t.tm_mday, 0, 0, 0, 0, 0, -1))


def build_risk_manager(settings) -> RiskManager:
    r = settings.risk
    return RiskManager(
        base_stake=r["base_stake"], max_stake=r["max_stake"],
        max_daily_loss=r["max_daily_loss"], max_drawdown=r["max_drawdown"],
        max_consecutive_losses=r["max_consecutive_losses"],
        max_trades_per_day=r["max_trades_per_day"],
        max_concurrent_trades=r["max_concurrent_trades"],
        cooldown_seconds=r["cooldown_seconds"])


def build_staking(settings) -> StakingEngine:
    s, r = settings.staking, settings.risk
    return StakingEngine(
        method=s["method"], base_stake=r["base_stake"], max_stake=r["max_stake"],
        kelly_fraction=s.get("kelly_fraction", 0.25),
        martingale_enabled=s["martingale_enabled"],
        martingale_factor=s.get("martingale_factor", 2.0),
        martingale_max_steps=s.get("martingale_max_steps", 3),
        min_consecutive_losses=s.get("martingale_trigger_losses", 2))


class LiveTrader:
    def __init__(self, settings, client, db, *, level2_model=None):
        self.settings = settings
        self.client = client
        self.db = db
        self.level2_model = level2_model
        self.risk = build_risk_manager(settings)
        self.staking = build_staking(settings)
        self.timeframe = settings["candles"]["timeframe_seconds"]
        self.symbols: list[str] = []
        self.builders: dict[str, CandleBuilder] = {}
        self.validators: dict[str, TickValidator] = {}
        self.pipelines: dict[str, SymbolPipeline] = {}
        self.queues: dict[str, asyncio.Queue] = {}
        self.open_symbols: set[str] = set()
        self.last_tick_at: dict[str, float] = {}
        self.last_payout_multiple: dict[str, float] = {}
        self._tasks: set[asyncio.Task] = set()
        self.started_at = time.time()

        # MC-calibrated expiry (strategy/expiry.py)
        ex = settings.expiry
        self.expiry_enabled = bool(ex["enabled"])
        self.calibrator = ExpiryCalibrator(
            horizons=settings.expiry_horizons, min_samples=ex["min_samples"],
            credible_quantile=ex["credible_quantile"], mc_draws=ex["mc_draws"],
            pool_symbols=ex["pool_symbols"])
        self.symbol_horizons: dict[str, list[Horizon]] = {}
        self._outcomes: list[dict] = []
        self.tracker = OutcomeTracker(settings.expiry_horizons, self._outcomes.append)
        self.deduper = SetupDeduper(settings["reversal"]["confirmation_window"] * self.timeframe)

    # ------------------------------------------------------------- helpers
    def _spawn(self, coro, name: str) -> asyncio.Task:
        task = asyncio.create_task(coro, name=name)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    @property
    def research_mode(self) -> bool:
        return self.settings.mode == "research"

    # ------------------------------------------------------------- startup
    async def discover_symbols(self) -> list[str]:
        usable = []
        for symbol in self.settings.symbols:
            ok, why = await self.client.verify_rise_fall_available(symbol, self.settings.currency)
            logger.info("contract check %s", why)
            if ok:
                usable.append(symbol)
            else:
                self.db.log_event("WARNING", "discovery", why)
        if not usable:
            raise RuntimeError("no configured symbol offers CALL/PUT -- refusing to run")
        self.symbols = usable
        for sym in usable:
            if self.expiry_enabled:
                try:
                    limits = await self.client.rise_fall_duration_limits(sym)
                except Exception as exc:  # noqa: BLE001 - unknown limits: offer all, proposals will tell
                    logger.warning("%s: could not read Deriv duration limits (%s)", sym, exc)
                    limits = []
                self.symbol_horizons[sym] = offered_horizons(self.settings.expiry_horizons, limits)
                logger.info("%s: candidate expiries %s (Deriv limits %s)", sym,
                            [h.label for h in self.symbol_horizons[sym]], limits or "unknown")
            self.builders[sym] = CandleBuilder(sym, self.timeframe)
            self.validators[sym] = TickValidator()
            self.pipelines[sym] = SymbolPipeline(sym, self.settings.raw)
        return usable

    def restore_state(self, balance: float) -> None:
        """Rebuild risk counters and reattach unsettled contracts."""
        self.risk.balance = balance
        self.risk.peak_balance = balance
        state = self.db.risk_state_since(local_day_start())
        open_trades = self.db.open_trades()
        self.risk.restore(daily_pnl=state["daily_pnl"], trades_today=state["trades_today"],
                          consecutive_losses=state["consecutive_losses"],
                          open_trades=len(open_trades))
        self.staking._consecutive_losses = state["consecutive_losses"]
        logger.info("restored risk state from journal: %s, %d open contract(s)%s",
                    state, len(open_trades),
                    f" -- EMERGENCY STOP: {self.risk.stop_reason}" if self.risk.emergency_stopped else "")
        for t in open_trades:
            self.open_symbols.add(t["symbol"])
            self._spawn(self._settle_and_release(t["contract_id"], t["symbol"],
                                                 t.get("duration"), t.get("duration_unit")),
                        f"settle-{t['contract_id']}")

    def refresh_calibration(self) -> None:
        since = time.time() - self.settings.expiry["lookback_days"] * 86400
        self.calibrator.load_counts(self.db.setup_outcome_counts(since))

    def flush_outcomes(self) -> None:
        if not self._outcomes:
            return
        rows, self._outcomes[:] = list(self._outcomes), []
        try:
            self.db.record_setup_outcomes(rows)
        except Exception as exc:  # noqa: BLE001
            logger.warning("could not store %d setup outcome(s): %s", len(rows), exc)

    async def seed_candles(self) -> None:
        seed_count = self.settings["candles"].get("seed_bars",
                                                  self.settings["candles"].get("max_lookback_bars", 400))
        for sym in self.symbols:
            try:
                raw_hist = await self.client.candle_history(sym, count=seed_count,
                                                            granularity=self.timeframe)
                seed = candles_from_history(sym, raw_hist, self.timeframe)
                n = self.builders[sym].seed_closed(seed)
            except Exception:
                logger.warning("%s: candle-history seed failed, starting unseeded "
                               "(warm-up will take longer)", sym, exc_info=True)
            else:
                logger.info("%s: seeded %d closed candles", sym, n)
                # Warm the percentile setup threshold on seed history (in a
                # thread: it's a few seconds of CPU and the socket must stay
                # serviced meanwhile).
                history = self.builders[sym].history()
                await asyncio.to_thread(self.pipelines[sym].warm, history)
                logger.info("%s: setup threshold warmed (%s)", sym,
                            getattr(self.pipelines[sym], "threshold_source", "absolute"))
                try:
                    for c in seed:
                        self.db.record_candle(c)
                except Exception:
                    logger.warning("%s: could not persist seeded history", sym, exc_info=True)
            self.queues[sym] = await self.client.subscribe_ticks(sym)
            self.last_tick_at[sym] = time.time()

    # --------------------------------------------------------------- loop
    async def handle_symbol(self, sym: str) -> None:
        while True:
            tick = await self.queues[sym].get()
            self.last_tick_at[sym] = time.time()
            v = self.validators[sym].validate(tick.epoch, tick.price)
            if not v.valid:
                self.db.log_event("WARNING", "data_quality", v.reason, {"symbol": sym})
                continue
            if self.expiry_enabled:
                self.tracker.on_tick(sym, tick.epoch, tick.price)
            newly_closed = self.builders[sym].add_tick(tick.epoch, tick.price)
            if newly_closed is None:
                continue
            await self.on_candle_closed(sym, newly_closed)

    async def on_candle_closed(self, sym: str, candle) -> None:
        self.db.record_candle(candle)
        pipeline = self.pipelines[sym]
        decision = pipeline.evaluate_on_close(self.builders[sym].history())
        signal_id = self.db.record_signal(decision)

        if self.expiry_enabled:
            self.flush_outcomes()
            q = qualifying_direction(decision)
            if q and self.deduper.accept(sym, q[0], q[1], candle.close_epoch):
                # follow this setup forward at every candidate expiry, traded or not
                self.tracker.start(symbol=sym, direction=q[0], confirmed=q[1],
                                   setup_epoch=candle.close_epoch,
                                   score=max(decision.bullish_score, decision.bearish_score),
                                   regime=decision.regime)

        if not decision.will_trade or sym in self.open_symbols:
            return
        if not self.client.is_connected:
            return

        direction = "bullish" if decision.decision == "TRADE_BULLISH" else "bearish"
        features = pipeline.features_for_direction(direction) if self.level2_model else None
        p_lower = None
        if self.level2_model is not None and features is not None:
            _, _, p_lower = self.level2_model.predict(features)
        stake, stake_why = self.staking.stake_for(
            balance=self.risk.balance, probability_lower_bound=p_lower,
            payout_multiple=self.last_payout_multiple.get(
                sym, self.settings["edge"]["min_payout_multiple"]))
        if stake <= 0:
            logger.info("%s: no stake (%s)", sym, stake_why)
            return

        options = None
        if self.expiry_enabled:
            options = self.calibrator.estimates(sym, direction, self.symbol_horizons.get(sym))
            if not options:
                logger.info("%s: no expiry has %d+ recorded outcomes yet -- default %s%s under "
                            "Level 1 rules", sym, self.calibrator.min_samples,
                            self.settings["contract"]["duration"],
                            self.settings["contract"]["duration_unit"])

        self.open_symbols.add(sym)
        c = self.settings["contract"]
        l2 = self.settings.level2
        ex = self.settings.expiry
        decision, edge, buy_resp = await execute_trade(
            self.client, self.db, decision=decision, signal_id=signal_id, symbol=sym,
            currency=self.settings.currency, duration=c["duration"],
            duration_unit=c["duration_unit"], stake=stake,
            min_payout_multiple=self.settings["edge"]["min_payout_multiple"],
            risk_manager=self.risk, research_mode=self.research_mode,
            max_proposal_age_seconds=self.settings.risk["max_proposal_age_seconds"],
            level2_model=self.level2_model, level2_features=features,
            level2_min_ev=l2["min_ev"], level2_require_lower_bound=l2["require_lower_bound_edge"],
            expiry_options=options, expiry_require_edge=ex["require_edge"],
            proposals_to_price=ex["proposals_to_price"])
        if edge is not None:
            self.last_payout_multiple[sym] = edge.proposal.payout_multiple
        self.db.update_signal_outcome(signal_id, decision)

        if buy_resp is not None:
            contract_id = buy_resp["contract_id"]
            logger.info("%s: bought %s contract %s stake=%.2f (%s)", sym, decision.decision,
                        contract_id, stake, edge.reason if edge else "")
            self._spawn(self._settle_and_release(contract_id, sym, buy_resp.get("_duration"),
                                                 buy_resp.get("_duration_unit")),
                        f"settle-{contract_id}")
        else:
            self.open_symbols.discard(sym)
            logger.info("%s: %s (%s)", sym, decision.reason_code, decision.explanation)

    def _expected_seconds(self, duration, unit) -> float | None:
        if duration is None or unit is None:
            return self.settings.contract_seconds()
        return Horizon(int(duration), str(unit)).approx_seconds(
            self.settings.expiry["seconds_per_tick"])

    async def _settle_and_release(self, contract_id, sym: str, duration=None, unit=None) -> None:
        try:
            poc = await monitor_settlement(
                self.client, self.db, self.risk, self.staking, contract_id,
                expected_seconds=self._expected_seconds(duration, unit),
                grace_seconds=self.settings.risk["settlement_grace_seconds"], logger=logger)
            logger.info("%s: contract %s settled %s pnl=%+.2f", sym, contract_id,
                        poc.get("status", ""), float(poc.get("profit", 0) or 0))
        finally:
            self.open_symbols.discard(sym)

    # ------------------------------------------------------- background
    async def watchdog(self, interval: float = 10.0) -> None:
        stale = self.settings.risk["stale_tick_seconds"]
        while True:
            await asyncio.sleep(interval)
            now = time.time()
            for sym in self.symbols:
                silent = now - self.last_tick_at.get(sym, now)
                if silent > stale and self.client.is_connected:
                    logger.warning("%s: no ticks for %.0fs -- re-subscribing", sym, silent)
                    self.db.log_event("WARNING", "stale_feed",
                                      f"{sym} silent {silent:.0f}s, re-subscribing")
                    try:
                        self.queues[sym] = await self.client.subscribe_ticks(sym)
                        self.last_tick_at[sym] = now
                    except Exception as exc:  # noqa: BLE001
                        logger.error("%s: re-subscribe failed: %s", sym, exc)

    async def refresh_balance(self) -> None:
        every = self.settings.operations["balance_refresh_seconds"]
        while True:
            await asyncio.sleep(every)
            try:
                self.risk.update_balance(await self.client.balance())
            except Exception as exc:  # noqa: BLE001
                logger.warning("balance refresh failed: %s", exc)

    async def reconcile_loop(self) -> None:
        every = self.settings.operations["reconcile_interval_seconds"]
        bars = self.settings["contract"]["duration_bars_approx"]
        while True:
            await asyncio.sleep(every)
            try:
                n = await asyncio.to_thread(self.db.reconcile_rejected_signals, duration_bars=bars)
                if n:
                    logger.info("reconciled %d rejected signal outcome(s)", n)
            except Exception as exc:  # noqa: BLE001
                logger.warning("reconcile pass failed: %s", exc)

    async def calibration_loop(self) -> None:
        while True:
            await asyncio.sleep(self.settings.expiry["refresh_seconds"])
            try:
                self.flush_outcomes()
                await asyncio.to_thread(self.refresh_calibration)
            except Exception as exc:  # noqa: BLE001
                logger.warning("expiry calibration refresh failed: %s", exc)

    async def status_loop(self) -> None:
        while True:
            await asyncio.sleep(STATUS_LOG_INTERVAL_SECONDS)
            logger.info("STATUS mode=%s risk=%s open=%s level2=%s", self.settings.mode,
                        self.risk.snapshot(), sorted(self.open_symbols),
                        self.level2_model.version if self.level2_model else "off")
            if self.expiry_enabled:
                logger.info("EXPIRY CALIBRATION\n%s", format_table(self.calibrator, self.symbols))

    # ---------------------------------------------------------------- run
    async def run(self) -> None:
        balance = await self.client.balance()
        await self.discover_symbols()
        self.restore_state(balance)
        if self.expiry_enabled:
            self.refresh_calibration()
            logger.info("EXPIRY CALIBRATION at start\n%s", format_table(self.calibrator, self.symbols))
        await self.seed_candles()
        logger.info("entering live loop (mode=%s, contract=%s%s, level2=%s)",
                    self.settings.mode, self.settings["contract"]["duration"],
                    self.settings["contract"]["duration_unit"],
                    self.level2_model.version if self.level2_model else "off")
        loops = [(self.watchdog(), "watchdog"), (self.refresh_balance(), "balance"),
                 (self.reconcile_loop(), "reconcile"), (self.status_loop(), "status")]
        if self.expiry_enabled:
            loops.append((self.calibration_loop(), "expiry-calibration"))
        for coro, name in loops:
            self._spawn(coro, name)
        await asyncio.gather(*(self.handle_symbol(s) for s in self.symbols))
