"""
Database (spec Section 49). SQLite for now; supabase/schema.sql mirrors this
exactly for the PostgreSQL migration path, same as the sibling Even/Odd bot.

Tables for candles/features/signals/trades/outcomes are created now even
though Level 1 populates only a subset of their columns -- model_versions,
model_performance, regime_history's drift-relevant columns, and
learning_events stay empty until Level 2/3 exist, but the schema is right
the first time rather than migrated twice.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

SCHEMA_VERSION = 5

SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL);

CREATE TABLE IF NOT EXISTS ticks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL, epoch REAL NOT NULL, price REAL NOT NULL,
    received_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ticks_symbol_epoch ON ticks(symbol, epoch);

CREATE TABLE IF NOT EXISTS candles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL, timeframe_seconds INTEGER NOT NULL,
    open_epoch INTEGER NOT NULL, close_epoch INTEGER NOT NULL,
    open REAL NOT NULL, high REAL NOT NULL, low REAL NOT NULL, close REAL NOT NULL,
    n_ticks INTEGER NOT NULL, has_volume INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_candles_symbol_close ON candles(symbol, close_epoch);

CREATE TABLE IF NOT EXISTS features (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL, symbol TEXT NOT NULL, candle_index INTEGER,
    atr REAL, atr_pct REAL, volatility_regime TEXT,
    rsi REAL, rsi_slope REAL, macd_hist REAL, roc REAL, ema_slope REAL,
    stretch_zscore REAL, stretch_score REAL,
    regime TEXT, adx REAL,
    detail TEXT     -- JSON: everything else (patterns, S/R zones, divergence)
);
CREATE INDEX IF NOT EXISTS idx_features_symbol_ts ON features(symbol, ts);

CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL, symbol TEXT NOT NULL,
    decision TEXT NOT NULL, reason_code TEXT NOT NULL, explanation TEXT,
    regime TEXT, volatility_regime TEXT,
    bullish_score REAL, bearish_score REAL, threshold_used REAL,
    regime_multiplier REAL, confirmed INTEGER, confirmation_reason TEXT,
    payout_multiple REAL
);
CREATE INDEX IF NOT EXISTS idx_signals_symbol_ts ON signals(symbol, ts);
CREATE INDEX IF NOT EXISTS idx_signals_reason ON signals(reason_code);

CREATE TABLE IF NOT EXISTS rejected_signals (
    -- Section 28/29: rejected setups tracked for counterfactual analysis.
    -- Populated at Level 1 (every NO_TRADE signal-level row). outcome_price,
    -- outcome_known_at, entry_price and would_have_won are filled in LATER
    -- by reconcile_rejected_signals() -- never at write time, so no future
    -- information leaks into the original row (Section 29).
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id INTEGER REFERENCES signals(id),
    ts REAL NOT NULL, symbol TEXT NOT NULL, reason_code TEXT NOT NULL,
    would_be_direction TEXT,
    entry_price REAL, outcome_price REAL, outcome_known_at REAL,
    would_have_won INTEGER, outcome_evaluated INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_rejected_unevaluated ON rejected_signals(outcome_evaluated) WHERE outcome_evaluated = 0;

CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id INTEGER REFERENCES signals(id),
    ts REAL NOT NULL, symbol TEXT NOT NULL, contract_id INTEGER UNIQUE,
    idempotency_key TEXT UNIQUE,
    contract_type TEXT, stake REAL, payout REAL, buy_price REAL,
    entry_spot REAL, exit_spot REAL,
    won INTEGER, pnl REAL, settled_at REAL, error TEXT
);
CREATE INDEX IF NOT EXISTS idx_trades_ts ON trades(ts);
CREATE INDEX IF NOT EXISTS idx_trades_open ON trades(symbol) WHERE settled_at IS NULL;

CREATE TABLE IF NOT EXISTS outcomes (
    -- Section 30: outcome attribution. Populated after settlement; category
    -- is left NULL (not a fabricated guess) when evidence is insufficient.
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id INTEGER REFERENCES trades(id),
    ts REAL NOT NULL, category TEXT, note TEXT
);

CREATE TABLE IF NOT EXISTS predictions (
    -- Reserved for Level 2's model probability output. Empty at Level 1.
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL, symbol TEXT NOT NULL, model_id TEXT,
    probability REAL, calibrated_probability REAL, model_version TEXT
);

CREATE TABLE IF NOT EXISTS model_versions (
    -- Reserved for Level 2/3 champion/challenger. Empty at Level 1.
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id TEXT NOT NULL, version TEXT NOT NULL,
    created_at REAL NOT NULL, training_period_start REAL, training_period_end REAL,
    features TEXT, parameters TEXT, calibration_method TEXT,
    promotion_status TEXT DEFAULT 'CANDIDATE'
);

CREATE TABLE IF NOT EXISTS model_performance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL, model_id TEXT, symbol TEXT,
    n INTEGER, brier REAL, log_loss REAL, accuracy REAL
);

CREATE TABLE IF NOT EXISTS regime_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL, symbol TEXT NOT NULL, regime TEXT NOT NULL,
    volatility_regime TEXT, adx REAL
);
CREATE INDEX IF NOT EXISTS idx_regime_history_symbol_ts ON regime_history(symbol, ts);

CREATE TABLE IF NOT EXISTS learning_events (
    -- Reserved for Level 3. Empty at Level 1.
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL, event_type TEXT, detail TEXT
);

CREATE TABLE IF NOT EXISTS system_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL, level TEXT, category TEXT, message TEXT, detail TEXT
);
CREATE INDEX IF NOT EXISTS idx_system_events_ts ON system_events(ts);

CREATE TABLE IF NOT EXISTS setup_outcomes (
    -- Real outcome of every qualifying setup at every candidate expiry
    -- (strategy/expiry.py). The evidence the MC expiry calibration uses.
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL, symbol TEXT NOT NULL, setup_epoch INTEGER NOT NULL,
    direction TEXT NOT NULL, confirmed INTEGER NOT NULL, score REAL, regime TEXT,
    horizon TEXT NOT NULL, entry_price REAL, exit_price REAL, won INTEGER NOT NULL,
    source TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_setup_outcomes_unique
    ON setup_outcomes(symbol, setup_epoch, direction, horizon);
CREATE INDEX IF NOT EXISTS idx_setup_outcomes_epoch ON setup_outcomes(setup_epoch);

CREATE TABLE IF NOT EXISTS parameter_changes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL, parameter TEXT NOT NULL, old_value TEXT, new_value TEXT,
    reason TEXT
);
"""


class Database:
    def __init__(self, path: str = "data/reversal.db"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._migrate()

    #: v1 -> v2: two columns added to rejected_signals so
    #: reconcile_rejected_signals() has somewhere to write. ALTER TABLE ADD
    #: COLUMN is the only cheap schema change SQLite supports; anything else
    #: would mean rewriting the whole table on every startup.
    _V2_COLUMNS = (
        ("entry_price", "REAL"),
        ("would_have_won", "INTEGER"),
    )

    _V4_PREDICTION_COLUMNS = (
        ("signal_id", "INTEGER"),
        ("probability_lower", "REAL"),
        ("payout_multiple", "REAL"),
        ("expected_value", "REAL"),
    )

    def _migrate(self) -> None:
        self.conn.executescript(SCHEMA)
        existing = {r["name"] for r in
                   self.conn.execute("PRAGMA table_info(rejected_signals)")}
        for col, decl in self._V2_COLUMNS:
            if col not in existing:
                self.conn.execute(
                    f"ALTER TABLE rejected_signals ADD COLUMN {col} {decl}")

        # v2 -> v3: main.py now persists candles seeded from tick_history on
        # startup, not just ones closed during the live loop -- and every
        # restart re-seeds from the last 5000 ticks, which overlaps candles
        # already saved from before the restart. Without a uniqueness
        # guard, that duplicates rows on every redeploy, and duplicate rows
        # corrupt reconcile_rejected_signals()'s "N candles later" query,
        # which counts ROWS, not distinct closes. Dedup first -- a pre-fix
        # deployment may already have written some via the live loop across
        # restarts of a different kind -- THEN add the index; the other
        # order raises a UNIQUE violation on the very duplicates being
        # fixed. Both steps are safe to run again on every startup: dedup
        # deletes nothing once there are no duplicates, and IF NOT EXISTS
        # skips index creation once it exists.
        self.conn.execute(
            "DELETE FROM candles WHERE id NOT IN "
            "(SELECT MIN(id) FROM candles GROUP BY symbol, close_epoch)")
        self.conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_candles_unique "
            "ON candles(symbol, close_epoch)")

        # v3 -> v4: Level 2 predictions are linked to the signal they priced
        # and carry the economics they were judged on.
        pred_cols = {r["name"] for r in self.conn.execute("PRAGMA table_info(predictions)")}
        for col, decl in self._V4_PREDICTION_COLUMNS:
            if col not in pred_cols:
                self.conn.execute(f"ALTER TABLE predictions ADD COLUMN {col} {decl}")

        # v4 -> v5: each trade records the expiry it was actually bought with
        # (chosen per trade by the MC expiry calibration).
        trade_cols = {r["name"] for r in self.conn.execute("PRAGMA table_info(trades)")}
        for col, decl in (("duration", "INTEGER"), ("duration_unit", "TEXT")):
            if col not in trade_cols:
                self.conn.execute(f"ALTER TABLE trades ADD COLUMN {col} {decl}")

        row = self.conn.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
        if row is None:
            self.conn.execute("INSERT INTO schema_version(version) VALUES (?)",
                              (SCHEMA_VERSION,))
        else:
            self.conn.execute("UPDATE schema_version SET version=?", (SCHEMA_VERSION,))
        self.conn.commit()

    def is_healthy(self) -> bool:
        try:
            self.conn.execute("SELECT 1").fetchone()
            return True
        except Exception:
            return False

    def close(self) -> None:
        self.conn.close()

    # -- writes ---------------------------------------------------------

    def record_candle(self, c) -> int:
        # INSERT OR IGNORE against idx_candles_unique (symbol, close_epoch):
        # main.py seeds candles from tick_history on every startup, which
        # overlaps whatever was already persisted before a restart. This is
        # what makes calling it again on the same candle a no-op instead of
        # a duplicate row -- see _migrate()'s v3 comment for why that matters.
        cur = self.conn.execute(
            """INSERT OR IGNORE INTO candles (symbol, timeframe_seconds, open_epoch,
               close_epoch, open, high, low, close, n_ticks, has_volume)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (c.symbol, c.timeframe_seconds, c.open_epoch, c.close_epoch,
             c.open, c.high, c.low, c.close, c.n_ticks, int(c.has_volume)))
        self.conn.commit()
        return cur.lastrowid

    def record_signal(self, decision) -> int:
        cur = self.conn.execute(
            """INSERT INTO signals (ts, symbol, decision, reason_code,
               explanation, regime, volatility_regime, bullish_score,
               bearish_score, threshold_used, regime_multiplier, confirmed,
               confirmation_reason, payout_multiple)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (decision.timestamp, decision.symbol, decision.decision,
             decision.reason_code, decision.explanation, decision.regime,
             decision.volatility_regime, decision.bullish_score,
             decision.bearish_score, decision.threshold_used,
             decision.regime_multiplier, int(decision.confirmed),
             decision.confirmation_reason, decision.payout_multiple))
        signal_id = cur.lastrowid
        if not decision.will_trade:
            direction = ("bullish" if decision.bullish_score >= decision.bearish_score
                        else "bearish")
            self.conn.execute(
                """INSERT INTO rejected_signals (signal_id, ts, symbol,
                   reason_code, would_be_direction)
                   VALUES (?,?,?,?,?)""",
                (signal_id, decision.timestamp, decision.symbol,
                 decision.reason_code, direction))
        self.conn.commit()
        return signal_id

    def update_signal_outcome(self, signal_id: int, decision) -> None:
        """Corrects the row record_signal() wrote BEFORE execute_trade ran.

        record_signal() is called with the signal-level decision -- before a
        proposal is even fetched -- because trades.signal_id needs a real
        foreign key to point at, and that row has to exist before
        execution can start. But a signal-level TRADE_BULLISH/TRADE_BEARISH
        can still be downgraded afterward by economics, risk, a bad
        proposal, or (in research mode) the research-mode gate itself --
        and without this call, `signals` would permanently show a trade
        that was never actually attempted. Called once, immediately after
        execute_trade() returns, with whatever decision it settled on.

        If the final decision is NO_TRADE, this also writes the
        rejected_signals row record_signal() skipped (it only writes one
        when the decision was ALREADY NO_TRADE at that earlier point) --
        so a setup that qualified and confirmed but was refused by
        economics or risk still lands in the counterfactual data, which is
        exactly the kind of case Level 2 most needs to see: not just
        "did the model's declines usually get skipped," but "was declining
        THIS one correct."
        """
        self.conn.execute(
            "UPDATE signals SET decision=?, reason_code=?, explanation=?, "
            "payout_multiple=? WHERE id=?",
            (decision.decision, decision.reason_code, decision.explanation,
             decision.payout_multiple, signal_id))
        if not decision.will_trade:
            direction = ("bullish" if decision.bullish_score >= decision.bearish_score
                        else "bearish")
            self.conn.execute(
                """INSERT INTO rejected_signals (signal_id, ts, symbol,
                   reason_code, would_be_direction)
                   VALUES (?,?,?,?,?)""",
                (signal_id, decision.timestamp, decision.symbol,
                 decision.reason_code, direction))
        self.conn.commit()

    def record_regime(self, symbol: str, regime_snap, volatility_regime: str) -> None:
        self.conn.execute(
            "INSERT INTO regime_history (ts, symbol, regime, volatility_regime, adx) "
            "VALUES (?,?,?,?,?)",
            (time.time(), symbol, regime_snap.regime, volatility_regime, regime_snap.adx))
        self.conn.commit()

    def record_trade_open(self, *, signal_id: int, symbol: str, contract_id: int,
                          idempotency_key: str, contract_type: str, stake: float,
                          payout: float, buy_price: float, entry_spot: float,
                          duration: int | None = None, duration_unit: str | None = None) -> int:
        cur = self.conn.execute(
            """INSERT INTO trades (signal_id, ts, symbol, contract_id,
               idempotency_key, contract_type, stake, payout, buy_price, entry_spot,
               duration, duration_unit)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (signal_id, time.time(), symbol, contract_id, idempotency_key,
             contract_type, stake, payout, buy_price, entry_spot, duration, duration_unit))
        self.conn.commit()
        return cur.lastrowid

    def record_trade_result(self, contract_id: int, *, won: bool, pnl: float,
                            exit_spot: float | None = None,
                            error: str | None = None) -> bool:
        """Idempotent: settles a trade at most once. Returns True only for
        the call that actually settled it, so risk/staking counters are
        updated exactly once even if settlement is observed twice (stream +
        poll, or again after a restart)."""
        cur = self.conn.execute(
            "UPDATE trades SET won=?, pnl=?, exit_spot=?, settled_at=?, error=? "
            "WHERE contract_id=? AND settled_at IS NULL",
            (int(won), pnl, exit_spot, time.time(), error, contract_id))
        self.conn.commit()
        return cur.rowcount > 0

    def open_trades(self) -> list[dict]:
        """Bought but not yet settled -- re-attached to settlement tracking
        on startup so a restart never orphans a live contract."""
        rows = self.conn.execute(
            "SELECT * FROM trades WHERE settled_at IS NULL AND contract_id IS NOT NULL "
            "ORDER BY ts").fetchall()
        return [dict(r) for r in rows]

    def known_contract_ids(self) -> set:
        return {r[0] for r in self.conn.execute(
            "SELECT contract_id FROM trades WHERE contract_id IS NOT NULL")}

    def record_risk_reset(self, note: str = "") -> None:
        self.log_event("INFO", "risk_reset", note or "operator reset")

    def risk_state_since(self, day_start: float) -> dict:
        """Today's realized P/L and trade count, plus the current losing
        streak counted only since the last operator risk reset."""
        row = self.conn.execute(
            "SELECT COALESCE(SUM(pnl),0) pnl FROM trades WHERE settled_at >= ?",
            (day_start,)).fetchone()
        n_today = self.conn.execute(
            "SELECT COUNT(*) FROM trades WHERE ts >= ?", (day_start,)).fetchone()[0]
        reset = self.conn.execute(
            "SELECT MAX(ts) FROM system_events WHERE category='risk_reset'").fetchone()[0] or 0.0
        streak = 0
        for r in self.conn.execute(
                "SELECT won FROM trades WHERE settled_at IS NOT NULL AND settled_at > ? "
                "ORDER BY settled_at DESC LIMIT 200", (reset,)):
            if r[0]:
                break
            streak += 1
        return {"daily_pnl": float(row["pnl"] or 0.0), "trades_today": int(n_today),
                "consecutive_losses": streak}

    def candle_symbols(self) -> list[str]:
        return [r[0] for r in self.conn.execute(
            "SELECT DISTINCT symbol FROM candles ORDER BY symbol")]

    def load_candles(self, symbol: str, limit: int | None = None) -> list:
        """Stored closed candles for one symbol, oldest first -- the input for
        real-data backtests and Level 2 training."""
        from data.candles import Candle
        q = ("SELECT * FROM candles WHERE symbol=? ORDER BY close_epoch DESC"
             + (" LIMIT ?" if limit else ""))
        args = (symbol, limit) if limit else (symbol,)
        rows = list(self.conn.execute(q, args))[::-1]
        return [Candle(symbol=r["symbol"], open_epoch=r["open_epoch"],
                       close_epoch=r["close_epoch"], open=r["open"], high=r["high"],
                       low=r["low"], close=r["close"], n_ticks=r["n_ticks"],
                       is_closed=True, has_volume=bool(r["has_volume"]),
                       timeframe_seconds=r["timeframe_seconds"]) for r in rows]

    def record_prediction(self, *, signal_id: int | None, symbol: str, model_id: str,
                          model_version: str, probability: float,
                          calibrated_probability: float, probability_lower: float,
                          payout_multiple: float | None, expected_value: float | None) -> None:
        self.conn.execute(
            "INSERT INTO predictions (ts, symbol, model_id, probability, "
            "calibrated_probability, model_version, signal_id, probability_lower, "
            "payout_multiple, expected_value) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (time.time(), symbol, model_id, probability, calibrated_probability,
             model_version, signal_id, probability_lower, payout_multiple, expected_value))
        self.conn.commit()

    def record_model_version(self, *, model_id: str, version: str, period_start: float,
                             period_end: float, features: list, parameters: dict,
                             calibration_method: str, promotion_status: str) -> None:
        self.conn.execute(
            "INSERT INTO model_versions (model_id, version, created_at, "
            "training_period_start, training_period_end, features, parameters, "
            "calibration_method, promotion_status) VALUES (?,?,?,?,?,?,?,?,?)",
            (model_id, version, time.time(), period_start, period_end,
             json.dumps(features), json.dumps(parameters), calibration_method,
             promotion_status))
        self.conn.commit()

    def record_setup_outcomes(self, rows: list[dict]) -> int:
        """Idempotent on (symbol, setup_epoch, direction, horizon): replaying
        the same history twice, or a live outcome for a candle already
        replayed, adds nothing."""
        cur = self.conn.executemany(
            "INSERT OR IGNORE INTO setup_outcomes (ts, symbol, setup_epoch, direction, "
            "confirmed, score, regime, horizon, entry_price, exit_price, won, source) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            [(time.time(), r["symbol"], r["setup_epoch"], r["direction"], int(r["confirmed"]),
              r.get("score"), r.get("regime"), r["horizon"], r.get("entry_price"),
              r.get("exit_price"), int(r["won"]), r["source"]) for r in rows])
        self.conn.commit()
        return cur.rowcount

    def setup_outcome_counts(self, since_epoch: float = 0) -> list[tuple]:
        """(symbol, direction, horizon, confirmed, wins, n) for the calibrator."""
        return [tuple(r) for r in self.conn.execute(
            "SELECT symbol, direction, horizon, confirmed, SUM(won), COUNT(*) "
            "FROM setup_outcomes WHERE setup_epoch >= ? "
            "GROUP BY symbol, direction, horizon, confirmed", (since_epoch,))]

    def dashboard(self, day_start: float) -> dict:
        """One-call operating summary for `--dashboard`."""
        c = self.conn
        cf = c.execute(
            "SELECT COUNT(*) n, AVG(would_have_won) r FROM rejected_signals "
            "WHERE outcome_evaluated=1").fetchone()
        preds = c.execute(
            "SELECT COUNT(*) n, AVG(calibrated_probability) p FROM predictions").fetchone()
        last_model = c.execute(
            "SELECT version, promotion_status, created_at FROM model_versions "
            "ORDER BY id DESC LIMIT 1").fetchone()
        return {
            "trades": self.trade_summary(),
            "today": self.risk_state_since(day_start),
            "open_trades": len(self.open_trades()),
            "signals_by_reason": self.reason_histogram(),
            "candles_stored": {s: c.execute("SELECT COUNT(*) FROM candles WHERE symbol=?",
                                            (s,)).fetchone()[0] for s in self.candle_symbols()},
            "counterfactual": {"evaluated": cf["n"] or 0,
                               "would_have_won_rate": cf["r"]},
            "predictions": {"n": preds["n"] or 0, "mean_probability": preds["p"]},
            "latest_model": dict(last_model) if last_model else None,
            "recent_errors": [dict(r) for r in c.execute(
                "SELECT ts, level, category, message FROM system_events "
                "WHERE level IN ('ERROR','CRITICAL','WARNING') ORDER BY ts DESC LIMIT 10")],
        }

    def log_event(self, level: str, category: str, message: str,
                 detail: dict | None = None) -> None:
        try:
            self.conn.execute(
                "INSERT INTO system_events (ts, level, category, message, detail) "
                "VALUES (?,?,?,?,?)",
                (time.time(), level, category, message,
                 json.dumps(detail) if detail else None))
            self.conn.commit()
        except Exception:
            pass   # logging must never take down an otherwise healthy bot

    def reconcile_rejected_signals(self, *, duration_bars: int = 5,
                                   batch_size: int = 5000) -> int:
        """Section 28/29's counterfactual analysis, made concrete.

        For every unevaluated rejected signal, finds the candle that was
        current at decision time (the last one CLOSED at or before the
        signal's timestamp) and the candle `duration_bars` closes after it
        for the same symbol -- the same candle-close approximation the
        backtest uses (config's contract.duration_bars_approx), so the two
        never silently disagree about what "settled" means. A row whose
        settlement candle doesn't exist yet (not enough time has passed) is
        left alone and picked up on a later call -- this is safe to run
        repeatedly and cheaply, e.g. on a schedule.

        Returns the number of rows reconciled this call.
        """
        rows = self.conn.execute(
            "SELECT id, symbol, ts, would_be_direction FROM rejected_signals "
            "WHERE outcome_evaluated = 0 ORDER BY ts LIMIT ?",
            (batch_size,)).fetchall()
        n = 0
        for row in rows:
            entry = self.conn.execute(
                "SELECT close, close_epoch FROM candles WHERE symbol=? AND "
                "close_epoch <= ? ORDER BY close_epoch DESC LIMIT 1",
                (row["symbol"], row["ts"])).fetchone()
            if entry is None:
                continue   # no candle history at/before this signal yet -- try later
            settlement = self.conn.execute(
                "SELECT close, close_epoch FROM candles WHERE symbol=? AND "
                "close_epoch > ? ORDER BY close_epoch ASC LIMIT 1 OFFSET ?",
                (row["symbol"], entry["close_epoch"], duration_bars - 1)).fetchone()
            if settlement is None:
                continue   # not enough candles have closed since this signal yet

            if row["would_be_direction"] == "bullish":
                would_have_won = settlement["close"] > entry["close"]
            else:
                would_have_won = settlement["close"] < entry["close"]

            self.conn.execute(
                "UPDATE rejected_signals SET entry_price=?, outcome_price=?, "
                "outcome_known_at=?, would_have_won=?, outcome_evaluated=1 "
                "WHERE id=?",
                (entry["close"], settlement["close"], settlement["close_epoch"],
                 int(would_have_won), row["id"]))
            n += 1
        self.conn.commit()
        return n

    # -- reads ------------------------------------------------------------

    def reason_histogram(self, symbol: str | None = None) -> dict:
        q = "SELECT reason_code, COUNT(*) n FROM signals"
        args = ()
        if symbol:
            q += " WHERE symbol=?"
            args = (symbol,)
        q += " GROUP BY reason_code ORDER BY n DESC"
        return {r["reason_code"]: r["n"] for r in self.conn.execute(q, args)}

    def trade_summary(self) -> dict:
        row = self.conn.execute(
            "SELECT COUNT(*) n, SUM(won) wins, COALESCE(SUM(pnl),0) pnl "
            "FROM trades WHERE settled_at IS NOT NULL").fetchone()
        n = row["n"] or 0
        wins = row["wins"] or 0
        return {"trades": n, "wins": wins, "pnl": float(row["pnl"] or 0.0),
                "win_rate": (wins / n) if n else float("nan")}
