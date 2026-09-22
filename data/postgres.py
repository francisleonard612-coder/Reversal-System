"""
PostgreSQL / Supabase persistence. Drop-in for data/database.Database,
selected via DB_BACKEND=postgres. Same fail-closed contract as the sibling
Even/Odd bot's app/storage/postgres.py: writes raise rather than swallow,
schema is applied once by hand via supabase/schema.sql (never DDL'd from
here), and connects through the Supabase POOLED port (6543), service role
credentials -- an anon connection would write nothing under the RLS
policies in that file.
"""
from __future__ import annotations

import json
import logging
import os
import time

logger = logging.getLogger(__name__)


class PostgresUnavailable(RuntimeError):
    pass


def _connect(dsn: str):
    try:
        import psycopg
    except ImportError as exc:
        raise PostgresUnavailable(
            "DB_BACKEND=postgres requires psycopg: pip install 'psycopg[binary]'"
        ) from exc
    return psycopg.connect(dsn, autocommit=True, connect_timeout=10)


class PostgresDatabase:
    def __init__(self, dsn: str | None = None):
        dsn = dsn or os.getenv("DATABASE_URL", "")
        if not dsn:
            raise PostgresUnavailable(
                "DATABASE_URL is empty; set it to the Supabase pooled URI "
                "(port 6543) or set DB_BACKEND=sqlite")
        if ":6543" not in dsn and "pooler" not in dsn:
            logger.warning(
                "DATABASE_URL does not look like the Supabase pooler (port "
                "6543); the direct port has a low connection cap")
        self.dsn = dsn
        self.conn = _connect(dsn)
        self._verify_schema()

    def _verify_schema(self) -> None:
        with self.conn.cursor() as cur:
            cur.execute("""
                select count(*) from information_schema.tables
                 where table_schema='public'
                   and table_name in ('signals','trades','rejected_signals',
                                      'regime_history','system_events')""")
            found = cur.fetchone()[0]
        if found < 5:
            raise PostgresUnavailable(
                f"only {found}/5 expected tables exist -- apply "
                f"supabase/schema.sql in the Supabase SQL editor first")

    def is_healthy(self) -> bool:
        try:
            with self.conn.cursor() as cur:
                cur.execute("select 1")
                cur.fetchone()
            return True
        except Exception:
            try:
                self.conn.close()
                self.conn = _connect(self.dsn)
                return True
            except Exception as exc:
                logger.error("database unhealthy: %s", exc)
                return False

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass

    def record_signal(self, decision) -> int:
        with self.conn.cursor() as cur:
            cur.execute("""
                insert into signals (ts, symbol, decision, reason_code,
                    explanation, regime, volatility_regime, bullish_score,
                    bearish_score, threshold_used, regime_multiplier,
                    confirmed, confirmation_reason, payout_multiple)
                values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                returning id""",
                (decision.timestamp, decision.symbol, decision.decision,
                 decision.reason_code, decision.explanation, decision.regime,
                 decision.volatility_regime, decision.bullish_score,
                 decision.bearish_score, decision.threshold_used,
                 decision.regime_multiplier, decision.confirmed,
                 decision.confirmation_reason, decision.payout_multiple))
            signal_id = cur.fetchone()[0]
            if not decision.will_trade:
                direction = ("bullish" if decision.bullish_score >= decision.bearish_score
                            else "bearish")
                cur.execute("""
                    insert into rejected_signals (signal_id, ts, symbol,
                        reason_code, would_be_direction)
                    values (%s,%s,%s,%s,%s)""",
                    (signal_id, decision.timestamp, decision.symbol,
                     decision.reason_code, direction))
            return signal_id

    def record_trade_open(self, *, signal_id, symbol, contract_id, idempotency_key,
                          contract_type, stake, payout, buy_price, entry_spot) -> int:
        with self.conn.cursor() as cur:
            cur.execute("""
                insert into trades (signal_id, ts, symbol, contract_id,
                    idempotency_key, contract_type, stake, payout, buy_price, entry_spot)
                values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) returning id""",
                (signal_id, time.time(), symbol, contract_id, idempotency_key,
                 contract_type, stake, payout, buy_price, entry_spot))
            return cur.fetchone()[0]

    def record_trade_result(self, contract_id, *, won, pnl, exit_spot=None,
                            error=None) -> None:
        with self.conn.cursor() as cur:
            cur.execute("""
                update trades set won=%s, pnl=%s, exit_spot=%s, settled_at=%s, error=%s
                 where contract_id=%s""",
                (won, pnl, exit_spot, time.time(), error, contract_id))

    def record_candle(self, c) -> None:
        with self.conn.cursor() as cur:
            cur.execute("""
                insert into candles (symbol, timeframe_seconds, open_epoch,
                    close_epoch, open, high, low, close, n_ticks, has_volume)
                values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (c.symbol, c.timeframe_seconds, c.open_epoch, c.close_epoch,
                 c.open, c.high, c.low, c.close, c.n_ticks, c.has_volume))

    def record_regime(self, symbol, regime_snap, volatility_regime) -> None:
        with self.conn.cursor() as cur:
            cur.execute("""
                insert into regime_history (ts, symbol, regime, volatility_regime, adx)
                values (%s,%s,%s,%s,%s)""",
                (time.time(), symbol, regime_snap.regime, volatility_regime,
                 regime_snap.adx))

    def log_event(self, level, category, message, detail=None) -> None:
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    insert into system_events (ts, level, category, message, detail)
                    values (%s,%s,%s,%s,%s)""",
                    (time.time(), level, category, message,
                     json.dumps(detail) if detail else None))
        except Exception as exc:
            logger.warning("could not log event (%s): %s", message, exc)

    def reconcile_rejected_signals(self, *, duration_bars: int = 5,
                                   batch_size: int = 5000) -> int:
        """Postgres mirror of data/database.Database.reconcile_rejected_signals
        -- same candle-close approximation, same safe-to-rerun semantics."""
        with self.conn.cursor() as cur:
            cur.execute(
                "select id, symbol, ts, would_be_direction from rejected_signals "
                "where outcome_evaluated = false order by ts limit %s",
                (batch_size,))
            rows = cur.fetchall()
            n = 0
            for signal_id, symbol, ts, would_be_direction in rows:
                cur.execute(
                    "select close, close_epoch from candles where symbol=%s "
                    "and close_epoch <= %s order by close_epoch desc limit 1",
                    (symbol, ts))
                entry = cur.fetchone()
                if entry is None:
                    continue
                entry_close, entry_epoch = entry
                cur.execute(
                    "select close, close_epoch from candles where symbol=%s "
                    "and close_epoch > %s order by close_epoch asc limit 1 offset %s",
                    (symbol, entry_epoch, duration_bars - 1))
                settlement = cur.fetchone()
                if settlement is None:
                    continue
                settle_close, settle_epoch = settlement

                would_have_won = (settle_close > entry_close if would_be_direction == "bullish"
                                  else settle_close < entry_close)
                cur.execute(
                    "update rejected_signals set entry_price=%s, outcome_price=%s, "
                    "outcome_known_at=%s, would_have_won=%s, outcome_evaluated=true "
                    "where id=%s",
                    (entry_close, settle_close, settle_epoch, would_have_won, signal_id))
                n += 1
            return n

    def reason_histogram(self, symbol: str | None = None) -> dict:
        q = "select reason_code, count(*) from signals where true"
        args: list = []
        if symbol:
            q += " and symbol=%s"
            args.append(symbol)
        q += " group by reason_code order by 2 desc"
        with self.conn.cursor() as cur:
            cur.execute(q, args)
            return {r[0]: r[1] for r in cur.fetchall()}

    def trade_summary(self) -> dict:
        with self.conn.cursor() as cur:
            cur.execute("""
                select count(*), count(*) filter (where won), coalesce(sum(pnl),0)
                  from trades where settled_at is not null""")
            n, wins, pnl = cur.fetchone()
        n = n or 0
        return {"trades": n, "wins": wins or 0, "pnl": float(pnl or 0.0),
                "win_rate": (wins / n) if n else float("nan")}


def open_database(backend: str | None = None, *, sqlite_path: str = "data/reversal.db"):
    backend = (backend or os.getenv("DB_BACKEND", "sqlite")).strip().lower()
    if backend == "postgres":
        return PostgresDatabase()
    if backend != "sqlite":
        raise ValueError(f"unknown DB_BACKEND {backend!r}: use sqlite or postgres")
    from data.database import Database
    return Database(sqlite_path)
