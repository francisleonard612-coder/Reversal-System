# Deploying the Reversal System to Railway with Supabase

Separate deployment from the sibling Even/Odd bot -- different Deriv
account permissions may be shared, but use a **separate Supabase project**
and a **separate Railway service**. These are two unrelated strategies;
sharing a database would mix two decision histories in one set of tables.

## 1. Supabase

1. New project. **SQL Editor -> New query ->** paste all of
   `supabase/schema.sql` -> **Run**. The bot verifies the required tables
   exist on startup and refuses to run otherwise.
2. **Project Settings -> Database -> Connection string -> URI**, the
   **pooled** one on port **6543**, not 5432 -- same reasoning as the
   sibling bot: this process holds one connection open for its whole life,
   and the direct port's connection cap is small.
3. Use the **service role** key. The schema's RLS policies give
   `authenticated` read-only; an anon connection writes nothing.

## 2. Railway

1. New project, deploy from repo (or `railway up`). `railway.json` sets the
   start command and restart policy.
2. **Settings -> Deploy -> Replicas: 1.** Two copies means two bots reading
   the same tick stream and potentially opening two positions from two
   different idempotency keys -- `max_concurrent_trades` is enforced
   per-process, in memory, and does nothing about a second process.
3. Set the variables below, deploy, and read the startup log for the
   contract-availability check (`contract check R_100: CALL and PUT
   confirmed available`) before assuming anything traded.

## 3. Variables

| Variable | Example | Notes |
|---|---|---|
| `DERIV_API_TOKEN` | `a1b2c3...` | Read + Trade scopes only. |
| `DERIV_APP_ID` | `1089` | Public demo id; register your own for live. |
| `TRADING_MODE` | `research` | `research` \| `demo` \| `live`. Start here -- research never calls buy(). |
| `DERIV_USE_REAL` | `false` | Must agree with `TRADING_MODE` -- see the two-switch guard below. |
| `DB_BACKEND` | `postgres` | `sqlite` loses history on every Railway redeploy. |
| `DATABASE_URL` | pooled Supabase URI | Service role, port 6543. |
| `SYMBOLS` | `R_100,R_75,R_50,R_25,R_10` | Comma-separated. Each is a candle builder + full feature pipeline -- more symbols is more compute per tick, not more signal. |
| `BASE_STAKE` / `MAX_STAKE` | `1.0` / `5.0` | Fixed currency, never a percentage at this layer. |
| `CONTRACT_DURATION` / `CONTRACT_DURATION_UNIT` | `5` / `m` | Must match the research horizon; see 4b. |
| `LEVEL2_ENABLED` | `false` | Needs a PROMOTED model file; see 4c. |
| `MAX_DAILY_LOSS` / `MAX_DRAWDOWN` | `25.0` / `50.0` | Hard risk stops; do not auto-clear. |
| `STAKING_METHOD` | `fixed` | Section 37: martingale has **no env override** in this repo -- see below. |

### Martingale has no environment override, deliberately

Unlike the sibling Even/Odd bot, `martingale_enabled` in this repo is read
**only** from `config/settings.yaml`, never from an environment variable
(see `config/loader.py`). Section 37 of the spec says "do NOT use martingale
by default" without the qualifier the Even/Odd bot's spec had -- turning it
on here requires editing the committed config file, so it can never be an
accidental deploy-config typo.

### The two-switch live guard

Same shape as the sibling bot: `TRADING_MODE=live` requires
`DERIV_USE_REAL=true`, and vice versa. Any disagreement is a hard startup
failure.

## 4. First-run checklist

### Confirm research mode actually refuses to trade before trusting it

Enforced by two independent checks -- see `strategy/decision_engine.py`'s
`apply_economics_and_risk` and `execution/executor.py`'s backstop
immediately before `buy()`. After your first deploy, confirm it in the data
itself rather than just trusting the config:
```sql
select reason_code, count(*) from signals
where reason_code = 'NO_TRADE_RESEARCH_MODE' group by reason_code;
select count(*) from trades;   -- must be 0 in research mode, always
```
If `trades` is ever non-empty while `TRADING_MODE=research`, stop the
deployment and treat it as a critical bug report, not a configuration
question.

1. **`research` mode, at least a day.** Confirms ticks arrive, candles
   close, and `signals` fills up. Query:
   ```sql
   select * from v_rejection_breakdown;
   ```
   Expect `NO_TRADE_NO_SETUP` and `NO_TRADE_HIGH_VOLATILITY` to dominate --
   Section 18's "trade frequency is not an optimization target" applies as
   much here as it does to the sibling bot. A quiet signals table is the
   expected output, not a sign something is broken.
2. **Check the regime distribution:**
   ```sql
   select * from v_regime_distribution;
   ```
   If `UNKNOWN` dominates, `regime.min_bars` (default 50) isn't being
   cleared -- check the candle timeframe against how much history has
   actually accumulated.
3. **`demo` mode**, once signals are flowing sensibly, to exercise the real
   proposal/buy/settlement path with no financial exposure.
4. **`live`** only with a stake you'd shrug at and risk limits you've
   actually decided on, not the shipped defaults.

## 4b. Upgrading an existing deployment

1. Re-run `supabase/schema.sql` in the SQL editor (idempotent). It adds the
   `idx_candles_unique` index the bot's startup check has always required
   but the file never created, and the Level 2 prediction columns.
2. The default contract is now **5 minutes**, not 5 ticks (see README). If
   you set `CONTRACT_DURATION`/`CONTRACT_DURATION_UNIT`, they must equal
   `duration_bars_approx` candles or startup refuses.
3. Risk limits now persist across restarts. If the bot is holding a
   consecutive-loss stop you've reviewed, clear it with
   `python main.py --reset-risk --note "..."` (Railway: run as a one-off
   command), then restart.

## 4b-2. Percentile setups and MC expiry (latest revision)

1. Re-run `supabase/schema.sql` again (adds `setup_outcomes`, trade
   `duration`/`duration_unit`, views `v_expiry_win_rates` and
   `v_trades_by_expiry`).
2. Backfill evidence (one-off Railway commands, or locally against the
   Supabase `DATABASE_URL`):
   ```bash
   python main.py --fetch-history --days 14
   python main.py --calibrate-expiry
   python main.py --expiry-report
   ```
3. `TRADING_MODE=demo` to see trades open. Optionally
   `EXPIRY_REQUIRE_EDGE=false` in demo only, to keep trading the best
   calibrated expiry before it has shown an edge.
4. Watch in Supabase:
   ```sql
   select * from v_expiry_win_rates where n >= 50;
   select * from v_trades_by_expiry;
   select reason_code, count(*) from signals group by 1 order by 2 desc;
   ```
   Break-even win rate is 1 / payout multiple (~0.513 at 1.95x).

## 4c. Level 2

```bash
python main.py --fetch-history --days 30   # locally or as a one-off Railway command
python main.py --evaluate
python main.py --train                     # writes level2.model_path
```
With `DB_BACKEND=postgres`, candles and model_versions land in Supabase, but
the model file itself is written to `level2.model_path` on local disk -- on
Railway that disk is wiped on redeploy. Train locally against the Supabase
`DATABASE_URL` and commit `data/level2_model.json`, or mount a Railway
volume at `data/`. Then set `LEVEL2_ENABLED=true`. A REJECTED model, a
missing file, or one trained under different settings is refused and the bot
runs on Level 1 rules, with the reason in the startup log.

```sql
select * from v_level2_calibration;   -- predicted vs actual win rate on real trades
```

## 5. What this build does not do yet

Level 3 (online learning, drift detection, champion/challenger) and
`--online-sim` are not implemented. Level 2 is implemented (see 4c) as a
single calibrated logistic model rather than an ensemble. Without a promoted
Level 2 model, `EdgeAssessment.expected_value` stays `None` and the
`min_payout_multiple` floor is the only economic check.

## 6. Operating queries

```sql
select * from v_rejection_breakdown;     -- why is it not trading?
select * from v_daily_pnl;               -- realized P/L (Level 1 only trades on confirmed setups)
select * from v_regime_distribution;     -- what regimes has each symbol actually been in?
select * from v_counterfactual_summary;  -- would-have-won rate for declined setups, once reconciled
select * from system_events where level in ('ERROR','WARNING') order by ts desc limit 50;
```

### Reconciling counterfactual outcomes for Level 2

`rejected_signals.would_have_won` stays NULL until reconciled -- run
periodically, e.g. as a Railway cron job or a manual pass before pulling
data for Level 2 work:
```bash
python main.py --reconcile
```
Idempotent and safe on a schedule: already-evaluated rows are skipped, and
rows whose settlement candle hasn't closed yet are simply left for the next
run. Without this, a Level 2 model trained on this data would only ever see
outcomes for the small fraction of setups that got traded.

## 7. Costs

Railway worker: ~$5/mo on Hobby, one always-on container, no HTTP service
needed. Supabase free tier (500 MB) holds signal + trade history for weeks
at typical Level-1 trade frequency, since the heavy per-candle detail lives
in `features`/`predictions` tables that stay mostly empty until Level 2.
