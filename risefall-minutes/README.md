# Rise/Fall minutes bot

Deriv Rise/Fall bot for 1-10 minute contracts, based on
`monicalashaythomas-coder/rise-and-fall-minutes` (`risefall-bot/`), with these changes:

- **LSTM removed** -- no torch, no trainer service. Gate 6 never vetoes.
- **Fall trades fixed** -- three separate bugs meant the original only ever
  traded Rise (261 of 261 qualifying trades in a 60-day replay):
  1. the Monte Carlo multiplied recent drift by the trade direction, inverting
     momentum for PUTs;
  2. the Hurst exponent (trending vs mean-reverting, no direction) was counted
     as a vote for Rise -- up 89% of the time, strongest voter in the trend
     regime. It now follows the recent move when persistent, fades it when not;
  3. `hmm_gbm_scan` ranked by |p - 0.5| so CALL always won the tie and Gate 5
     vetoed borderline PUTs. It now ranks by signed edge.
  After the fixes a 700-check sample on R_100 gave 16 Falls and 21 Rises.
- **Fixed stake** (`FIXED_STAKE`, default 0.35) and **martingale off**
  (`MARTINGALE_MAX_STEPS=0`).
- **Filters are Railway variables** -- see `.env.example`.
- **`SYMBOLS`** variable to choose symbols (the original universe never
  included RDBULL/RDBEAR).

## What the evidence says (read before going live)

Replayed on 60 days of R_100, R_75, RDBEAR, RDBULL (research/risefall_stack_test.py):
raw direction accuracy 49-51% at every horizon; full filters 53.3% on 261 trades
(95% range 47-59%, all Rise, driven by RDBULL's built-in upward drift); minimal
filters 50.3% on 447 trades. Break-even is ~51.3% at a 1.95 payout and ~53.2% at
RDBULL's 1.88. Tick data shows no pattern at all. Run this on **demo** and judge
it from `bot_trade_log`.

## One-time Supabase setup

```sql
CREATE TABLE IF NOT EXISTS bot_trade_log (
    id          BIGSERIAL PRIMARY KEY,
    ts          TIMESTAMPTZ DEFAULT now(),
    symbol      TEXT,
    direction   INTEGER,
    step        INTEGER,
    stake       REAL,
    won         BOOLEAN,
    profit      REAL,
    p_up        REAL,
    confidence  REAL,
    duration    INTEGER,
    layer_votes JSONB,
    n_agree     INTEGER,
    n_disagree  INTEGER
);

CREATE TABLE IF NOT EXISTS bot_symbol_state (
    symbol         TEXT PRIMARY KEY,
    reliability    REAL,
    threshold      REAL,
    step0_wins     INTEGER DEFAULT 0,
    step0_total    INTEGER DEFAULT 0,
    layer_weights  JSONB  DEFAULT '{}',
    payout_history JSONB  DEFAULT '[]',
    updated_at     TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS bot_global_state (
    key        TEXT PRIMARY KEY,
    value      JSONB,
    updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS bot_gate_config (
    key        TEXT PRIMARY KEY,
    value      REAL,
    updated_at TIMESTAMPTZ DEFAULT now()
);
```

## Deploy (Railway)

1. New service from the Reversal-System repo, branch `risefall-minutes`,
   **Root Directory** `risefall-minutes`.
2. Add the variables from `.env.example` (at minimum `DERIV_APP_ID`,
   `DERIV_API_TOKEN`, `SUPABASE_URL`, `SUPABASE_KEY`). `SUPABASE_URL` is the
   project's https REST URL and `SUPABASE_KEY` the service_role key -- not the
   Postgres connection string the other bots use.
3. Deploy. It bootstraps ticks, runs a calibration pass, then starts scanning.
