# Deriv Reversal Intelligence System -- Level 1 + Level 2

A deterministic reversal-trading engine for Deriv Rise/Fall (CALL/PUT)
contracts on synthetic indices, built to the attached 69-section
specification. This is a **separate system** from the sibling Even/Odd
digit-parity bot -- different contract family (direction, not last-digit
parity), different feature space (candles and technical indicators, not
digit statistics), sharing only the connection layer, risk manager, and
deployment shape.

## What changed in this revision (read first)

- **Live contract now matches what research measures.** Live traded 5-TICK
  contracts while the backtest, `--reconcile` and Level 2 labels all measured
  direction 5 one-minute CANDLES later. Default is now a 5-minute contract,
  and startup refuses a config where the two disagree (tick contracts are
  allowed but flagged as unvalidated). **This changes live trades.**
- **Risk controls that existed but never ran are wired:** `register_open`
  was never called (so `max_trades_per_day`, `cooldown_seconds` and the
  concurrency limit never triggered), several `settings.yaml` risk/staking
  values never reached the code, and `max_proposal_age_seconds` /
  `stale_tick_seconds` were unused.
- **Risk limits survive restarts.** Today's P/L, trade count and losing
  streak are rebuilt from the journal on startup; `--reset-risk` is the
  explicit operator reset. Unsettled contracts are re-attached after a restart.
- **No fake losses.** A settlement that didn't arrive within 120s was booked
  as a LOSS with pnl 0 (and any reconnect mid-trade caused exactly that).
  Now it polls until Deriv reports the real result; booking is idempotent.
- **Ambiguous buys reconciled** via `portfolio` instead of forgotten.
- **Supabase schema fixed:** `idx_candles_unique`, required by the bot's own
  startup check, was missing from `supabase/schema.sql`, so a Postgres
  deploy could not start. Re-run the schema file (idempotent).
- **Level 2 implemented:** `--evaluate`, `--train`, and a live EV gate.
- **Real-data research:** `--fetch-history` downloads Deriv candles;
  `--backtest` / `--walk-forward` use them (or `--data-file`), and only fall
  back to the synthetic walk with a loud warning.
- Also: stale-feed watchdog, periodic balance refresh, reconcile on a timer,
  `--dashboard`, `.env.example`. 133 tests (was 97), including the Postgres
  backend against a real Postgres (`TEST_DATABASE_URL`).

## Setups and expiry (latest revision)

**Why it wasn't opening trades.** The setup threshold was an absolute 55
(77 in trends). On random-walk-like data (what Deriv volatility indices are
built to be), ranging markets almost never score 55, and the high readings
that do occur are nearly all counter-trend, where 77 applies. In testing,
12,000 bars produced **zero** setups -- and on data with a real reversal
effect planted in it, also zero.

**Now: setups are the strongest readings on each symbol's own history.**
`reversal.threshold_mode: percentile` -- a setup needs a score in the top 2%
of the last day's bars (top 1% in transitions, 0.5% against a trend), from
prior bars only, never below `min_absolute_score`. On the planted-reversal
data this found setups that won ~96%; on pure noise it finds setups that win
about half the time -- which is why quality is then *measured*, not assumed.

**MC-calibrated expiry, ticks and candles** (`strategy/expiry.py`). Every
qualifying setup is followed forward and its real outcome recorded at each
candidate expiry -- 3/5/7/10 ticks and 1/2/3/5 minutes by default, limited to
what Deriv offers per symbol (`setup_outcomes` table). For each expiry, Monte
Carlo draws from the posterior win rate give a conservative (Bonferroni-
adjusted) P(win); the best few are priced with real proposals and the trade
is bought at the expiry with the best worst-case EV.

- Until an expiry has `min_samples` (200) outcomes the bot trades the default
  5-minute contract under Level 1 rules, so trades open from day one.
- `--calibrate-expiry` backfills minute expiries from stored candles (run it
  after `--fetch-history`); tick expiries are learned live.
- `expiry.require_edge` (default on): once calibrated, a trade needs positive
  worst-case EV. Settings were chosen by simulation: noise passes ~1.3% of
  the time; a true 56% edge passes ~50% at 200 samples, ~94% at 800. In
  **demo** you can set `EXPIRY_REQUIRE_EDGE=false` to keep trading the
  best-calibrated expiry while evidence builds -- never do that live.
- Supabase: `v_expiry_win_rates` (outcomes per expiry), `v_trades_by_expiry`.

## Recommended workflow

```bash
python main.py --fetch-history --days 30          # real candles into the DB
python main.py --walk-forward --symbol R_100 --payout-multiple 1.95
python main.py --evaluate                          # does a model beat Level 1 out of sample?
python main.py --train                             # saves PROMOTED or REJECTED
python main.py --calibrate-expiry                  # backfill minute-expiry outcomes
python main.py --expiry-report                     # MC calibration per symbol/direction
# research mode for days; then --dashboard / v_level2_calibration before demo, then live
```

Deriv's volatility indices are generated to behave like random walks. Expect
`--evaluate` to show little or no out-of-sample edge, and a REJECTED model --
that result is the system working, not failing. It is far cheaper to learn
it here than from the account balance.

## Level 2 (implemented)

`models/level2.py`. Logistic regression on the same features the live
pipeline computes (one shared function, `models.dataset.compute_evidence`),
Platt-calibrated on the newest 20% of data.

- **Promotion is earned:** on every held-out walk-forward block the model must
  correlate *positively* with real outcomes above `promotion_corr_floor` AND
  beat the "always predict the base rate" forecaster on Brier score. Otherwise
  it is saved as REJECTED and the live bot refuses it.
- **Live EV gate** (when `level2.enabled` and the model is PROMOTED): EV =
  P(win) x real quoted payout multiple - 1 must be >= `min_ev`, and by default
  the model's conservative *lower bound* on P(win) must clear break-even.
  Every priced setup is written to `predictions`; `v_level2_calibration`
  compares predicted vs actual win rate on real trades.
- **Fingerprinted:** changing any feature/horizon setting invalidates a saved
  model until retrained. Artifacts are plain JSON (no pickle).
- Kelly staking now has the probability bound it needs.

Level 3 (online learning, drift detection, champion/challenger) and
`--online-sim` remain unimplemented.

## Original Level 1 notes

The spec defines three levels. This repository implements **Level 1
completely and correctly** -- deterministic reversal logic, no probability
model, no learning -- and stages Level 2 (calibrated probability + ML
ensemble) and Level 3 (online learning, drift detection, champion/
challenger) as the next phase, per the spec's own Section 2 progression:
validated baseline first, complexity only once it's earned.

```
RAW TICKS -> CANDLES (closed only) -> FEATURES -> REGIME -> REVERSAL SCORE
    -> SETUP -> CONFIRMATION -> NO-TRADE ENGINE -> PROPOSAL -> RISK -> BUY
    -> SETTLEMENT -> DATABASE
```

`docs/STRATEGY_SPECIFICATION.md` is the single source of truth for every
formula. `pine/reversal_strategy.pine` derives from the same document --
neither implementation invents a formula the other doesn't have.

## Project layout

```
deriv/client.py           connection layer: OTP auth, rate limiting,
                           reconnection, idempotent buy (ported from the
                           sibling Even/Odd bot; adapted for CALL/PUT and
                           float tick prices instead of text)
data/
  candles.py               tick -> candle construction, no fabricated volume
  validation.py             tick-level data quality checks
  database.py               SQLite, full Section 49 schema
  postgres.py               Supabase/Postgres backend + factory
features/
  volatility.py             ATR, Wilder smoothing, Bollinger inputs
  momentum.py                RSI, MACD, ROC, EMA slope
  structure.py                swing points (lagged, anti-lookahead), S/R zones
  engine.py                    orchestrates the above into one snapshot per candle
regime/detector.py         ADX-based TREND_UP/DOWN/RANGE/TRANSITION/UNKNOWN
reversal/
  stretch.py                statistical stretch (z-score, EMA distance, %B, percentile)
  exhaustion.py               extreme vs decelerating-extreme distinction
  price_action.py              rejection, engulfing, failed breaks, HL/LH
  support_resistance.py         zone-proximity evidence
  divergence.py                  RSI/MACD divergence, optional evidence
  scoring.py                     documented weighted blend -> 0-100 score
  confirmation.py                 setup -> confirmation state machine
strategy/
  edge_engine.py             Level 1's honest (probability-free) economics
  decision_engine.py           the NO-TRADE engine
  pipeline.py                    per-symbol orchestration
execution/executor.py       CALL/PUT proposal -> buy -> settle
risk/manager.py, staking.py  ported from the sibling bot; martingale off,
                              no env override (see config/loader.py)
backtest/simulator.py       Level 1 backtest + walk-forward, real decision path
pine/reversal_strategy.pine  PineScript mirror
tests/                       86 tests
```

## Why Level 1 has no expected value

Sections 25/26 require a model-implied probability to compute EV from.
Level 1 is defined (Section 3) as pure deterministic logic with no
probability estimation -- that's explicitly Level 2's job. Rather than
invent a probability to produce a number (which Section 62 forbids
outright: "do not create... fake probabilities... fabricated
profitability"), `strategy/edge_engine.py`'s `EdgeAssessment.expected_value`
is `None`, always, at this level. What Level 1 checks instead is a
**payout-quality floor** -- read the module's docstring before assuming a
"poor economics" refusal means what it would on a bot with a real
calibrated probability.

## A real bug found and fixed while building this

The feature engine originally received the *entire* growing candle history
on every closed candle, and swing/structure detection rescans its whole
input each call -- making a backtest quadratic in candle count (a
5,000-candle run took over a minute). Fixed by capping the window handed
downstream at `candles.max_lookback_bars` (400, comfortably above the
largest individual lookback). That fix introduced a second, subtler bug:
`reversal/confirmation.py` originally derived a pending setup's age by
subtracting array indices across calls, and once the lookback window slides
(which happens almost immediately), both the stored and current index
converge to the same constant (`len(window)-1`) -- so age would read as
permanently zero and a setup would never confirm or expire. Fixed by
tracking age as an explicit bar-count the pipeline maintains
(`PendingSetup.bars_waited`), decoupled entirely from array indexing.
`tests/test_confirmation.py::test_age_is_independent_of_bar_index_across_a_sliding_window`
pins this.

A third bug, found on review rather than by a test: `main.py`'s live loop
never actually called `db.record_candle()` -- the table existed and the
write method existed, but nothing connected them, so candle history (the
raw material Level 2 needs, and the only way to reconstruct what happened
after a rejected signal) would not have been saved at all. Fixed; individual
ticks are still deliberately NOT persisted -- 60s candles are the
granularity everything downstream needs, and storing every tick would burn
through a free-tier database fast for no corresponding benefit.

A fourth, more serious one: **`TRADING_MODE=research` was logged but never
actually enforced.** Nothing in the live trade path checked `settings.mode`
before calling `execute_trade()` -> `client.buy()`. Since account selection
is driven separately by `DERIV_USE_REAL` (which picks demo vs real, not
whether trading happens), research mode as originally shipped would have
placed real orders against the demo account -- not the "evaluates
everything, never trades" behavior it was described as having. Found when
directly asked "will research mode place actual trades" and verified by
reading the trade path rather than assumed. Fixed with two independent
checks: `strategy/decision_engine.py`'s `apply_economics_and_risk` downgrades
a would-be trade to `NO_TRADE_RESEARCH_MODE` after economics and risk both
pass (so the log shows what it would have traded), and
`execution/executor.py` refuses to call `buy()` as a hard backstop that does
not trust the decision object's own state -- a bug that bypassed the first
check would still be caught by the second.
`tests/test_executor.py::test_backstop_refuses_even_if_the_first_gate_is_bypassed`
proves the second layer works independently of the first by monkeypatching
the first one to fail open.

## Using this data for Level 2 -- what's captured and what isn't yet

Once deployed, `signals` (every decision, taken or not) and `candles`
(every closed bar) accumulate continuously; `trades` captures the real
outcome of anything actually bought.

**Counterfactual outcomes for rejected signals are reconciled, not
auto-computed at decision time.** Section 29 forbids writing a future
outcome into the original row -- so `rejected_signals.would_have_won` stays
NULL until `python main.py --reconcile` runs, which joins each rejected
signal against the candle history `contract.duration_bars_approx` bars
later and fills in whether that direction would have won. Safe to run
repeatedly (already-evaluated rows are skipped) and safe to run before
enough candles exist (those rows are simply left for the next call) --
schedule it as a periodic Railway job, or run it by hand before pulling
data for Level 2. `v_counterfactual_summary` in Supabase gives the
would-have-won rate per reason code once it's run.

This is what actually closes the survivorship-bias gap: without it, a
Level 2 model would only ever see the outcomes of the tiny fraction of
setups that got traded, never what happened after the much larger set the
bot correctly or incorrectly declined.

## Running it

```bash
pip install -r requirements.txt
cp .env.example .env        # fill in DERIV_API_TOKEN at minimum
pytest tests/ -q             # 133 tests (Postgres ones need TEST_DATABASE_URL)

python main.py --status
python main.py --fetch-history --days 30
python main.py --backtest --symbol R_100 --payout-multiple 1.95
python main.py --walk-forward --symbol R_100
python main.py --evaluate | --train
python main.py --dashboard
python main.py --reconcile   # also runs automatically every 5 min while live
python main.py --reset-risk --note "reviewed losses"
python main.py               # live loop, mode from TRADING_MODE
```

`--online-sim` (Level 3) prints a clear "not implemented" message rather than
doing something fake. See "Recommended workflow" above for the full CLI.

## Deployment

Railway + Supabase: see **[DEPLOY.md](DEPLOY.md)**. Separate Supabase
project and Railway service from the sibling Even/Odd bot -- these are
unrelated strategies and should not share a database. Replicas must stay at
1, same reasoning as the sibling bot's deployment guide.

## Honesty notes carried through the whole build

- No fabricated volume on synthetic instruments (`has_volume=False`,
  explicit, never a placeholder 0 or 1).
- Every swing point is confirmed only after `swing_window` bars on both
  sides -- a swing cannot appear before it could actually have been known
  (`features/structure.py`).
- Reversal score weights are validated to sum to 1.0 at startup; an
  unnormalized weight set fails loudly rather than being silently rescaled.
- The backtest's assumed payout multiple is named in every report --
  historical proposals aren't available, so it's a disclosed assumption,
  never presented as a fetched quote.
- Rejected signals are recorded for future counterfactual analysis
  (`rejected_signals` table) with outcome fields left NULL until a later
  reconciliation pass -- never filled at decision time, so no future
  information leaks backward into the original row.

## Disclaimers

Level 1's own philosophy (Section 18): the system must be comfortable
saying NO TRADE, and trade frequency is not an optimization target. This is
not a claim of profitability -- none is made, and none should be assumed
until Level 2's calibration and out-of-sample evaluation exist to actually
measure one.
