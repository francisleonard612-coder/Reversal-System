-- ===========================================================================
-- Deriv Reversal Intelligence System -- Supabase (PostgreSQL) schema
--
-- Apply once, in the Supabase SQL editor, BEFORE the first deploy. Mirrors
-- data/database.py's SQLite schema table-for-table; a separate project from
-- the sibling Even/Odd bot's Supabase instance, since these are two
-- unrelated strategies with independent data.
-- ===========================================================================

create table if not exists schema_version (version integer primary key);
insert into schema_version (version) values (1) on conflict (version) do nothing;

create table if not exists ticks (
    id bigserial primary key,
    symbol text not null, epoch double precision not null, price double precision not null,
    received_at double precision not null
);
create index if not exists idx_ticks_symbol_epoch on ticks(symbol, epoch);

create table if not exists candles (
    id bigserial primary key,
    symbol text not null, timeframe_seconds integer not null,
    open_epoch bigint not null, close_epoch bigint not null,
    open double precision not null, high double precision not null,
    low double precision not null, close double precision not null,
    n_ticks integer not null, has_volume boolean not null default false
);
create index if not exists idx_candles_symbol_close on candles(symbol, close_epoch);
-- Required by record_candle()'s ON CONFLICT (symbol, close_epoch) and checked
-- by the bot at startup (data/postgres.py). Previously MISSING from this file,
-- so a Supabase deploy following DEPLOY.md refused to start. Dedup first
-- (restarts before this fix could have written duplicates), then index --
-- the other order fails on the very duplicates being fixed. Idempotent.
delete from candles a using candles b
 where a.symbol = b.symbol and a.close_epoch = b.close_epoch and a.id > b.id;
create unique index if not exists idx_candles_unique on candles(symbol, close_epoch);

create table if not exists signals (
    id bigserial primary key,
    ts double precision not null, created_at timestamptz not null default now(),
    symbol text not null, decision text not null, reason_code text not null,
    explanation text, regime text, volatility_regime text,
    bullish_score double precision, bearish_score double precision,
    threshold_used double precision, regime_multiplier double precision,
    confirmed boolean, confirmation_reason text, payout_multiple double precision
);
create index if not exists idx_signals_symbol_ts on signals(symbol, ts desc);
create index if not exists idx_signals_reason on signals(reason_code);

create table if not exists rejected_signals (
    -- Section 28/29: counterfactual tracking. entry_price, outcome_price,
    -- outcome_known_at, would_have_won are filled by a LATER reconciliation
    -- pass, never at write time.
    id bigserial primary key,
    signal_id bigint references signals(id),
    ts double precision not null, symbol text not null, reason_code text not null,
    would_be_direction text,
    entry_price double precision, outcome_price double precision,
    outcome_known_at double precision, would_have_won boolean,
    outcome_evaluated boolean default false
);
create index if not exists idx_rejected_unevaluated on rejected_signals(symbol)
    where outcome_evaluated = false;

create table if not exists trades (
    id bigserial primary key,
    signal_id bigint references signals(id),
    ts double precision not null, created_at timestamptz not null default now(),
    symbol text not null, contract_id bigint unique,
    idempotency_key text unique,
    contract_type text, stake double precision, payout double precision,
    buy_price double precision, entry_spot double precision, exit_spot double precision,
    won boolean, pnl double precision, settled_at double precision, error text
);
create index if not exists idx_trades_ts on trades(ts desc);
create index if not exists idx_trades_open on trades(symbol) where settled_at is null;

create table if not exists outcomes (
    id bigserial primary key,
    trade_id bigint references trades(id),
    ts double precision not null, category text, note text
);

create table if not exists predictions (
    -- Level 2 reserved. Empty at Level 1.
    id bigserial primary key,
    ts double precision not null, symbol text not null, model_id text,
    probability double precision, calibrated_probability double precision,
    model_version text
);

create table if not exists model_versions (
    -- Level 2/3 champion/challenger reserved. Empty at Level 1.
    id bigserial primary key,
    model_id text not null, version text not null,
    created_at timestamptz not null default now(),
    training_period_start double precision, training_period_end double precision,
    features jsonb, parameters jsonb, calibration_method text,
    promotion_status text default 'CANDIDATE'
);

create table if not exists model_performance (
    id bigserial primary key,
    ts double precision not null, model_id text, symbol text,
    n integer, brier double precision, log_loss double precision, accuracy double precision
);

create table if not exists regime_history (
    id bigserial primary key,
    ts double precision not null, symbol text not null, regime text not null,
    volatility_regime text, adx double precision
);
create index if not exists idx_regime_history_symbol_ts on regime_history(symbol, ts desc);

create table if not exists learning_events (
    -- Level 3 reserved. Empty at Level 1.
    id bigserial primary key,
    ts double precision not null, event_type text, detail jsonb
);

create table if not exists system_events (
    id bigserial primary key,
    ts double precision not null, level text, category text, message text, detail jsonb
);
create index if not exists idx_system_events_ts on system_events(ts desc);

create table if not exists parameter_changes (
    id bigserial primary key,
    ts double precision not null, parameter text not null,
    old_value text, new_value text, reason text
);

-- v4: Level 2 predictions are linked to the signal they priced and carry
-- the economics they were judged on. Idempotent -- safe to re-run.
alter table predictions add column if not exists signal_id bigint references signals(id);
alter table predictions add column if not exists probability_lower double precision;
alter table predictions add column if not exists payout_multiple double precision;
alter table predictions add column if not exists expected_value double precision;
create index if not exists idx_predictions_signal on predictions(signal_id);

-- v5: MC-calibrated expiry (strategy/expiry.py). Every qualifying setup's
-- real outcome at every candidate expiry, and the expiry each trade used.
create table if not exists setup_outcomes (
    id bigserial primary key,
    ts double precision not null, symbol text not null, setup_epoch bigint not null,
    direction text not null, confirmed boolean not null, score double precision,
    regime text, horizon text not null, entry_price double precision,
    exit_price double precision, won boolean not null, source text not null
);
create unique index if not exists idx_setup_outcomes_unique
    on setup_outcomes(symbol, setup_epoch, direction, horizon);
create index if not exists idx_setup_outcomes_epoch on setup_outcomes(setup_epoch);
alter table trades add column if not exists duration integer;
alter table trades add column if not exists duration_unit text;

-- ===========================================================================
-- Views
-- ===========================================================================
create or replace view v_rejection_breakdown as
select symbol, reason_code,
       count(*) n,
       round(avg(bullish_score)::numeric, 2) mean_bullish_score,
       round(avg(bearish_score)::numeric, 2) mean_bearish_score,
       round(avg(threshold_used)::numeric, 2) mean_threshold
from signals
where decision = 'NO_TRADE'
group by symbol, reason_code
order by n desc;

create or replace view v_daily_pnl as
select date_trunc($$day$$, to_timestamp(ts)) as day_bucket, symbol,
       count(*) trades, count(*) filter (where won) wins,
       round(sum(pnl)::numeric, 2) pnl
from trades
where settled_at is not null
group by 1, 2
order by 1 desc;

create or replace view v_regime_distribution as
select symbol, regime, volatility_regime, count(*) n
from regime_history
group by symbol, regime, volatility_regime
order by symbol, n desc;

-- Section 28/29's counterfactual analysis, once reconcile_rejected_signals()
-- has run: does the reversal score actually predict what WOULD have
-- happened on setups the bot declined? This is the Level 2 training
-- signal, not just an operator curiosity -- a model trained only on the
-- tiny fraction that got traded is badly survivorship-biased.
-- Level 2 honesty check: does the model's predicted P(win) match what
-- actually happened on trades it approved? Buckets of 5 percentage points.
create or replace view v_level2_calibration as
select round((floor(p.calibrated_probability * 20) / 20)::numeric, 2) as predicted_bucket,
       count(*) trades,
       round(avg(case when t.won then 1.0 else 0.0 end)::numeric, 4) actual_win_rate,
       round(avg(p.expected_value)::numeric, 4) mean_predicted_ev,
       round(sum(t.pnl)::numeric, 2) pnl
from predictions p
join trades t on t.signal_id = p.signal_id
where t.settled_at is not null
group by 1
order by 1;

-- Which expiry does each kind of setup actually resolve best at?
create or replace view v_expiry_win_rates as
select symbol, direction, horizon, confirmed, source,
       count(*) n,
       round(avg(case when won then 1.0 else 0.0 end)::numeric, 4) win_rate
from setup_outcomes
group by symbol, direction, horizon, confirmed, source
order by symbol, direction, confirmed desc, win_rate desc;

-- Real results by the expiry each trade was bought with.
create or replace view v_trades_by_expiry as
select symbol, duration || coalesce(duration_unit, '') as expiry,
       count(*) trades, count(*) filter (where won) wins,
       round(avg(case when won then 1.0 else 0.0 end)::numeric, 4) win_rate,
       round(sum(pnl)::numeric, 2) pnl
from trades where settled_at is not null
group by 1, 2 order by 1, 2;

create or replace view v_counterfactual_summary as
select symbol, reason_code,
       count(*) filter (where outcome_evaluated) n_evaluated,
       count(*) filter (where not outcome_evaluated) n_pending,
       round(avg(case when would_have_won then 1.0 else 0.0 end)
             filter (where outcome_evaluated)::numeric, 4) would_have_won_rate
from rejected_signals
group by symbol, reason_code
order by n_evaluated desc;

-- ===========================================================================
-- Row Level Security -- the bot connects with the service role key, which
-- bypasses RLS. These policies exist so an anon/dashboard connection can
-- read but never write.
-- ===========================================================================
alter table signals            enable row level security;
alter table trades             enable row level security;
alter table rejected_signals   enable row level security;
alter table regime_history     enable row level security;
alter table system_events      enable row level security;
alter table setup_outcomes     enable row level security;

do $$
declare t text;
begin
    foreach t in array array['signals','trades','rejected_signals',
                             'regime_history','system_events','setup_outcomes']
    loop
        execute format('drop policy if exists %I on %I', 'read_only_authenticated', t);
        execute format('create policy %I on %I for select to authenticated using (true)',
                       'read_only_authenticated', t);
    end loop;
end $$;
