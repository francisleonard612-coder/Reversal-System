# Strategy Specification — Level 1 Deterministic Reversal Baseline

This is the single source of truth for Level 1. `features/`, `regime/`,
`reversal/` in Python and `pine/reversal_strategy.pine` are both
implementations OF this document — neither invents a formula the other
doesn't have. If Python and Pine ever disagree, one of them has a bug
relative to this file, not to each other.

Every parameter named here lives in `config/settings.yaml` under the same
key. Nothing below is a hard-coded magic number in source.

## 1. Candle construction

Built from Deriv ticks, not fetched as candles from Deriv directly (Deriv's
own candle endpoint exists but tick-built candles guarantee we know exactly
which ticks are inside each bar, which matters for anti-repainting below).

- `timeframe_seconds` (default 60): a candle closes when a tick's epoch
  crosses a boundary at `epoch // timeframe_seconds`.
- OHLC from tick quotes within the bar; **no volume** — Deriv synthetics have
  none, and `has_volume=False` is recorded explicitly rather than a
  fabricated 0 or 1, per Section 5's requirement not to invent it.
- A candle is CLOSED the instant the first tick of the next bar arrives.
  Every feature below reads only CLOSED candles — the candle currently
  forming is never used as an input, which is the anti-repainting rule in
  Section 19 stated as code rather than prose.

## 2. Statistical stretch (`reversal/stretch.py`)

All computed on **closing prices** of the last `stretch_lookback` (default
20) closed candles, at decision time = the candle that just closed.

- `sma = mean(close[-lookback:])`
- `sd = stdev(close[-lookback:])` (sample stdev, n-1)
- `zscore = (close[-1] - sma) / sd` (0 if `sd == 0`)
- `ema_fast = EMA(close, ema_fast_period)` (default 9)
- `ema_distance = (close[-1] - ema_fast) / atr` (ATR-normalized, see below)
- `bollinger_pct_b = (close[-1] - lower_band) / (upper_band - lower_band)`
  where bands are `sma ± bollinger_k * sd` (default k=2.0); `0.5` if the
  band width is 0
- `percentile_rank = rank of close[-1] within close[-percentile_lookback:]`
  (default lookback 100), in `[0, 1]`

`stretch_score` (bullish = oversold, bearish = overbought), each term
clamped to `[-1, 1]` before averaging:
```
stretch_score = mean(
    clamp(-zscore / stretch_z_cap, -1, 1),
    clamp(-ema_distance / stretch_atr_cap, -1, 1),
    clamp((0.5 - bollinger_pct_b) * 2, -1, 1),
    clamp((0.5 - percentile_rank) * 2, -1, 1),
)
```
Positive = oversold (bullish reversal evidence). Negative = overbought.
`stretch_z_cap` default 3.0, `stretch_atr_cap` default 2.5.

## 3. Volatility (`features/volatility.py`)

- `true_range[i] = max(high[i]-low[i], |high[i]-close[i-1]|, |low[i]-close[i-1]|)`
- `atr = EMA(true_range, atr_period)` (default 14, Wilder smoothing:
  `atr[i] = atr[i-1] + (tr[i] - atr[i-1]) / atr_period`)
- `atr_pct = atr / close[-1]` — used for cross-instrument comparability
- `volatility_regime`: `HIGH_VOLATILITY` if `atr_pct` is above the
  `vol_high_percentile` (default 80th) of its own trailing
  `vol_lookback` (default 200) distribution; `LOW_VOLATILITY` below the
  `vol_low_percentile` (default 20th); else neutral. Percentiles are
  computed from CLOSED history only.

## 4. Momentum / exhaustion (`reversal/exhaustion.py`, `features/momentum.py`)

- `rsi period` default 14, Wilder smoothing (same recursive form as ATR,
  applied to average gain/loss).
- `rsi_slope = rsi[-1] - rsi[-rsi_slope_lookback]` (default lookback 3)
- `macd = EMA(close, 12) - EMA(close, 26)`; `macd_signal = EMA(macd, 9)`;
  `macd_hist = macd - macd_signal`
- `roc = (close[-1] - close[-roc_period]) / close[-roc_period]`
  (default `roc_period` 10)
- `ema_slope = (ema_fast[-1] - ema_fast[-ema_slope_lookback]) / ema_fast[-ema_slope_lookback]`

**Extreme vs exhaustion (Section 11's explicit distinction).** An
"extreme" reading (RSI > 70, say) is not by itself exhaustion — a strong
trend keeps RSI extreme for many bars. Exhaustion additionally requires the
extreme to be **decelerating**:
```
bullish_exhaustion = (rsi[-1] < rsi_oversold) and (rsi_slope > 0)
                      and (macd_hist[-1] > macd_hist[-2])   # histogram turning up
bearish_exhaustion = (rsi[-1] > rsi_overbought) and (rsi_slope < 0)
                      and (macd_hist[-1] < macd_hist[-2])
```
`rsi_oversold` default 30, `rsi_overbought` default 70. Exhaustion is
boolean; its evidence weight in the reversal score (Section 7 below) is
fixed regardless of how extreme the reading is, deliberately — a strategy
that scaled the weight with RSI distance would reward chasing depth rather
than rewarding the deceleration itself.

## 5. Price action (`reversal/price_action.py`)

All defined on CLOSED candles only; a "pattern" spanning bar `i` requires
bar `i` to be closed.

- **Rejection candle**: `upper_wick = high - max(open, close)`,
  `lower_wick = min(open, close) - low`, `body = |close - open|`,
  `range = high - low`. Bullish rejection: `lower_wick > rejection_wick_ratio
  * body` (default ratio 2.0) and `lower_wick > rejection_range_min *
  range` (default 0.5). Bearish rejection is the mirror on `upper_wick`.
- **Engulfing**: bullish if `close[-1] > open[-2]` and `open[-1] < close[-2]`
  and `body[-1] > body[-2]` and `close[-2] < open[-2]` (prior candle was
  bearish). Bearish is the mirror.
- **Failed breakout / breakdown**: price closes beyond a recent swing
  high/low (see Section 6) within `failure_lookback` bars (default 5) and
  then closes back inside it — `high[j] > swing_high` for some `j` in the
  window, followed by `close[-1] < swing_high`.
- **Higher-low / lower-high**: using swing points from Section 6, a
  higher-low is `swing_low[-1] > swing_low[-2]` while price is still below
  the prior swing high (incipient structure shift, not yet a full break).

Each detected pattern contributes a fixed evidence weight (Section 7); no
single pattern is sufficient alone (Section 12), enforced by the scoring
blend rather than by any single pattern gating the decision.

## 6. Support / resistance and structure (`features/structure.py`)

- **Swing point**: bar `i` is a swing high if `high[i]` is the maximum of
  `high[i-swing_window : i+swing_window+1]` (default `swing_window` 3).
  This is a `2*swing_window+1`-bar centered window, which means a swing
  point is only CONFIRMED `swing_window` bars after it occurs — the
  anti-lookahead delay is explicit and documented at the call site, per
  Section 19. Swing low is the mirror on `low`.
- **Zone**: swing points within `zone_merge_pct` (default 0.15%) of price
  of each other are merged into one zone. Zone strength = number of merged
  touches, decayed by recency (`zone_decay_half_life` bars, default 100):
  `strength = sum(0.5 ** (age / half_life) for each touch)`.
- **Distance to nearest zone**: `(close[-1] - zone_price) / atr`,
  ATR-normalized so it compares across instruments and volatility regimes.
- **Reclaim / loss**: price closes back above a zone it had been below (or
  the reverse) within `reclaim_lookback` bars (default 5).

## 7. Divergence (`reversal/divergence.py`)

Compares swing points in price against the same-indexed value of RSI and
MACD histogram. Regular bullish divergence: `price` makes a lower low
across two confirmed swing lows while `RSI` (or `MACD hist`) makes a
higher low at those same two bars. Regular bearish is the mirror on swing
highs. Divergence is optional evidence (Section 14) — its weight in the
score is nonzero only when at least two confirmed swing points exist to
compare; otherwise it contributes 0, not a penalty.

## 8. Regime (`regime/detector.py`)

- `adx`-style trend strength: Wilder's ADX, period `adx_period` (default
  14).
- `TREND_UP`: `ema_slope > trend_slope_min` (default 0.001) and
  `adx > adx_trend_min` (default 25)
- `TREND_DOWN`: mirror, `ema_slope < -trend_slope_min`
- `RANGE`: `adx < adx_range_max` (default 20)
- `TRANSITION`: neither of the above (ADX between the two thresholds, or
  slope near zero with high ADX — a directionless-but-forceful market)
- `HIGH_VOLATILITY` / `LOW_VOLATILITY`: from Section 3, layered on top of
  the directional state (a regime is reported as e.g. `TREND_UP` with a
  separate `volatility_regime` field, not one enum trying to hold both)
- `UNKNOWN`: insufficient closed history (`< regime_min_bars`, default 50)

**Regime effect on reversal requirements (Section 15's explicit rule):** a
reversal against `TREND_UP`/`TREND_DOWN` requires a HIGHER minimum reversal
score than a reversal inside `RANGE`. Concretely, the score threshold used
by the decision engine is multiplied by `regime_threshold_multiplier[regime]`
(config-driven; defaults: TREND_* = 1.4, RANGE = 1.0, TRANSITION = 1.2,
UNKNOWN = refuse entirely — regime unknown is a hard gate, not a soft
multiplier, since there is no basis to price the trade's risk at all).

## 9. Reversal scoring (`reversal/scoring.py`)

```
bullish_reversal_score =
    w_stretch    * max(stretch_score, 0)
  + w_exhaustion * (1.0 if bullish_exhaustion else 0.0)
  + w_price_action * price_action_bullish_evidence   # in [0,1], see below
  + w_sr         * sr_bullish_evidence                # in [0,1]
  + w_divergence * (1.0 if bullish_divergence else 0.0)
```
mirrored for `bearish_reversal_score` using the bearish evidence terms.
Each `w_*` is configured in `config/settings.yaml` under
`reversal.weights` and MUST sum to 1.0 (validated at startup — an
unnormalized weight set is a startup failure, not a silent rescale,
because a silent rescale would make the score's 0-100 meaning drift
without anyone deciding that). Final score is `100 * bullish_reversal_score`
(or bearish), clamped to `[0, 100]`.

`price_action_bullish_evidence` is itself the count of triggered bullish
patterns from Section 5, divided by the number of pattern types checked
(so partial pattern agreement is partial evidence, not all-or-nothing).
`sr_bullish_evidence` is 1.0 if price is within `sr_proximity_atr`
(default 1.0 ATR) of a support zone with `strength >= sr_min_strength`
(default 2.0), scaled down linearly to 0 at `sr_proximity_atr * 3`.

## 10. Setup vs confirmation (`reversal/confirmation.py`)

A **setup** exists when `reversal_score >= setup_threshold` (default 55,
pre-regime-multiplier). A setup is NOT tradeable by itself.

**Confirmation** requires, within `confirmation_window` bars (default 3)
after the setup bar, at least one of:
- a closed candle beyond the setup-bar's high (bullish) / low (bearish)
  in the reversal direction — "follow-through"
- price closing back on the correct side of `ema_fast` (EMA reclaim/loss)
- `macd_hist` crossing zero in the reversal direction

A setup that ages past `confirmation_window` without confirmation is
DISCARDED, not held open indefinitely — Section 19's "never allow future
information" implies the reverse too: a setup can't be validated by
something that happens arbitrarily far in the future and still be called
the same decision.

## 11. NO-TRADE conditions (Section 18, implemented in `strategy/decision_engine.py`)

Refuse (NO TRADE) when any of:
- regime is `UNKNOWN`
- no confirmed setup, or setup unconfirmed within its window
- `reversal_score < regime_threshold_multiplier[regime] * setup_threshold`
- bullish and bearish scores both clear threshold in the same window
  (conflicting signals — refuse rather than pick arbitrarily)
- volatility_regime is `HIGH_VOLATILITY` and `disable_on_high_vol` is true
  (default true — abnormal volatility invalidates the ATR-normalized
  thresholds this whole spec relies on)
- data quality checks (Section 7 of the master spec) fail
- expected value (see EV engine) is not positive
- any risk-engine limit is at capacity

## 12. Data requirements

Minimum closed candles before ANY signal is evaluated:
`max(percentile_lookback, vol_lookback, regime_min_bars) + swing_window`
(with current defaults: 200 + 3 = 203). Below this, the pipeline emits
`NO_TRADE_MINIMUM_DATA` and computes nothing else — not a partial score.
