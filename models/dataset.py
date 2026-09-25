"""
Level 2 training data (spec Sections 21-23).

THE ONE RULE THIS MODULE EXISTS TO ENFORCE: every feature a Level 2 model
trains on must be computed by the SAME functions the live pipeline calls
(features/, regime/, reversal/) -- never a reimplementation "for training
purposes." A model trained on features computed one way and served on
features computed a slightly different way is a model that quietly stops
meaning what its training metrics said it meant. build_training_examples()
calls exactly the functions strategy/pipeline.py calls, in the same order,
with the same config -- it is a replay harness, not a parallel feature
engine.

LABELS ARE FORWARD-LOOKING AND COMPUTED FROM REAL SETTLEMENT, NEVER
ASSUMED. "Won" means the direction was correct `duration_bars` closed
candles later -- the same candle-close approximation the backtest and
reconcile_rejected_signals() already use (config's
contract.duration_bars_approx), so all three agree on what "settled" means.
A bar too close to the end of the available history to have a real
settlement outcome gets label=None and is excluded, never guessed.

ONE EXAMPLE PER BAR, NOT TWO. Every bar has both a bullish and a bearish
reading; using both as independent training rows would make two rows from
one bar highly correlated (their shared inputs mostly cancel, since
bullish_score and bearish_score are near-complements of the same evidence)
and would inflate the effective sample size claimed by any walk-forward
split. Each bar contributes one example: whichever direction currently
dominates (bull_score vs bear_score), with that direction's evidence as
the feature vector and that direction's real forward outcome as the label.
This is also what the earlier score/outcome correlation checks in this
conversation used, so results here are comparable to those.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from data.candles import Candle
from features.engine import build_features
from regime.detector import compute_regime
from reversal.divergence import compute_divergence
from reversal.exhaustion import compute_exhaustion
from reversal.price_action import compute_price_action
from reversal.scoring import ReversalWeights, score_reversal
from reversal.stretch import compute_stretch
from reversal.support_resistance import sr_bearish_evidence, sr_bullish_evidence

#: The feature names produced by to_feature_dict(), in a fixed order.
#: Models read this list rather than dict key order, which Python
#: guarantees is insertion order but which is easy to accidentally change
#: while editing this file -- an explicit list is the thing that can't
#: silently drift between training and inference.
FEATURE_NAMES = [
    "stretch_signed", "zscore_signed", "ema_distance_signed",
    "bollinger_pct_b_signed", "percentile_rank_signed",
    "exhaustion", "price_action_evidence", "sr_evidence", "divergence",
    "atr_pct", "rsi_signed", "rsi_slope_signed", "macd_hist_signed",
    "roc_signed", "ema_slope_signed", "adx", "regime_multiplier",
    "high_volatility", "low_volatility", "n_candles_available",
]


@dataclass(frozen=True)
class TrainingExample:
    symbol: str
    epoch: int
    direction: str              # "bullish" | "bearish" -- whichever dominated
    blended_score: float        # Level 1's own score, for comparison, not used as a model input
    features: dict               # name -> float, keys == FEATURE_NAMES
    label: bool | None           # None if not yet settleable (excluded from training)


def _sign(direction: str, value: float) -> float:
    """Flips a bearish-side reading onto the same scale as bullish, so
    'stretch_signed' means the same thing (evidence FOR the dominant
    direction) regardless of which direction that is. Without this, the
    model would have to learn the sign flip itself from twice the data."""
    return value if direction == "bullish" else -value


def _to_feature_dict(direction: str, feat, stretch, momentum_signed_extra,
                     exhaustion, pa_evidence, sr_evidence, divergence_val,
                     regime_snap, mult, volatility_regime, n_available) -> dict:
    m = feat.momentum
    return {
        "stretch_signed": _sign(direction, stretch.stretch_score),
        "zscore_signed": _sign(direction, -stretch.zscore),   # negative z = oversold = bullish evidence
        "ema_distance_signed": _sign(direction, -stretch.ema_distance),
        "bollinger_pct_b_signed": _sign(direction, 0.5 - stretch.bollinger_pct_b),
        "percentile_rank_signed": _sign(direction, 0.5 - stretch.percentile_rank),
        "exhaustion": 1.0 if exhaustion else 0.0,
        "price_action_evidence": pa_evidence,
        "sr_evidence": sr_evidence,
        "divergence": 1.0 if divergence_val else 0.0,
        "atr_pct": feat.volatility.atr_pct,
        "rsi_signed": _sign(direction, 50.0 - m.rsi),
        "rsi_slope_signed": _sign(direction, -m.rsi_slope),
        "macd_hist_signed": _sign(direction, -m.macd_hist),
        "roc_signed": _sign(direction, -m.roc),
        "ema_slope_signed": _sign(direction, -m.ema_slope),
        "adx": regime_snap.adx,
        "regime_multiplier": mult,
        "high_volatility": 1.0 if volatility_regime == "HIGH_VOLATILITY" else 0.0,
        "low_volatility": 1.0 if volatility_regime == "LOW_VOLATILITY" else 0.0,
        "n_candles_available": float(n_available),
    }


def build_training_examples(candles: list[Candle], cfg: dict, *,
                            max_lookback_override: int | None = None
                            ) -> list[TrainingExample]:
    """Replays `candles` (already CLOSED-only, chronological, one symbol)
    through the same feature/regime/scoring calls the live pipeline makes,
    producing one TrainingExample per bar once enough history exists.

    `max_lookback_override` exists only for tests -- production callers
    should rely on cfg's own candles.max_lookback_bars, the same value the
    live pipeline uses, so a test can never accidentally validate against a
    window size training wouldn't actually see live.
    """
    if not candles:
        return []
    symbol = candles[0].symbol
    weights = ReversalWeights(**cfg["reversal"]["weights"])
    weights.validate()
    max_lookback = (max_lookback_override if max_lookback_override is not None
                    else cfg["candles"]["max_lookback_bars"])
    duration_bars = cfg["contract"]["duration_bars_approx"]

    closes = [c.close for c in candles]
    examples: list[TrainingExample] = []

    for i in range(len(candles)):
        sub = candles[: i + 1]
        window = sub[-max_lookback:] if len(sub) > max_lookback else sub
        feat = build_features(window, cfg)
        if feat is None or not feat.sufficient_data:
            continue

        regime_snap = compute_regime(
            feat.high, feat.low, feat.close_series,
            ema_slope_value=feat.momentum.ema_slope,
            adx_period=cfg["regime"]["adx_period"],
            trend_slope_min=cfg["regime"]["trend_slope_min"],
            adx_trend_min=cfg["regime"]["adx_trend_min"],
            adx_range_max=cfg["regime"]["adx_range_max"],
            min_bars=cfg["regime"]["min_bars"])
        stretch = compute_stretch(
            feat.close_series, atr=feat.volatility.atr,
            lookback=cfg["stretch"]["lookback"],
            ema_fast_period=cfg["stretch"]["ema_fast_period"],
            bollinger_k=cfg["stretch"]["bollinger_k"],
            percentile_lookback=cfg["stretch"]["percentile_lookback"],
            z_cap=cfg["stretch"]["z_cap"], atr_cap=cfg["stretch"]["atr_cap"])
        exhaustion = compute_exhaustion(
            feat.momentum, rsi_oversold=cfg["momentum"]["rsi_oversold"],
            rsi_overbought=cfg["momentum"]["rsi_overbought"])
        pa = compute_price_action(
            feat.open_series, feat.high, feat.low, feat.close_series,
            swing_window=cfg["structure"]["swing_window"],
            rejection_wick_ratio=cfg["price_action"]["rejection_wick_ratio"],
            rejection_range_min=cfg["price_action"]["rejection_range_min"],
            failure_lookback=cfg["price_action"]["failure_lookback"])
        srb = sr_bullish_evidence(
            feat.structure, proximity_atr=cfg["structure"]["sr_proximity_atr"],
            min_strength=cfg["structure"]["sr_min_strength"])
        srs = sr_bearish_evidence(
            feat.structure, proximity_atr=cfg["structure"]["sr_proximity_atr"],
            min_strength=cfg["structure"]["sr_min_strength"])
        div = compute_divergence(
            feat.high, feat.low, feat.rsi_series, feat.macd_hist_series,
            swing_window=cfg["structure"]["swing_window"])
        score = score_reversal(stretch=stretch, exhaustion=exhaustion,
                               price_action=pa, sr_bullish=srb, sr_bearish=srs,
                               divergence=div, weights=weights)
        mult = cfg["regime"]["threshold_multiplier"].get(regime_snap.regime, 1.0)

        if score.bullish >= score.bearish:
            direction = "bullish"
            pa_evidence = pa.bullish_count() / 4.0
            sr_evidence = srb
            divergence_val = div.bullish_divergence
            exhaustion_val = exhaustion.bullish_exhaustion
            blended = score.bullish
        else:
            direction = "bearish"
            pa_evidence = pa.bearish_count() / 4.0
            sr_evidence = srs
            divergence_val = div.bearish_divergence
            exhaustion_val = exhaustion.bearish_exhaustion
            blended = score.bearish

        features = _to_feature_dict(
            direction, feat, stretch, None, exhaustion_val, pa_evidence,
            sr_evidence, divergence_val, regime_snap, mult,
            feat.volatility.volatility_regime, len(sub))

        settle_idx = i + duration_bars
        label = None
        if settle_idx < len(closes):
            entry, exitp = closes[i], closes[settle_idx]
            label = (exitp > entry) if direction == "bullish" else (exitp < entry)

        examples.append(TrainingExample(
            symbol=symbol, epoch=candles[i].close_epoch, direction=direction,
            blended_score=blended, features=features, label=label))

    return examples


def to_matrix(examples: list[TrainingExample]):
    """Feature matrix + label vector for sklearn, dropping unlabeled
    (not-yet-settleable) examples. Kept as a separate step from
    build_training_examples so callers can inspect unlabeled examples
    (e.g. the most recent few bars) without them silently vanishing."""
    import numpy as np
    labeled = [e for e in examples if e.label is not None]
    X = np.array([[e.features[name] for name in FEATURE_NAMES] for e in labeled])
    y = np.array([1.0 if e.label else 0.0 for e in labeled])
    epochs = np.array([e.epoch for e in labeled])
    return X, y, epochs, labeled
