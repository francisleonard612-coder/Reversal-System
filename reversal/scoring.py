"""
Reversal scoring (spec Section 16, STRATEGY_SPECIFICATION.md Section 9).

A plain documented weighted sum, deliberately -- Section 16 asks for
"configurable weights... document the mathematical calculation," not for a
learned or opaque blend. The weights are validated to sum to 1.0 at startup
(see config.py); a set that doesn't sum to 1.0 fails loudly rather than
being silently renormalized, since a silent rescale would drift what the
0-100 score means without anyone deciding that.
"""
from __future__ import annotations

from dataclasses import dataclass

from reversal.divergence import DivergenceSnapshot
from reversal.exhaustion import ExhaustionSnapshot
from reversal.price_action import PriceActionSnapshot
from reversal.stretch import StretchSnapshot

_PATTERN_TYPES = 4   # bullish_count()/bearish_count() range over this many


@dataclass(frozen=True)
class ReversalWeights:
    stretch: float = 0.30
    exhaustion: float = 0.25
    price_action: float = 0.20
    support_resistance: float = 0.15
    divergence: float = 0.10

    def validate(self) -> None:
        total = (self.stretch + self.exhaustion + self.price_action
                 + self.support_resistance + self.divergence)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"reversal.weights must sum to 1.0, got {total:.4f} "
                f"({self.stretch}+{self.exhaustion}+{self.price_action}+"
                f"{self.support_resistance}+{self.divergence}) -- fix "
                f"config/settings.yaml rather than relying on a rescale")


@dataclass(frozen=True)
class ReversalScore:
    bullish: float    # [0, 100]
    bearish: float    # [0, 100]
    detail: dict       # per-component evidence, for logging/diagnostics


def score_reversal(*, stretch: StretchSnapshot, exhaustion: ExhaustionSnapshot,
                   price_action: PriceActionSnapshot,
                   sr_bullish: float, sr_bearish: float,
                   divergence: DivergenceSnapshot,
                   weights: ReversalWeights) -> ReversalScore:
    weights.validate()

    pa_bullish = price_action.bullish_count() / _PATTERN_TYPES
    pa_bearish = price_action.bearish_count() / _PATTERN_TYPES

    bullish = (
        weights.stretch * max(stretch.stretch_score, 0.0)
        + weights.exhaustion * (1.0 if exhaustion.bullish_exhaustion else 0.0)
        + weights.price_action * pa_bullish
        + weights.support_resistance * sr_bullish
        + weights.divergence * (1.0 if divergence.bullish_divergence else 0.0)
    )
    bearish = (
        weights.stretch * max(-stretch.stretch_score, 0.0)
        + weights.exhaustion * (1.0 if exhaustion.bearish_exhaustion else 0.0)
        + weights.price_action * pa_bearish
        + weights.support_resistance * sr_bearish
        + weights.divergence * (1.0 if divergence.bearish_divergence else 0.0)
    )

    detail = {
        "stretch_score": stretch.stretch_score,
        "bullish_exhaustion": exhaustion.bullish_exhaustion,
        "bearish_exhaustion": exhaustion.bearish_exhaustion,
        "pa_bullish_evidence": pa_bullish,
        "pa_bearish_evidence": pa_bearish,
        "sr_bullish_evidence": sr_bullish,
        "sr_bearish_evidence": sr_bearish,
        "bullish_divergence": divergence.bullish_divergence,
        "bearish_divergence": divergence.bearish_divergence,
        "divergence_evaluable": divergence.evaluable,
    }
    return ReversalScore(bullish=max(0.0, min(100.0, 100.0 * bullish)),
                         bearish=max(0.0, min(100.0, 100.0 * bearish)),
                         detail=detail)
