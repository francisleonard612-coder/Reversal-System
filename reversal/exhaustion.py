"""
Exhaustion, distinguished from a mere extreme reading (spec Section 11,
STRATEGY_SPECIFICATION.md Section 4).

An extreme RSI reading persists for many bars in a strong trend -- treating
"RSI > 70" alone as reversal evidence would fire constantly during exactly
the trends it should be most cautious about. Exhaustion additionally
requires the extreme to be DECELERATING: RSI turning back from its extreme,
AND the MACD histogram corroborating that momentum is fading, not just
price.
"""
from __future__ import annotations

from dataclasses import dataclass

from features.momentum import MomentumSnapshot


@dataclass(frozen=True)
class ExhaustionSnapshot:
    bullish_exhaustion: bool
    bearish_exhaustion: bool


def compute_exhaustion(m: MomentumSnapshot, *, rsi_oversold: float = 30.0,
                       rsi_overbought: float = 70.0) -> ExhaustionSnapshot:
    bullish = (m.rsi < rsi_oversold and m.rsi_slope > 0
              and m.macd_hist > m.macd_hist_prev)
    bearish = (m.rsi > rsi_overbought and m.rsi_slope < 0
              and m.macd_hist < m.macd_hist_prev)
    return ExhaustionSnapshot(bullish_exhaustion=bullish, bearish_exhaustion=bearish)
