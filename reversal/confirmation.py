"""
Setup vs confirmation (spec Section 17, STRATEGY_SPECIFICATION.md Section
10). A setup indicates developing conditions; confirmation is what makes it
tradeable. A setup that ages past its window without confirming is
DISCARDED -- Section 19's anti-lookahead principle read forward: a decision
can't be validated by something arbitrarily far in the future and still be
called the same decision.

AGE IS TRACKED SEPARATELY FROM ARRAY INDEXING, DELIBERATELY. The pipeline
passes a bounded, sliding lookback window to the feature engine (see
strategy/pipeline.py's max_lookback_bars), so the array position of "the
current bar" is constant (`len(window)-1`) on every call once the window is
full -- it cannot be subtracted against a stored index from an earlier call
to recover elapsed time, because both indices converge to the same value.
`age` is instead an explicit bar-count the caller maintains and passes in;
`bar_index` is used only to read the last position out of the arrays.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PendingSetup:
    direction: str          # "bullish" | "bearish"
    setup_high: float
    setup_low: float
    score_at_setup: float
    bars_waited: int = 0    # incremented by the caller once per closed candle


@dataclass(frozen=True)
class ConfirmationResult:
    confirmed: bool
    reason: str


def check_confirmation(setup: PendingSetup, *, age: int, bar_index: int,
                       high: list[float], low: list[float], close: list[float],
                       ema_fast: list[float], macd_hist: list[float],
                       confirmation_window: int = 3) -> ConfirmationResult:
    if age > confirmation_window:
        return ConfirmationResult(False, f"setup expired after {age} bars, discarded")
    if age == 0:
        return ConfirmationResult(False, "setup just formed, awaiting confirmation")

    i = bar_index
    if setup.direction == "bullish":
        follow_through = close[i] > setup.setup_high
        ema_reclaim = i < len(ema_fast) and close[i] > ema_fast[i]
        macd_cross = (i > 0 and i < len(macd_hist)
                     and macd_hist[i] > 0 and macd_hist[i - 1] <= 0)
    else:
        follow_through = close[i] < setup.setup_low
        ema_reclaim = i < len(ema_fast) and close[i] < ema_fast[i]
        macd_cross = (i > 0 and i < len(macd_hist)
                     and macd_hist[i] < 0 and macd_hist[i - 1] >= 0)

    if follow_through:
        return ConfirmationResult(True, "follow-through beyond setup bar extreme")
    if ema_reclaim:
        return ConfirmationResult(True, "EMA reclaim/loss in reversal direction")
    if macd_cross:
        return ConfirmationResult(True, "MACD histogram crossed zero in reversal direction")
    return ConfirmationResult(False, f"awaiting confirmation (bar {age}/{confirmation_window})")
