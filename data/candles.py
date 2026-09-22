"""
Candle construction from Deriv ticks (spec Sections 1, 5, 6, 19).

Built from ticks rather than fetched from Deriv's own candle endpoint. The
reason is anti-repainting (Section 19), not a preference: building candles
ourselves means we know exactly which ticks are inside each bar and exactly
when a bar closes, so "use only CLOSED candles" (the rule every feature in
this system follows) is enforceable in code rather than assumed about a
third-party aggregation we don't control.

NO FABRICATED VOLUME (Section 5). Deriv synthetics have none. `has_volume`
is False on every candle this module produces; nothing downstream should
ever read a volume field as if it were real.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Candle:
    symbol: str
    open_epoch: int          # first tick's epoch in the bar
    close_epoch: int         # last tick's epoch in the bar (bar is CLOSED once this is set)
    open: float
    high: float
    low: float
    close: float
    n_ticks: int
    is_closed: bool = False
    has_volume: bool = False   # always False -- see module docstring
    timeframe_seconds: int = 60


class CandleBuilder:
    """One instance per symbol. Feed ticks in strict chronological order;
    out-of-order epochs are rejected rather than silently reordered, since a
    reordered tick stream would corrupt which candle a tick actually
    belongs to.
    """

    def __init__(self, symbol: str, timeframe_seconds: int = 60):
        self.symbol = symbol
        self.timeframe_seconds = timeframe_seconds
        self._current: Candle | None = None
        self._last_epoch: float = -1.0
        self.closed: list[Candle] = []

    def _bucket(self, epoch: float) -> int:
        return int(epoch) // self.timeframe_seconds

    def add_tick(self, epoch: float, price: float) -> Candle | None:
        """Returns the newly CLOSED candle if this tick started a new bar,
        else None. The caller decides what "closed" triggers (feature
        recompute, signal evaluation); this class only tracks boundaries.
        """
        if epoch < self._last_epoch:
            raise ValueError(
                f"{self.symbol}: tick epoch {epoch} is before the last seen "
                f"epoch {self._last_epoch} -- ticks must arrive in order")
        self._last_epoch = epoch

        bucket = self._bucket(epoch)
        newly_closed: Candle | None = None

        if self._current is None:
            self._current = Candle(
                symbol=self.symbol, open_epoch=int(epoch), close_epoch=int(epoch),
                open=price, high=price, low=price, close=price, n_ticks=1,
                timeframe_seconds=self.timeframe_seconds)
            self._current_bucket = bucket
            return None

        if bucket != self._current_bucket:
            self._current.is_closed = True
            self.closed.append(self._current)
            newly_closed = self._current
            self._current = Candle(
                symbol=self.symbol, open_epoch=int(epoch), close_epoch=int(epoch),
                open=price, high=price, low=price, close=price, n_ticks=1,
                timeframe_seconds=self.timeframe_seconds)
            self._current_bucket = bucket
        else:
            c = self._current
            c.high = max(c.high, price)
            c.low = min(c.low, price)
            c.close = price
            c.close_epoch = int(epoch)
            c.n_ticks += 1

        return newly_closed

    def history(self, n: int | None = None) -> list[Candle]:
        """CLOSED candles only -- the forming candle is never returned, so a
        caller cannot accidentally read a bar that hasn't finished yet."""
        return self.closed[-n:] if n else list(self.closed)

    @property
    def current_forming(self) -> Candle | None:
        """Exposed ONLY for monitoring/dashboards. No feature or decision
        code may call this -- see the module docstring."""
        return self._current


def candles_from_ticks(symbol: str, epochs: list[float], prices: list[float],
                       timeframe_seconds: int = 60) -> list[Candle]:
    """Batch helper for historical replay (backtest, collector). The forming
    (final, possibly incomplete) candle is intentionally dropped -- a
    backtest must never evaluate against a bar that wasn't actually closed
    at that point in history.
    """
    builder = CandleBuilder(symbol, timeframe_seconds)
    for e, p in zip(epochs, prices):
        builder.add_tick(e, p)
    return builder.closed
