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

import time
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

    def seed_closed(self, candles: list[Candle]) -> int:
        """Prime this builder with already-CLOSED candles (typically from
        candles_from_history() below), instead of rebuilding them
        tick-by-tick from raw history. Must be called before this builder
        has seen any tick -- seeding a builder mid-stream risks silently
        reordering or duplicating bars, so that case is rejected outright
        rather than guessed at.

        Every candle is validated (closed, correct symbol, correct
        timeframe, strictly increasing) before anything is accepted --
        a malformed seed would silently corrupt every feature computed
        downstream from `.closed`, and that is a far worse failure mode
        than raising here. Returns the number of candles actually seeded.
        """
        if self._current is not None or self.closed:
            raise ValueError(
                f"{self.symbol}: seed_closed must be called before any "
                f"tick is added to this builder")
        if not candles:
            return 0
        prev: Candle | None = None
        for i, c in enumerate(candles):
            if not c.is_closed:
                raise ValueError(
                    f"{self.symbol}: seed_closed candle at index {i} is "
                    f"not marked closed")
            if c.symbol != self.symbol:
                raise ValueError(
                    f"{self.symbol}: seed_closed candle at index {i} has "
                    f"symbol {c.symbol!r}, expected {self.symbol!r}")
            if c.timeframe_seconds != self.timeframe_seconds:
                raise ValueError(
                    f"{self.symbol}: seed_closed candle at index {i} has "
                    f"timeframe_seconds={c.timeframe_seconds}, expected "
                    f"{self.timeframe_seconds}")
            if prev is not None and c.open_epoch <= prev.open_epoch:
                raise ValueError(
                    f"{self.symbol}: seed_closed candles are not "
                    f"strictly increasing in open_epoch at index {i}")
            prev = c
        self.closed = list(candles)
        # Guards subsequent add_tick() calls: any live tick this builder
        # sees is guaranteed to arrive at or after "now" at seed time,
        # which is strictly after every seeded candle's close_epoch (see
        # candles_from_history's still-forming filter) -- so this can
        # never falsely reject a genuine live tick as out-of-order.
        self._last_epoch = float(candles[-1].close_epoch)
        return len(self.closed)

    def history(self, n: int | None = None) -> list[Candle]:
        """CLOSED candles only -- the forming candle is never returned, so a
        caller cannot accidentally read a bar that hasn't finished yet."""
        return self.closed[-n:] if n else list(self.closed)

    @property
    def current_forming(self) -> Candle | None:
        """Exposed ONLY for monitoring/dashboards. No feature or decision
        code may call this -- see the module docstring."""
        return self._current


def candles_from_history(symbol: str, historical: list, timeframe_seconds: int = 60,
                         now: float | None = None) -> list[Candle]:
    """Convert deriv.client.HistoricalCandle rows (server-aggregated OHLC,
    ascending epoch order expected) into CLOSED Candle objects suitable
    for CandleBuilder.seed_closed().

    The most recent row is dropped unless a FULL timeframe period has
    elapsed since its open epoch -- Deriv's history endpoint can include
    the still-forming current candle, and treating that as closed would
    let a feature computation see a bar before it actually finished (the
    exact repainting bug the rest of this module exists to prevent).
    Rows that are out of order or duplicate an epoch already seen are
    dropped rather than raising, since a defensive re-check here should
    never crash a cold start over a single bad upstream row -- the
    caller's own seed_closed() still validates strictly before accepting
    anything.
    """
    now = time.time() if now is None else now
    out: list[Candle] = []
    prev_epoch: int | None = None
    for hc in historical:
        if hc.epoch + timeframe_seconds > now:
            continue   # still forming -- never treat as closed
        if prev_epoch is not None and hc.epoch <= prev_epoch:
            continue   # defensive: non-increasing/duplicate row
        out.append(Candle(
            symbol=symbol, open_epoch=hc.epoch,
            close_epoch=hc.epoch + timeframe_seconds - 1,
            open=hc.open, high=hc.high, low=hc.low, close=hc.close,
            n_ticks=0,   # server-aggregated -- real per-candle tick count is unknown
            is_closed=True, has_volume=False,
            timeframe_seconds=timeframe_seconds))
        prev_epoch = hc.epoch
    return out


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
