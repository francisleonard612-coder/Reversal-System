"""
Data quality engine (spec Section 7). Bad data must not silently enter the
candle builder or the feature pipeline -- every check here runs before a
tick is handed to data/candles.py.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TickValidation:
    valid: bool
    reason: str = ""


class TickValidator:
    """Stateful per-symbol: duplicate and gap detection need the previous
    tick to compare against."""

    def __init__(self, *, max_gap_seconds: float = 30.0,
                price_jump_sigma: float = 8.0, jump_window: int = 200):
        self.max_gap_seconds = max_gap_seconds
        self.price_jump_sigma = price_jump_sigma
        self.jump_window = jump_window
        self._last_epoch: float | None = None
        self._last_price: float | None = None
        self._recent_abs_returns: list[float] = []

    def validate(self, epoch: float, price: float) -> TickValidation:
        if price != price or price <= 0:   # NaN or non-positive
            return TickValidation(False, f"invalid price {price!r}")
        if self._last_epoch is not None and epoch < self._last_epoch:
            return TickValidation(False,
                f"out-of-order tick: epoch {epoch} < last seen {self._last_epoch}")
        if self._last_epoch is not None and epoch == self._last_epoch:
            return TickValidation(False, "duplicate tick (same epoch as previous)")

        if self._last_epoch is not None:
            gap = epoch - self._last_epoch
            if gap > self.max_gap_seconds:
                # Not fatal by itself -- flagged so the caller can decide
                # whether to treat the stream as stale (Section 6), but the
                # tick's own price is still usable once the gap is known.
                reason = f"gap of {gap:.1f}s since last tick (stream may be stale)"
            else:
                reason = ""
            if self._last_price and self._recent_abs_returns:
                ret = abs(price - self._last_price) / self._last_price
                window = self._recent_abs_returns[-self.jump_window:]
                if len(window) >= 20:
                    mean = sum(window) / len(window)
                    var = sum((r - mean) ** 2 for r in window) / len(window)
                    sd = var ** 0.5
                    if sd > 0 and ret > mean + self.price_jump_sigma * sd:
                        return TickValidation(
                            False, f"price jump {ret:.4%} exceeds "
                                  f"{self.price_jump_sigma} sigma of recent returns")
        else:
            reason = ""

        if self._last_price is not None:
            self._recent_abs_returns.append(abs(price - self._last_price) / self._last_price
                                            if self._last_price else 0.0)
            self._recent_abs_returns = self._recent_abs_returns[-self.jump_window:]
        self._last_epoch = epoch
        self._last_price = price
        return TickValidation(True, reason)
