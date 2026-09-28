"""
Monte Carlo-calibrated contract expiry (ticks AND candle-minutes).

THE QUESTION THIS ANSWERS: given a confirmed reversal setup, over what
horizon does the reversal actually play out -- 5 ticks, 20 ticks, 2 minutes,
5 minutes? A fixed expiry either cuts a slow reversal off before it
completes or holds a fast one long enough for noise to take it back.

HOW, WITHOUT GUESSING:

1. EVIDENCE. Every qualifying setup (traded or not) is followed forward and
   its real outcome recorded at EVERY candidate horizon (`setup_outcomes`
   table): tick horizons from the live tick stream, minute horizons from
   live ticks and, for a head start, from replaying stored candle history
   (`python main.py --calibrate-expiry`). One setup -> one row per horizon.

2. MONTE CARLO CALIBRATION. For each horizon, draw the win rate from its
   Beta(wins+1, losses+1) posterior (`mc_draws` samples) and take a LOW
   quantile as a conservative P(win). The quantile is divided by the number
   of horizons being compared (Bonferroni): picking the best of eight noisy
   estimates otherwise systematically picks whichever got luckiest.

3. PRICING. The top horizons by that lower bound are priced with REAL
   Deriv proposals (tick and minute contracts pay differently), and the
   expiry with the best worst-case EV -- p_lower x payout_multiple - 1 --
   wins. With `require_edge`, a trade needs that worst case to be positive.

4. NO DATA, NO PRETENDING. A horizon with fewer than `min_samples` outcomes
   is not eligible. With no eligible horizon the bot falls back to the
   default contract under plain Level 1 rules, so trades still open while
   evidence accumulates -- and those trades generate more evidence.

Sample selection prefers CONFIRMED setups (the population actually traded);
it falls back to all qualified setups for a horizon only while confirmed
ones are too few, and says which it used.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

import numpy as np

_HORIZON_RE = re.compile(r"^\s*(\d+)\s*([tsm])\s*$", re.I)


@dataclass(frozen=True)
class Horizon:
    value: int
    unit: str      # "t" ticks | "s" seconds | "m" minutes

    @property
    def label(self) -> str:
        return f"{self.value}{self.unit}"

    @property
    def is_ticks(self) -> bool:
        return self.unit == "t"

    @property
    def seconds(self) -> int | None:
        """Wall-clock length; None for tick horizons."""
        mult = {"s": 1, "m": 60}.get(self.unit)
        return None if mult is None else self.value * mult

    def approx_seconds(self, seconds_per_tick: float = 2.0) -> float:
        return self.value * seconds_per_tick if self.is_ticks else float(self.seconds)

    @classmethod
    def parse(cls, text: str) -> "Horizon":
        m = _HORIZON_RE.match(str(text))
        if not m:
            raise ValueError(f"bad horizon {text!r}: use e.g. 5t, 30s, 3m")
        return cls(int(m.group(1)), m.group(2).lower())


def parse_horizons(items) -> list[Horizon]:
    hs = [Horizon.parse(x) for x in items]
    if len({h.label for h in hs}) != len(hs):
        raise ValueError("duplicate expiry horizons")
    return hs


_DUR_RE = re.compile(r"^\s*(\d+)\s*([tsmhd])\s*$", re.I)


def _to_units(text: str) -> tuple[str, float] | None:
    """"10t" -> ("t", 10); "15s" -> ("s", 15); "1d" -> ("s", 86400)."""
    m = _DUR_RE.match(str(text))
    if not m:
        return None
    v, u = int(m.group(1)), m.group(2).lower()
    if u == "t":
        return "t", v
    return "s", v * {"s": 1, "m": 60, "h": 3600, "d": 86400}[u]


def offered_horizons(horizons: list[Horizon], limits: list[tuple]) -> list[Horizon]:
    """Only the horizons inside a Deriv-reported (min, max) duration range
    of the same kind (ticks vs time). With no limits known, all are kept."""
    if not limits:
        return list(horizons)
    ranges = []
    for lo, hi in limits:
        a, b = _to_units(lo), _to_units(hi)
        if a and b and a[0] == b[0]:
            ranges.append((a[0], a[1], b[1]))
    if not ranges:
        return list(horizons)
    keep = []
    for h in horizons:
        kind, val = ("t", h.value) if h.is_ticks else ("s", h.seconds)
        if any(k == kind and lo <= val <= hi for k, lo, hi in ranges):
            keep.append(h)
    return keep


# ------------------------------------------------------------ calibration

@dataclass
class HorizonEstimate:
    horizon: Horizon
    wins: int
    n: int
    population: str          # "confirmed" | "qualified" | "pooled-confirmed" | ...
    p_mean: float
    p_lower: float

    def describe(self) -> str:
        return (f"{self.horizon.label}: {self.wins}/{self.n} won ({self.population}), "
                f"MC p={self.p_mean:.3f} lower={self.p_lower:.3f}")


@dataclass
class ExpiryCalibrator:
    """Stateless apart from outcome counts, which the live loop refreshes
    from the database on a timer."""
    horizons: list[Horizon]
    min_samples: int = 40
    credible_quantile: float = 0.10
    mc_draws: int = 4000
    pool_symbols: bool = True
    seed: int = 0
    # {(symbol, direction, horizon_label, confirmed:bool): (wins, n)}
    counts: dict = field(default_factory=dict)
    refreshed_at: float = 0.0

    def load_counts(self, rows) -> None:
        """rows: iterable of (symbol, direction, horizon_label, confirmed, wins, n)."""
        self.counts = {(s, d, h, bool(c)): (int(w), int(n)) for s, d, h, c, w, n in rows}
        self.refreshed_at = time.time()

    def _sample(self, symbol: str, direction: str, label: str) -> tuple[int, int, str] | None:
        def get(sym, confirmed):
            if sym is None:
                w = n = 0
                for (s, d, h, c), (ww, nn) in self.counts.items():
                    if d == direction and h == label and (c or not confirmed):
                        w, n = w + ww, n + nn
                return w, n
            w1, n1 = self.counts.get((sym, direction, label, True), (0, 0))
            if confirmed:
                return w1, n1
            w0, n0 = self.counts.get((sym, direction, label, False), (0, 0))
            return w1 + w0, n1 + n0

        chain = [(symbol, True, "confirmed"), (symbol, False, "qualified")]
        if self.pool_symbols:
            chain += [(None, True, "pooled-confirmed"), (None, False, "pooled-qualified")]
        for sym, confirmed, name in chain:
            w, n = get(sym, confirmed)
            if n >= self.min_samples:
                return w, n, name
        return None

    def estimates(self, symbol: str, direction: str,
                  horizons: list[Horizon] | None = None) -> list[HorizonEstimate]:
        """Eligible horizons, best conservative P(win) first. `horizons`
        narrows the set (e.g. to what Deriv offers on this symbol) -- and
        the Bonferroni split counts only those actually compared."""
        rng = np.random.default_rng(self.seed)
        found = []
        for h in (horizons if horizons is not None else self.horizons):
            s = self._sample(symbol, direction, h.label)
            if s is not None:
                found.append((h, *s))
        if not found:
            return []
        # Bonferroni: the lower bound must hold for whichever horizon we end
        # up picking out of all of them, not just for one chosen in advance.
        q = self.credible_quantile / len(found)
        out = []
        for h, w, n, pop in found:
            draws = rng.beta(w + 1, n - w + 1, size=self.mc_draws)
            out.append(HorizonEstimate(h, w, n, pop, float(draws.mean()),
                                       float(np.quantile(draws, q))))
        out.sort(key=lambda e: -e.p_lower)
        return out


# ---------------------------------------------------------- live tracking

@dataclass
class _Pending:
    symbol: str
    direction: str
    confirmed: bool
    setup_epoch: int
    score: float
    regime: str
    entry_price: float | None = None
    entry_epoch: float | None = None
    ticks_after_entry: int = 0
    last_price: float | None = None
    last_epoch: float | None = None
    done: set = field(default_factory=set)


class OutcomeTracker:
    """Follows each qualifying setup forward tick by tick and emits one
    outcome per horizon. Entry is the first tick AFTER the setup (what a
    contract bought at that moment would use as its entry spot); a tick
    horizon of k exits k ticks after entry, a time horizon exits on the last
    tick at or before entry + duration."""

    def __init__(self, horizons: list[Horizon], sink):
        self.horizons = horizons
        self.sink = sink            # callable(dict) -> None
        self.pending: dict[str, list[_Pending]] = {}

    def start(self, *, symbol, direction, confirmed, setup_epoch, score, regime) -> None:
        self.pending.setdefault(symbol, []).append(_Pending(
            symbol, direction, bool(confirmed), int(setup_epoch), float(score), regime))

    def on_tick(self, symbol: str, epoch: float, price: float) -> None:
        items = self.pending.get(symbol)
        if not items:
            return
        keep = []
        for p in items:
            if p.entry_price is None:
                p.entry_price, p.entry_epoch = price, epoch
                p.last_price, p.last_epoch = price, epoch
                keep.append(p)
                continue
            p.ticks_after_entry += 1
            for h in self.horizons:
                if h.label in p.done:
                    continue
                if h.is_ticks and p.ticks_after_entry == h.value:
                    self._emit(p, h, price)
                elif not h.is_ticks and epoch > p.entry_epoch + h.seconds:
                    self._emit(p, h, p.last_price)   # last tick at/before expiry
            p.last_price, p.last_epoch = price, epoch
            if len(p.done) < len(self.horizons):
                keep.append(p)
        self.pending[symbol] = keep

    def _emit(self, p: _Pending, h: Horizon, exit_price: float) -> None:
        p.done.add(h.label)
        won = exit_price > p.entry_price if p.direction == "bullish" else exit_price < p.entry_price
        self.sink({"symbol": p.symbol, "setup_epoch": p.setup_epoch, "direction": p.direction,
                   "confirmed": p.confirmed, "score": p.score, "regime": p.regime,
                   "horizon": h.label, "entry_price": p.entry_price, "exit_price": exit_price,
                   "won": bool(won), "source": "live"})


# ------------------------------------------------------ historical replay

class SetupDeduper:
    """A setup that stays qualified for several bars while awaiting
    confirmation would otherwise be counted as several independent samples
    of what is really one event, overstating the evidence. Unconfirmed
    re-qualifications of the same direction within `window_seconds` are
    skipped; confirmations are always kept."""

    def __init__(self, window_seconds: float):
        self.window = window_seconds
        self.last: dict = {}

    def accept(self, symbol: str, direction: str, confirmed: bool, epoch: float) -> bool:
        key = (symbol, direction)
        prev = self.last.get(key)
        if not confirmed and prev is not None and epoch - prev <= self.window:
            return False
        self.last[key] = epoch
        return True


def qualifying_direction(decision) -> tuple[str, bool] | None:
    """(direction, confirmed) if this decision is a setup worth following,
    else None. A setup qualifies when one direction cleared its
    regime-adjusted threshold -- confirmed (TRADE_*) or still awaiting
    confirmation."""
    if decision.decision == "TRADE_BULLISH":
        return "bullish", True
    if decision.decision == "TRADE_BEARISH":
        return "bearish", True
    if decision.reason_code == "NO_TRADE_UNCONFIRMED":
        return ("bullish" if decision.bullish_score >= decision.bearish_score else "bearish"), False
    return None


def replay_setup_outcomes(candles: list, cfg: dict, horizons: list[Horizon]) -> list[dict]:
    """Minute horizons only (stored history has no ticks). Entry = the setup
    candle's close, exit = the close `k` candles later -- same approximation
    the backtest uses. Tick horizons are learned live."""
    from strategy.pipeline import SymbolPipeline
    tf = cfg["candles"]["timeframe_seconds"]
    minute_h = [h for h in horizons if not h.is_ticks and h.seconds % tf == 0]
    if not candles or not minute_h:
        return []
    pl = SymbolPipeline(candles[0].symbol, cfg)
    max_lookback = cfg["candles"].get("max_lookback_bars", 400)
    dedup = SetupDeduper(cfg["reversal"]["confirmation_window"] * tf)
    out = []
    for i in range(len(candles)):
        d = pl.evaluate_on_close(candles[max(0, i + 1 - max_lookback): i + 1])
        q = qualifying_direction(d)
        if q is None:
            continue
        direction, confirmed = q
        if not dedup.accept(candles[i].symbol, direction, confirmed, candles[i].close_epoch):
            continue
        entry = candles[i].close
        for h in minute_h:
            j = i + h.seconds // tf
            if j >= len(candles):
                continue
            exitp = candles[j].close
            out.append({"symbol": candles[i].symbol, "setup_epoch": candles[i].close_epoch,
                        "direction": direction, "confirmed": confirmed,
                        "score": max(d.bullish_score, d.bearish_score), "regime": d.regime,
                        "horizon": h.label, "entry_price": entry, "exit_price": exitp,
                        "won": (exitp > entry) if direction == "bullish" else (exitp < entry),
                        "source": "replay"})
    return out


def format_table(cal: ExpiryCalibrator, symbols: list[str]) -> str:
    """Human-readable calibration table for logs / --expiry-report."""
    lines = []
    for sym in symbols:
        for direction in ("bullish", "bearish"):
            ests = cal.estimates(sym, direction)
            if not ests:
                lines.append(f"{sym} {direction}: no expiry has {cal.min_samples}+ outcomes yet")
                continue
            lines.append(f"{sym} {direction}:")
            lines.extend("   " + e.describe() for e in ests)
    return "\n".join(lines)
