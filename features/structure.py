"""
Structure and support/resistance (spec Section 13, STRATEGY_SPECIFICATION.md
Section 6).

THE ANTI-LOOKAHEAD RULE, MADE MECHANICAL. A swing high at bar i needs
`swing_window` bars on BOTH sides to confirm -- which means it cannot be
known until `swing_window` bars after it happened. `find_swings` returns
swings already offset by that lag: the caller never sees a swing point
before it could actually have been known. This is what Section 13's "avoid
lookahead bias... do not use future-confirmed pivots in a way that makes
historical signals unrealistically early" means as code.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SwingPoint:
    index: int          # index into the candle series this was computed from
    price: float
    kind: str            # "high" | "low"


def find_swings(high: list[float], low: list[float], *,
                swing_window: int = 3) -> list[SwingPoint]:
    """Only returns swings that are fully confirmable given the data passed
    in -- i.e. up to `len(high) - swing_window - 1`. A caller passing the
    full CLOSED history up to "now" therefore gets exactly the swings that
    could have been known at "now", automatically.
    """
    n = len(high)
    swings: list[SwingPoint] = []
    if n < 2 * swing_window + 1:
        return swings
    for i in range(swing_window, n - swing_window):
        window_high = high[i - swing_window: i + swing_window + 1]
        if high[i] == max(window_high) and window_high.count(high[i]) == 1:
            swings.append(SwingPoint(i, high[i], "high"))
        window_low = low[i - swing_window: i + swing_window + 1]
        if low[i] == min(window_low) and window_low.count(low[i]) == 1:
            swings.append(SwingPoint(i, low[i], "low"))
    return swings


@dataclass
class Zone:
    price: float           # merged (average) price of the zone
    kind: str               # "support" | "resistance"
    strength: float
    touches: int
    last_touch_index: int


def build_zones(swings: list[SwingPoint], *, current_index: int,
                zone_merge_pct: float = 0.0015,
                zone_decay_half_life: float = 100.0) -> list[Zone]:
    """Merges nearby swing points into zones and scores strength by
    recency-decayed touch count, per the spec formula:
    strength = sum(0.5 ** (age / half_life) for each touch).
    """
    highs = sorted((s for s in swings if s.kind == "high"), key=lambda s: s.price)
    lows = sorted((s for s in swings if s.kind == "low"), key=lambda s: s.price)
    zones: list[Zone] = []
    for kind, points in (("resistance", highs), ("support", lows)):
        cluster: list[SwingPoint] = []
        for p in points:
            if cluster and abs(p.price - cluster[-1].price) / max(cluster[-1].price, 1e-9) > zone_merge_pct:
                zones.append(_finalize_zone(cluster, kind, current_index, zone_decay_half_life))
                cluster = []
            cluster.append(p)
        if cluster:
            zones.append(_finalize_zone(cluster, kind, current_index, zone_decay_half_life))
    return zones


def _finalize_zone(cluster: list[SwingPoint], kind: str, current_index: int,
                   half_life: float) -> Zone:
    price = sum(p.price for p in cluster) / len(cluster)
    strength = sum(0.5 ** (max(current_index - p.index, 0) / half_life) for p in cluster)
    last_touch = max(p.index for p in cluster)
    return Zone(price=price, kind=kind, strength=strength, touches=len(cluster),
               last_touch_index=last_touch)


@dataclass(frozen=True)
class StructureSnapshot:
    nearest_support: Zone | None
    nearest_resistance: Zone | None
    distance_to_support_atr: float
    distance_to_resistance_atr: float
    reclaimed_resistance: bool   # closed back below a zone it had broken above
    lost_support: bool


def nearest_zone(zones: list[Zone], price: float, kind: str) -> Zone | None:
    candidates = [z for z in zones if z.kind == kind]
    if not candidates:
        return None
    return min(candidates, key=lambda z: abs(z.price - price))


def compute_structure(high: list[float], low: list[float], close: list[float],
                      *, atr: float, swing_window: int = 3,
                      zone_merge_pct: float = 0.0015,
                      zone_decay_half_life: float = 100.0,
                      reclaim_lookback: int = 5) -> StructureSnapshot:
    n = len(close)
    if n < 2 * swing_window + 1 or atr <= 0:
        return StructureSnapshot(None, None, float("inf"), float("inf"), False, False)

    swings = find_swings(high, low, swing_window=swing_window)
    zones = build_zones(swings, current_index=n - 1, zone_merge_pct=zone_merge_pct,
                        zone_decay_half_life=zone_decay_half_life)
    price = close[-1]
    support = nearest_zone(zones, price, "support")
    resistance = nearest_zone(zones, price, "resistance")

    dist_support = abs(price - support.price) / atr if support else float("inf")
    dist_resistance = abs(price - resistance.price) / atr if resistance else float("inf")

    window = close[-reclaim_lookback - 1:-1] if n > reclaim_lookback else close[:-1]
    reclaimed_resistance = bool(
        resistance and any(c > resistance.price for c in window) and price < resistance.price)
    lost_support = bool(
        support and any(c < support.price for c in window) and price > support.price)

    return StructureSnapshot(
        nearest_support=support, nearest_resistance=resistance,
        distance_to_support_atr=dist_support,
        distance_to_resistance_atr=dist_resistance,
        reclaimed_resistance=reclaimed_resistance, lost_support=lost_support)
