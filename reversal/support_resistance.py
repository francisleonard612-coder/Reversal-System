"""
Support/resistance evidence (spec Section 13, STRATEGY_SPECIFICATION.md
Section 9). Zone detection itself lives in features/structure.py, since
swing points feed both structure and price-action failed-break detection;
this module only converts a StructureSnapshot into the [0,1] evidence terms
the reversal score consumes.
"""
from __future__ import annotations

from features.structure import StructureSnapshot


def _proximity_evidence(distance_atr: float, *, proximity_atr: float) -> float:
    """1.0 at the zone, linearly down to 0.0 at 3x the proximity radius."""
    if distance_atr == float("inf"):
        return 0.0
    far = proximity_atr * 3.0
    if distance_atr >= far:
        return 0.0
    return max(0.0, 1.0 - distance_atr / far)


def sr_bullish_evidence(s: StructureSnapshot, *, proximity_atr: float = 1.0,
                        min_strength: float = 2.0) -> float:
    if s.nearest_support is None or s.nearest_support.strength < min_strength:
        base = 0.0
    else:
        base = _proximity_evidence(s.distance_to_support_atr, proximity_atr=proximity_atr)
    if s.lost_support:
        base = max(base, 0.6)   # reclaim is meaningful evidence even off-zone
    return base


def sr_bearish_evidence(s: StructureSnapshot, *, proximity_atr: float = 1.0,
                        min_strength: float = 2.0) -> float:
    if s.nearest_resistance is None or s.nearest_resistance.strength < min_strength:
        base = 0.0
    else:
        base = _proximity_evidence(s.distance_to_resistance_atr, proximity_atr=proximity_atr)
    if s.reclaimed_resistance:
        base = max(base, 0.6)
    return base
