"""
Contract economics (spec Sections 25, 26, 38) -- Level 1 scope only.

READ THIS BEFORE ADDING A PROBABILITY TO THIS FILE. Sections 25 and 26 both
require a MODEL-IMPLIED PROBABILITY: "EV = P(win) x net_win - P(loss) x
loss" and "compare model-implied probability vs probability required by
contract economics." Level 1 is defined in Section 3 as pure deterministic
logic with no probability estimation -- that's explicitly Level 2's job
(Section 21: "the system estimates P(success | current market state)").

So Level 1 has no P(win) to compute a true expected value from, and Section
62 forbids exactly the shortcut of inventing one ("do not create... fake
probabilities... fabricated profitability"). A win rate borrowed from a
backtest bucket would BE a probability estimate wearing a disguise, and a
flat assumed probability (0.5, or a hand-picked "optimistic" number) would
be worse -- confidently wrong rather than honestly absent.

What Level 1 CAN do honestly: verify the contract's quoted economics aren't
poor enough to make even a well-timed signal not worth it. That's a payout
floor, not an EV. `EdgeAssessment.expected_value` is `None` here and stays
`None` until Level 2 supplies a calibrated probability to compute it from --
the field is spelled out as absent rather than populated with 0.0 or
anything else that could be mistaken for "computed and found neutral".
"""
from __future__ import annotations

import time
from dataclasses import dataclass


class ProposalError(RuntimeError):
    pass


@dataclass(frozen=True)
class Proposal:
    contract_type: str        # CALL | PUT
    symbol: str
    stake: float
    payout: float
    ask_price: float
    currency: str
    proposal_id: str
    received_at: float = 0.0

    @property
    def payout_multiple(self) -> float:
        return self.payout / self.stake if self.stake > 0 else 0.0

    def is_stale(self, now: float | None = None, max_age_seconds: float = 5.0) -> bool:
        now = time.time() if now is None else now
        return (now - self.received_at) > max_age_seconds


@dataclass(frozen=True)
class EdgeAssessment:
    proposal: Proposal
    expected_value: float | None    # None at Level 1 -- see module docstring
    payout_floor_met: bool
    reason: str
    # Level 2 only (None when no promoted model is active)
    probability: float | None = None
    probability_lower: float | None = None
    ev_ok: bool = True

    @property
    def economically_sound(self) -> bool:
        return self.payout_floor_met and self.ev_ok


def assess_level1_economics(proposal: Proposal, *,
                            min_payout_multiple: float = 1.80) -> EdgeAssessment:
    """Level 1's honest substitute for an EV check: is the quoted payout
    above a configured floor? This is NOT expected value -- see the module
    docstring for why Level 1 doesn't compute one."""
    met = proposal.payout_multiple >= min_payout_multiple
    reason = (f"payout {proposal.payout_multiple:.3f}x >= floor "
             f"{min_payout_multiple:.3f}x" if met else
             f"payout {proposal.payout_multiple:.3f}x below floor "
             f"{min_payout_multiple:.3f}x -- refusing without a probability "
             f"model to weigh it against")
    return EdgeAssessment(proposal=proposal, expected_value=None,
                          payout_floor_met=met, reason=reason)


def assess_level2_economics(proposal: Proposal, *, probability: float,
                            probability_lower: float, min_payout_multiple: float = 1.80,
                            min_ev: float = 0.0,
                            require_lower_bound_edge: bool = True) -> EdgeAssessment:
    """Sections 25/26 made real, once a PROMOTED Level 2 model supplies a
    calibrated P(win): EV per unit staked = p * payout_multiple - 1, priced
    against the actual proposal's payout. The Level 1 payout floor still
    applies on top.

    With `require_lower_bound_edge`, the conservative lower bound of P(win)
    must also exceed break-even (1 / payout_multiple) -- an edge that only
    exists at the point estimate is within the model's own error bars.
    """
    m = proposal.payout_multiple
    floor_met = m >= min_payout_multiple
    ev = probability * m - 1.0
    breakeven = 1.0 / m if m > 0 else 1.0
    reasons = []
    if not floor_met:
        reasons.append(f"payout {m:.3f}x below floor {min_payout_multiple:.3f}x")
    if ev < min_ev:
        reasons.append(f"EV {ev:+.4f} below minimum {min_ev:+.4f} (p={probability:.4f})")
    if require_lower_bound_edge and probability_lower <= breakeven:
        reasons.append(f"P(win) lower bound {probability_lower:.4f} does not clear "
                       f"break-even {breakeven:.4f}")
    ev_ok = ev >= min_ev and (not require_lower_bound_edge or probability_lower > breakeven)
    reason = ("; ".join(reasons) if reasons else
              f"p={probability:.4f} (lower {probability_lower:.4f}) vs break-even "
              f"{breakeven:.4f}, EV {ev:+.4f} at {m:.3f}x")
    return EdgeAssessment(proposal=proposal, expected_value=ev, payout_floor_met=floor_met,
                          reason=reason, probability=probability,
                          probability_lower=probability_lower, ev_ok=ev_ok)
