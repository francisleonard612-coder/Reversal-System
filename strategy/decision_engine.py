"""
NO-TRADE decision engine (spec Section 18, STRATEGY_SPECIFICATION.md
Section 11).

Simpler than the sibling Even/Odd engine's opportunity-score layer, and
deliberately so: Level 1 has exactly one continuous quantity per direction
(the reversal score from reversal/scoring.py, itself already a documented
weighted blend of every piece of evidence), not an ensemble of independent
models needing their own floor-capped aggregation. Adding another
aggregation layer on top of an already-aggregated score would just be
complexity Level 1 doesn't need -- Section 2's progression explicitly says
not to add sophistication ahead of a proven need for it. Level 2/3, with
their model ensembles and calibration, are where that kind of layering
earns its place.

Every condition in Section 18's list maps to a named check below:
  stretch/exhaustion insufficient  -> folded into reversal_score (already blended)
  regime unsuitable                -> regime UNKNOWN is hard; TREND_* raises the bar
  signals conflict                 -> both directions clearing threshold at once
  volatility abnormal              -> HIGH_VOLATILITY hard gate (configurable)
  confirmation missing             -> unconfirmed setup
  probability insufficient / EV negative -> Level 1's payout-floor proxy (edge_engine.py)
  data quality poor                -> tick validation failures / insufficient bars
  execution conditions poor        -> stale proposal, API disconnected
  risk limits reached              -> delegated to risk/manager.py
"""
from __future__ import annotations

from dataclasses import dataclass, field

NO_TRADE_MINIMUM_DATA = "NO_TRADE_MINIMUM_DATA"
NO_TRADE_REGIME_UNKNOWN = "NO_TRADE_REGIME_UNKNOWN"
NO_TRADE_HIGH_VOLATILITY = "NO_TRADE_HIGH_VOLATILITY"
NO_TRADE_NO_SETUP = "NO_TRADE_NO_SETUP"
NO_TRADE_UNCONFIRMED = "NO_TRADE_UNCONFIRMED"
NO_TRADE_CONFLICTING_SIGNALS = "NO_TRADE_CONFLICTING_SIGNALS"
NO_TRADE_BELOW_THRESHOLD = "NO_TRADE_BELOW_THRESHOLD"
NO_TRADE_POOR_ECONOMICS = "NO_TRADE_POOR_ECONOMICS"
NO_TRADE_BAD_PROPOSAL = "NO_TRADE_BAD_PROPOSAL"
NO_TRADE_STALE_PROPOSAL = "NO_TRADE_STALE_PROPOSAL"
NO_TRADE_API = "NO_TRADE_API"
NO_TRADE_DATA_QUALITY = "NO_TRADE_DATA_QUALITY"
NO_TRADE_RISK = "NO_TRADE_RISK"
NO_TRADE_RESEARCH_MODE = "NO_TRADE_RESEARCH_MODE"

TRADE_BULLISH = "TRADE_BULLISH"   # -> CALL
TRADE_BEARISH = "TRADE_BEARISH"   # -> PUT


@dataclass
class Decision:
    symbol: str
    timestamp: float
    decision: str = "NO_TRADE"
    reason_code: str = "PENDING"
    explanation: str = ""

    regime: str = "UNKNOWN"
    volatility_regime: str = "NORMAL"
    bullish_score: float = 0.0
    bearish_score: float = 0.0
    threshold_used: float = 0.0
    regime_multiplier: float = 1.0
    confirmed: bool = False
    confirmation_reason: str = ""
    payout_multiple: float | None = None

    @property
    def will_trade(self) -> bool:
        return self.decision in (TRADE_BULLISH, TRADE_BEARISH)


def evaluate_signal(*, symbol: str, timestamp: float, sufficient_data: bool,
                    regime: str, volatility_regime: str,
                    bullish_score: float, bearish_score: float,
                    setup_threshold: float, regime_multiplier: float,
                    disable_on_high_vol: bool,
                    confirmed_direction: str | None,
                    confirmation_reason: str) -> Decision:
    """Pure function: score -> decision, no I/O, no risk/economics. The
    caller layers economics (edge_engine) and risk (risk/manager) on top of
    a TRADE_* result before it can actually execute -- this function only
    answers "does the signal itself qualify," matching Section 18's split
    between signal-level and execution-level NO-TRADE causes.
    """
    d = Decision(symbol=symbol, timestamp=timestamp, regime=regime,
                volatility_regime=volatility_regime,
                bullish_score=bullish_score, bearish_score=bearish_score,
                regime_multiplier=regime_multiplier,
                threshold_used=setup_threshold * regime_multiplier)

    if not sufficient_data:
        d.reason_code = NO_TRADE_MINIMUM_DATA
        d.explanation = "fewer closed candles than the configured minimum"
        return d

    if regime == "UNKNOWN":
        d.reason_code = NO_TRADE_REGIME_UNKNOWN
        d.explanation = "regime could not be classified -- no basis to price risk"
        return d

    if disable_on_high_vol and volatility_regime == "HIGH_VOLATILITY":
        d.reason_code = NO_TRADE_HIGH_VOLATILITY
        d.explanation = "volatility regime is abnormal; ATR-normalized thresholds are unreliable here"
        return d

    bull_qualifies = bullish_score >= d.threshold_used
    bear_qualifies = bearish_score >= d.threshold_used

    if bull_qualifies and bear_qualifies:
        d.reason_code = NO_TRADE_CONFLICTING_SIGNALS
        d.explanation = (f"both directions cleared threshold {d.threshold_used:.1f} "
                         f"(bullish {bullish_score:.1f}, bearish {bearish_score:.1f}) "
                         f"-- refusing rather than picking arbitrarily")
        return d

    if not bull_qualifies and not bear_qualifies:
        d.reason_code = NO_TRADE_NO_SETUP
        d.explanation = (f"neither direction cleared threshold {d.threshold_used:.1f} "
                         f"(bullish {bullish_score:.1f}, bearish {bearish_score:.1f})")
        return d

    candidate_direction = "bullish" if bull_qualifies else "bearish"

    if confirmed_direction != candidate_direction:
        d.reason_code = NO_TRADE_UNCONFIRMED
        d.explanation = f"{candidate_direction} setup qualified but not confirmed: {confirmation_reason}"
        return d

    d.confirmed = True
    d.confirmation_reason = confirmation_reason
    d.decision = TRADE_BULLISH if candidate_direction == "bullish" else TRADE_BEARISH
    d.reason_code = d.decision
    d.explanation = (f"{candidate_direction} score {bullish_score if bull_qualifies else bearish_score:.1f} "
                     f">= threshold {d.threshold_used:.1f} ({regime}, multiplier "
                     f"{regime_multiplier:.2f}), confirmed: {confirmation_reason}")
    return d


def apply_economics_and_risk(decision: Decision, *, edge_assessment,
                             risk_decision, research_mode: bool = False) -> Decision:
    """Second pass, once a real proposal and a real risk check exist.
    Mirrors the Even/Odd engine's two-phase gate for the same reason: the
    signal-level decision (above) is cheap and needs no API call, so it
    runs first and only a genuine candidate reaches a proposal request.

    `research_mode` is checked LAST, deliberately -- after economics and
    risk both pass -- so a research-mode log line reads "this would have
    traded, and here's the full economics it cleared" rather than an early
    exit that never shows whether the rest of the pipeline agreed. This is
    ONE of two independent places this gate is enforced; execution/executor.py
    refuses to call buy() in research mode regardless of what this function
    returns, so a bug in either layer alone cannot place a real order.
    """
    if not decision.will_trade:
        return decision

    if edge_assessment is None:
        decision.decision = "NO_TRADE"
        decision.reason_code = NO_TRADE_BAD_PROPOSAL
        decision.explanation = "no valid proposal obtained"
        return decision

    decision.payout_multiple = edge_assessment.proposal.payout_multiple
    if edge_assessment.proposal.is_stale():
        decision.decision = "NO_TRADE"
        decision.reason_code = NO_TRADE_STALE_PROPOSAL
        decision.explanation = "proposal too old to execute safely"
        return decision

    if not edge_assessment.economically_sound:
        decision.decision = "NO_TRADE"
        decision.reason_code = NO_TRADE_POOR_ECONOMICS
        decision.explanation = edge_assessment.reason
        return decision

    if risk_decision is None or not getattr(risk_decision, "allowed", False):
        decision.decision = "NO_TRADE"
        decision.reason_code = NO_TRADE_RISK
        decision.explanation = getattr(risk_decision, "reason", "risk not evaluated")
        return decision

    if research_mode:
        decision.decision = "NO_TRADE"
        decision.reason_code = NO_TRADE_RESEARCH_MODE
        decision.explanation = (
            f"research mode: would have traded ({edge_assessment.reason}, "
            f"payout {decision.payout_multiple:.3f}x) -- buy() is never called")
        return decision

    return decision
