"""
REGIME-CONDITIONAL ROUTING + ASYMMETRIC CONVICTION SIZING
==========================================================
A philosophy change for risefall_bot_v4_hmm_gbm.py.

WHAT THIS REPLACES
──────────────────
The current bot is ensemble-consensus: all 17 layers vote, bayesian_fusion
averages them into p_up, and a trade fires when enough layers agree. Every
layer votes in every market, weighted the same way regardless of whether
that layer's underlying assumption currently holds.

That's the flaw. An OU mean-reversion layer is informative in a ranging
market and actively misleading in a trending one. Averaging it in during a
trend doesn't just add noise — it adds *biased* noise, pulling p_up toward
a reversion that isn't coming. Same in reverse for ARFIMA/Kalman during
chop. The ensemble dilutes its own best signals with layers that are
structurally wrong for the current conditions.

THE NEW PHILOSOPHY — two changes, both requested:

  (1) REGIME-CONDITIONAL ROUTING
      Classify the market state FIRST. Then consult only the layers whose
      assumptions hold in that state, and ignore the rest entirely — not
      down-weighted, ignored. A layer that is structurally wrong right now
      contributes nothing rather than contributing error.

  (2) ASYMMETRIC CONVICTION SIZING
      Stop treating every qualifying trade as equal. Rise/Fall payout is
      structurally against you (~0.95 win vs 1.00 loss, needing >51.3% to
      break even), so you cannot get asymmetry from the contract. You get
      it from POSITION SIZE: concentrate capital in the rare high-conviction
      setups where the regime-appropriate layers strongly concur, and stake
      little or nothing on marginal ones. Wins land big, losses land small,
      and the P&L becomes asymmetric even though each individual contract
      is not.

      This is NOT martingale. Martingale sizes by past losses (loss-chasing).
      This sizes by present conviction, and is completely indifferent to
      what the previous trade did.

REGIMES
───────
Classified on two axes the bot already computes every cycle:

    Hurst (h)          → trending (persistent) vs ranging (anti-persistent)
    Realized vol (σ)   → quiet vs volatile, measured against the symbol's
                         own recent baseline (self-relative, so this works
                         across instruments with different native vol)

    ┌──────────────┬─────────────────────┬──────────────────────┐
    │              │  QUIET              │  VOLATILE            │
    ├──────────────┼─────────────────────┼──────────────────────┤
    │ TRENDING     │  TREND_QUIET        │  TREND_VOLATILE      │
    │ (h > 0.55)   │  → trend-following  │  → breakout layers   │
    ├──────────────┼─────────────────────┼──────────────────────┤
    │ RANGING      │  RANGE_QUIET        │  RANGE_VOLATILE      │
    │ (h < 0.45)   │  → mean-reversion   │  → SKIP (chop)       │
    └──────────────┴─────────────────────┴──────────────────────┘

    h in [0.45, 0.55] → NEUTRAL → SKIP. No regime means no layer set is
    justified, and trading the full ensemble there is exactly the
    behaviour this change exists to stop.

    RANGE_VOLATILE is skipped outright: anti-persistent price with large
    moves is chop. Mean-reversion layers fire constantly and get run over;
    trend layers have no trend to find. It is the one quadrant where the
    honest answer is that there is no edge to extract.

WHY THESE LAYER→REGIME ASSIGNMENTS
───────────────────────────────────
Each layer is assigned by what it structurally measures, not by backtest
fitting (there is no walk-forward validation behind these — see CAVEATS):

  TREND_QUIET — persistence is real and price is orderly enough to follow:
    markov     : directional persistence, literally a continuation measure
    hmm        : regime state lean, informative when a state is sustained
    hurst      : the persistence measure itself
    arfima     : long-memory continuation
    kalman     : filtered trend estimate; needs low noise to be stable
    adx        : trend strength × direction, purpose-built for this
    copula     : cross-sectional co-movement, holds up in orderly markets

  TREND_VOLATILE — direction is real but moves are violent; want layers
  that handle discontinuity rather than assume smoothness:
    hawkes     : self-exciting intensity, built for clustered bursts
    jump_dir   : explicit jump direction detection
    sr         : support/resistance — breakout/continuation mode here
    adx        : trend strength survives volatility
    hmm        : regime states capture volatile-trending as a state
    kalman     : kept but it is the weakest member here (noise hurts it)

  RANGE_QUIET — anti-persistent and orderly; reversion actually completes:
    ou         : Ornstein-Uhlenbeck, the canonical mean-reversion layer
    rsi        : overbought/oversold oscillator
    srsi       : stochastic RSI, same family
    boll       : Bollinger band deviation, a reversion measure
    zscore     : standardized deviation from mean
    sr         : support/resistance — fade mode here (opposite of above)
    post_jump  : post-jump reversion, explicitly a reversion layer

  Deliberately excluded everywhere: transfer entropy (te). It measures
  cross-symbol information flow, which is a different question from "what
  regime is THIS symbol in" and does not belong to any single regime's
  assumption set. It is not deleted from the bot — it simply is not part
  of the regime-routed decision.

CAVEATS — read before trusting this
────────────────────────────────────
  · The layer→regime map is REASONED, NOT VALIDATED. It comes from what
    each layer mathematically measures, not from walk-forward evidence
    that these specific groupings outperform on these specific symbols.
    It is a hypothesis with a clear rationale, and it should be treated
    as one until live results say otherwise.
  · Regime thresholds (0.45/0.55 Hurst, vol percentile) are starting
    points, not tuned values.
  · Conviction sizing concentrates risk by design. A high-conviction read
    that is wrong costs more than a flat-stake loss would have. The whole
    approach is a bet that conviction correlates with win rate — if it
    does not, this sizing makes outcomes worse, not better. That
    correlation is measurable (see conviction_outcome_report) and SHOULD
    be measured before raising max_mult.
"""

from typing import Dict, List, Optional, Tuple
import math


# ── Regime labels ───────────────────────────────────────────────────────

REGIME_TREND_QUIET    = "TREND_QUIET"
REGIME_TREND_VOLATILE = "TREND_VOLATILE"
REGIME_RANGE_QUIET    = "RANGE_QUIET"
REGIME_RANGE_VOLATILE = "RANGE_VOLATILE"   # skip
REGIME_NEUTRAL        = "NEUTRAL"          # skip

TRADEABLE_REGIMES = {
    REGIME_TREND_QUIET,
    REGIME_TREND_VOLATILE,
    REGIME_RANGE_QUIET,
}

# Layer index → name, matching _layer_votes order in the bot exactly.
LAYER_NAMES = [
    "markov", "hmm", "hawkes", "ou", "hurst", "arfima", "kalman",
    "copula", "rsi", "srsi", "adx", "boll", "zscore", "te",
    "jump_dir", "post_jump", "sr",
]

REGIME_LAYERS: Dict[str, List[str]] = {
    REGIME_TREND_QUIET: [
        "markov", "hmm", "hurst", "arfima", "kalman", "adx", "copula",
    ],
    REGIME_TREND_VOLATILE: [
        "hawkes", "jump_dir", "sr", "adx", "hmm", "kalman",
    ],
    REGIME_RANGE_QUIET: [
        "ou", "rsi", "srsi", "boll", "zscore", "sr", "post_jump",
    ],
    REGIME_RANGE_VOLATILE: [],
    REGIME_NEUTRAL:       [],
}


# ── Regime classification ───────────────────────────────────────────────

def classify_regime(hurst: float,
                    sigma_now: float,
                    sigma_baseline: float,
                    cfg: Optional[dict] = None) -> Tuple[str, str]:
    """
    Returns (regime, human_readable_reason).

    hurst          : Hurst exponent, already computed by the bot as `h`
    sigma_now      : current realized volatility
    sigma_baseline : that symbol's own recent median/typical volatility.
                     Using a SELF-RELATIVE baseline rather than an absolute
                     threshold is what makes this work unchanged across
                     instruments with wildly different native vol scales
                     (1HZ10V vs R_75 differ by orders of magnitude).
    """
    cfg = cfg or {}
    h_trend = cfg.get("regime_hurst_trend", 0.55)
    h_range = cfg.get("regime_hurst_range", 0.45)
    vol_mult = cfg.get("regime_vol_multiple", 1.35)

    if sigma_baseline <= 0:
        return REGIME_NEUTRAL, "no vol baseline yet"

    vol_ratio = sigma_now / sigma_baseline
    volatile  = vol_ratio >= vol_mult

    if h_range <= hurst <= h_trend:
        return (REGIME_NEUTRAL,
                f"H={hurst:.3f} in neutral band [{h_range}, {h_trend}] "
                f"— no regime, no justified layer set")

    if hurst > h_trend:
        if volatile:
            return (REGIME_TREND_VOLATILE,
                    f"H={hurst:.3f} trending, vol={vol_ratio:.2f}x baseline")
        return (REGIME_TREND_QUIET,
                f"H={hurst:.3f} trending, vol={vol_ratio:.2f}x baseline (quiet)")

    if volatile:
        return (REGIME_RANGE_VOLATILE,
                f"H={hurst:.3f} anti-persistent with vol={vol_ratio:.2f}x "
                f"baseline — chop, no edge")
    return (REGIME_RANGE_QUIET,
            f"H={hurst:.3f} anti-persistent, vol={vol_ratio:.2f}x baseline (quiet)")


# ── Regime-routed vote extraction ───────────────────────────────────────

def regime_votes(layer_votes: List[float], regime: str) -> Dict[str, float]:
    """
    Takes the bot's full _layer_votes list (17 signed floats, same order as
    LAYER_NAMES) and returns ONLY the votes belonging to this regime's
    layer set, keyed by name.

    Layers outside the regime are dropped entirely — not down-weighted.
    That is the point: a layer whose assumption does not currently hold
    contributes nothing rather than contributing bias.
    """
    if len(layer_votes) != len(LAYER_NAMES):
        raise ValueError(
            f"layer_votes has {len(layer_votes)} entries but LAYER_NAMES has "
            f"{len(LAYER_NAMES)} — these must stay in sync with the bot's "
            f"_layer_votes list. If a layer was added/removed there, update "
            f"LAYER_NAMES and REGIME_LAYERS here to match."
        )
    wanted = set(REGIME_LAYERS.get(regime, []))
    return {name: float(v)
            for name, v in zip(LAYER_NAMES, layer_votes)
            if name in wanted}


# ── Conviction ──────────────────────────────────────────────────────────

def compute_conviction(votes: Dict[str, float],
                       cfg: Optional[dict] = None) -> Tuple[float, int, str]:
    """
    Returns (conviction, direction, reason).

    conviction ∈ [0, 1]. direction ∈ {+1, -1, 0}.

    Conviction is the product of two things that must BOTH be high:

      strength  = normalised |mean(votes)|   how strongly the layers lean
      agreement = fraction agreeing          how unanimously they lean

    STRENGTH NORMALISATION — this was a real bug, fixed after live data.
    Originally strength was the raw |mean(votes)|, which assumed layer
    votes span roughly [-1, 1]. They do not. Measured across 276 live
    readings, individual votes ran ±0.06 to ±0.54 and the resulting
    strength had a MEDIAN of 0.106 and a MAXIMUM of 0.334. Multiplying
    that by an agreement fraction (<= 1.0) drove conviction lower still,
    so a floor of 0.35 was mathematically unreachable — it produced
    exactly zero trades out of 276 opportunities, not rarely but never.

    The fix: divide the mean by the largest |vote| in the active set, so
    strength measures "how aligned are these layers relative to the
    loudest one among them" and genuinely spans [0, 1]. A single layer
    shouting alone still scores low (the mean stays small relative to
    that layer's own magnitude); several layers leaning together scores
    high. That is the property we actually wanted.

    Multiplying strength by agreement rather than averaging them is
    deliberate. A set of layers that all agree weakly (unanimous but
    near-zero) and a set that disagrees violently (strong but split) are
    BOTH low-conviction and both should size small. Averaging would let
    one mask the other; multiplying requires genuine agreement AND
    genuine strength.

    Layers voting exactly 0.0 are treated as abstentions — excluded from
    the agreement fraction rather than counted as disagreement, since a
    neutral read is not evidence against.
    """
    cfg = cfg or {}
    if not votes:
        return 0.0, 0, "no layers active for this regime"

    vals = list(votes.values())
    mean_vote = sum(vals) / len(vals)
    direction = 1 if mean_vote > 0 else (-1 if mean_vote < 0 else 0)
    if direction == 0:
        return 0.0, 0, "regime layers net exactly neutral"

    # Abstentions excluded from the agreement denominator.
    non_zero = [v for v in vals if abs(v) > 1e-9]
    if not non_zero:
        return 0.0, 0, "all regime layers abstained"

    # Require a minimum number of layers actually voting. Without this, a
    # single loud layer with the rest abstaining scores respectably (1 of 1
    # agreeing = 100% agreement, and mean/max = 1.0 when it is the only
    # voter) — one layer is not a consensus, and sizing up on it defeats
    # the point of routing to a layer SET.
    min_voters = cfg.get("conviction_min_voters", 3)
    if len(non_zero) < min_voters:
        return 0.0, 0, (f"only {len(non_zero)} layer(s) voting, need "
                        f">={min_voters} — not a consensus")

    agreeing  = sum(1 for v in non_zero if (v > 0) == (direction > 0))
    agreement = agreeing / len(non_zero)

    # Normalise against the loudest active layer so strength spans [0,1]
    # regardless of the absolute scale the layers happen to vote on.
    max_mag = max(abs(v) for v in non_zero)
    strength = min(1.0, abs(mean_vote) / max_mag) if max_mag > 0 else 0.0

    conviction = strength * agreement

    side = "CALL" if direction > 0 else "PUT"
    reason = (f"{side} strength={strength:.3f} (mean={abs(mean_vote):.3f} / "
              f"max_layer={max_mag:.3f}) agreement={agreement:.2f} "
              f"({agreeing}/{len(non_zero)} non-abstaining) "
              f"conviction={conviction:.3f}")
    return conviction, direction, reason


def conviction_stake(conviction: float,
                     base_stake: float,
                     cfg: Optional[dict] = None) -> Tuple[float, str]:
    """
    Maps conviction → stake. This is where the asymmetry actually happens.

    Below `floor`, returns 0.0 — do not trade. Marginal reads are not
    small trades, they are no trades; taking them at any size is what
    erodes the account between the good setups.

    Above the floor, conviction is rescaled to [0,1] across the remaining
    range and mapped linearly onto [min_mult, max_mult]. Linear rather
    than exponential is a deliberately conservative choice: exponential
    sizing on a signal whose conviction→win-rate correlation is UNPROVEN
    (see module caveats) would concentrate risk on an assumption that has
    not been demonstrated yet.
    """
    cfg = cfg or {}
    floor    = cfg.get("conviction_floor",   0.20)
    min_mult = cfg.get("conviction_min_mult", 0.5)
    max_mult = cfg.get("conviction_max_mult", 3.0)
    max_stake = cfg.get("conviction_max_stake", 0.0)  # 0 = uncapped

    if conviction < floor:
        return 0.0, (f"conviction {conviction:.3f} < floor {floor:.2f} "
                     f"— no trade (marginal reads are skipped, not sized down)")

    span = max(1e-9, 1.0 - floor)
    scaled = (conviction - floor) / span            # 0 at floor, 1 at perfect
    mult   = min_mult + scaled * (max_mult - min_mult)
    stake  = base_stake * mult
    if max_stake > 0:
        stake = min(stake, max_stake)
    stake = round(stake, 2)
    return stake, (f"conviction {conviction:.3f} → {mult:.2f}x base "
                   f"→ stake ${stake:.2f}")


# ── Full decision ───────────────────────────────────────────────────────

def regime_decision(hurst: float,
                    sigma_now: float,
                    sigma_baseline: float,
                    layer_votes: List[float],
                    base_stake: float,
                    cfg: Optional[dict] = None) -> dict:
    """
    One call, the whole philosophy. Returns a dict:

        {
          "trade":      bool,
          "regime":     str,
          "direction":  +1 / -1 / 0,
          "conviction": float,
          "stake":      float,
          "reasons":    [str, ...]      # full audit trail for the log
        }

    Every rejection path returns trade=False WITH the reason recorded, so
    the logs always explain why nothing fired rather than going silent.
    """
    cfg = cfg or {}
    reasons = []

    regime, why = classify_regime(hurst, sigma_now, sigma_baseline, cfg)
    reasons.append(f"regime={regime} ({why})")

    if regime not in TRADEABLE_REGIMES:
        reasons.append("regime is not tradeable — waiting")
        return {"trade": False, "regime": regime, "direction": 0,
                "conviction": 0.0, "stake": 0.0, "reasons": reasons}

    votes = regime_votes(layer_votes, regime)
    active = ", ".join(f"{k}={v:+.2f}" for k, v in votes.items())
    reasons.append(f"active layers ({len(votes)}): {active}")
    ignored = [n for n in LAYER_NAMES if n not in votes]
    reasons.append(f"ignored ({len(ignored)}): {', '.join(ignored)}")

    conviction, direction, conv_why = compute_conviction(votes, cfg)
    reasons.append(conv_why)

    if direction == 0:
        return {"trade": False, "regime": regime, "direction": 0,
                "conviction": conviction, "stake": 0.0, "reasons": reasons}

    stake, stake_why = conviction_stake(conviction, base_stake, cfg)
    reasons.append(stake_why)

    return {"trade": stake > 0, "regime": regime, "direction": direction,
            "conviction": conviction, "stake": stake, "reasons": reasons}


# ── Measuring whether the core assumption actually holds ────────────────

def conviction_outcome_report(trades: List[dict],
                              buckets: int = 4) -> str:
    """
    The whole approach rests on one unproven assumption: that higher
    conviction means a higher win rate. If that correlation is absent,
    conviction sizing actively makes results WORSE than flat staking,
    because it puts more money on reads that are no better than average.

    This bins completed trades by conviction and reports win rate per bin,
    so the assumption gets checked against real outcomes instead of taken
    on faith. Feed it dicts with "conviction" and "won" keys.

    Read it as: win rate should climb monotonically across buckets. If it
    is flat, drop conviction_max_mult to 1.0 (flat staking) until it
    isn't. If it INVERTS, the layer→regime map is wrong for these symbols.
    """
    if not trades:
        return "no completed trades yet"

    scored = [t for t in trades if "conviction" in t and "won" in t]
    if not scored:
        return "no trades carry conviction+won fields"

    scored.sort(key=lambda t: t["conviction"])
    n = len(scored)
    size = max(1, n // buckets)
    lines = [f"Conviction → win-rate check ({n} trades):"]
    for i in range(0, n, size):
        chunk = scored[i:i+size]
        if not chunk:
            continue
        wins = sum(1 for t in chunk if t["won"])
        lo = chunk[0]["conviction"]
        hi = chunk[-1]["conviction"]
        lines.append(f"  conviction {lo:.2f}-{hi:.2f}: "
                     f"{wins}/{len(chunk)} = {wins/len(chunk)*100:.1f}% win rate")
    lines.append("  (want: win rate rising across buckets. Flat → set "
                 "conviction_max_mult=1.0. Inverted → layer map is wrong.)")
    return "\n".join(lines)
