"""
Historical simulator and walk-forward testing (spec Sections 40, 41).

Drives strategy/pipeline.py directly -- the same SymbolPipeline used live --
so a bug in the decision path shows up identically here and in production,
the same discipline the sibling Even/Odd bot's simulator follows.

NO PROBABILITY, NO EV, NO CALIBRATION METRICS HERE. Level 1 has none of
those (see strategy/edge_engine.py); this simulator reports trade-level
metrics only (Section 44's "trading metrics" list), not the "prediction
metrics" list, which requires a probability to score against and doesn't
exist until Level 2.

SETTLEMENT IS SIMULATED ON DIRECTION, NOT A FABRICATED WIN RATE. A CALL
"wins" if the close price `duration` bars after entry is higher than the
close at entry; PUT the opposite. This is what a Rise/Fall contract
literally pays on (direction relative to entry spot), so it's not an
assumption stacked on top of the real contract terms -- it's the real
contract terms, applied to historical closes instead of a live settlement
feed. The payout multiple is a CONFIGURED ASSUMPTION (`payout_multiple`
below), since historical proposals aren't available; it's named in every
report, same discipline as the Even/Odd simulator's assumed-payout warning.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from strategy.pipeline import SymbolPipeline


@dataclass
class SimulationResult:
    n_candles: int
    n_evaluated: int
    n_trades: int
    wins: int
    losses: int
    pnl: float
    stake: float
    payout_multiple: float
    reason_counts: dict = field(default_factory=dict)
    max_losing_streak: int = 0
    max_drawdown: float = 0.0

    @property
    def win_rate(self) -> float:
        return self.wins / self.n_trades if self.n_trades else float("nan")

    def report(self) -> str:
        L = ["=" * 68,
             f"LEVEL 1 BACKTEST  candles={self.n_candles} evaluated={self.n_evaluated} "
             f"trades={self.n_trades}",
             f"  ASSUMED payout {self.payout_multiple:.3f}x (historical proposals "
             f"unavailable -- this is a configured assumption, not a fetched quote)",
             f"  win rate {self.win_rate:.4f}  pnl {self.pnl:+.2f}  "
             f"max losing streak {self.max_losing_streak}  max dd {self.max_drawdown:.2f}"]
        if self.reason_counts:
            L.append("  no-trade reasons:")
            for code, n in sorted(self.reason_counts.items(), key=lambda kv: -kv[1])[:10]:
                L.append(f"    {code:<32}{n}")
        L.append("=" * 68)
        return "\n".join(L)


def run_backtest(candles: list, cfg: dict, *, stake: float = 1.0,
                 payout_multiple: float = 1.85, warmup_bars: int = 210,
                 contract_duration_bars: int = 5) -> SimulationResult:
    """`contract_duration_bars` mirrors config's contract.duration when
    duration_unit is ticks-on-closed-candles for simulation purposes; for a
    real tick-count Rise/Fall contract the live executor uses actual tick
    counts, not candle counts -- this simulator approximates settlement at
    candle-close granularity, which is coarser than live and is disclosed
    here rather than presented as identical.
    """
    pipe = SymbolPipeline("SIM", cfg)
    n_eval = trades = wins = losses = 0
    pnl = peak = max_dd = 0.0
    streak = max_streak = 0
    from collections import Counter
    reasons: Counter = Counter()

    n = len(candles)
    for i in range(warmup_bars, n):
        sub = candles[: i + 1]
        decision = pipe.evaluate_on_close(sub)
        n_eval += 1

        if not decision.will_trade:
            reasons[decision.reason_code] += 1
            continue

        settle_idx = i + contract_duration_bars
        if settle_idx >= n:
            reasons["NO_TRADE_SIM_HORIZON"] += 1
            continue   # can't settle within the available history -- excluded, not counted as a loss

        entry_close = candles[i].close
        exit_close = candles[settle_idx].close
        if decision.decision == "TRADE_BULLISH":
            won = exit_close > entry_close
        else:
            won = exit_close < entry_close

        trades += 1
        if won:
            wins += 1
            pnl += stake * (payout_multiple - 1)
            streak = 0
        else:
            losses += 1
            pnl -= stake
            streak += 1
            max_streak = max(max_streak, streak)
        peak = max(peak, pnl)
        max_dd = max(max_dd, peak - pnl)

    return SimulationResult(
        n_candles=n, n_evaluated=n_eval, n_trades=trades, wins=wins, losses=losses,
        pnl=pnl, stake=stake, payout_multiple=payout_multiple,
        reason_counts=dict(reasons), max_losing_streak=max_streak, max_drawdown=max_dd)


@dataclass
class WalkForwardReport:
    blocks: list[SimulationResult] = field(default_factory=list)

    @property
    def total_trades(self) -> int:
        return sum(b.n_trades for b in self.blocks)

    @property
    def total_pnl(self) -> float:
        return sum(b.pnl for b in self.blocks)

    @property
    def pooled_win_rate(self) -> float:
        n = self.total_trades
        return sum(b.wins for b in self.blocks) / n if n else float("nan")

    def report(self) -> str:
        L = [f"WALK-FORWARD: {len(self.blocks)} blocks"]
        for i, b in enumerate(self.blocks, 1):
            L.append(f"\n--- block {i} ---")
            L.append(b.report())
        L.append(f"\nPOOLED  trades={self.total_trades}  "
                 f"win_rate={self.pooled_win_rate:.4f}  pnl={self.total_pnl:+.2f}")
        return "\n".join(L)


def walk_forward(candles: list, cfg: dict, *, n_blocks: int = 5, stake: float = 1.0,
                 payout_multiple: float = 1.85, warmup_bars: int = 210,
                 contract_duration_bars: int = 5) -> WalkForwardReport:
    """Section 41: train/validate/test/roll-forward. Level 1 has no
    training step (it's deterministic), so "train" here means only "build up
    enough closed-candle history to clear warmup_bars" -- there is no model
    state carried between blocks to leak, which is the one simplification
    Level 1's lack of learning legitimately buys.
    """
    block = len(candles) // n_blocks
    if block < warmup_bars + contract_duration_bars + 20:
        raise ValueError(
            f"{len(candles)} candles over {n_blocks} blocks gives {block} per "
            f"block; too few to clear warmup ({warmup_bars}) and settle trades. "
            f"Use more data or fewer blocks.")
    report = WalkForwardReport()
    for b in range(n_blocks):
        segment = candles[b * block: (b + 1) * block]
        if len(segment) < warmup_bars + contract_duration_bars + 20:
            break
        report.blocks.append(run_backtest(
            segment, cfg, stake=stake, payout_multiple=payout_multiple,
            warmup_bars=warmup_bars, contract_duration_bars=contract_duration_bars))
    return report
