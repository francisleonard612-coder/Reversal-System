"""Binance USD-M perpetuals research: is there a strategy that makes money
after taker fees, slippage and funding, judged on data it was not tuned on?

Data: 4h candles + funding for every USDT perpetual incl. delisted
(research/binance_download.py). Decisions at a bar's close use only data up
to that close; the position earns the NEXT bar's return and pays/receives
the funding settled during that bar. Costs are charged on every change in
position (taker fee + a liquidity-dependent slippage).

Discovery = before 2024-01-01 (choose settings); validation = 2024-01-01 on
(judge them). One pick per strategy family goes to validation; Bonferroni
over the families.
"""
from __future__ import annotations

import glob
import itertools
import math
import os
import sys

import numpy as np
import pandas as pd

BAR = 4 * 3600
PER_DAY = 6
SPLIT = pd.Timestamp("2024-01-01", tz="UTC").timestamp()
FEE_BP = 5.0                       # VIP0 taker, per side
MIN_ADV = 20e6                     # $/day traded to be in the universe
MIN_AGE = 60 * PER_DAY             # bars since listing (skip launch chaos)


# ------------------------------------------------------------------ panel
def build_panel(root="4h"):
    closes, qvs, funds = {}, {}, {}
    for p in sorted(glob.glob(f"{root}/klines/*.csv.xz")):
        s = os.path.basename(p)[:-7]
        d = pd.read_csv(p)
        if len(d) < MIN_AGE + 50:
            continue
        T = d.t.values + BAR                      # index by bar CLOSE time
        closes[s] = pd.Series(d.close.values, index=T)
        qvs[s] = pd.Series(d.quote_volume.values, index=T)
        fp = f"{root}/funding/{s}.csv.xz"
        if os.path.exists(fp):
            f = pd.read_csv(fp)
            if len(f):
                Tf = (np.ceil(f.t.values / BAR) * BAR).astype(np.int64)   # funding in (T-4h, T]
                funds[s] = pd.Series(f.funding_rate.values, index=Tf).groupby(level=0).sum()
    C = pd.DataFrame(closes).sort_index()
    idx = np.arange(C.index.min(), C.index.max() + BAR, BAR)
    C = C.reindex(idx)
    Q = pd.DataFrame(qvs).reindex(idx)
    Fd = pd.DataFrame(funds).reindex(index=idx, columns=C.columns).fillna(0.0)
    return C.astype("float64"), Q.astype("float64"), Fd.astype("float64")


def derived(C, Q):
    R = C / C.shift(1) - 1
    age = C.notna().cumsum()
    adv = Q.rolling(30 * PER_DAY, min_periods=10 * PER_DAY).mean() * PER_DAY
    U = (age >= MIN_AGE) & (adv >= MIN_ADV) & C.notna()
    slip = pd.DataFrame(np.select([adv > 1e9, adv > 2e8, adv > 5e7], [1.0, 3.0, 6.0], 12.0),
                        index=C.index, columns=C.columns)
    sig = R.rolling(30 * PER_DAY, min_periods=10 * PER_DAY).std()
    return R, U, slip, sig


# ------------------------------------------------------------ backtester
def backtest(W, R, Fd, slip):
    """W: target weights decided at each row (NaN->0). Returns per-bar net,
    gross, funding and cost series (aligned to the bar the P&L is earned)."""
    W = W.fillna(0.0)
    Rn = R.shift(-1).fillna(0.0)                      # next bar's return
    Fn = Fd.shift(-1).fillna(0.0)                     # funding settled during next bar
    gross = (W * Rn).sum(axis=1)
    fund = -(W * Fn).sum(axis=1)                      # longs pay positive funding
    cost = ((W - W.shift(1).fillna(0.0)).abs() * (FEE_BP + slip) * 1e-4).sum(axis=1)
    net = gross + fund - cost
    sh = lambda s: s.shift(1).fillna(0.0)             # P&L belongs to the NEXT bar
    return pd.DataFrame({"net": sh(net), "gross": sh(gross), "fund": sh(fund), "cost": cost,
                         "expo": W.abs().sum(axis=1), "turn": (W - W.shift(1).fillna(0.0)).abs().sum(axis=1)})


def hold_daily(W):
    """Rebalance once a day (00:00 UTC close), hold in between."""
    daily = (W.index % 86400) == 0
    return W.where(pd.Series(daily, index=W.index), np.nan).ffill()


def hold_weekly(W):
    wk = ((W.index % (7 * 86400)) == 4 * 86400)       # Monday 00:00 UTC
    return W.where(pd.Series(wk, index=W.index), np.nan).ffill()


def normalize(W):
    g = W.abs().sum(axis=1).replace(0, np.nan)
    return W.div(g, axis=0).fillna(0.0)


# ---------------------------------------------------------- strategies
def ts_trend(C, U, sig, L, only=None, long_only=False):
    m = C / C.shift(L) - 1
    s = np.sign(m)
    if long_only:
        s = s.clip(lower=0)
    W = (s / sig).where(U)
    if only:
        W = W[[c for c in only if c in W.columns]].reindex(columns=C.columns)
    return normalize(W)


def xs_rank(sigl, U, frac=0.2, min_n=10):
    x = sigl.where(U)
    n = x.notna().sum(axis=1)
    r = x.rank(axis=1, pct=True)
    long_ = (r > 1 - frac).astype(float)
    short = (r <= frac).astype(float)
    W = long_.div(long_.sum(axis=1).replace(0, np.nan), axis=0) * 0.5 \
        - short.div(short.sum(axis=1).replace(0, np.nan), axis=0) * 0.5
    return W.where(n >= min_n, 0.0).fillna(0.0)


def xs_momentum(C, U, L, sign=1):
    return xs_rank(sign * (C / C.shift(L) - 1), U)


def carry(Fd, U, L):
    avgf = Fd.rolling(L, min_periods=max(2, L // 2)).mean()
    return xs_rank(-avgf, U)          # long lowest funding, short highest


def event(R, U, sig, k, H, sign):
    """sign=+1: buy after a crash (R < -k sigma); sign=-1: short after a pump."""
    z = R / sig.shift(1)
    trig = ((z < -k) if sign > 0 else (z > k)) & U
    active = trig.astype(float).rolling(H, min_periods=1).max() > 0
    W = active.astype(float) * sign
    return normalize(W)


def configs(C, U, sig, Fd, R):
    L_opts = {"1d": 6, "3d": 18, "7d": 42, "14d": 84, "30d": 180}
    for (ln, L) in L_opts.items():
        yield "TREND_ALL", f"L={ln}", lambda L=L: hold_daily(ts_trend(C, U, sig, L))
        yield "TREND_LONGONLY", f"L={ln}", lambda L=L: hold_daily(ts_trend(C, U, sig, L, long_only=True))
        yield "TREND_BTCETH", f"L={ln}", lambda L=L: hold_daily(ts_trend(C, U, sig, L, only=["BTCUSDT", "ETHUSDT"]))
        for fq, hold in (("daily", hold_daily), ("weekly", hold_weekly)):
            yield "XS_MOMENTUM", f"L={ln} {fq}", lambda L=L, hold=hold: hold(xs_momentum(C, U, L, 1))
            yield "XS_REVERSAL", f"L={ln} {fq}", lambda L=L, hold=hold: hold(xs_momentum(C, U, L, -1))
    for (ln, L) in (("1d", 6), ("3d", 18), ("7d", 42)):
        for fq, hold in (("daily", hold_daily), ("weekly", hold_weekly)):
            yield "FUNDING_CARRY", f"avg={ln} {fq}", lambda L=L, hold=hold: hold(carry(Fd, U, L))
    for k, H in itertools.product((3, 4, 5), (1, 3, 6)):
        yield "CRASH_BUY", f"k={k} hold={H*4}h", lambda k=k, H=H: event(R, U, sig, k, H, +1)
        yield "PUMP_FADE", f"k={k} hold={H*4}h", lambda k=k, H=H: event(R, U, sig, k, H, -1)


# ---------------------------------------------------------------- stats
def stats(bt: pd.DataFrame, lo, hi):
    b = bt[(bt.index >= lo) & (bt.index < hi)]
    d = b.groupby(b.index // 86400).sum()
    wk = b.groupby(b.index // (7 * 86400)).net.sum()
    if len(d) < 20 or d.net.std() == 0:
        return None
    t = wk.mean() / (wk.std(ddof=1) / math.sqrt(len(wk))) if wk.std() > 0 else 0.0
    return dict(days=len(d), ann_ret=d.net.mean() * 365, ann_gross=d.gross.mean() * 365,
                ann_fund=d.fund.mean() * 365, ann_cost=d.cost.mean() * 365,
                sharpe=d.net.mean() / d.net.std() * math.sqrt(365),
                maxdd=float((d.net.cumsum() - d.net.cumsum().cummax()).min()),
                t=t, p=0.5 * math.erfc(t / math.sqrt(2)),
                in_mkt=float((b.expo > 0).mean()), turn_per_day=b.turn.sum() / len(d))


def run(C, Q, Fd, out="perp_out", tag=""):
    os.makedirs(out, exist_ok=True)
    R, U, slip, sig = derived(C, Q)
    lo, hi = C.index.min(), C.index.max() + BAR
    rows = []
    for fam, name, make in configs(C, U, sig, Fd, R):
        bt = backtest(make(), R, Fd, slip)
        dsc, val = stats(bt, lo, SPLIT), stats(bt, SPLIT, hi)
        if dsc and val:
            rows.append(dict(family=fam, config=name, **{f"d_{k}": v for k, v in dsc.items()},
                             **{f"v_{k}": v for k, v in val.items()}))
        print(f"{fam:15} {name:16} disc SR {dsc['sharpe'] if dsc else float('nan'):+.2f}  "
              f"val SR {val['sharpe'] if val else float('nan'):+.2f}", file=sys.stderr, flush=True)
    T = pd.DataFrame(rows)
    T.to_csv(f"{out}/all_configs{tag}.csv", index=False)

    # benchmark
    bh = pd.DataFrame(0.0, index=C.index, columns=C.columns)
    bh["BTCUSDT"] = 1.0
    bhb = backtest(bh, R, Fd, slip)
    b_d, b_v = stats(bhb, lo, SPLIT), stats(bhb, SPLIT, hi)

    picks = T.loc[T.groupby("family").d_sharpe.idxmax()].sort_values("d_sharpe", ascending=False)
    alpha = 0.05 / len(picks)
    L = []
    P = L.append
    P(f"BINANCE PERPS RESEARCH {tag}-- net of taker fee {FEE_BP:g}bp/side + slippage + funding")
    P(f"coins in data: {C.shape[1]}   universe: listed >= {MIN_AGE // PER_DAY}d and >= ${MIN_ADV/1e6:.0f}M/day volume")
    P(f"discovery {pd.to_datetime(lo, unit='s').date()} -> 2023-12-31   validation 2024-01-01 -> {pd.to_datetime(hi, unit='s').date()}")
    P(f"families: {len(picks)}  Bonferroni alpha {alpha:.4f} (one-sided, weekly returns)\n")
    P("Returns are per year at 1x gross exposure (long+short notional = equity).")
    P(f"{'family':15} {'best config (discovery)':22} {'disc SR':>7} | {'val SR':>6} {'val ret':>8} {'gross':>7} {'fund':>6} {'cost':>7} {'maxDD':>7} {'in mkt':>6} {'p':>8}  verdict")
    for _, r in picks.iterrows():
        v = "VALIDATED" if (r.v_p < alpha and r.v_ann_ret > 0) else ("promising" if r.v_p < 0.05 and r.v_ann_ret > 0 else "")
        P(f"{r.family:15} {r.config:22} {r.d_sharpe:>+7.2f} | {r.v_sharpe:>+6.2f} {r.v_ann_ret:>+8.1%} {r.v_ann_gross:>+7.1%} "
          f"{r.v_ann_fund:>+6.1%} {-r.v_ann_cost:>+7.1%} {r.v_maxdd:>+7.1%} {r.v_in_mkt:>6.0%} {r.v_p:>8.1e}  {v}")
    P(f"\nBenchmark: hold BTC long   disc SR {b_d['sharpe']:+.2f}   val SR {b_v['sharpe']:+.2f}  val ret {b_v['ann_ret']:+.1%}/yr  maxDD {b_v['maxdd']:+.1%}")
    P("\nEvery config, validation Sharpe (no selection) -- median / share > 0 per family:")
    for fam, g in T.groupby("family"):
        P(f"  {fam:15} n={len(g):>2}  median val SR {g.v_sharpe.median():+.2f}  share>0 {(g.v_sharpe > 0).mean():.0%}   "
          f"median disc SR {g.d_sharpe.median():+.2f}")
    txt = "\n".join(L)
    open(f"{out}/summary{tag}.txt", "w").write(txt)
    print(txt)
    return T


if __name__ == "__main__":
    C, Q, Fd = build_panel(sys.argv[1] if len(sys.argv) > 1 else "4h")
    run(C, Q, Fd)
