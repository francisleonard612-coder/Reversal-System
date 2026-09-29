"""CFD research: do directional trades with stop-loss / take-profit / time exit
make money after trading costs on Deriv forex, stock indices and BTC?

Uses the 5-minute closes already downloaded (candles/<SYM>_5m.csv).
No lookahead: every signal at bar i uses closes <= i, entry is at close i,
exits are checked on later closes only (close-only data, so a stop is filled
at the first close beyond it -- this includes realistic slippage).

Results are in R units: 1R = the stop distance (a vol-scaled risk unit), so a
bot risking a fixed % per trade earns exactly these numbers.

Discovery = first 60% of each symbol's history, validation = last 40%.
Configs are chosen on discovery only and judged on validation only, with a
Bonferroni correction over everything taken to validation.
"""
from __future__ import annotations

import glob
import itertools
import math
import os
import sys

import numpy as np
import pandas as pd

BAR = 300
DISC_FRAC = 0.6

# Round-trip cost ASSUMPTIONS in basis points of price (spread + commission).
# Verify against your account; the report also prints the break-even cost.
MAJORS = {"frxEURUSD", "frxGBPUSD", "frxUSDJPY", "frxUSDCHF", "frxAUDUSD", "frxUSDCAD"}
EXOTIC = {"frxUSDMXN", "frxUSDPLN"}
US_IDX = {"OTC_SPC", "OTC_NDX", "OTC_DJI"}


def cost_bp(sym: str) -> float:
    if sym in MAJORS:
        return 1.0
    if sym in EXOTIC:
        return 6.0
    if sym.startswith("frx"):
        return 2.5
    if sym in US_IDX:
        return 1.0
    if sym.startswith("OTC_"):
        return 2.0
    return 5.0  # crypto


def asset_class(sym: str) -> str:
    return "fx" if sym.startswith("frx") else "index" if sym.startswith("OTC_") else "crypto"


# ----------------------------------------------------------------- data
def load_grid(path: str):
    df = pd.read_csv(path)
    df["g"] = (df.epoch // BAR) * BAR
    df = df.drop_duplicates("g", keep="last").sort_values("g")
    t0, t1 = int(df.g.iloc[0]), int(df.g.iloc[-1])
    grid = np.arange(t0, t1 + BAR, BAR)
    lc = pd.Series(np.log(df.close.values), index=df.g.values).reindex(grid).values
    return grid, lc


def features(t, lc):
    """Computed on trading bars only (closed-market gaps squeezed out), so
    session-traded indices get full-length windows; results are mapped back
    to the 5-minute grid. Holds still may not cross a gap (see simulate)."""
    v = np.flatnonzero(np.isfinite(lc))
    x = lc[v]
    r = np.diff(x, prepend=np.nan)
    sig_v = np.sqrt(pd.Series(r ** 2).rolling(288, min_periods=200).mean().values)
    L = pd.Series(x)
    fv = {"sig": sig_v}
    for k in (12, 48, 288):
        fv[f"z{k}"] = (x - L.shift(k).values) / (sig_v * math.sqrt(k))
    for n in (48, 288):
        fv[f"hi{n}"] = L.shift(1).rolling(n, min_periods=n).max().values
        fv[f"lo{n}"] = L.shift(1).rolling(n, min_periods=n).min().values
    f = {}
    for k, arr in fv.items():
        full = np.full(len(lc), np.nan)
        full[v] = arr
        f[k] = full
    f["hour"] = (t % 86400) / 3600.0
    return f


def signals(f, lc, sym):
    """name -> direction array (+1 long, -1 short, 0 none) at each bar."""
    out = {}
    for k, th in itertools.product((12, 48), (2.0, 3.0)):
        z = np.nan_to_num(f[f"z{k}"])
        out[f"MR_z{k}_{th:g}"] = np.where(z > th, -1, np.where(z < -th, 1, 0))
        out[f"MOM_z{k}_{th:g}"] = -out[f"MR_z{k}_{th:g}"]
    for n in (48, 288):
        hi, lo = f[f"hi{n}"], f[f"lo{n}"]
        with np.errstate(invalid="ignore"):
            out[f"BO_{n}"] = np.where(lc > hi, 1, np.where(lc < lo, -1, 0))
    z = np.nan_to_num(f["z288"])
    hourly = (np.arange(len(lc)) % 12) == 0
    out["TSM_z288"] = np.where(hourly & (np.abs(z) > 1.5), np.sign(z), 0).astype(int)
    if sym.startswith("frx"):  # 21:00 UTC rollover: spreads blow out, prices glitch
        bad = (f["hour"] >= 20.5) & (f["hour"] < 22.0)
        for v in out.values():
            v[bad] = 0
    return out


EXITS = [(h, a, b) for h in (12, 48, 144) for (a, b) in ((1, 1), (1, 2), (2, 1), (0, 0))]
# h = max hold in bars (1h / 4h / 12h); stop = a*sig*sqrt(h), target = b*sig*sqrt(h);
# (0,0) = no stop/target, time exit only (risk unit still sig*sqrt(h)).


def simulate(t, lc, sig, direction, h, a, b, cbp):
    """Non-overlapping trades. Returns entry_time, gross_R, net_R, gross_logret."""
    n = len(lc)
    idx = np.flatnonzero(direction != 0)
    idx = idx[(idx + h < n) & np.isfinite(sig[idx]) & np.isfinite(lc[idx]) & (sig[idx] > 0)]
    if len(idx) == 0:
        return None
    path = lc[idx[:, None] + np.arange(1, h + 1)]
    ok = np.isfinite(path).all(axis=1)          # no gaps (weekends / closed market) inside the hold
    idx, path = idx[ok], path[ok]
    if len(idx) == 0:
        return None
    d = direction[idx][:, None]
    unit = sig[idx] * math.sqrt(h)
    move = d * (path - lc[idx][:, None]) / unit[:, None]          # in R
    if a > 0:
        hit_sl = move <= -a
        hit_tp = move >= b
        any_hit = hit_sl | hit_tp
        first = np.where(any_hit.any(axis=1), any_hit.argmax(axis=1), h - 1)
        # R is measured in stop-distance units when a stop exists
        risk = a
    else:
        first = np.full(len(idx), h - 1)
        risk = 1.0
    exit_R = move[np.arange(len(idx)), first] / risk
    exit_bar = idx + first + 1
    # greedy non-overlap
    keep, busy_until = [], -1
    for j, i in enumerate(idx):
        if i > busy_until:
            keep.append(j)
            busy_until = exit_bar[j]
    keep = np.array(keep)
    idx, exit_R, unit = idx[keep], exit_R[keep], unit[keep]
    cost_R = (cbp * 1e-4) / (unit * risk)
    gross_log = exit_R * unit * risk
    return t[idx], exit_R, exit_R - cost_R, gross_log


# ------------------------------------------------------------ statistics
def daily(ts, x):
    if len(ts) == 0:
        return np.array([])
    return pd.Series(x).groupby(ts // 86400).sum().values


def tstat(v):
    v = np.asarray(v, float)
    if len(v) < 5 or v.std(ddof=1) == 0:
        return 0.0, 1.0
    t = v.mean() / (v.std(ddof=1) / math.sqrt(len(v)))
    p = 0.5 * math.erfc(t / math.sqrt(2))          # one-sided, normal approx (days >= 30)
    return t, p


# ------------------------------------------------------------------ main
def run(candle_dir="candles", out_dir="cfd_out", syms=None):
    os.makedirs(out_dir, exist_ok=True)
    paths = sorted(glob.glob(f"{candle_dir}/*_5m.csv"))
    rows = []
    for p in paths:
        sym = os.path.basename(p)[:-7]
        if syms and sym not in syms:
            continue
        t, lc = load_grid(p)
        f = features(t, lc)
        sigs = signals(f, lc, sym)
        split = t[0] + DISC_FRAC * (t[-1] - t[0])
        cbp = cost_bp(sym)
        for name, dirn in sigs.items():
            for (h, a, b) in EXITS:
                res = simulate(t, lc, f["sig"], dirn, h, a, b, cbp)
                if res is None:
                    continue
                et, gR, nR, glog = res
                for part, m in (("disc", et < split), ("val", et >= split)):
                    if m.sum() == 0:
                        continue
                    rows.append(dict(sym=sym, cls=asset_class(sym), strat=name, fam=name.split("_")[0],
                                     h=h, a=a, b=b, part=part, n=int(m.sum()), cost_bp=cbp,
                                     gross_R=gR[m].mean(), net_R=nR[m].mean(),
                                     gross_bp=glog[m].mean() * 1e4,
                                     win=(nR[m] > 0).mean(),
                                     _t=et[m], _g=gR[m], _n=nR[m], _gl=glog[m]))
        print(f"{sym}: done", file=sys.stderr, flush=True)
    return rows


def report(rows, out_dir="cfd_out", k_pooled=3, min_n=30):
    R = pd.DataFrame(rows)
    lines = []
    P = lines.append
    P("CFD RESEARCH -- directional trades, stop/target/time exits, net of assumed costs")
    P(f"symbols: {R.sym.nunique()}   configs per symbol: {R.strat.nunique()} signals x {len(EXITS)} exits")
    P("R = multiples of the risk unit (stop distance). break-even cost = gross bps per trade.\n")

    # -------- level 1: pooled per asset class (one config, all symbols of the class)
    picks = []
    for cls, g in R.groupby("cls"):
        d = g[g.part == "disc"]
        scored = []
        for key, gg in d.groupby(["strat", "h", "a", "b"]):
            ts = np.concatenate(gg._t.values); nr = np.concatenate(gg._n.values)
            if len(ts) < min_n * 3:
                continue
            tt, _ = tstat(daily(ts, nr))
            scored.append((tt, key))
        scored.sort(reverse=True)
        for tt, key in scored[:k_pooled]:
            picks.append(("pooled", cls, key, tt))
    # -------- level 2: per symbol, best config per signal family
    for (sym, fam), g in R[R.part == "disc"].groupby(["sym", "fam"]):
        g = g[g.n >= min_n]
        if g.empty:
            continue
        best, bt = None, -9
        for _, r in g.iterrows():
            tt, _ = tstat(daily(r._t, r._n))
            if tt > bt:
                bt, best = tt, r
        if bt > 2.0:   # only take a symbol-level pick to validation if discovery was convincing
            picks.append(("symbol", sym, (best.strat, best.h, best.a, best.b), bt))

    m = max(len(picks), 1)
    alpha = 0.05 / m
    P(f"picks taken to validation: {m}  (Bonferroni alpha {alpha:.1e})\n")
    hdr = f"{'level':7} {'who':10} {'signal':12} {'hold':>4} {'SL':>3} {'TP':>3} {'disc t':>6} | {'val n':>6} {'days':>5} {'gross R':>8} {'net R':>7} {'gross bp':>8} {'cost bp':>7} {'val t':>6} {'p':>8}  verdict"
    P(hdr)
    out = []
    V = R[R.part == "val"]
    for level, who, (strat, h, a, b), dt in picks:
        sel = V[(V.strat == strat) & (V.h == h) & (V.a == a) & (V.b == b)]
        sel = sel[sel.cls == who] if level == "pooled" else sel[sel.sym == who]
        if sel.empty:
            continue
        ts = np.concatenate(sel._t.values); gR = np.concatenate(sel._g.values)
        nR = np.concatenate(sel._n.values); gl = np.concatenate(sel._gl.values)
        dly = daily(ts, nR)
        vt, p = tstat(dly)
        cb = float(np.average(sel.cost_bp, weights=sel.n))
        verdict = "VALIDATED" if p < alpha and nR.mean() > 0 else ("promising" if p < 0.05 and nR.mean() > 0 else "")
        rec = dict(level=level, who=who, signal=strat, hold_h=h * 5 / 60, sl=a, tp=b, disc_t=dt,
                   val_n=len(nR), days=len(dly), gross_R=gR.mean(), net_R=nR.mean(),
                   gross_bp=gl.mean() * 1e4, cost_bp=cb, val_t=vt, p=p, verdict=verdict)
        out.append(rec)
        P(f"{level:7} {who:10} {strat:12} {h*5/60:>3g}h {a:>3} {b:>3} {dt:>6.2f} | {len(nR):>6} {len(dly):>5} "
          f"{gR.mean():>+8.3f} {nR.mean():>+7.3f} {gl.mean()*1e4:>+8.2f} {cb:>7.1f} {vt:>6.2f} {p:>8.1e}  {verdict}")

    # -------- family summary: every config, every symbol, validation, gross (no selection at all)
    P("\nUNSELECTED BASELINE (validation, all symbols x all exits, per family, gross and net):")
    for (cls, fam), g in V.groupby(["cls", "fam"]):
        gR = np.concatenate(g._g.values); nR = np.concatenate(g._n.values); gl = np.concatenate(g._gl.values)
        P(f"  {cls:6} {fam:4} trades {len(gR):>7}  gross {gR.mean():+.3f}R ({gl.mean()*1e4:+.2f}bp)  net {nR.mean():+.3f}R")

    pd.DataFrame(out).to_csv(f"{out_dir}/validation.csv", index=False)
    R.drop(columns=[c for c in R.columns if c.startswith("_")]).to_csv(f"{out_dir}/all_configs.csv", index=False)
    txt = "\n".join(lines)
    open(f"{out_dir}/summary.txt", "w").write(txt)
    print(txt)
    return out


if __name__ == "__main__":
    cdir = sys.argv[1] if len(sys.argv) > 1 else "candles"
    odir = sys.argv[2] if len(sys.argv) > 2 else "cfd_out"
    report(run(cdir, odir), odir)
