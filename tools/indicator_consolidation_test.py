"""
Can EMA / MACD / ADX / Bollinger / efficiency-ratio / variance-ratio spot a
consolidation that CONTINUES long enough for an Ends Between trade to win?

    python tools/indicator_consolidation_test.py                  # RDBULL, RDBEAR, 60 days
    python tools/indicator_consolidation_test.py --symbols R_10 --days 30
    python tools/indicator_consolidation_test.py --selftest       # no network: proves the test works

THE QUESTION. Ends Between pays when the price at expiry is inside the range.
Deriv prices that at the symbol's normal volatility. So a "consolidation"
flag is only worth something if, when it is ON, the next H minutes end
inside a range MORE often than usual.

HOW. For every indicator, "consolidation" = its most range-like 20% of
minutes (cut-offs fixed on the first 60% of the history). At each decision
minute (non-overlapping, one every H minutes) we place a symmetric range
sized to win 80% of the time at the symbol's normal volatility, and compare:

    win rate when the flag is ON  vs  win rate overall

on the LAST 40% of the history only (the part the cut-offs never saw).
"edge" = that difference in percentage points = the most this filter could
add to ER's win rate. PASS needs a positive edge whose p-value survives a
Bonferroni correction over every indicator x horizon tested.

Indicators (all on 1-minute candles, using only bars up to the decision):
  ER      Kaufman efficiency ratio (20): net move / total movement; low = choppy
  BBW     Bollinger band width (20, 2sd) as a percentile of the last 500 bars; low = squeeze
  ATRR    ATR(14) / ATR(100); low = volatility compressed
  ADX     ADX(14); low = no trend
  EMA     |EMA9 - EMA21| / ATR14 plus |EMA21 slope|; low = averages tangled
  MACD    |MACD histogram| / ATR14; low = no momentum
  MACDX   MACD(12,26) zero-crossings in the last 30 bars; high = see-saw
  VR      variance ratio of 5-minute vs 1-minute returns over 60 bars; low = pulls back
  COMBO   at least 3 of the above flags at once

Numpy only. Candles cached in data/cache/ so re-runs are instant.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HORIZONS = (2, 3, 5, 10)
TARGET_HIT = 0.80
K_BARRIER = 1.2815516        # P(|Z| < k) = 0.80
FLAG_FRAC = 0.20
DISC_FRAC = 0.60
WARMUP = 600                 # bars before the first decision (BBW percentile needs 500)
IND = ("ER", "BBW", "ATRR", "ADX", "EMA", "MACD", "MACDX", "VR")
LOW_IS_CALM = {"ER": True, "BBW": True, "ATRR": True, "ADX": True, "EMA": True,
               "MACD": True, "MACDX": False, "VR": True}


# ------------------------------------------------------------------ data
def _cache(sym, days):
    return os.path.join("data", "cache", f"{sym}_{days:g}d_1m_ohlc.csv")


def load_cached(sym, days, max_age_h=6.0):
    p = _cache(sym, days)
    if not os.path.exists(p) or time.time() - os.path.getmtime(p) > max_age_h * 3600:
        return None
    a = np.loadtxt(p, delimiter=",", skiprows=1)
    return a[:, 0].astype(np.int64), a[:, 1], a[:, 2], a[:, 3], a[:, 4]


def save_cache(sym, days, ep, o, h, l, c):
    p = _cache(sym, days)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "open", "high", "low", "close"])
        w.writerows(zip(ep.tolist(), o.tolist(), h.tolist(), l.tolist(), c.tolist()))


async def fetch(symbols, days):
    from config.loader import Settings
    from main import _make_client

    out, need = {}, []
    for s in symbols:
        c = load_cached(s, days)
        if c is not None:
            print(f"{s}: using cached candles ({len(c[0]):,})")
            out[s] = c
        else:
            need.append(s)
    if not need:
        return out
    client = _make_client(Settings())
    await client.connect()
    try:
        for s in need:
            print(f"{s}: downloading {days:g} days of 1-minute candles ...", flush=True)
            raw = await client.candle_history_paged(s, int(days * 1440), granularity=60)
            if not raw:
                print(f"{s}: no data returned -- check the symbol name")
                continue
            raw.sort(key=lambda x: x.epoch)
            arr = [np.array([getattr(x, k) for x in raw], dtype=float) for k in ("open", "high", "low", "close")]
            ep = np.array([x.epoch for x in raw], dtype=np.int64)
            save_cache(s, days, ep, *arr)
            out[s] = (ep, *arr)
            print(f"{s}: {len(ep):,} candles ({(ep[-1] - ep[0]) / 86400:.1f} days)")
    finally:
        await client.close()
    return out


# ------------------------------------------------------------ indicators
def ema(x, n):
    a = 2.0 / (n + 1)
    out = np.empty_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out


def wilder(x, n):
    out = np.full_like(x, np.nan)
    if len(x) < n:
        return out
    out[n - 1] = np.nanmean(x[:n])
    for i in range(n, len(x)):
        out[i] = out[i - 1] + (x[i] - out[i - 1]) / n
    return out


def rolling_sum(x, n):
    cs = np.concatenate([[0.0], np.cumsum(x)])
    out = np.full(len(x), np.nan)
    out[n - 1:] = cs[n:] - cs[:-n]
    return out


def rolling_std(x, n):
    m = rolling_sum(x, n) / n
    m2 = rolling_sum(x * x, n) / n
    return np.sqrt(np.maximum(m2 - m * m, 0.0))


def rolling_pct_rank(x, n):
    """Percentile of x[i] within x[i-n+1..i] (inclusive)."""
    out = np.full(len(x), np.nan)
    from numpy.lib.stride_tricks import sliding_window_view
    if len(x) < n:
        return out
    w = sliding_window_view(x, n)
    out[n - 1:] = (w <= w[:, -1:]).mean(axis=1)
    return out


def indicators(o, h, l, c):
    n = len(c)
    prev_c = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    atr14, atr100 = wilder(tr, 14), wilder(tr, 100)
    f = {}
    # efficiency ratio
    net = np.full(n, np.nan)
    net[20:] = np.abs(c[20:] - c[:-20])
    path = rolling_sum(np.abs(np.diff(c, prepend=c[0])), 20)
    f["ER"] = net / np.where(path > 0, path, np.nan)
    # Bollinger width percentile
    sd = rolling_std(c, 20)
    ma = rolling_sum(c, 20) / 20
    bbw = 4 * sd / ma
    f["BBW"] = rolling_pct_rank(np.nan_to_num(bbw, nan=np.nanmedian(bbw)), 500)
    f["BBW"][:520] = np.nan
    f["ATRR"] = atr14 / atr100
    # ADX(14)
    up, dn = h - np.concatenate([[h[0]], h[:-1]]), np.concatenate([[l[0]], l[:-1]]) - l
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    pdi = 100 * wilder(pdm, 14) / atr14
    mdi = 100 * wilder(mdm, 14) / atr14
    dx = 100 * np.abs(pdi - mdi) / np.where(pdi + mdi > 0, pdi + mdi, np.nan)
    f["ADX"] = wilder(np.nan_to_num(dx), 14)
    # EMA compression: gap + slope, both in ATR units
    e9, e21 = ema(c, 9), ema(c, 21)
    slope = np.full(n, np.nan)
    slope[5:] = np.abs(e21[5:] - e21[:-5])
    f["EMA"] = (np.abs(e9 - e21) + slope) / atr14
    # MACD
    macd = ema(c, 12) - ema(c, 26)
    sig = ema(macd, 9)
    f["MACD"] = np.abs(macd - sig) / atr14
    cross = np.concatenate([[0.0], (np.sign(macd[1:]) != np.sign(macd[:-1])).astype(float)])
    f["MACDX"] = rolling_sum(cross, 30)
    # variance ratio VR(5) over the last 60 one-minute returns
    r = np.diff(np.log(c), prepend=np.log(c[0]))
    r5 = rolling_sum(r, 5)
    v1 = rolling_std(r, 60) ** 2
    v5 = rolling_std(np.nan_to_num(r5), 60) ** 2
    f["VR"] = v5 / (5 * np.where(v1 > 0, v1, np.nan))
    for k in f:
        f[k][:WARMUP] = np.nan
    return f


# -------------------------------------------------------------- analysis
def binom_p_greater(k_on, n_on, p0):
    """One-sided p that a hit rate >= k_on/n_on arises when the true rate is p0
    (normal approximation with continuity correction; n_on is large)."""
    if n_on == 0:
        return 1.0
    z = (k_on - 0.5 - n_on * p0) / math.sqrt(n_on * p0 * (1 - p0))
    return 0.5 * math.erfc(z / math.sqrt(2))


def analyse(sym, ep, o, h, l, c, quiet=False):
    f = indicators(o, h, l, c)
    lc = np.log(c)
    n = len(c)
    contiguous = np.concatenate([[True], np.diff(ep) == 60])
    bad_run = np.cumsum(~contiguous)                  # changes whenever a gap occurs
    r = np.diff(lc)
    sigma = float(np.std(r[np.abs(r) < 20 * np.std(r)]))
    split = int(n * DISC_FRAC)

    # cut-offs from the discovery part only
    cut = {}
    for k in IND:
        x = f[k][WARMUP:split]
        x = x[np.isfinite(x)]
        cut[k] = np.quantile(x, FLAG_FRAC if LOW_IS_CALM[k] else 1 - FLAG_FRAC)

    def flags_at(idx):
        fl = {}
        for k in IND:
            v = f[k][idx]
            fl[k] = (v <= cut[k]) if LOW_IS_CALM[k] else (v >= cut[k])
            fl[k] &= np.isfinite(v)
        fl["COMBO"] = sum(fl[k].astype(int) for k in IND) >= 3
        return fl

    results = []
    for H in HORIZONS:
        idx = np.arange(WARMUP, n - H, H)
        idx = idx[bad_run[idx + H] == bad_run[idx - 120]]            # no gap in lookback or hold
        width = K_BARRIER * sigma * math.sqrt(H)
        win = np.abs(lc[idx + H] - lc[idx]) < width
        fl = flags_at(idx)
        val = idx >= split
        base_disc, base_val = win[~val].mean(), win[val].mean()
        for k, m in fl.items():
            on_v = m & val
            n_on, k_on = int(on_v.sum()), int(win[on_v].sum())
            hit_on = k_on / n_on if n_on else float("nan")
            d_on = m & ~val
            results.append(dict(sym=sym, H=H, ind=k, disc_edge=(win[d_on].mean() - base_disc) if d_on.any() else np.nan,
                                n_on=n_on, n_val=int(val.sum()), share_on=n_on / max(val.sum(), 1),
                                hit_all=base_val, hit_on=hit_on, edge=hit_on - base_val,
                                p=binom_p_greater(k_on, n_on, base_val)))
    return results, sigma, n


def report(all_res, meta):
    m = len(all_res)
    alpha = 0.05 / max(m, 1)
    print(f"\nTests in validation: {m}  ->  Bonferroni alpha {alpha:.1e}")
    for sym in dict.fromkeys(r["sym"] for r in all_res):
        sigma, n = meta[sym]
        print(f"\n{'=' * 86}\n{sym}   {n:,} one-minute candles ({n / 1440:.1f} days), per-minute vol {sigma * 100:.4f}%")
        print("Range sized to win 80% at normal volatility. 'edge' = extra win rate when the flag is ON (validation).")
        print(f"  {'flag':6} " + "".join(f"{f'{H}m edge':>12} {'p':>8} {'on':>5}  " for H in HORIZONS) + " verdict")
        rows = [r for r in all_res if r["sym"] == sym]
        for k in IND + ("COMBO",):
            line, passes, helps = f"  {k:6} ", 0, 0
            for H in HORIZONS:
                r = next(x for x in rows if x["ind"] == k and x["H"] == H)
                line += f"{r['edge'] * 100:>+10.1f}p {r['p']:>8.1e} {r['share_on']:>5.0%}  "
                if r["edge"] > 0 and r["p"] < alpha:
                    passes += 1
                if r["edge"] >= 0.03 and r["p"] < alpha:
                    helps += 1
            verdict = ("USEFUL (+3p or more, significant)" if helps else
                       "real but small" if passes else "no edge")
            line += f" {verdict}"
            print(line)
        base = next(x for x in rows if x["ind"] == "ER" and x["H"] == 5)
        print(f"  (overall 5m win rate in validation: {base['hit_all']:.3f}; 'on' = share of minutes flagged)")
    good = [r for r in all_res if r["edge"] >= 0.03 and r["p"] < alpha]
    print("\n" + "=" * 86)
    if good:
        best = max(good, key=lambda r: r["edge"])
        print(f"RESULT: {len(good)} indicator/horizon pairs add 3+ points of win rate, significant after correction.")
        print(f"        Best: {best['sym']} {best['ind']} at {best['H']}m, +{best['edge'] * 100:.1f} points "
              f"({best['hit_on']:.3f} vs {best['hit_all']:.3f}).  -> worth adding to ER as a filter.")
    else:
        print("RESULT: no indicator adds a meaningful, significant win-rate edge. A consolidation filter")
        print("        would cut the number of trades without making them more likely to win.")


# ------------------------------------------------------------- self-test
def simulate(kind, n=60 * 1440, seed=3):
    rng = np.random.default_rng(seed)
    if kind == "random-walk":
        r = rng.normal(0, 1e-4, n)
    elif kind == "vol-clustering":                    # GARCH(1,1)
        r, var = np.empty(n), 1e-8
        for t in range(n):
            r[t] = math.sqrt(var) * rng.standard_normal()
            var = 1e-8 * 0.02 + 0.10 * r[t] ** 2 + 0.88 * var
    else:                                             # regimes: persistent ranges alternate with trends
        r = np.empty(n)
        lp, centre, in_range, left = 0.0, 0.0, True, 0
        for t in range(n):
            if left == 0:
                in_range = not in_range
                left = int(rng.exponential(120)) + 30
                centre = lp
                drift = rng.choice([-1, 1]) * 3e-5
            e = rng.normal(0, 1e-4)
            step = (-0.2 * (lp - centre) + e) if in_range else (drift + e)
            lp += step
            r[t] = step
            left -= 1
    lc = np.concatenate([[0.0], np.cumsum(r)])
    c = 1000 * np.exp(lc)
    # synthetic intrabar high/low from a few sub-steps
    noise = np.abs(rng.normal(0, np.std(r) * 0.5, len(c)))
    o = np.concatenate([[c[0]], c[:-1]])
    h = np.maximum(o, c) * np.exp(noise)
    l = np.minimum(o, c) * np.exp(-noise)
    ep = 1_700_000_000 + 60 * np.arange(len(c))
    return ep, o, h, l, c


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--symbols", default="RDBULL,RDBEAR")
    ap.add_argument("--days", type=float, default=60)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--csv", default=os.path.join("data", "indicator_consolidation.csv"))
    a = ap.parse_args()
    if a.selftest:
        print("SELF-TEST on simulated data (no network). Expect: random-walk -> no edge; "
              "vol-clustering -> BBW/ATRR useful; range-regimes -> ER/ADX/EMA useful.")
        data = {k: simulate(k) for k in ("random-walk", "vol-clustering", "range-regimes")}
    else:
        data = asyncio.run(fetch([s.strip() for s in a.symbols.split(",") if s.strip()], a.days))
    all_res, meta = [], {}
    for sym, arrs in data.items():
        res, sigma, n = analyse(sym, *arrs)
        all_res += res
        meta[sym] = (sigma, n)
    report(all_res, meta)
    if not a.selftest and all_res:
        os.makedirs(os.path.dirname(a.csv), exist_ok=True)
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(all_res[0].keys()))
            w.writeheader()
            w.writerows(all_res)
        print(f"\nDetails saved to {a.csv} -- send this file (or a screenshot of the output) to Claude.")


if __name__ == "__main__":
    main()
