"""
Does consolidation PERSIST on these symbols?  (ER / Ends Between feasibility test)

    python tools/consolidation_test.py                       # RDBULL, RDBEAR, 14 days
    python tools/consolidation_test.py --symbols R_10,1HZ10V --days 21
    python tools/consolidation_test.py --selftest            # no network: proves the test works

WHY THIS MATTERS. An Ends Between contract wins when the price stays quiet
until expiry. Deriv already charges for the volatility it expects, so being
quiet NOW only helps if a quiet recent stretch predicts a quieter-than-usual
NEXT few minutes (volatility clustering). If it does, ER can win by trading
the calm stretches. If it doesn't, a calm stretch is luck and there is
nothing to forecast.

WHAT IT MEASURES, per symbol and contract length H (2, 3, 5, 10 minutes):

1. Clustering: autocorrelation of |1-minute returns| (a pure random walk
   gives ~0 at every lag).
2. Persistence: split windows into fifths by the last 20 minutes' volatility.
   Does the calmest fifth stay calmer over the next H minutes? Reported as
   future volatility in the calm fifth / the wild fifth, with a bootstrap
   95% interval. No clustering -> ~1.00.
3. The trade itself: a symmetric Ends Between range sized to win ~80% of the
   time on average. How much MORE often does it win in the calm fifth? That
   extra win rate is the most a consolidation filter could add, if Deriv
   prices every window at the symbol's fixed volatility.
4. Forecastability: out-of-sample R^2 of a HAR volatility forecast (last 5,
   20 and 60 minutes) against simply using the average.

Windows are non-overlapping so the intervals are honest. Nothing is traded;
the only network use is reading candle history. Fetched candles are cached
in data/cache/ so re-runs are instant.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import os
import sys
import time
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HORIZONS = (2, 3, 5, 10)
LOOKBACK = 20            # minutes of "recent" volatility, as ER's regime detector uses
TARGET_HIT = 0.80        # size the test range to win this often on average
BOOT = 1000
RNG = np.random.default_rng(7)


# ----------------------------------------------------------------- data

def _cache_path(sym: str, days: float) -> str:
    return os.path.join("data", "cache", f"{sym}_{days:g}d_1m.csv")


def load_cached(sym: str, days: float, max_age_h: float = 6.0):
    p = _cache_path(sym, days)
    if not os.path.exists(p) or (time.time() - os.path.getmtime(p)) > max_age_h * 3600:
        return None
    with open(p) as f:
        rows = list(csv.reader(f))[1:]
    return np.array([int(r[0]) for r in rows]), np.array([float(r[1]) for r in rows])


def save_cache(sym: str, days: float, epochs, closes) -> None:
    p = _cache_path(sym, days)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "close"])
        w.writerows(zip(epochs.tolist(), closes.tolist()))


async def fetch(symbols: list[str], days: float) -> dict:
    from config.loader import Settings
    from main import _make_client

    out, need = {}, []
    for s in symbols:
        c = load_cached(s, days)
        if c is not None:
            print(f"{s}: using cached candles ({len(c[0])})")
            out[s] = c
        else:
            need.append(s)
    if not need:
        return out
    settings = Settings()
    client = _make_client(settings)
    await client.connect()
    try:
        for s in need:
            raw = await client.candle_history_paged(s, int(days * 1440), granularity=60)
            if not raw:
                print(f"{s}: no data returned -- check the symbol name")
                continue
            ep = np.array([c.epoch for c in raw])
            cl = np.array([c.close for c in raw], dtype=float)
            save_cache(s, days, ep, cl)
            out[s] = (ep, cl)
            print(f"{s}: fetched {len(ep)} candles ({(ep[-1] - ep[0]) / 86400:.1f} days)")
    finally:
        await client.close()
    return out


# ------------------------------------------------------------- analysis

def contiguous_returns(epochs: np.ndarray, closes: np.ndarray) -> list[np.ndarray]:
    """1-minute log returns split into gap-free segments (a missing minute
    must not be read as one giant 'return')."""
    r = np.diff(np.log(closes))
    ok = np.diff(epochs) == 60
    segs, cur = [], []
    for ret, good in zip(r, ok):
        if good:
            cur.append(ret)
        elif cur:
            segs.append(np.array(cur))
            cur = []
    if cur:
        segs.append(np.array(cur))
    return [s for s in segs if len(s) > LOOKBACK + 60]


def acf_abs(segs, lags=(1, 2, 5, 10, 20, 30)):
    a = np.concatenate(segs)
    x = np.abs(a) - np.abs(a).mean()
    den = (x * x).sum()
    return {k: float((x[:-k] * x[k:]).sum() / den) for k in lags}, 1.96 / np.sqrt(len(a))


@dataclass
class HResult:
    h: int
    n: int
    ratio: float          # future vol calm fifth / wild fifth
    ratio_ci: tuple
    hit_all: float
    hit_calm: float
    edge: float           # hit_calm - hit_all
    edge_ci: tuple
    fifth_vol: list       # mean future vol per fifth, calm -> wild (relative to overall)


def windows(segs, h):
    """Non-overlapping (past 60 returns, next h returns) pairs."""
    past60, fut = [], []
    for s in segs:
        t = 60
        while t + h <= len(s):
            past60.append(s[t - 60:t])
            fut.append(s[t:t + h])
            t += h
    return np.array(past60), np.array(fut)


def analyse_h(segs, h, sigma):
    p60, fut = windows(segs, h)
    rv_past = np.sqrt((p60[:, -LOOKBACK:] ** 2).mean(axis=1))
    rv_fut = np.sqrt((fut ** 2).mean(axis=1))
    move = np.abs(fut.sum(axis=1)) / (sigma * np.sqrt(h))
    z = float(np.quantile(move, TARGET_HIT))          # range half-width, in sigma units
    win = move < z
    q = np.quantile(rv_past, [0.2, 0.4, 0.6, 0.8])
    fifth = np.searchsorted(q, rv_past)               # 0 = calmest
    calm, wild = fifth == 0, fifth == 4

    def stats(idx):
        f, cm, wd = rv_fut[idx], calm[idx], wild[idx]
        return f[cm].mean() / f[wd].mean(), win[idx][cm].mean() - win[idx].mean()

    ratio, edge = stats(np.arange(len(win)))
    boots = np.array([stats(RNG.integers(0, len(win), len(win))) for _ in range(BOOT)])
    ci = lambda col: (float(np.percentile(boots[:, col], 2.5)), float(np.percentile(boots[:, col], 97.5)))
    base = rv_fut.mean()
    return HResult(h, len(win), float(ratio), ci(0), float(win.mean()), float(win[calm].mean()),
                   float(edge), ci(1), [float(rv_fut[fifth == k].mean() / base) for k in range(5)])


def har_oos_r2(segs, h=5):
    p60, fut = windows(segs, h)
    lv = lambda a: np.log(np.sqrt((a ** 2).mean(axis=1)) + 1e-12)
    X = np.column_stack([np.ones(len(p60)), lv(p60[:, -5:]), lv(p60[:, -20:]), lv(p60)])
    y = lv(fut)
    cut = int(len(y) * 0.6)
    beta, *_ = np.linalg.lstsq(X[:cut], y[:cut], rcond=None)
    pred = X[cut:] @ beta
    sse = ((y[cut:] - pred) ** 2).sum()
    sst = ((y[cut:] - y[:cut].mean()) ** 2).sum()
    return float(1 - sse / sst)


def verdict(r5: HResult) -> str:
    persists = r5.ratio_ci[1] < 0.95                  # calm fifth reliably quieter later
    usable = r5.edge_ci[0] > 0.01                     # calm windows win reliably more often
    if persists and usable:
        return (f"CONSOLIDATION PERSISTS. Calm stretches stay calmer and win "
                f"{r5.edge * 100:+.1f} points more often. Worth building into ER.")
    if persists:
        return ("Some clustering, but too weak to lift the win rate reliably. "
                "Unlikely to beat Deriv's pricing.")
    return ("NO PERSISTENCE. A calm stretch does not predict a calm next few minutes "
            "here, so no consolidation filter can beat Deriv's price on this symbol.")


def report(sym: str, epochs, closes) -> None:
    segs = contiguous_returns(epochs, closes)
    if not segs:
        print(f"\n{sym}: not enough gap-free data")
        return
    allr = np.concatenate(segs)
    sigma = float(allr.std())
    acf, band = acf_abs(segs)
    print(f"\n{'=' * 78}\n{sym}   {len(allr):,} one-minute returns, "
          f"{len(allr) / 1440:.1f} days, per-minute vol {sigma * 100:.4f}%")
    print(f"  |return| autocorrelation (random walk ~0, noise band +/-{band:.3f}):  "
          + "  ".join(f"lag{k}={v:+.3f}" for k, v in acf.items()))
    print(f"  {'H':>4} {'windows':>8} {'calm/wild vol':>14} {'95% CI':>15} "
          f"{'win avg':>8} {'win calm':>9} {'edge':>7} {'95% CI':>16}")
    res = {}
    for h in HORIZONS:
        r = analyse_h(segs, h, sigma)
        res[h] = r
        print(f"  {h:>3}m {r.n:>8} {r.ratio:>14.3f} ({r.ratio_ci[0]:.2f}-{r.ratio_ci[1]:.2f}) "
              f"{r.hit_all:>8.3f} {r.hit_calm:>9.3f} {r.edge * 100:>+6.1f}p "
              f"({r.edge_ci[0] * 100:+.1f} to {r.edge_ci[1] * 100:+.1f})")
    print("  future vol by fifth of recent vol, calm -> wild (5m, 1.00 = average): "
          + "  ".join(f"{v:.2f}" for v in res[5].fifth_vol))
    print(f"  HAR forecast out-of-sample R^2 (5m): {har_oos_r2(segs):+.3f}  (0 = no better than average)")
    print(f"  VERDICT: {verdict(res[5])}")


# ------------------------------------------------------------- self-test

def simulate(kind: str, n=20160, seed=1):
    rng = np.random.default_rng(seed)
    if kind == "constant":
        r = rng.normal(0, 1e-4, n)
    else:                                              # GARCH(1,1): real volatility clustering
        r = np.empty(n)
        var = 1e-8
        for t in range(n):
            r[t] = np.sqrt(var) * rng.standard_normal()
            var = 1e-8 * 0.02 + 0.10 * r[t] ** 2 + 0.88 * var
    ep = 1_700_000_000 + 60 * np.arange(n + 1)
    return ep, 1000 * np.exp(np.concatenate([[0], np.cumsum(r)]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--symbols", default="RDBULL,RDBEAR")
    ap.add_argument("--days", type=float, default=14)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        print("SELF-TEST on simulated data (no network). Expect: constant -> NO PERSISTENCE, "
              "clustered -> CONSOLIDATION PERSISTS.")
        for kind in ("constant", "clustered"):
            report(f"simulated-{kind}", *simulate(kind))
        return
    data = asyncio.run(fetch([s.strip() for s in a.symbols.split(",") if s.strip()], a.days))
    for sym, (ep, cl) in data.items():
        report(sym, ep, cl)
    print("\nNote: 'edge' assumes Deriv prices every window at the symbol's fixed volatility.")


if __name__ == "__main__":
    main()
