"""
Is there ANY predictable pattern in Deriv tick-by-tick moves?
(Feasibility for tick Rise/Fall: CALL wins if the price N ticks later is
strictly higher than the entry tick.)

    python research/tick_test.py data/cache/R_100_ticks.csv data/cache/RDBULL_ticks.csv ...
    python research/tick_test.py --selftest

Per symbol:
  1. Autocorrelation of tick changes at lags 1-20 (independent steps -> ~0).
  2. Runs test: are up/down streaks longer or shorter than chance?
  3. Predictors: momentum and reversal of the last k ticks (k = 1,2,3,5,10),
     streak-continuation/-reversal after 3+ same-direction ticks, all scored
     as Rise/Fall over the next N = 1,3,5,10 ticks. Each rule is CHOSEN on the
     first 60% of the data and SCORED on the last 40%, vs ~51.28% break-even
     (95% payout), Bonferroni-corrected over every rule x horizon x symbol.
  4. Long-only baseline (always Rise), to expose drift (RDBULL/RDBEAR).
"""
from __future__ import annotations

import math
import sys

import numpy as np

BREAK_EVEN = 1 / 1.95
HORIZONS = (1, 3, 5, 10)
SPLIT = 0.6


def load(path):
    a = np.genfromtxt(path, delimiter=",", names=True)
    t, p = a["epoch"], a["price"]
    o = np.argsort(t)
    return t[o], p[o]


def p_greater(k, n, p0):
    if n == 0:
        return 1.0
    z = (k - 0.5 - n * p0) / math.sqrt(n * p0 * (1 - p0))
    return 0.5 * math.erfc(z / math.sqrt(2))


def acf(x, lag):
    x = x - x.mean()
    return float(np.dot(x[:-lag], x[lag:]) / np.dot(x, x))


def runs_test(s):
    """Wald-Wolfowitz on the sign sequence (zeros dropped). Returns z
    (negative = streaks LONGER than chance, positive = more alternation)."""
    s = s[s != 0]
    n1, n2 = (s > 0).sum(), (s < 0).sum()
    n = n1 + n2
    runs = 1 + np.count_nonzero(s[1:] != s[:-1])
    mu = 2 * n1 * n2 / n + 1
    var = 2 * n1 * n2 * (2 * n1 * n2 - n) / (n * n * (n - 1))
    return (runs - mu) / math.sqrt(var), runs / n, (mu) / n


def signals(p):
    """dict name -> direction array (+1 CALL, -1 PUT, 0 none) at each tick."""
    d = np.sign(np.diff(p, prepend=p[0]))
    out = {}
    for k in (1, 2, 3, 5, 10):
        mv = np.zeros_like(p)
        mv[k:] = np.sign(p[k:] - p[:-k])
        out[f"momentum_{k}"] = mv
        out[f"reversal_{k}"] = -mv
    # streak of 3+ same-direction ticks
    streak = np.zeros_like(p)
    run = 0
    for i in range(1, len(p)):
        run = run + 1 if d[i] != 0 and d[i] == d[i - 1] else (1 if d[i] != 0 else 0)
        if run >= 3:
            streak[i] = d[i]
    out["streak3_continue"] = streak
    out["streak3_reverse"] = -streak
    out["always_rise"] = np.ones_like(p)
    return out


def analyse(name, t, p):
    n = len(p)
    d = np.diff(p)
    split = int(n * SPLIT)
    lines = [f"\n{'=' * 78}\n{name}: {n:,} ticks over {(t[-1] - t[0]) / 3600:.1f} h, "
             f"zero-change ticks {np.mean(d == 0):.1%}"]
    band = 1.96 / math.sqrt(len(d))
    ac = {k: acf(d, k) for k in (1, 2, 3, 5, 10, 20)}
    lines.append(f"  tick-change autocorrelation (independent ~0, noise band +/-{band:.4f}): "
                 + "  ".join(f"lag{k}={v:+.4f}" for k, v in ac.items()))
    z, r_obs, r_exp = runs_test(np.sign(d))
    lines.append(f"  runs test: z={z:+.2f} (|z|>3 = real pattern; negative = streaky, positive = choppy)")
    rows = []
    for rule, sig in signals(p).items():
        for H in HORIZONS:
            idx = np.arange(20, n - H)
            dirn = sig[idx]
            take = dirn != 0
            idx, dirn = idx[take], dirn[take]
            win = np.sign(p[idx + H] - p[idx]) == dirn       # tie loses
            disc, val = idx < split, idx >= split
            rows.append(dict(sym=name, rule=rule, H=H, disc=win[disc].mean() if disc.any() else np.nan,
                             n=int(val.sum()), k=int(win[val].sum()), win=win[val].mean() if val.any() else np.nan))
    return lines, rows


def report(datasets):
    out, allrows = [], []
    for name, (t, p) in datasets.items():
        lines, rows = analyse(name, t, p)
        out += lines
        allrows += rows
    m = len(allrows)
    alpha = 0.05 / m
    out.append(f"\n{m} rule x horizon x symbol tests -> Bonferroni alpha {alpha:.1e}")
    out.append("Win rates below are on the LAST 40% of each symbol's ticks (not used to pick anything).")
    for name in datasets:
        R = [r for r in allrows if r["sym"] == name]
        # best rule per horizon chosen on discovery
        out.append(f"\n  {name}:  best rule per horizon (picked on first 60%), scored on last 40%")
        for H in HORIZONS:
            cand = [r for r in R if r["H"] == H and r["rule"] != "always_rise"]
            best = max(cand, key=lambda r: r["disc"])
            pv = p_greater(best["k"], best["n"], BREAK_EVEN)
            base = next(r for r in R if r["H"] == H and r["rule"] == "always_rise")
            flag = "EDGE" if pv < alpha and best["win"] > BREAK_EVEN else ""
            out.append(f"    {H:>2} ticks: {best['rule']:17} disc {best['disc']:.4f} -> val {best['win']:.4f} "
                       f"(n={best['n']:,}) p={pv:.1e} {flag}   | always-Rise {base['win']:.4f}")
    edges = [r for r in allrows if r["n"] and r["win"] > BREAK_EVEN and p_greater(r["k"], r["n"], BREAK_EVEN) < alpha]
    out.append("\n" + "=" * 78)
    if edges:
        out.append(f"RESULT: {len(edges)} rule(s) beat break-even on held-out ticks after correction:")
        for r in sorted(edges, key=lambda r: -r["win"])[:10]:
            out.append(f"  {r['sym']} {r['rule']} {r['H']} ticks: {r['win']:.4f} on {r['n']:,}")
    else:
        out.append("RESULT: nothing beats the ~51.3% break-even on held-out ticks. Tick Rise/Fall has no edge "
                   "from recent-tick patterns on these symbols.")
    txt = "\n".join(out)
    print(txt)
    return txt


def synth(kind, n=80000, seed=0):
    rng = np.random.default_rng(seed)
    e = rng.normal(0, 1, n)
    if kind == "planted":                     # mild tick-level momentum: AR(1) phi=0.08 on changes
        x = np.zeros(n)
        for i in range(1, n):
            x[i] = 0.08 * x[i - 1] + e[i]
        e = x
    p = 1000 + np.round(np.cumsum(e) * 0.05, 2)
    return np.arange(n, dtype=float), p


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        report({"SIM-random": synth("null"), "SIM-planted-momentum": synth("planted", seed=1)})
    else:
        import os
        report({os.path.basename(f).replace("_ticks.csv", "").split("-")[-1]: load(f) for f in sys.argv[1:]})
