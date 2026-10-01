"""
Does the Rise-and-Fall-Minutes signal stack actually predict direction?
(github.com/monicalashaythomas-coder/rise-and-fall-minutes, risefall-bot/)

Replays THEIR code, unmodified, over historical 1-minute candles:
  compute_features -> fuse_signal -> regime_decision -> monte_carlo_duration
  -> MIN_EXP_WIN_RATE -> Gate 1 layer vote -> Gate 2 entropy -> Gate 3
  multi-timeframe -> Gate 4 bootstrap -> Gate 5 HMM/GBM borderline check
exactly in the order try_minute_gates_candidate() runs them. Gate 6 (LSTM)
is off: their trainer has no model that beats chance (their own README),
and LSTM_ENABLED=false is their documented setting for that case.

What a "fresh" bot has, so does this replay: no meta-learner (needs 200
settled trades), identity confidence calibration, default thresholds. Their
wall-clock gate auto-tuning is not replayed (it keys off real time).

Each qualifying signal is scored as a Rise/Fall contract: entry = that
minute's close, exit = close `duration` minutes later (the duration THEIR
Monte Carlo picked), CALL wins if exit > entry, PUT if exit < entry.
One open contract per symbol at a time (like the live bot).
Break-even for Deriv Rise/Fall at ~95% payout is 1/1.95 = 51.28%.

Usage:
  python research/risefall_stack_test.py --repo /path/to/rise-and-fall-minutes \
      --data data/cache/RDBULL_60d_1m_ohlc.csv data/cache/RDBEAR_60d_1m_ohlc.csv
  python research/risefall_stack_test.py --repo ... --selftest   # synthetic: null + planted edge
"""
from __future__ import annotations

import argparse
import contextlib
import io
import math
import os
import sys
import time
import types
from multiprocessing import Pool

import numpy as np

BREAK_EVEN = 1 / 1.95
FIT_WINDOW = 400          # minute bars the models/features see (live bot: ~200-400)
REFIT_EVERY = 120         # minutes between model refits (live: scheduled calibration ~2h)
DURATIONS_DIAG = (1, 2, 3, 5, 10)

B = None                  # their bot module, loaded per process
RC = None                 # their regime_conviction module


# --------------------------------------------------------------- loading
def load_their_code(repo: str):
    """Import their bot with torch/LSTM stubbed (only the disabled Gate 6 uses them)."""
    global B, RC
    os.environ.setdefault("LSTM_ENABLED", "false")
    if "torch" not in sys.modules:
        fake_torch = types.ModuleType("torch")
        fake_torch.Tensor = type("Tensor", (), {})   # scipy's array-API probe looks for torch.Tensor
        sys.modules["torch"] = fake_torch
    for net in ("websockets", "requests"):          # only used by their live trading/IO code
        try:
            __import__(net)
        except ImportError:
            sys.modules[net] = types.ModuleType(net)
    lstm = types.ModuleType("risefall_lstm_model")
    lstm.WINDOW_SIZE_TICKS, lstm.WINDOW_SIZE_MINUTES = 200, 200
    lstm.CANDIDATE_DURATIONS_TICKS = [1, 3, 5, 7, 10]
    lstm.CANDIDATE_DURATIONS_MINUTES = [1, 2, 3, 5, 10]

    class RiseFallWinClassifier:  # never instantiated with LSTM off
        pass
    lstm.RiseFallWinClassifier = RiseFallWinClassifier
    lstm.lstm_duration_scan = lambda *a, **k: None
    sys.modules["risefall_lstm_model"] = lstm
    sys.path.insert(0, os.path.join(repo, "risefall-bot"))
    with contextlib.redirect_stdout(io.StringIO()):
        import risefall_bot_v4_hmm_gbm as bot
        import regime_conviction as rc
    B, RC = bot, rc
    return bot


class View:
    """Same interface as their MinuteBarView."""
    def __init__(self, symbol, epochs, prices):
        self.symbol, self._e, self._p, self.tick_dt = symbol, epochs, prices, 60.0

    def prices(self):
        return self._p

    def epochs(self):
        return self._e

    def returns(self):
        p = self._p
        return np.diff(p) / p[:-1] if len(p) > 1 else np.array([])

    def mean_tick_dt(self):
        return 60.0

    def has_data(self, n=30):
        return len(self._p) >= n


# -------------------------------------------------------------- pipeline
def gates(view, models, feats, state, sym, minimal: bool):
    """try_minute_gates_candidate(), step for step. Returns (candidate|None, reason)."""
    feats["recent_call_ratio"] = 0.5
    p_up, conf = B.fuse_signal(feats, state, sym)
    d = 1 if p_up > 0.5 else -1
    mret = view.returns()

    regime_res = None
    if B.REGIME_ROUTING_ENABLED:
        sigma_now = float(np.std(mret[-30:])) if len(mret) >= 30 else 0.0
        sigma_base = float(np.median([abs(r) for r in mret[-200:]])) * 1.253 if len(mret) >= 60 else 0.0
        regime_res = B.regime_decision(hurst=feats.get("hurst", 0.5), sigma_now=sigma_now,
                                       sigma_baseline=sigma_base, layer_votes=feats["layer_votes"],
                                       base_stake=B.MIN_STAKE, cfg=B.REGIME_CFG)
        if not regime_res["trade"]:
            return None, "regime", p_up, conf, d

    dur, ewr = B.monte_carlo_duration(view.prices(), mret, d, feats,
                                      B.CANDIDATE_DURATIONS_MINUTES, models=models)
    if ewr < B.MIN_EXP_WIN_RATE:
        return None, "mc_win_rate", p_up, conf, d

    if not minimal:
        ok, *_ = B.passes_layer_gate(feats, d, regime=regime_res["regime"] if regime_res else None)
        if not ok:
            return None, "gate1_layers", p_up, conf, d
    pe_ok, _ = B.entropy_gate_passes(view.prices())
    if not pe_ok:
        return None, "gate2_entropy", p_up, conf, d
    tf_agree, _ = B.multi_timeframe_confluence(view.prices(), d)
    if tf_agree < B.MIN_TF_AGREEMENT:
        return None, "gate3_timeframes", p_up, conf, d
    if not minimal:
        bs_ok, _ = B.meta_ensemble_agrees(mret, d, dur, ewr)
        if not bs_ok:
            return None, "gate4_bootstrap", p_up, conf, d
    mc = B.hmm_gbm_scan(view.prices(), mret, B.CANDIDATE_DURATIONS_MINUTES,
                        hmm_model=getattr(models, "hmm_model", None))
    score = conf * state.reliability.get(sym, 1.0)
    thr = state.per_symbol_threshold.get(sym, state.adaptive_threshold)
    if mc["direction"] != d and score < B.MC_BORDERLINE_MULTIPLIER * thr:
        return None, "gate5_mc_borderline", p_up, conf, d
    return {"direction": d, "duration": int(dur), "p_up": p_up, "conf": conf, "ewr": ewr,
            "regime": regime_res["regime"] if regime_res else None}, "TRADE", p_up, conf, d


def run_symbol(args):
    repo, sym, data, step, max_evals = args
    load_their_code(repo)
    ep, px = data[sym]
    others = {s: v for s, v in data.items()}
    state = B.TradeState()
    n = len(px)
    maxd = max(DURATIONS_DIAG)
    models, last_fit = None, -10 ** 9
    busy = {"default": -1, "minimal": -1}
    trades, diag, reasons = [], [], {}
    t0, evals = time.time(), 0
    sink = io.StringIO()
    for t in range(FIT_WINDOW, n - maxd - 1, step):
        if max_evals and evals >= max_evals:
            break
        # skip windows with gaps (missing minutes)
        if ep[t] - ep[t - FIT_WINDOW + 1] != 60 * (FIT_WINDOW - 1) or ep[t + maxd] - ep[t] != 60 * maxd:
            continue
        lo = t - FIT_WINDOW + 1
        view = View(sym, ep[lo:t + 1], px[lo:t + 1])
        with contextlib.redirect_stdout(sink):
            if t - last_fit >= REFIT_EVERY:
                models = B.fit_symbol_models(view)
                last_fit = t
                if not getattr(models, "fitted", False):
                    models = None
            if models is None:
                continue
            rwd = {}
            for s, (e2, p2) in others.items():
                j = np.searchsorted(e2, ep[t], side="right") - 1
                if j >= 201 and e2[j] == ep[t]:
                    seg = p2[j - 200:j + 1]
                    rwd[s] = np.diff(seg) / seg[:-1]
            feats = B.compute_features(view, models, rwd)
            if feats is None:
                continue
            evals += 1
            for mode in ("default", "minimal"):
                cand, why, p_up, conf, d = gates(view, models, dict(feats), state, sym, mode == "minimal")
                if mode == "default":
                    reasons[why] = reasons.get(why, 0) + 1
                    outs = [int(np.sign(px[t + k] - px[t]) * d > 0) for k in DURATIONS_DIAG]
                    m = 1 if px[t] > px[t - 10] else -1          # scoring sanity check: plain 10-min momentum
                    mom = [int(np.sign(px[t + k] - px[t]) * m > 0) for k in DURATIONS_DIAG]
                    diag.append((ep[t], p_up, conf, d, *outs, *mom))
                if cand and t >= busy[mode]:
                    D = cand["duration"]
                    win = int(np.sign(px[t + D] - px[t]) * cand["direction"] > 0)
                    trades.append((mode, ep[t], cand["direction"], D, cand["p_up"], cand["conf"],
                                   cand["ewr"], cand["regime"], win))
                    busy[mode] = t + D
        sink.seek(0)
        sink.truncate()
    secs = time.time() - t0
    return sym, trades, diag, reasons, evals, secs


# -------------------------------------------------------------- stats
def binom_p_greater(k, n, p0):
    if n == 0:
        return 1.0
    z = (k - 0.5 - n * p0) / math.sqrt(n * p0 * (1 - p0))
    return 0.5 * math.erfc(z / math.sqrt(2))


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def report(results, days_by_sym, out_csv=None):
    lines = []
    P = lines.append
    P("RISE-AND-FALL-MINUTES SIGNAL STACK -- replayed on Deriv 1-minute history")
    P(f"break-even at ~95% payout: {BREAK_EVEN:.4f}\n")
    all_trades = []
    for sym, trades, diag, reasons, evals, secs in results:
        all_trades += [(sym,) + t for t in trades]
        P(f"{sym}: {evals:,} evaluations over {days_by_sym[sym]:.1f} days ({secs / 60:.1f} min compute)")
        tot = sum(reasons.values())
        P("  where signals stopped (default mode): " + ", ".join(
            f"{k} {v / tot:.0%}" for k, v in sorted(reasons.items(), key=lambda x: -x[1])))
        D = np.array([r[4:9] for r in diag], dtype=float) if diag else np.zeros((0, 5))
        M = np.array([r[9:14] for r in diag], dtype=float) if diag else np.zeros((0, 5))
        if len(D):
            P("  raw fused direction (every evaluation, no gates) hit rate by horizon: " + "  ".join(
                f"{k}m {D[:, i].mean():.4f}" for i, k in enumerate(DURATIONS_DIAG)))
            P("  scoring check -- plain 10-minute momentum, same scoring:            " + "  ".join(
                f"{k}m {M[:, i].mean():.4f}" for i, k in enumerate(DURATIONS_DIAG)))
            conf = np.array([r[2] for r in diag])
            top = conf >= np.quantile(conf, 0.8)
            P("  same, top-20% confidence only:                                   " + "  ".join(
                f"{k}m {D[top, i].mean():.4f}" for i, k in enumerate(DURATIONS_DIAG)))
    P("")
    tests = []
    for mode in ("default", "minimal"):
        T = [t for t in all_trades if t[1] == mode]
        k, n = sum(t[-1] for t in T), len(T)
        lo, hi = wilson(k, n)
        p = binom_p_greater(k, n, BREAK_EVEN)
        tpd = n / max(sum(days_by_sym.values()), 1e-9)
        P(f"[{mode.upper()} filters] trades {n:,} ({tpd:.1f}/symbol-day)  win {k / max(n, 1):.4f}  "
          f"95% CI {lo:.4f}-{hi:.4f}  p(beats break-even) {p:.2e}")
        tests.append(p)
        for sym in days_by_sym:
            Ts = [t for t in T if t[0] == sym]
            ks, ns = sum(t[-1] for t in Ts), len(Ts)
            P(f"    {sym:10} n={ns:>6}  win {ks / max(ns, 1):.4f}")
        for D in sorted({t[4] for t in T}):
            Td = [t for t in T if t[4] == D]
            kd, nd = sum(t[-1] for t in Td), len(Td)
            P(f"    {D:>2}m       n={nd:>6}  win {kd / max(nd, 1):.4f}")
    alpha = 0.05 / len(tests)
    P("")
    best = min(tests)
    if best < alpha:
        P(f"VERDICT: an edge above break-even survives (p={best:.1e} < {alpha:.3f}). Worth building on.")
    else:
        P("VERDICT: no edge above break-even. Building a bot on this stack is not supported by the data.")
    txt = "\n".join(lines)
    print(txt)
    if out_csv:
        import csv
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["symbol", "mode", "epoch", "direction", "duration", "p_up", "confidence",
                        "mc_exp_win_rate", "regime", "win"])
            w.writerows(all_trades)
    return txt


# ---------------------------------------------------------------- data
def load_csv(path):
    a = np.genfromtxt(path, delimiter=",", names=True)
    ep = a["epoch"].astype(np.int64)
    px = a["close"].astype(float)
    order = np.argsort(ep)
    ep, px = ep[order], px[order]
    keep = np.concatenate([[True], np.diff(ep) > 0])
    return ep[keep], px[keep]


def synth(kind, n, seed):
    rng = np.random.default_rng(seed)
    r = rng.normal(0, 1e-4, n)
    if kind == "planted":       # 10-minute momentum: next minute leans with the last 10 minutes
        out = r.copy()
        for i in range(10, n):
            out[i] = r[i] + 0.15 * np.sign(out[i - 10:i].sum()) * 1e-4
        r = out
    ep = 1_700_000_000 + 60 * np.arange(n + 1)
    return ep, 1000 * np.exp(np.concatenate([[0.0], np.cumsum(r)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="path to a clone of rise-and-fall-minutes")
    ap.add_argument("--data", nargs="*", default=[], help="1-minute CSVs with epoch,close columns")
    ap.add_argument("--step", type=int, default=3, help="evaluate every N minutes")
    ap.add_argument("--max-evals", type=int, default=0, help="cap per symbol (0 = all)")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default="risefall_stack_trades.csv")
    a = ap.parse_args()

    if a.selftest:
        data = {"NULL": synth("null", 6000, 1), "PLANTED": synth("planted", 6000, 2)}
    else:
        data = {}
        for p in a.data:
            sym = os.path.basename(p).split("_")[0]
            data[sym] = load_csv(p)
    days = {s: (v[0][-1] - v[0][0]) / 86400 for s, v in data.items()}
    jobs = [(a.repo, s, data, a.step, a.max_evals) for s in data]
    with Pool(min(a.workers, len(jobs))) as pool:
        results = pool.map(run_symbol, jobs)
    report(results, days, None if a.selftest else a.out)


if __name__ == "__main__":
    main()
