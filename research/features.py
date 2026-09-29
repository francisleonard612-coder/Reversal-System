"""Feature/label builder for forex mean-reversion research. Every feature at
decision bar i uses only bars <= i. Labels: Rise/Fall outcome of a CALL at
horizons, from bar-i close to bar-(i+h) close (strictly up = CALL wins)."""
import glob, os
import numpy as np
import pandas as pd

BAR = 300
H = {15: 3, 30: 6, 60: 12, 120: 24, 240: 48}


def load(sym):
    df = pd.read_csv(f"candles/{sym}_5m.csv")
    df = df.drop_duplicates("epoch").sort_values("epoch").reset_index(drop=True)
    return df


def build(sym, step=3):
    df = load(sym)
    t = df.epoch.values.astype(np.int64)
    c = df.close.values.astype(float)
    lc = np.log(c)
    n = len(c)
    pos = pd.Series(np.arange(n), index=t)

    def back(k):
        """index of bar exactly k bars earlier in time, else -1"""
        tgt = t - k * BAR
        j = pos.reindex(tgt).values
        return np.where(np.isnan(j), -1, j).astype(int)

    r = np.full(n, np.nan)
    j1 = back(1)
    ok = j1 >= 0
    r[ok] = lc[ok] - lc[j1[ok]]
    r2 = pd.Series(r ** 2)
    rv12 = np.sqrt(r2.rolling(12, min_periods=8).mean().values)
    rv288 = np.sqrt(r2.rolling(288, min_periods=200).mean().values)
    rv2016 = np.sqrt(r2.rolling(2016, min_periods=1000).mean().values)   # ~1 week

    feats = {"sym": sym, "t": t}
    for k in (3, 6, 12, 24, 48, 96, 288):
        jk = back(k)
        z = np.full(n, np.nan)
        m = jk >= 0
        z[m] = (lc[m] - lc[jk[m]]) / (rv288[m] * np.sqrt(k))
        feats[f"z{k}"] = z
    for k in (48, 288):
        ma = pd.Series(lc).rolling(k, min_periods=int(k * .8)).mean().values
        feats[f"dev{k}"] = (lc - ma) / (rv288 * np.sqrt(k / 3))
    # range position over last 48 / 288 bars
    for k in (48, 288):
        hi = pd.Series(c).rolling(k, min_periods=int(k * .8)).max().values
        lo = pd.Series(c).rolling(k, min_periods=int(k * .8)).min().values
        feats[f"rp{k}"] = (c - lo) / np.where(hi > lo, hi - lo, np.nan) - 0.5
    feats["calm"] = np.log(rv12 / rv288)
    feats["volreg"] = np.log(rv288 / rv2016)
    hr = (t % 86400) / 3600.0
    feats["hsin"], feats["hcos"] = np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24)
    feats["hour"] = hr
    feats["dow"] = ((t // 86400) + 4) % 7          # 0 = Monday
    for m, h in H.items():
        jf = pos.reindex(t + h * BAR).values
        y = np.full(n, np.nan)
        okf = ~np.isnan(jf)
        jf2 = jf[okf].astype(int)
        y[okf] = (c[jf2] > c[okf]).astype(float)
        y[okf & False] = np.nan
        feats[f"up{m}"] = y
        # tie -> CALL loses AND PUT loses; keep a tie flag
        tie = np.zeros(n, bool)
        tie[np.where(okf)[0]] = c[jf2] == c[okf]
        feats[f"tie{m}"] = tie
    out = pd.DataFrame(feats)
    # decide every `step` bars (15 minutes), on the quarter hour
    out = out[(out.t // BAR) % step == step - 1]
    return out.dropna(subset=["z48", "z288", "calm", "volreg", "dev288"])


if __name__ == "__main__":
    syms = sorted(os.path.basename(p)[:-7] for p in glob.glob("candles/frx*_5m.csv"))
    frames = [build(s) for s in syms]
    X = pd.concat(frames, ignore_index=True)
    X.to_parquet("fx_features.parquet") if False else X.to_pickle("fx_features.pkl")
    print(X.shape, X.sym.nunique(), "pairs")
    print(pd.to_datetime(X.t.min(), unit="s"), pd.to_datetime(X.t.max(), unit="s"))
