import numpy as np, pandas as pd
HOLD = int(pd.Timestamp("2026-06-01", tz="UTC").timestamp())
X = pd.read_pickle("fx_features.pkl")
DEV = X[X.t < HOLD].copy()
HO = X[X.t >= HOLD].copy()
def fade_win(df, zcol, m):
    """win of betting AGAINST the sign of zcol at horizon m (ties lose)."""
    up = df[f"up{m}"].values; tie = df[f"tie{m}"].values
    bet_up = df[zcol].values < 0
    return np.where(tie, 0.0, np.where(bet_up, up, 1 - up))
