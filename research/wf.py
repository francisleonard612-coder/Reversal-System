import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from common import *
FEATS = ["z3","z6","z12","z24","z48","z96","z288","dev48","dev288","rp48","rp288",
         "calm","volreg","hsin","hcos","dow","symc"]
X["symc"] = X.sym.astype("category").cat.codes
DEV["symc"] = X.loc[DEV.index,"symc"]; HO["symc"] = X.loc[HO.index,"symc"]
BE = {15:0.559, 30:0.550, 60:0.550, 120:0.550, 240:0.560}
def model():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.03, max_leaf_nodes=15,
        min_samples_leaf=400, l2_regularization=1.0, categorical_features=[FEATS.index("symc")],
        random_state=0)
def walk_forward(df, m, months):
    """train on everything before each month, predict that month."""
    out=[]
    for ms in months:
        a=int(pd.Timestamp(ms,tz="UTC").timestamp()); b=int((pd.Timestamp(ms,tz="UTC")+pd.offsets.MonthBegin()).timestamp())
        tr=df[(df.t<a-86400)].dropna(subset=[f"up{m}"]); te=df[(df.t>=a)&(df.t<b)].dropna(subset=[f"up{m}"])
        # embargo 1 day before test; labels in train end before test starts
        clf=model().fit(tr[FEATS], tr[f"up{m}"])
        p=clf.predict_proba(te[FEATS])[:,1]
        o=te[["sym","t",f"up{m}",f"tie{m}"]].copy(); o["p"]=p; out.append(o)
    return pd.concat(out)
def score(o, m, thr):
    side_up = o.p>=0.5; conf=np.maximum(o.p,1-o.p)
    sel=o[conf>=thr]; su=side_up[conf>=thr]
    win=np.where(sel[f"tie{m}"],0,np.where(su,sel[f"up{m}"],1-sel[f"up{m}"]))
    return len(sel), win.mean() if len(sel) else np.nan, sel
