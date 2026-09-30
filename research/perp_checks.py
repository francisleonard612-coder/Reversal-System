import numpy as np, pandas as pd, perp_research as P, sys
C,Q,F=pd.read_pickle('panel.pkl')
R=C/C.shift(1)-1
print("largest in-universe 4h moves:", R.stack().abs().sort_values().tail(3).round(2).to_dict())
rng=np.random.default_rng(1)
def rebuild(Rn):
    Cn=(1+Rn.fillna(0)).cumprod().where(C.notna()); return Cn
Rn=R.copy()
for c in R.columns:
    v=R[c].values; m=np.isfinite(v); w=v[m].copy(); rng.shuffle(w); v2=v.copy(); v2[m]=w; Rn[c]=v2
Fz=F*0
mode=sys.argv[1]
if mode=="null":
    P.run(rebuild(Rn),Q,Fz,out="perp_null",tag="(NULL: shuffled returns) ")
else:
    # planted: 7-day trend persists: next 4h return gets +0.08*sigma in the direction of the 7d move
    sig=Rn.rolling(180,min_periods=60).std()
    Cn=rebuild(Rn); m=np.sign(Cn/Cn.shift(42)-1)
    # build sequentially-consistent approximation: add drift using the null path's 7d sign
    Rp=Rn+0.08*sig.shift(1)*m.shift(1)
    P.run(rebuild(Rp),Q,Fz,out="perp_plant",tag="(PLANTED 7d trend) ")
