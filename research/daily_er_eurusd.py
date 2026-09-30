"""Quick feasibility: can we forecast whether EURUSD ends a 24h window inside a range,
better than a pricer that uses slow (22-day) volatility? One year of 5m closes."""
import numpy as np, pandas as pd, math
d=pd.read_csv("candles/frxEURUSD_5m.csv"); d["g"]=(d.epoch//300)*300
d=d.drop_duplicates("g",keep="last").set_index("g").close
idx=np.arange(d.index[0],d.index[-1]+300,300); c=d.reindex(idx)
lc=np.log(c.values); r=np.diff(lc,prepend=np.nan)
t=idx; hour=(t%86400)/3600
# rollover spike around 21:00 UTC: drop those returns from vol estimates
r_clean=np.where((hour>=20.9)&(hour<21.25),np.nan,r)
valid=np.isfinite(r_clean)
sq=pd.Series(r_clean[valid]**2)
def rv(n,minp):                                        # rolling over TRADING bars only (weekends squeezed out)
    out=np.full(len(r_clean),np.nan); out[valid]=np.sqrt(sq.rolling(n,min_periods=minp).mean().values*288)
    return pd.Series(out).ffill().values
rv1,rv5,rv22=rv(288,200),rv(288*5,900),rv(288*22,4000)
H=288; K=1.2815516
rows=[]
for i in range(len(t)):
    if t[i]%3600: continue                           # hourly entries
    j=i+H
    if j>=len(t) or not np.isfinite(lc[i]) or not np.isfinite(lc[j]): continue
    dow=((t[i]//86400)+3)%7
    if dow>=5: continue
    if np.isnan(lc[i:j+1]).mean()>0.2: continue      # spans weekend/closure
    if not (np.isfinite(rv1[i]) and np.isfinite(rv22[i])): continue
    rows.append((t[i],hour[i],abs(lc[j]-lc[i]),rv1[i],rv5[i],rv22[i]))
X=pd.DataFrame(rows,columns=["t","hour","absret","rv1","rv5","rv22"]).dropna()
split=X.t.quantile(0.6); D,V=X[X.t<split],X[X.t>=split]
# HAR on log |ret| scale: predict log(absret) from log rv's (fit on discovery)
def feats(Z): return np.column_stack([np.ones(len(Z)),np.log(Z.rv1),np.log(Z.rv5),np.log(Z.rv22)])
y=np.log(D.absret+1e-6); beta=np.linalg.lstsq(feats(D),y,rcond=None)[0]
# scale so forecast sigma is calibrated: absret ~ sigma*|Z| -> E|Z|=0.798
Dp=np.exp(feats(D)@beta); scale=np.median(D.absret/Dp)/0.6745
V=V.assign(sig_f=np.exp(feats(V)@beta)*scale)
naive_scale=np.median(D.absret/D.rv22)/0.6745
V=V.assign(sig_n=V.rv22*naive_scale)
V["win_naive"]=V.absret< K*V.sig_n            # range a slow-vol pricer thinks wins 80%
V["ratio"]=V.sig_f/V.sig_n
V["q"]=pd.qcut(V.ratio,5,labels=["calmest","2","3","4","wildest"])
days=V.t.floordiv(86400).nunique()
print(f"validation: {len(V)} hourly entries over {days} trading days  ({pd.to_datetime(V.t.min(),unit='s').date()} -> {pd.to_datetime(V.t.max(),unit='s').date()})")
print(f"range sized to win 80% using 22-day vol; overall it won {V.win_naive.mean():.3f}")
g=V.groupby("q",observed=True)
out=pd.DataFrame({"entries":g.size(),"days":g.t.apply(lambda x:x.floordiv(86400).nunique()),
                  "forecast/naive":g.ratio.mean().round(2),"win rate":g.win_naive.mean().round(3)})
print(out.to_string())
# day-level (independent) test for the calm quintile: one observation per day = mean win that day
cal=V[V.q=="calmest"]; per_day=cal.groupby(cal.t//86400).win_naive.mean()
base=V.win_naive.mean(); se=per_day.std(ddof=1)/math.sqrt(len(per_day))
print(f"calmest fifth: {per_day.mean():.3f} vs {base:.3f} overall, per-day SE {se:.3f}, z={(per_day.mean()-base)/se:+.2f}")
# forecast skill: correlation of log forecast with log realized
print("corr(log forecast, log realized |24h move|): HAR %.2f  naive %.2f"%(
    np.corrcoef(np.log(V.sig_f),np.log(V.absret+1e-6))[0,1],np.corrcoef(np.log(V.sig_n),np.log(V.absret+1e-6))[0,1]))
