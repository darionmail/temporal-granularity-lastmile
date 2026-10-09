"""Section 4.4: validate the BHH tour approximation against NN+2-opt on real Depot01 courier-days (Oct 2025)."""
import pandas as pd, numpy as np
from engine import tour_bhh; from tsp import nn2opt
d=pd.read_pickle('deliv.pkl'); z=d[(d.depot=='Depot01')&(d.dow<5)&(d.m==10)]
rows=[]
for (day,c),g in z.groupby(['date','cid']):
    if len(g)<10: continue
    P=g.sort_values('t')[['x','y']].values
    rows.append(dict(n=len(g),actual=np.hypot(*np.diff(P,axis=0).T).sum()/1000,tsp=nn2opt(P)/1000,bhh=tour_bhh(P)/1000))
V=pd.DataFrame(rows); V.to_csv('bhh_validation.csv',index=False)
print(V.corr(method='spearman').round(3)); print('bhh/tsp',(V.bhh/V.tsp).median(),'actual/tsp',(V.actual/V.tsp).median())

# All depots: random sample of 150 courier-days (>=10 parcels) per depot, 2025 weekdays
d=pd.read_pickle('deliv.pkl'); d=d[d.dow<5]; rows=[]
for dep,g0 in d.groupby('depot'):
    cd=g0.groupby(['date','cid']).size(); cd=cd[cd>=10].sample(150,random_state=7)
    for (day,c) in cd.index:
        g=g0[(g0.date==day)&(g0.cid==c)]; P=g.sort_values('t')[['x','y']].values
        rows.append(dict(depot=dep,n=len(g),actual=np.hypot(*np.diff(P,axis=0).T).sum()/1000,tsp=nn2opt(P)/1000,bhh=tour_bhh(P)/1000))
V=pd.DataFrame(rows); V.to_csv('bhh_validation_all.csv',index=False)
print(V.groupby('depot').apply(lambda s: pd.Series(dict(rho=s[['bhh','tsp']].corr('spearman').iloc[0,1],bhh_tsp=(s.bhh/s.tsp).median(),act_tsp=(s.actual/s.tsp).median()))))
