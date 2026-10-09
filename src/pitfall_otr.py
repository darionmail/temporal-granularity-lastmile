"""Section 5.4: reproduce the courier-identity pitfall on the original Oct/Dec Depot01 samples.
Needs the two original CSVs (paths in load.py)."""
from load import *; from hexlib import *
import numpy as np, pandas as pd
from scipy.cluster.vq import kmeans2
from scipy.optimize import linear_sum_assignment
res=[]
for m in ['oct','dec']:
    d=load(m)
    def otr(lab):
        out=0
        for c,g in d.groupby(lab):
            Q=g[['x','y']].values; cc,s=snap_cover_cap(Q); out+=(~inside(Q,cc,s)).sum()
        return 100*out/len(d)
    def otr_daily(lab):
        out=0
        for key,g in d.groupby([d.date,lab]):
            Q=g[['x','y']].values; cc,s=snap_cover_cap(Q); out+=(~inside(Q,cc,s)).sum()
        return 100*out/len(d)
    res.append(dict(month=m,variant='real',monthly=otr(d.cid),daily=otr_daily(d.cid)))
    for seed in range(5):
        rng=np.random.default_rng(seed); lr=pd.Series(index=d.index,dtype=float); la=lr.copy(); prev=None
        for day,g in d.groupby('date'):
            ids=sorted(g.cid.unique()); k=len(ids); X=g[['x','y']].values
            C,l=kmeans2(X,X[rng.choice(len(X),k,replace=False)],minit='matrix',seed=int(rng.integers(1e6)))
            lr[g.index]=np.array(ids)[l]
            if prev is None: slots=np.arange(k); prev=C.copy()
            else:
                cost=np.linalg.norm(C[:,None]-prev[None],axis=2); r,cc=linear_sum_assignment(cost)
                slots=np.full(k,-1); slots[r]=cc
                extra=np.where(slots<0)[0]; slots[extra]=len(prev)+np.arange(len(extra))
                newprev=np.vstack([prev,np.zeros((len(extra),2))]); newprev[slots]=C; prev=newprev
            la[g.index]=slots[l]
        res.append(dict(month=m,variant=f'random{seed}',monthly=otr(lr),daily=otr_daily(lr)))
        res.append(dict(month=m,variant=f'aligned{seed}',monthly=otr(la),daily=otr_daily(la)))
R=pd.DataFrame(res); R.to_csv('pitfall_otr.csv',index=False); print(R)
