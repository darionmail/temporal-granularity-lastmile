"""Timing re-run of the rolling-horizon experiment (Algorithm 1) on BU-day loads.
Mirrors rq3_experiment.py: same BU set, string ordering, rng sequence (seed 42), K_t, eps, delta, lambdas."""
import numpy as np, pandas as pd, lzma, sys, time, datetime as dt
from scipy.optimize import linear_sum_assignment
import engine as E
DELTA=1/7; LAMS=[250,1000,5000]; TARGET=70; BU=250; S3=np.sqrt(3)
def hex_center(q,r,s): return s*S3*(q+r/2), s*1.5*r
def load(path,depot):
    rows=[]; cur=None; bus=None
    for line in lzma.open(path,'rt'):
        line=line.rstrip('\n')
        if line.startswith('#'):
            p=line[1:].split(';'); cur=p[0]; bus=[tuple(map(int,x.split(','))) for x in p[1:]]; continue
        if cur!=depot: continue
        doy,rest=line.split(':',1); date=dt.date(2025,1,1)+dt.timedelta(int(doy)-1)
        for x in rest.split(';'):
            i,n=map(int,x.split(',')); q,r=bus[i]; rows.append((date,f'{q}_{r}',n))
    d=pd.DataFrame(rows,columns=['date','bu','n']); d['m']=pd.to_datetime(d.date).dt.month; return d
def bu_xy(ids):
    q,r=np.array([list(map(int,b.split('_'))) for b in ids]).T; return np.c_[hex_center(q,r,BU)]
def run(depot,path):
    d=load(path,depot); rng=np.random.default_rng(seed=42); T=[]; I=[]
    for t in range(1,12):
        tr=d[d.m==t]; te=d[d.m==t+1]
        ids=sorted(set(tr.bu)|set(te.bu)); bpos={b:i for i,b in enumerate(ids)}; X=bu_xy(ids)
        wtr=np.zeros(len(ids)); c=tr.groupby('bu').n.sum(); wtr[[bpos[b] for b in c.index]]=c.values/tr.date.nunique()
        K=max(2,int(round(tr.n.sum()/tr.date.nunique()/TARGET)))
        m=wtr>0; t0=time.perf_counter(); labF,CF=E.cap_kmeans(X[m],wtr[m],K,rng=rng); tF=time.perf_counter()-t0
        T.append(dict(depot=depot,month=t,date=None,step='FIX',sec=tF,n=int(m.sum()),K=K,nvar=int(m.sum())*K))
        labFix=np.argmin(((X[:,None]-CF[None])**2).sum(2),1); labFix[np.where(m)[0]]=labF
        Cw=CF.copy(); Cc_prev=None
        for day,g in te.groupby('date'):
            wd=np.zeros(len(ids)); wd[[bpos[b] for b in g.bu]]=g.n.values; act=wd>0; n=int(act.sum()); V=wd.sum(); mu=V/K
            t0=time.perf_counter(); cert=wd.max()>mu*(1+DELTA); tP1=time.perf_counter()-t0
            for lam in LAMS:
                n0=len(E.LOG); t0=time.perf_counter()
                E.assign_lp(X[act],wd[act],CF,eps=DELTA,home=labFix[act],lam=lam,integer=True)
                s=time.perf_counter()-t0; rec=E.LOG[n0]
                T.append(dict(depot=depot,month=t,date=str(day),step=f'PEN{lam}',sec=s,n=n,K=K,nvar=n*K,status=rec['status'],fallback=rec['fallback'],cert=cert,p1_sec=tP1))
            t0=time.perf_counter(); l,Cw=E.cap_kmeans(X[act],wd[act],K,C0=Cw,rng=rng); s=time.perf_counter()-t0
            T.append(dict(depot=depot,month=t,date=str(day),step='DWS',sec=s,n=n,K=K,nvar=n*K))
            t0=time.perf_counter(); l,Cc=E.cap_kmeans(X[act],wd[act],K,rng=rng); s1=time.perf_counter()-t0
            t0=time.perf_counter()
            if Cc_prev is not None:
                r_,c_=linear_sum_assignment(np.linalg.norm(Cc[:,None]-Cc_prev[None],axis=2)); perm=np.empty(K,int); perm[r_]=c_; Cc=Cc[np.argsort(perm)]
            s2=time.perf_counter()-t0; Cc_prev=Cc
            T.append(dict(depot=depot,month=t,date=str(day),step='DCS',sec=s1+s2,hung_sec=s2,n=n,K=K,nvar=n*K))
        print(depot,t,flush=True)
    pd.DataFrame(T).to_csv(f'timing_{depot}.csv',index=False)
if __name__=='__main__': run(sys.argv[1],sys.argv[2])
