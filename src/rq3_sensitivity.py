import pandas as pd, numpy as np, sys, time
from scipy.optimize import linear_sum_assignment
from engine import *
from prep import hex_center
import os
DELTA=1/7; LAMS=[1000]; TARGET=int(os.environ.get('TARGET',70)); BU=int(os.environ.get('BU',250)); SEED=int(os.environ.get('SEED',42)); ENGINE=os.environ.get('ENGINE','mcf')
NB=[(1,0),(-1,0),(0,1),(0,-1),(1,-1),(-1,1)]
def bu_xy(ids):
    q,r=np.array([list(map(int,b.split('_'))) for b in ids]).T; return np.c_[hex_center(q,r,BU)]
def repair(lab,load_bu,ids,idx,C,mu,maxmoves=60):
    lab=lab.copy(); K=len(C); L=np.bincount(lab,weights=load_bu,minlength=K)
    lo,hi=mu*(1-DELTA),mu*(1+DELTA); moved=np.zeros(len(lab),bool)
    qr=[tuple(map(int,b.split('_'))) for b in ids]
    nbrs=[[idx[(q+a,r+b)] for a,b in NB if (q+a,r+b) in idx] for q,r in qr]
    X=bu_xy(ids)
    for _ in range(maxmoves):
        if ((L>=lo)&(L<=hi)).all(): break
        best=None; bestgain=0
        dev=np.abs(L-mu)
        for b in np.where(load_bu>0)[0]:
            k=lab[b]
            for nb in nbrs[b]:
                j=lab[nb]
                if j==k: continue
                nk,nj=L[k]-load_bu[b],L[j]+load_bu[b]
                gain=max(dev[k],dev[j])-max(abs(nk-mu),abs(nj-mu))
                if gain>bestgain+1e-9 or (best is not None and abs(gain-bestgain)<1e-9 and np.hypot(*(X[b]-C[j]))<best[3]):
                    if gain>1e-9: best=(b,k,j,np.hypot(*(X[b]-C[j]))); bestgain=gain
        if best is None: break
        b,k,j,_=best; lab[b]=j; L[k]-=load_bu[b]; L[j]+=load_bu[b]; moved[b]=True
    return lab,moved
def metrics(P,lab_p,K,mu,cont):
    L=np.bincount(lab_p,minlength=K).astype(float)
    tours=sum(tour_bhh(P[lab_p==k]) for k in range(K))/1000
    comp=np.mean([np.hypot(*(P[lab_p==k]-np.median(P[lab_p==k],0)).T).mean() for k in range(K) if (lab_p==k).sum()>0])
    return dict(cv=L.std()/L.mean(),inband=((L>=mu*(1-DELTA)-1e-6)&(L<=mu*(1+DELTA)+1e-6)).mean(),
                inband_abs=((L>=60)&(L<=80)).mean(),maxdev=np.abs(L-mu).max()/mu,tour_km=tours,comp_m=comp,cont=cont)
def run(depot,months,seed=None,annual=False):
    seed=SEED if seed is None else seed
    d=pd.read_pickle('deliv.pkl'); d=d[(d.depot==depot)&(d.dow<5)].copy()
    if BU!=250:
        from prep import hex_axial
        q,r=hex_axial(d.x.values,d.y.values,BU); d['bu']=pd.Series(q,index=d.index).astype(str)+'_'+pd.Series(r,index=d.index).astype(str)
    out=[]; rng=np.random.default_rng(seed)
    for t in months:
        tr=d[d.m==(1 if annual else t)]; te=d[d.m==t+1]
        if len(tr)==0 or len(te)==0: continue
        ids=sorted(set(tr.bu)|set(te.bu)); idx={tuple(map(int,b.split('_'))):i for i,b in enumerate(ids)}; bpos={b:i for i,b in enumerate(ids)}
        X=bu_xy(ids)
        wtr=np.zeros(len(ids)); c=tr.groupby('bu').size(); wtr[[bpos[b] for b in c.index]]=c.values/tr.date.nunique()
        trK=d[d.m==t]; K=max(2,int(round(len(trK)/trK.date.nunique()/TARGET)))
        m=wtr>0; labF,CF=cap_generic(X[m],wtr[m],K,rng=rng,engine=ENGINE)
        labFix=np.argmin(((X[:,None]-CF[None])**2).sum(2),1); labFix[np.where(m)[0]]=labF
        prev={k:None for k in ['FIX','DWS','DCS','OPR']+[f'PEN{l}' for l in LAMS]}; lastlab={k:{} for k in prev}; Cw=CF.copy(); Cc_prev=None
        for day,g in te.groupby('date'):
            P=g[['x','y']].values; bi=g.bu.map(bpos).values
            wd=np.bincount(bi,minlength=len(ids)).astype(float); mu=len(g)/K; act=wd>0
            labs={}
            labs['FIX']=labFix
            for lam in LAMS:
                l=assign_lp(X[act],wd[act],CF,eps=DELTA,home=labFix[act],lam=lam,integer=True); tmp=labFix.copy(); tmp[act]=l; labs[f'PEN{lam}']=tmp
            l,Cw=cap_generic(X[act],wd[act],K,C0=Cw,rng=rng,engine=ENGINE); tmp=labFix.copy(); tmp[act]=l; labs['DWS']=tmp
            l,Cc=cap_generic(X[act],wd[act],K,rng=rng,engine=ENGINE)
            if Cc_prev is not None:
                r_,c_=linear_sum_assignment(np.linalg.norm(Cc[:,None]-Cc_prev[None],axis=2)); perm=np.empty(K,int); perm[r_]=c_; l=perm[l]; Cc=Cc[np.argsort(perm)]
            Cc_prev=Cc; tmp=np.full(len(ids),-1); tmp[act]=l; labs['DCS']=tmp
            for name,lab in labs.items():
                lp=lab[bi]
                # continuity: same district as last day this BU was served
                same=[lastlab[name].get(b) for b in bi]; ok=[s is not None for s in same]
                cont=np.mean([s==x for s,x in zip(same,lp) if s is not None]) if any(ok) else np.nan
                for b in np.unique(bi): lastlab[name][b]=lab[b]
                r=metrics(P,lp,K,mu,cont); r.update(depot=depot,train=(1 if annual else t),test=t+1,date=day,design=name,K=K,V=len(g),moved_share=(wd[act][lab[act]!=labFix[act]].sum()/wd.sum())); out.append(r)
            # operator
            cids=g.cid.values; u,lp=np.unique(cids,return_inverse=True); kd=len(u)
            same=[lastlab['OPR'].get(b) for b in bi]; cont=np.mean([s==x for s,x in zip(same,cids) if s is not None]) if any(s is not None for s in same) else np.nan
            for b,cc in zip(bi,cids): lastlab['OPR'][b]=cc
            r=metrics(P,lp,kd,len(g)/kd,cont); r.update(depot=depot,train=(1 if annual else t),test=t+1,date=day,design='OPR',K=kd,V=len(g),moved_share=0); out.append(r)
        print(depot,t,'done',flush=True)
    return pd.DataFrame(out)
if __name__=='__main__':
    dep=sys.argv[1]; annual=len(sys.argv)>2 and sys.argv[2]=='annual'
    R=run(dep,range(1,12),annual=annual)
    R.to_csv(f"sens_{dep}_BU{BU}_T{TARGET}_S{SEED}_{ENGINE}.csv",index=False)
