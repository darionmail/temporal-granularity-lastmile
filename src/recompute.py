import pandas as pd, numpy as np, glob, sys
DELTA=1/7
for f in sorted(glob.glob('rq3i_*.csv')):
    tag=f[5:-4]; R=pd.read_csv(f); C=pd.read_csv(f'cent_{tag}.csv.gz')
    g=C.groupby(['date','design']).n.apply(list).to_dict()
    ib=[];i5=[]
    for _,r in R.iterrows():
        L=np.array(g.get((r.date,r.design),[]),float); L=np.r_[L,np.zeros(max(0,int(r.K)-len(L)))]; mu=r.V/r.K
        ib.append(((L>=mu*(1-DELTA)-1e-6)&(L<=mu*(1+DELTA)+1e-6)).mean())
        i5.append(((L>=mu*0.95-1e-6)&(L<=mu*1.05+1e-6)).mean() if r.design!='OPR' else np.nan)
    R['inband_old']=R.inband; R['inband']=ib; R['in5']=i5
    R.to_csv(f'rq3f_{tag}.csv',index=False)
    print(tag, R.groupby('design')[['inband_old','inband']].mean().round(4).T.to_string())
