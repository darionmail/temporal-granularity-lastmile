import pandas as pd, numpy as np, sys, glob
from stats import paired
pd.set_option('display.width',220)
cols=['cv','inband','inband_abs','tour_km','comp_m','cont','moved_share']
files=sorted(f for f in glob.glob('rq3_*.csv') if 'annual' not in f)
allS=[]
for f in files:
    R=pd.read_csv(f); dep=R.depot.iloc[0]
    print('=====',dep,'test days',R.date.nunique(),'K median',R[R.design=='FIX'].K.median())
    print(R.groupby('design')[cols].mean().round(3).to_string())
    for m in ['inband','cv','tour_km','cont']:
        W=R.pivot_table(index='date',columns='design',values=m)
        for a,b in [('DWS','FIX'),('PEN1000','FIX'),('PEN1000','DWS')]:
            r=paired(W[a],W[b]); r.update(depot=dep,metric=m,comp=f'{a} - {b}'); allS.append(r)
S=pd.DataFrame(allS); S['p_bonf']=np.minimum(S.p*len(S),1)
p=S.p.values; o=np.argsort(p); m=len(p); q=np.empty(m); q[o]=np.minimum.accumulate((p[o]*m/np.arange(1,m+1))[::-1])[::-1]; S['q_BH']=np.minimum(q,1)
S.to_csv('paired_tests.csv',index=False)
print(S[['depot','metric','comp','n','HL','CI_lo','CI_hi','W','p_bonf','q_BH','r_rb']].round(4).to_string())
