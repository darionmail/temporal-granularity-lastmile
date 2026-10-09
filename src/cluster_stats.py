import pandas as pd, numpy as np, glob, sys
from scipy.stats import wilcoxon
pref=sys.argv[1] if len(sys.argv)>1 else 'rq3_'
R=pd.concat([pd.read_csv(f) for f in glob.glob(pref+'*.csv') if 'annual' not in f])
R['date']=pd.to_datetime(R.date); R['week']=R.date.dt.isocalendar().week.astype(int); R['month']=R.date.dt.month
comps=[('DWS','FIX'),('PEN1000','FIX'),('PEN1000','DWS')]
rng=np.random.default_rng(2026); B=5000
rows=[]
for m in ['inband','cv','total_km','cont']:
    W=R.pivot_table(index=['depot','date','week','month'],columns='design',values=m).reset_index()
    for a,b in comps:
        W['d']=W[a]-W[b]; D=W.dropna(subset=['d'])
        for scope,S in [('All',D)]+[(dep,D[D.depot==dep]) for dep in sorted(D.depot.unique())]:
            wk=S.groupby('week').d.agg(['sum','count']); ws=wk['sum'].values; wc=wk['count'].values; nW=len(wk)
            idx=rng.integers(0,nW,(B,nW)); bs=ws[idx].sum(1)/wc[idx].sum(1)
            mm=S.groupby('month').d.mean(); wm=S.groupby('week').d.mean()
            p_w=wilcoxon(wm).pvalue if (wm!=0).any() else 1.0
            p_m=wilcoxon(mm).pvalue if (mm!=0).any() else 1.0
            rows.append(dict(metric=m,comp=f'{a} - {b}',scope=scope,n_days=len(S),n_weeks=nW,mean_diff=S.d.mean(),
                boot_lo=np.percentile(bs,2.5),boot_hi=np.percentile(bs,97.5),p_week_wilcoxon=p_w,
                months_same_sign=f"{(np.sign(mm)==np.sign(S.d.mean())).sum()}/{len(mm)}",p_month_wilcoxon=p_m))
T=pd.DataFrame(rows); T.to_csv('cluster_robust.csv',index=False)
pd.set_option('display.width',250)
print(T[T.scope=='All'].round(4).to_string())
print(T[T.scope!='All'][['metric','comp','scope','mean_diff','boot_lo','boot_hi','p_week_wilcoxon','months_same_sign']].round(4).to_string())
