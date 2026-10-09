import pandas as pd, numpy as np
d=pd.read_pickle('deliv.pkl')
wd=d[d.dow<5]
rows=[]
for dep,g in wd.groupby('depot'):
    cd=g.groupby(['cid','date'])[['x','y']].median().reset_index().sort_values(['cid','date'])
    # consecutive working days of same courier
    cd['disp']=np.hypot(cd.groupby('cid').x.diff(),cd.groupby('cid').y.diff())
    # purity: 250m BU, monthly
    pur=[];
    for m,gm in g.groupby('m'):
        c=gm.groupby(['bu','cid']).size(); tot=c.groupby(level=0).sum(); top=c.groupby(level=0).max()
        pur.append(top.sum()/tot.sum())
    # BU familiarity: share of BU-day deliveries by the BU's modal courier of the PREVIOUS month
    fam=[]
    for m in range(2,13):
        a=g[g.m==m-1]; b=g[g.m==m]
        mode=a.groupby('bu').cid.agg(lambda s:s.value_counts().index[0])
        bb=b[b.bu.isin(mode.index)]; fam.append((bb.cid.values==mode.loc[bb.bu].values).mean())
    # demand variability: daily parcels per BU (BUs with >=1/day avg)
    bd=g.groupby(['bu','date']).size().unstack(fill_value=0)
    mu=bd.mean(1); cv=(bd.std(1)/mu)[mu>=2]
    # workload
    w=g.groupby(['date','cid']).size()
    rows.append(dict(depot=dep,parcels=len(g),couriers=g.cid.nunique(),
      drift_med=cd.disp.median(),drift_p75=cd.disp.quantile(.75),drift_gt2km=(cd.disp>2000).mean()*100,
      purity=np.mean(pur)*100,prev_month_mode_share=np.mean(fam)*100,
      bu_cv_med=cv.median(), w_med=w.median(), w_cv=(w.groupby(level=0).std()/w.groupby(level=0).mean()).median()))
R=pd.DataFrame(rows).round(2); print(R.to_string()); R.to_csv('rq1_stability.csv',index=False)
# RQ2 feasibility for Depot01: [60,80] with actual k_d, and median k
# Table 5a diagnostics for all four depots on the 11 out-of-sample test months.
# Monthly roster for test month m is built from the mean weekday volume in month m-1.
FOUR=['Depot01','Depot02','Depot03','Depot04']
q=[]
for dep in FOUR:
    gd=d[(d.depot==dep)&(d.dow<5)].groupby('date').agg(V=('cid','size'),k_obs=('cid','nunique'),m=('m','first')).reset_index()
    mean_by_month=gd.groupby('m').V.mean()
    for _,r in gd[gd.m>=2].iterrows():
        K_month=max(2,int(round(mean_by_month.loc[r.m-1]/70)))
        feas_month=(r.V>=60*K_month) and (r.V<=80*K_month)
        under=r.V<60*K_month; over=r.V>80*K_month
        feas_obs=(r.V>=60*r.k_obs) and (r.V<=80*r.k_obs)
        k0=max(1,int(np.floor(r.V/70))); k1=max(1,int(np.ceil(r.V/70)))
        feas_daily=any((r.V>=60*k) and (r.V<=80*k) for k in {k0,k1})
        q.append(dict(depot=dep,date=r.date,month=int(r.m),V=int(r.V),K_month=K_month,
                      k_observed=int(r.k_obs),feasible_monthly=feas_month,under_load=under,
                      over_load=over,feasible_observed=feas_obs,feasible_daily_roster=feas_daily))
Q=pd.DataFrame(q)
Q.to_csv('rq2_feasibility_all_depots.csv',index=False)
S=Q.groupby('depot').agg(test_days=('date','size'),
    feasible_monthly=('feasible_monthly','mean'),under_load=('under_load','mean'),
    over_load=('over_load','mean'),feasible_observed=('feasible_observed','mean'),
    feasible_daily_roster=('feasible_daily_roster','mean'))
allrow=pd.DataFrame([dict(test_days=len(Q),feasible_monthly=Q.feasible_monthly.mean(),
    under_load=Q.under_load.mean(),over_load=Q.over_load.mean(),
    feasible_observed=Q.feasible_observed.mean(),feasible_daily_roster=Q.feasible_daily_roster.mean())],index=['All'])
S=pd.concat([S,allrow]); pct=['feasible_monthly','under_load','over_load','feasible_observed','feasible_daily_roster']
S[pct]=100*S[pct]
S.round(1).to_csv('rq2_feasibility_summary.csv')
print('\nRQ2 feasibility, % of test weekdays')
print(S.round(1).to_string())
