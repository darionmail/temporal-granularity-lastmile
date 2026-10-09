import pandas as pd, numpy as np, itertools, os
FOUR=['Depot01','Depot02','Depot03','Depot04']; DES=['FIX','PEN250','PEN1000','PEN5000','DWS','DCS']
R=pd.read_csv('R4road.csv')
def dayroad(dep,var='',intra_mult=1.0):
    f=f'road_{dep}'+(f'_{var}' if var else '')+'.csv'
    X=pd.read_csv(f); X['km']=X.route_km+intra_mult*X.intra_km
    g=X.groupby(['date','design']).agg(km=('km','sum'),mn=('route_min','sum'),routes=('k','size')).reset_index(); g['depot']=dep; return g
# (a) cost table
A=pd.concat([dayroad(d) for d in FOUR]); A=A[A.design.isin(DES+['OPR'])]
T=A.groupby(['depot','design']).agg(km=('km','mean'),mn=('mn','mean'),routes=('routes','mean')).reset_index()
T['km_route']=T.km/T.routes; print('== (a) cost by depot and design'); print(T.pivot(index='design',columns='depot',values=['km','mn']).round(1).to_string())
T.to_csv('road_cost_table.csv',index=False)
# (b) within-cell multiplier
print('== (b) intra multiplier: mean daily % difference vs comparator')
for mult in [0,1,2]:
    B=pd.concat([dayroad(d,intra_mult=mult) for d in FOUR]); W=B.pivot_table(index=['depot','date'],columns='design',values='km')
    print(mult, {f'{a}-{b}':round(100*(W[a]/W[b]-1).mean(),2) for a,b in [('DWS','FIX'),('PEN250','FIX'),('PEN1000','FIX'),('PEN5000','FIX'),('PEN1000','DWS')]}, 'intra share of FIX km %.1f%%'%(100*(1-pd.concat([dayroad(d,intra_mult=0) for d in FOUR]).query("design=='FIX'").km.sum()/B.query("design=='FIX'").km.sum()) if mult==1 else 0))
# (c) ranking agreement road vs straight-line (total_km)
print('== (c) road vs straight-line agreement')
Wr=R.pivot_table(index=['depot','date'],columns='design',values='road_km'); Ws=R.pivot_table(index=['depot','date'],columns='design',values='total_km')
rows=[]
for a,b in itertools.combinations(DES,2):
    dr=Wr[a]-Wr[b]; ds=Ws[a]-Ws[b]
    rows.append(dict(pair=f'{a} vs {b}',sign_agree=100*(np.sign(dr)==np.sign(ds)).mean(),corr=np.corrcoef(dr,ds)[0,1],road_mean=dr.mean(),sl_mean=ds.mean()))
C=pd.DataFrame(rows); print(C.round(3).to_string())
best_r=Wr[DES].idxmin(axis=1); best_s=Ws[DES].idxmin(axis=1); print('same lowest-travel design: %.1f%% of depot-days'%(100*(best_r==best_s).mean()))
rank_r=Wr[DES].rank(axis=1); rank_s=Ws[DES].rank(axis=1)
from scipy.stats import kendalltau
kt=[kendalltau(rank_r.iloc[i],rank_s.iloc[i])[0] for i in range(len(rank_r))]; print('Kendall tau per depot-day median %.2f'%np.nanmedian(kt))
# pooled mean ranking
print('mean rank road',rank_r.mean().round(2).to_dict()); print('mean rank straight',rank_s.mean().round(2).to_dict())
C.to_csv('road_vs_straight.csv',index=False)
# (d) speed variants and seed noise, if available
if all(os.path.exists(f'road_{d}_uniform25.csv') and os.path.exists(f'road_{d}_congested.csv') and os.path.exists(f'road_{d}_seed1.csv') for d in FOUR):
    print('== (d) speeds')
    for var in ['','uniform25','congested']:
        B=pd.concat([dayroad(d,var) for d in FOUR]); W=B.pivot_table(index=['depot','date'],columns='design',values='mn')
        print(var or 'base', 'FIX min/day %.0f'%W['FIX'].mean(), {f'{a}-{b}':round(100*(W[a]/W[b]-1).mean(),2) for a,b in [('DWS','FIX'),('PEN1000','FIX'),('PEN1000','DWS')]},
              'PEN1000-DWS min/day %.2f'%(W['PEN1000']-W['DWS']).mean())
    print('== (e) solver noise (seed 0 vs seed 1)')
    B0=pd.concat([dayroad(d) for d in FOUR]); B1=pd.concat([dayroad(d,'seed1') for d in FOUR])
    M=B0.merge(B1,on=['depot','date','design'],suffixes=('0','1'))
    M['dk']=M.km1-M.km0; print('daily total km diff between seeds: mean %.3f, sd %.3f, |rel| median %.3f%%, p95 %.3f%%'%(M.dk.mean(),M.dk.std(),100*(M.dk.abs()/M.km0).median(),100*(M.dk.abs()/M.km0).quantile(.95)))
    W0=M.pivot_table(index=['depot','date'],columns='design',values='km0'); W1=M.pivot_table(index=['depot','date'],columns='design',values='km1')
    for a,b in [('DWS','FIX'),('PEN1000','FIX'),('PEN1000','DWS'),('PEN250','DWS')]:
        d0=W0[a]-W0[b]; d1=W1[a]-W1[b]
        print(f'{a}-{b}: sign agreement {100*(np.sign(d0)==np.sign(d1)).mean():.1f}%, mean diff seed0 {d0.mean():.2f} seed1 {d1.mean():.2f} km')
