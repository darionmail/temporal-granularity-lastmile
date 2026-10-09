import pandas as pd, numpy as np, sys
from stats import paired
FOUR=['Depot01','Depot02','Depot03','Depot04']
def daylevel(tag):
    R=pd.read_csv(f'road_{tag}.csv'); R['km']=R.route_km+R.intra_km
    return R.groupby(['date','design']).agg(road_km=('km','sum'),road_min=('route_min','sum'),routes=('k','size')).reset_index()
parts=[]
for dep in FOUR:
    Q=pd.read_csv(f'rq3r_{dep}.csv'); Q['date']=Q.date.astype(str).str[:10]
    F=pd.read_csv(f'rq3f_{dep}.csv'); F['date']=F.date.astype(str).str[:10]
    Q=Q.merge(F[['date','design','inband','stem_km','total_km']].rename(columns={'inband':'inband_f'}),on=['date','design'])
    Q=Q.merge(daylevel(dep),on=['date','design'],how='left'); parts.append(Q)
R=pd.concat(parts); R['inband']=R.inband_f
idx=pd.MultiIndex.from_frame(R[['depot','date']])
for col,src in [('ridx','road_km'),('midx','road_min'),('sidx','total_km')]:
    fx=R[R.design=='FIX'].set_index(['depot','date'])[src]; R[col]=100*R[src].values/fx.reindex(idx).values
R.to_csv('R4road.csv',index=False)
print('missing road',R.road_km.isna().sum())
print(R.groupby(['depot','design'])[['ridx','midx','sidx']].mean().round(1).unstack(0).to_string())
print(R.groupby('design')[['road_km','road_min','ridx','midx','sidx']].mean().round(2))
res=[]
for m in ['road_km','road_min']:
    W=R.pivot_table(index=['depot','date'],columns='design',values=m)
    for a,b in [('DWS','FIX'),('PEN1000','FIX'),('PEN1000','DWS'),('PEN250','DWS'),('PEN5000','DWS'),('DCS','DWS')]:
        r=paired(W[a],W[b]); r.update(metric=m,comp=f'{a} - {b}',mean=(W[a]-W[b]).mean()); res.append(r)
S=pd.DataFrame(res); print(S[['metric','comp','n','HL','CI_lo','CI_hi','p','r_rb','mean']].round(4).to_string())
dl=[]
for dep in FOUR:
    W=R[R.depot==dep].pivot_table(index='date',columns='design',values='road_km')
    for a,b in [('DWS','FIX'),('PEN1000','FIX'),('PEN1000','DWS')]:
        r=paired(W[a],W[b]); r.update(depot=dep,comp=f'{a} - {b}'); dl.append(r)
D=pd.DataFrame(dl); D['p_bonf']=np.minimum(D.p*48,1); print(D[['depot','comp','HL','CI_lo','CI_hi','p_bonf','r_rb']].round(3).to_string())
F=R[R.design=='FIX']; Wc=R.pivot_table(index=['depot','date'],columns='design',values='cont'); rc=paired(Wc['PEN1000'],Wc['DWS'])
Wk=R.pivot_table(index=['depot','date'],columns='design',values='road_km'); rk=paired(Wk['PEN1000'],Wk['DWS'])
Wm=R.pivot_table(index=['depot','date'],columns='design',values='road_min'); rm=paired(Wm['PEN1000'],Wm['DWS'])
par=rc['HL']*F.V.mean()
print('breakeven: +%.2f km +%.2f min for %.1f parcels -> %.0f m, %.1f s per parcel'%(rk['HL'],rm['HL'],par,1000*rk['HL']/par,60*rm['HL']/par))
print('FIX mean road km %.1f, min %.1f, per route km %.1f'%(F.road_km.mean(),F.road_min.mean(),(F.road_km/F.routes).mean()))
