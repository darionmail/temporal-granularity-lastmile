import pandas as pd, numpy as np, glob
from stats import paired
from scipy.stats import wilcoxon
FOUR=['Depot01','Depot02','Depot03','Depot04']
R=pd.concat([pd.read_csv(f) for f in glob.glob('rq3f_*.csv') if 'annual' not in f])
R=R[R.depot.isin(FOUR)]
idx=pd.MultiIndex.from_frame(R[['depot','date']])
for col,src in [('tidx_total','total_km'),('tidx_local','tour_km')]:
    fx=R[R.design=='FIX'].set_index(['depot','date'])[src]; R[col]=100*R[src].values/fx.reindex(idx).values
print('== days',R[R.design=='FIX'].shape[0])
P=R.groupby('design').agg(inband=('inband','mean'),cv=('cv','mean'),cont=('cont','mean'),tt=('tidx_total','mean'),tl=('tidx_local','mean'),moved=('moved_share','mean'))
P[['inband','cont','moved']]*=100; print(P.round(2))
print('FIX inband range', R[R.design=='FIX'].groupby('depot').inband.mean().mul(100).round(1).to_dict())
print('FIX cv range', R[R.design=='FIX'].groupby('depot').cv.mean().round(2).to_dict())
print('PEN cv',R[R.design=='PEN1000'].groupby('depot').cv.mean().round(3).to_dict(),'DWS cv',R[R.design=='DWS'].groupby('depot').cv.mean().round(3).to_dict())
print('OPR cv',R[R.design=='OPR'].groupby('depot').cv.mean().round(2).to_dict())
print('PEN local tour idx by lam', R[R.design.str.startswith('PEN')].groupby('design').tidx_local.mean().round(1).to_dict())
F=R[R.design=='FIX']; print('mean V %.1f mean K %.2f'%(F.V.mean(),F.K.mean()))
# paired pooled
res=[]
for m in ['inband','cv','total_km','cont']:
    W=R.pivot_table(index=['depot','date'],columns='design',values=m)
    for a,b in [('DWS','FIX'),('PEN1000','FIX'),('PEN1000','DWS'),('DCS','DWS')]:
        r=paired(W[a],W[b]); r.update(metric=m,comp=f'{a} - {b}'); res.append(r)
S=pd.DataFrame(res); S['p_bonf']=np.minimum(S.p*20,1)
print(S[['metric','comp','n','HL','CI_lo','CI_hi','p','p_bonf','r_rb']].round(4).to_string())
# depot-level family
dl=[]
for dep in FOUR:
    for m in ['inband','cv','total_km','cont']:
        W=R[R.depot==dep].pivot_table(index='date',columns='design',values=m)
        for a,b in [('DWS','FIX'),('PEN1000','FIX'),('PEN1000','DWS')]:
            r=paired(W[a],W[b]); r.update(depot=dep,metric=m,comp=f'{a} - {b}'); dl.append(r)
D=pd.DataFrame(dl); D['p_bonf']=np.minimum(D.p*len(D),1)
print('depot-level sig',(D.p_bonf<0.05).sum(),'of',len(D)); print(D[D.p_bonf>=0.05][['depot','metric','comp']].to_string())
s=D[D.p_bonf<0.05]; print('min |r| sig',s.r_rb.abs().min())
D.to_csv('paired_depot_4.csv',index=False); S.to_csv('paired_pooled_4.csv',index=False)
# break-even
W=R.pivot_table(index=['depot','date'],columns='design',values='total_km'); 
r=paired(W['PEN1000'],W['DWS']); Wc=R.pivot_table(index=['depot','date'],columns='design',values='cont')
rc=paired(Wc['PEN1000'],Wc['DWS'])
print('breakeven: km HL %.2f cont HL %.4f parcels %.1f m/parcel %.1f sec %.1f'%(r['HL'],rc['HL'],rc['HL']*F.V.mean(),1000*r['HL']/(rc['HL']*F.V.mean()),1000*r['HL']/(rc['HL']*F.V.mean())/(20/3.6)))
R.to_csv('R4.csv',index=False)
