"""Break-even (Section 6.1): PEN(lambda=1 km) vs DWS, mean differences on the depot-days where continuity is defined."""
import pandas as pd
FOUR=['Depot01','Depot02','Depot03','Depot04']
R=pd.read_csv('R4road.csv'); P=lambda c: R.pivot_table(index=['depot','date'],columns='design',values=c,dropna=False)
km,mn=P('road_km'),P('road_min'); cont=P('cont').reindex(km.index)
F=R[R.design=='FIX'].set_index(['depot','date']); V=F.V.reindex(km.index); K=F.K.reindex(km.index)
dc=cont.PEN1000-cont.DWS; ok=dc.notna(); par=(dc*V)[ok].mean()
def var(v):
    X=pd.concat([pd.read_csv(f'road_{d}_{v}.csv').assign(depot=d) for d in FOUR]).groupby(['depot','date','design']).route_min.sum().unstack()
    return (X.PEN1000-X.DWS).reindex(km.index)[ok].mean()
dk=(km.PEN1000-km.DWS)[ok].mean(); dm=(mn.PEN1000-mn.DWS)[ok].mean(); u=var('uniform25'); c=var('congested')
print(f'{ok.sum()} depot-days; V {V[ok].mean():.0f}, K {K[ok].mean():.1f}; +{dk:.2f} km, +{dm:.2f} min (uniform {u:.2f}, congested {c:.2f}) for {par:.1f} parcels')
print(f'per retained parcel: {1000*dk/par:.0f} m, {60*dm/par:.1f} s free flow, {60*u/par:.1f} s uniform, {60*c/par:.1f} s congested')
