import pandas as pd, numpy as np, sys
B=sys.argv[1] if len(sys.argv)>1 else '.'
FOUR=['Depot01','Depot02','Depot03','Depot04']
def W(var='',mult=1,val='km'):
    X=pd.concat([pd.read_csv(f'{B}/road_{d}'+(f'_{var}' if var else '')+'.csv').assign(depot=d) for d in FOUR])
    X['km']=X.route_km+mult*X.intra_km; X['mn']=X.route_min
    return X.groupby(['depot','date','design'])[val].sum().unstack()
R=pd.read_csv(f'{B}/R4road.csv'); S=R.pivot_table(index=['depot','date'],columns='design',values='total_km')
rows=[]
def add(name,w,unit):
    r={'setting':name}
    for a,b,c in [('DWS','FIX','DWS-FIX %'),('PEN1000','FIX','PEN-FIX %'),('PEN1000','DWS','PEN-DWS %')]: r[c]=100*(w[a]/w[b]-1).mean()
    r['PEN-DWS abs']=(w.PEN1000-w.DWS).mean(); r['unit']=unit; rows.append(r)
add('Baseline: road km, within-cell term ×1',W(),'km'); add('Within-cell term ×0',W(mult=0),'km'); add('Within-cell term ×2',W(mult=2),'km')
add('Different heuristic seed',W('seed1'),'km')
add('Driving time, mapped speeds (free flow)',W(val='mn'),'min'); add('Driving time, uniform 25 km/h',W('uniform25',val='mn'),'min')
add('Driving time, congested (60% of mapped, ≤ 40 km/h)',W('congested',val='mn'),'min'); add('Straight-line estimate (Section 4.4)',S,'km')
out=pd.DataFrame(rows); print(out.round(4).to_string())
if B=='.': out.to_csv('road_sensitivity.csv',index=False)
