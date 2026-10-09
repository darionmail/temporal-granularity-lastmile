import pickle, numpy as np, pandas as pd, sys, time
from routing import route_multi, intra
tag=sys.argv[1]; dep=tag.replace('_annual',''); var=sys.argv[2] if len(sys.argv)>2 else ''; seed=int(sys.argv[3]) if len(sys.argv)>3 else 0
M=pickle.load(open('road_mats.pkl','rb'))[dep]; D=M['D']; T=M['T_'+var] if var.startswith(('uniform','congested')) else M['T']; circ=M['circ']
bix={b:i+1 for i,b in enumerate(M['bu'])}
A=pd.read_csv(f'assign_{tag}.csv.gz')
d=pd.read_pickle('deliv.pkl'); d=d[(d.depot==dep)&(d.dow<5)]
d['dstr']=d.date.dt.strftime('%Y-%m-%d')
# intra-BU distance (road-adjusted BHH) per BU-day, and per BU-courier-day for OPR
t0=time.time()
ib={k:intra(g[['x','y']].values)*circ for k,g in d.groupby(['dstr','bu'])}
ibc={k:intra(g[['x','y']].values)*circ for k,g in d.groupby(['dstr','bu','cid'])}
print(tag,'intra done',time.time()-t0,flush=True)
A['date']=A.date.astype(str).str[:10]
rows=[]; t0=time.time()
for (day,des,k),g in A.groupby(['date','design','k']):
    nodes=[0]+[bix[b] for b in g.bu]
    km,mn,_=route_multi(nodes,D,T,seed=seed)
    if des=='OPR': ik=sum(ibc.get((day,b,k),0.0) for b in g.bu)
    else: ik=sum(ib.get((day,b),0.0) for b in g.bu)
    rows.append((day,des,k,g.n.sum(),len(nodes)-1,km,mn,ik))
R=pd.DataFrame(rows,columns=['date','design','k','parcels','stops','route_km','route_min','intra_km'])
R.to_csv(f'road_{tag}'+(f'_{var}' if var else '')+'.csv',index=False); print(tag,'routes',len(R),'time',time.time()-t0,flush=True)
