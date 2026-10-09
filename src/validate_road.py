import pickle, numpy as np, pandas as pd
from routing import route_multi, intra
Ms=pickle.load(open('road_mats.pkl','rb'))
d=pd.read_pickle('deliv.pkl'); d=d[d.dow<5]
rows=[]
for dep,g0 in d.groupby('depot'):
    M=Ms[dep]; D,T=M['D'],M['T']; circ=M['circ']; bix={b:i+1 for i,b in enumerate(M['bu'])}
    cd=g0.groupby(['date','cid']).size(); cd=cd[cd>=10].sample(150,random_state=11)
    for (day,c) in cd.index:
        g=g0[(g0.date==day)&(g0.cid==c)].sort_values('t')
        seq=[bix[b] for b in g.bu]; seq=[s for i,s in enumerate(seq) if i==0 or s!=seq[i-1]]
        path=[0]+seq+[0]; act=sum(D[a,b] for a,b in zip(path[:-1],path[1:]))
        acts=sum(T[a,b] for a,b in zip(path[:-1],path[1:]))
        nodes=[0]+sorted(set(seq)); km,mn,_=route_multi(nodes,D,T)
        ik=sum(intra(gg[['x','y']].values)*circ for _,gg in g.groupby('bu'))
        span=(g.t.max()-g.t.min()).total_seconds()/60
        rows.append(dict(depot=dep,n=len(g),stops=len(nodes)-1,revisits=len(seq)-(len(nodes)-1),actual_km=act+ik,model_km=km+ik,actual_min=acts,model_min=mn,span_min=span))
V=pd.DataFrame(rows); V.to_csv('road_validation.csv',index=False)
V['ratio']=V.actual_km/V.model_km
print(V.groupby('depot').agg(n=('ratio','size'),model_km=('model_km','median'),actual_km=('actual_km','median'),ratio=('ratio','median'),revisits=('revisits','median'),model_min=('model_min','median'),span=('span_min','median')).round(2))
print('pooled ratio median %.3f IQR %.3f-%.3f; model km median %.1f'%(V.ratio.median(),V.ratio.quantile(.25),V.ratio.quantile(.75),V.model_km.median()))
