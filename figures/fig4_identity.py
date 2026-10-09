import pandas as pd, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial import ConvexHull
from scipy.cluster.vq import kmeans2
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.edgecolor':'#8a8984','axes.linewidth':0.6}); COL=['#2a78d6','#eb6834','#1baf7a','#eda100','#e87ba4']
d=pd.read_pickle('deliv.pkl'); z=d[(d.depot=='Depot01')&(d.m==10)]; days=[pd.Timestamp('2025-10-07'),pd.Timestamp('2025-10-08')]; top=z.cid.value_counts().index[:5]
rng=np.random.default_rng(42); lab={}
for day in days:
    g=z[z.date==day]; ids=sorted(g.cid.unique()); X=g[['x','y']].values; C,l=kmeans2(X,X[rng.choice(len(X),len(ids),replace=False)],minit='matrix',seed=int(rng.integers(1e6))); lab[day]=pd.Series(np.array(ids)[l],index=g.index)
fig,ax=plt.subplots(1,2,figsize=(7.2,4.1),sharex=True,sharey=True); x0,y0=z.x.median(),z.y.median()
for a,title,src in [(ax[0],'(a) Operator: real courier assignment',None),(ax[1],'(b) Daily k-means, arbitrary courier labels',lab)]:
    a.scatter((z.x-x0)/1000,(z.y-y0)/1000,s=0.3,c='#d9d8d3',rasterized=True,lw=0)
    for i,c in enumerate(top):
        for j,day in enumerate(days):
            g=z[z.date==day]; ids=g.cid if src is None else src[day]; P=g.loc[ids[ids==c].index,['x','y']].values
            if len(P)<3: continue
            h=ConvexHull(P); v=np.r_[h.vertices,h.vertices[0]]; a.plot((P[v,0]-x0)/1000,(P[v,1]-y0)/1000,color=COL[i],lw=1.6 if j==0 else 1.2,ls='-' if j==0 else (0,(3,2)))
    a.set_title(title,fontsize=9,loc='left'); a.set_aspect('equal'); a.tick_params(labelsize=8,colors='#52514e')
    for s in ['top','right']: a.spines[s].set_visible(False)
qx=((z.x-x0)/1000).quantile([.002,.998]).values; qy=((z.y-y0)/1000).quantile([.002,.998]).values; ax[0].set_xlim(qx[0]-.3,qx[1]+.3); ax[0].set_ylim(qy[0]-.3,qy[1]+.3); ax[0].set_xlabel('km (east)'); ax[1].set_xlabel('km (east)'); ax[0].set_ylabel('km (north)')
from matplotlib.lines import Line2D
fig.legend([Line2D([],[],color='#52514e',lw=1.6),Line2D([],[],color='#52514e',lw=1.2,ls=(0,(3,2)))],['7 Oct 2025','8 Oct 2025'],loc='lower center',bbox_to_anchor=(0.5,0.045),ncol=2,frameon=False,fontsize=8)
fig.text(0.5,0.015,'Colours: the same five courier IDs in both panels. Grey: all October deliveries, Depot01.',ha='center',fontsize=7.5,color='#52514e')
fig.tight_layout(rect=(0,0.12,1,0.98)); out=Path(__file__).resolve().parent/'output'; out.mkdir(exist_ok=True); fig.savefig(out/'Figure_4_identity.png',dpi=300)
