import pickle, numpy as np, pandas as pd, time
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from depots import DXY
net=pickle.load(open('roadnet.pkl','rb'))
keep=net['keep']; XY=net['XY']; u,v=net['u'],net['v']
m=keep[u]&keep[v]; n=len(XY)
Gd=csr_matrix((net['length'][m]+1e-6,(u[m],v[m])),shape=(n,n))
Gt=csr_matrix((net['tt'][m]+1e-6,(u[m],v[m])),shape=(n,n))
kidx=np.where(keep)[0]; tree=cKDTree(XY[kidx])
d=pd.read_pickle('deliv.pkl'); d=d[d.dow<5]
out={}
for dep,g in d.groupby('depot'):
    t=time.time()
    bc=g.groupby('bu')[['x','y']].mean()
    pts=np.vstack([DXY[dep][None,:],bc.values])
    snapd,si=tree.query(pts); nodes=kidx[si]
    D=dijkstra(Gd,directed=True,indices=nodes)[:,nodes]/1000
    T=dijkstra(Gt,directed=True,indices=nodes)[:,nodes]/60
    E=np.hypot(*(pts[:,None,:]-pts[None,:,:]).transpose(2,0,1))/1000
    off=~np.eye(len(pts),dtype=bool); ok=E>0.3
    circ=np.median((D[off&ok])/(E[off&ok]))
    print(dep,len(bc),'BUs; snap dist median %.0f m p95 %.0f m; inf %d; circuity median %.3f; %.1fs'%(np.median(snapd),np.percentile(snapd,95),np.isinf(D).sum(),circ,time.time()-t),flush=True)
    out[dep]=dict(bu=list(bc.index),D=D,T=T,E=E,circ=circ,snap=snapd)
pickle.dump(out,open('road_mats.pkl','wb'))
