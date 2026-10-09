"""Driving time along the distance-minimising path (lexicographic Dijkstra: minimise length, then time).
Weight = length_cm * 1e7 + time_ms, exact integers in float64; L and T are recovered from the path sum."""
import pickle, numpy as np, time
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from depots import DXY
net=pickle.load(open('roadnet.pkl','rb')); keep=net['keep']; u,v=net['u'],net['v']; m=keep[u]&keep[v]; n=len(net['XY'])
L=net['length'][m]; tt=net['tt'][m]; sp=L/tt*3.6
TT={'T':tt,'T_uniform25':L/(25/3.6),'T_congested':L/(np.minimum(0.6*sp,40)/3.6)}
M=pickle.load(open('road_mats_fastest.pkl','rb'))
from scipy.spatial import cKDTree
import pandas as pd
d=pd.read_pickle('deliv.pkl'); d=d[d.dow<5]
kidx=np.where(keep)[0]; tree=cKDTree(net['XY'][kidx])
for dep,g in d.groupby('depot'):
    bc=g.groupby('bu')[['x','y']].mean(); assert list(bc.index)==M[dep]['bu']
    pts=np.vstack([DXY[dep][None,:],bc.values]); _,si=tree.query(pts); nodes=kidx[si]
    for key,t in TT.items():
        t0=time.time(); w=np.round(L*100)*1e7+np.round(t*1000); w[w==0]=1
        G=csr_matrix((w,(u[m],v[m])),shape=(n,n))
        S=dijkstra(G,directed=True,indices=nodes)[:,nodes]
        Lc=np.floor(S/1e7); Tm=S-Lc*1e7
        Dk=Lc/1e5; dev=np.abs(Dk-M[dep]['D']).max()
        M[dep]['Tfast'+key[1:]]=M[dep][key]; M[dep][key]=Tm/60000
        print(dep,key,'max |D_lex - D| %.4f km'%dev,'mean T/Tfast %.3f'%np.nanmean(M[dep][key][M[dep]['Tfast'+key[1:]]>0]/M[dep]['Tfast'+key[1:]][M[dep]['Tfast'+key[1:]]>0]),'%.0fs'%(time.time()-t0),flush=True)
pickle.dump(M,open('road_mats.pkl','wb'))
