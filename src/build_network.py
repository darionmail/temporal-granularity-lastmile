import osmium, numpy as np, pandas as pd, pickle, time, re, os, sys
from pathlib import Path
from pyproj import Transformer
pbf_arg = sys.argv[1] if len(sys.argv) > 1 else os.environ.get('OSM_PBF')
if not pbf_arg:
    raise FileNotFoundError('Provide the Croatia .osm.pbf path as the first argument or set OSM_PBF.')
PBF = Path(pbf_arg)
if not PBF.is_file():
    raise FileNotFoundError(f'OpenStreetMap extract not found: {PBF}')
d=pd.read_pickle('deliv.pkl'); d=d
tr_inv=Transformer.from_crs(3765,4326,always_xy=True)
lon,lat=tr_inv.transform(d.x.values,d.y.values)
W,E_,S,N=lon.min()-0.03,lon.max()+0.03,lat.min()-0.02,lat.max()+0.02
print('bbox',W,E_,S,N)
DRIVE={'motorway':110,'trunk':80,'primary':50,'secondary':50,'tertiary':50,'unclassified':40,'residential':30,'living_street':10,'service':20,'road':30,
       'motorway_link':60,'trunk_link':50,'primary_link':40,'secondary_link':40,'tertiary_link':30}
class H(osmium.SimpleHandler):
    def __init__(s): super().__init__(); s.edges=[]; s.coords={}
    def way(s,w):
        hw=w.tags.get('highway')
        if hw not in DRIVE: return
        if w.tags.get('access') in ('no',) or w.tags.get('motor_vehicle') in ('no',) or w.tags.get('area')=='yes': return
        try: pts=[(n.ref,n.lon,n.lat) for n in w.nodes]
        except osmium.InvalidLocationError: return
        if not any(W<=x<=E_ and S<=y<=N for _,x,y in pts): return
        ms=w.tags.get('maxspeed','')
        m=re.match(r'^(\d+)',ms); sp=float(m.group(1)) if m else DRIVE[hw]
        ow=w.tags.get('oneway','no'); jr=w.tags.get('junction','')
        fwd=True; bwd=not(ow in('yes','1','true') or jr in('roundabout','circular') or hw in('motorway',))
        if ow=='-1': fwd,bwd=False,True
        for r,x,y in pts: s.coords[r]=(x,y)
        for (a,_,_),(b,_,_) in zip(pts[:-1],pts[1:]):
            if fwd: s.edges.append((a,b,sp))
            if bwd: s.edges.append((b,a,sp))
t=time.time(); h=H(); h.apply_file(str(PBF),locations=True); print('parsed',time.time()-t,'edges',len(h.edges),'nodes',len(h.coords))
ids=np.array(list(h.coords.keys())); ix={r:i for i,r in enumerate(ids)}
ll=np.array([h.coords[r] for r in ids]); tr=Transformer.from_crs(4326,3765,always_xy=True)
X,Y=tr.transform(ll[:,0],ll[:,1]); XY=np.c_[X,Y]
E=np.array([(ix[a],ix[b],sp) for a,b,sp in h.edges])
u=E[:,0].astype(int); v=E[:,1].astype(int); sp=E[:,2]
length=np.hypot(*(XY[u]-XY[v]).T); tt=length/(sp/3.6)
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
n=len(ids); G=csr_matrix((length+1e-6,(u,v)),shape=(n,n))
nc,lab=connected_components(G,directed=True,connection='strong'); big=np.bincount(lab).argmax(); keep=lab==big
print('SCC nodes',keep.sum(),'of',n)
pickle.dump(dict(XY=XY,u=u,v=v,length=length,tt=tt,keep=keep),open('roadnet.pkl','wb'))
