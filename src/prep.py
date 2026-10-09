import pandas as pd, numpy as np
from pyproj import Transformer
S3=np.sqrt(3)
def hex_axial(x,y,s):
    q=(S3/3*x - y/3)/s; r=(2/3*y)/s
    cx,cz=q,r; cy=-cx-cz
    rx,ry,rz=np.round(cx),np.round(cy),np.round(cz)
    dx,dy,dz=np.abs(rx-cx),np.abs(ry-cy),np.abs(rz-cz)
    m1=(dx>dy)&(dx>dz); m2=(~m1)&(dy>dz); m3=~(m1|m2)
    rx=np.where(m1,-ry-rz,rx); rz=np.where(m3,-rx-ry,rz)
    return rx.astype(int), rz.astype(int)
def hex_center(q,r,s): return s*S3*(q+r/2), s*1.5*r
def load_facility_map(path='facility_map_private.csv'):
    m=pd.read_csv(path, dtype={'FacilityCode':str})
    need={'FacilityCode','depot'}
    if not need.issubset(m.columns):
        raise ValueError(f'{path} must contain columns: FacilityCode,depot')
    m=m[m.depot.isin(['Depot01','Depot02','Depot03','Depot04'])].copy()
    if m.depot.nunique()!=4:
        raise ValueError(f'{path} must map exactly Depot01-Depot04')
    return dict(zip(m.FacilityCode.astype(str),m.depot))

def build(BU_SIDE=250, facility_map='facility_map_private.csv'):
    DEPOTS=load_facility_map(facility_map)
    d=pd.read_pickle('year.pkl')
    d['FacilityCode']=d.FacilityCode.astype(str)
    d=d[d.FacilityCode.isin(DEPOTS)].copy()
    tr=Transformer.from_crs(4326,3765,always_xy=True)
    d['x'],d['y']=tr.transform(d.EventGeoX.values,d.EventGeoY.values)
    d['date']=d.t.dt.normalize(); d['dow']=d.t.dt.dayofweek
    d['depot']=d.FacilityCode.map(DEPOTS)
    d['cid']=d.UserID.astype(float).astype(int)
    d['q'],d['r']=hex_axial(d.x.values,d.y.values,BU_SIDE)
    d['bu']=d.q.astype(str)+'_'+d.r.astype(str)
    d=d.sort_values(['depot','date','cid','t'])
    d[['depot','date','m','dow','t','cid','x','y','q','r','bu','ShipmentItemBarcode']].to_pickle('deliv.pkl')
    return d
if __name__=='__main__':
    d=build()
    g=d.groupby(['depot','date']).agg(V=('cid','size'),k=('cid','nunique'),dow=('dow','first')).reset_index()
    print(g.groupby(['depot',g.dow<5]).agg(days=('V','size'),V=('V','median'),k=('k','median')))
    print('Sundays',(g.dow==6).sum())
