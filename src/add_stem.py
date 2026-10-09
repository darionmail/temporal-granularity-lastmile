import pandas as pd, numpy as np, glob
from depots import DXY
for f in sorted(glob.glob('rq3f_*.csv')):
    tag=f[5:-4]; R=pd.read_csv(f); C=pd.read_csv(f'cent_{tag}.csv.gz')
    dep=R.depot.iloc[0]; dx,dy=DXY[dep]
    C['stem']=2*np.hypot(C.cx-dx,C.cy-dy)/1000
    S=C.groupby(['date','design']).stem.sum().rename('stem_km').reset_index()
    R=R.drop(columns=[c for c in ['stem_km','total_km'] if c in R]).merge(S,on=['date','design'],how='left')
    R['total_km']=R.tour_km+R.stem_km
    R.to_csv(f,index=False)
    print(tag, R.groupby('design')[['tour_km','stem_km','total_km']].mean().round(1).T.to_string())
