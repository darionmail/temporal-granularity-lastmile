import os
from pathlib import Path
import pandas as pd, numpy as np

LAT0,LON0=45.8150,15.9819
DATA_DIR=Path(os.environ.get('PITFALL_DATA_DIR','private_identity_samples'))
FILES={'oct':'depot01_october_sample.csv','dec':'depot01_december_sample.csv'}

def load(m):
    if m not in FILES:
        raise ValueError(f'Unknown sample {m!r}; expected one of {list(FILES)}')
    f=DATA_DIR/FILES[m]
    if not f.exists():
        raise FileNotFoundError(
            f'Missing {f}. Set PITFALL_DATA_DIR to the directory containing the restricted '             'October/December Depot01 samples.'
        )
    d=pd.read_csv(f)
    d=d.drop(columns=[c for c in d.columns if c in('UserFirstName','UserLastName')])
    d['t']=pd.to_datetime(d.EventDatetime); d['date']=d.t.dt.normalize()
    d['lon']=d.EventGeoX; d['lat']=d.EventGeoY
    d['x']=(d.lon-LON0)*111320*np.cos(np.radians(LAT0)); d['y']=(d.lat-LAT0)*111132.9
    d['cid']=d.UserID.astype(int)
    return d
