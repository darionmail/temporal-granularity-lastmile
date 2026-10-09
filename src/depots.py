from pathlib import Path
import os
import pandas as pd
import numpy as np
from pyproj import Transformer

EXPECTED = ['Depot01','Depot02','Depot03','Depot04']

def load_depots(path=None):
    path = Path(path or os.environ.get('DEPOT_COORDS_FILE','depots_private.csv'))
    if not path.exists():
        raise FileNotFoundError(
            f'Missing {path}. Copy depots_private_template.csv to depots_private.csv '             'and fill the four private depot coordinates.'
        )
    d = pd.read_csv(path)
    if not {'depot','lat','lon'}.issubset(d.columns):
        raise ValueError('Depot coordinate file must contain columns: depot,lat,lon')
    d = d[d.depot.isin(EXPECTED)].copy()
    if sorted(d.depot.unique()) != EXPECTED:
        raise ValueError('Depot coordinate file must contain exactly Depot01-Depot04')
    tr = Transformer.from_crs(4326,3765,always_xy=True)
    return {r.depot: np.array(tr.transform(float(r.lon),float(r.lat))) for _,r in d.iterrows()}

DXY = load_depots()
