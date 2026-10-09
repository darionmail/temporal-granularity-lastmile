"""Step 0: read the name-free extract and cache it (run once)."""
import pandas as pd, sys
src = sys.argv[1] if len(sys.argv) > 1 else 'zagreb_2025_extract_noNames.csv.gz'
d = pd.read_csv(src, dtype={'UserID': str, 'FacilityCode': str})
d['t'] = pd.to_datetime(d.EventDatetime); d['m'] = d.t.dt.month
d.to_pickle('year.pkl')
