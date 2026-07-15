"""
Prepares December dataset for comparison with October.

Isti kriteriji kao za October:
- Distributivno podrucje: Zagreb West distribution area
- Kuriri s >5000 dostava godisnje (u istom podrucju)
- Mjesec: December 2025

Run: python3 pripremi_December.py
"""
import pandas as pd
import numpy as np

INPUT_FILE = config["data"].get("input_file_raw", "data/raw_deliveries.csv")
OUTPUT_FILE = config["data"].get("input_file_december", "data/december_input.csv")

FACILITY_NAME = "Zagreb West distribution area"
TARGET_MONTH = "2025-12"
MIN_ANNUAL_DELIVERIES = 5000

# Koordinatna projekcija (isti kao u preprocessing)
ZAGREB_LAT = 45.8150
ZAGREB_LON = 15.9819
M_LAT = 111320.0
M_LON = 111320.0 * np.cos(np.radians(ZAGREB_LAT))

print("=" * 60)
print("PRIPREMA PROSINACKOG UZORKA")
print("=" * 60)

zagreb_df = pd.read_csv(INPUT_FILE, encoding="utf-8", low_memory=False)
zagreb_df["EventDatetime"] = pd.to_datetime(zagreb_df["EventDatetime"], errors="coerce")
zagreb_df["month"] = zagreb_df["EventDatetime"].dt.to_period("M")

# Filter na facility
facility_df = zagreb_df[zagreb_df["FacilityName"] == FACILITY_NAME].copy()
print(f"\nTotal redova u {FACILITY_NAME}: {len(facility_df):,}")

# Kvalificirani kuriri (>5000 godisnje)
per_courier_total = facility_df.groupby("UserID").size()
qualifying_ids = set(per_courier_total[per_courier_total > MIN_ANNUAL_DELIVERIES].index)
print(f"Kvalificiranih couriers (>5000/god): {len(qualifying_ids)}")

# Filter na December i kvalificirane kurire
final_df = facility_df[
    (facility_df["UserID"].isin(qualifying_ids)) &
    (facility_df["month"].astype(str) == TARGET_MONTH)
].copy()

# Projekcija koordinata
final_df["x_m"] = (final_df["EventGeoX"] - ZAGREB_LON) * M_LON
final_df["y_m"] = (final_df["EventGeoY"] - ZAGREB_LAT) * M_LAT

# Sortiraj i spremi
cols = ["ShipmentItemBarcode", "EventDatetime", "UserID",
        "UserFirstName", "UserLastName", "EventGeoX", "EventGeoY",
        "x_m", "y_m", "FacilityName", "GroupAreaName", "dist_from_center_km"]

# dist_from_center_km mozda nije u ovom datasetu - dodaj ga
R = 6371.0
lat = np.radians(final_df["EventGeoY"])
lon = np.radians(final_df["EventGeoX"])
lat0 = np.radians(ZAGREB_LAT)
lon0 = np.radians(ZAGREB_LON)
dlat = lat - lat0
dlon = lon - lon0
a = np.sin(dlat/2)**2 + np.cos(lat0)*np.cos(lat)*np.sin(dlon/2)**2
final_df["dist_from_center_km"] = 2*R*np.arcsin(np.sqrt(a))

available_cols = [c for c in cols if c in final_df.columns]
final_df = final_df[available_cols].sort_values(
    ["EventDatetime", "UserID"]
).reset_index(drop=True)

# Dodaj date kolonu
final_df["date"] = pd.to_datetime(final_df["EventDatetime"]).dt.date

final_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

print(f"\nProsinacki uzorak:")
print(f"  Total redova: {len(final_df):,}")
print(f"  Number couriers: {final_df['UserID'].nunique()}")
print(f"  Number days: {final_df['date'].nunique()}")
print(f"  Datumski raspon: {final_df['date'].min()} do {final_df['date'].max()}")

# Dnevni volume
daily = final_df.groupby("date").size()
print(f"\nDnevni volume (December):")
print(f"  Min: {daily.min()}, Max: {daily.max()}, Median: {daily.median():.0f}, Mean: {daily.mean():.1f}")

# Feasibility check [60,80]
k = final_df["UserID"].nunique()
infeasible = ((daily < k*60) | (daily > k*80)).sum()
print(f"\nFeasibility [60,80] s k={k} couriers:")
print(f"  Infeasible days: {infeasible} od {daily.nunique()} ({100*infeasible/daily.nunique():.1f}%)")
print(f"  Feasibility window: [{k*60}, {k*80}] parcels/dan")

print(f"\nSaved: {OUTPUT_FILE}")
print("\nDONE.")
