"""
Analysis stabilnosti teritorija dan-na-dan (Territory Stability Analysis)

Za svaki kurir i svaki par uzastopnih days racuna:
1. Centroid displacement (m) - koliko se centar heksagona pomakne
2. Overlap between consecutive days (%) - what fraction of territory remains the same
3. Promjena velicine (side length heksagona)

Takodjer racuna agregiranu "prostornu volatilnost" po kuriru kroz cijeli mjesec.

Run: python3 stability_analysis.py
"""
import os
import numpy as np
import pandas as pd
from shapely import wkt
import yaml
import os

with open(os.path.join(os.path.dirname(__file__), "..", "config.yaml")) as f:
    config = yaml.safe_load(f)


RESULTS_DIR = config["data"]["results_dir"]
OUT_PATH = os.path.join(RESULTS_DIR, "stability_analysis.csv")
OUT_SUMMARY = os.path.join(RESULTS_DIR, "stability_summary.csv")

# Koristimo Method 1 (Greedy k-means) kao primarnu metodu
hex_df = pd.read_csv(os.path.join(RESULTS_DIR, "method1_hexagons.csv"))
hex_df["date"] = pd.to_datetime(hex_df["date"])
hex_df = hex_df.sort_values(["UserID", "date"])

# Ucitaj i poligone
hex_df["polygon"] = hex_df["polygon_wkt"].apply(wkt.loads)

dates = sorted(hex_df["date"].unique())
couriers = sorted(hex_df["UserID"].unique())

records = []

for cid in couriers:
    c_data = hex_df[hex_df["UserID"] == cid].set_index("date")

    for i in range(len(dates) - 1):
        d1, d2 = dates[i], dates[i + 1]

        if d1 not in c_data.index or d2 not in c_data.index:
            continue

        r1 = c_data.loc[d1]
        r2 = c_data.loc[d2]

        # Centroid displacement
        dx = r2["center_x"] - r1["center_x"]
        dy = r2["center_y"] - r1["center_y"]
        displacement_m = np.sqrt(dx**2 + dy**2)

        # Spatial overlap between consecutive days
        poly1 = r1["polygon"]
        poly2 = r2["polygon"]
        try:
            intersection_area = poly1.intersection(poly2).area
            union_area = poly1.union(poly2).area
            iou = intersection_area / union_area if union_area > 0 else 0.0
            overlap_pct = 100 * intersection_area / poly1.area if poly1.area > 0 else 0.0
        except Exception:
            iou = np.nan
            overlap_pct = np.nan

        # Side length change
        delta_s = abs(r2["side_length_m"] - r1["side_length_m"])

        records.append({
            "UserID": cid,
            "date_from": d1,
            "date_to": d2,
            "centroid_displacement_m": displacement_m,
            "overlap_pct": overlap_pct,
            "iou": iou,
            "side_length_from": r1["side_length_m"],
            "side_length_to": r2["side_length_m"],
            "delta_side_length_m": delta_s,
        })

df = pd.DataFrame(records)
df.to_csv(OUT_PATH, index=False)

print("=" * 60)
print("TERRITORY DAY-TO-DAY STABILITY ANALYSIS")
print("=" * 60)
print(f"\nNumber parova uzastopnih days: {len(df):,}")
print(f"Number couriers: {len(couriers)}")

print("\n--- Centroid Displacement (m) ---")
print(df["centroid_displacement_m"].describe().round(1))

print("\n--- Overlap s prethodnim danom (%) ---")
print(df["overlap_pct"].describe().round(1))

print("\n--- IoU (Intersection over Union) ---")
print(df["iou"].describe().round(3))

print("\n--- Promjena stranice heksagona (m) ---")
print(df["delta_side_length_m"].describe().round(1))

# Korisna klasifikacija displacement-a
print("\n--- Distribucija centroid displacement-a ---")
bins = [0, 250, 500, 1000, 2000, float("inf")]
labels = ["<250m", "250-500m", "500m-1km", "1-2km", ">2km"]
df["displacement_bin"] = pd.cut(df["centroid_displacement_m"], bins=bins, labels=labels)
print(df["displacement_bin"].value_counts().sort_index())
pct = df["displacement_bin"].value_counts(normalize=True).sort_index() * 100
for label, val in pct.items():
    print(f"  {label}: {val:.1f}%")

# Summary po kuriru (prostorna volatilnost)
print("\n--- Prostorna volatilnost po kuriru (srednja displacement) ---")
summary = df.groupby("UserID").agg(
    mean_displacement_m=("centroid_displacement_m", "mean"),
    median_displacement_m=("centroid_displacement_m", "median"),
    mean_overlap_pct=("overlap_pct", "mean"),
    mean_iou=("iou", "mean"),
    n_transitions=("centroid_displacement_m", "count"),
).round(1)
print(summary.sort_values("mean_displacement_m", ascending=False).to_string())
summary.to_csv(OUT_SUMMARY, index=True)

print(f"\nSaved: {OUT_PATH}")
print(f"Saved: {OUT_SUMMARY}")
print("\nDONE.")
