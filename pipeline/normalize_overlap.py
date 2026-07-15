"""
Normalizirani overlap - preklapanje kao postotak ukupne teritorijalne površine.

Za svaki dan i svaku metodu racuna:
  overlap_pct = (ukupni overlap između parova) / (ukupna površina svih heksagona tog dana) * 100

Pokreni: python3 normaliziraj_overlap.py
"""
import os
import numpy as np
import pandas as pd
from shapely import wkt

RESULTS_DIR = "/podaci/data_fixed/rezultati"

METHODS = {
    "Greedy k-means":        "method1_hexagons.csv",
    "Min-Cost Flow":         "method2_mcf_baseline_hexagons.csv",
    "Min-Cost Flow (tuned)": "method2_mcf_tuned_hexagons.csv",
    "Weighted Voronoi":      "method3_voronoi_baseline_hexagons.csv",
    "Weighted Voronoi (tuned)": "method3_voronoi_tuned_hexagons.csv",
}

print("=" * 60)
print("NORMALIZIRANI OVERLAP PO METODI")
print("=" * 60)

summary = []

for method, fname in METHODS.items():
    hex_df = pd.read_csv(os.path.join(RESULTS_DIR, fname))
    hex_df["polygon"] = hex_df["polygon_wkt"].apply(wkt.loads)

    daily_overlap_pct = []

    for date, day_hex in hex_df.groupby("date"):
        polys = list(day_hex["polygon"])
        areas = [p.area for p in polys]
        total_area_m2 = sum(areas)

        overlap_m2 = 0.0
        for i in range(len(polys)):
            for j in range(i + 1, len(polys)):
                if polys[i].intersects(polys[j]):
                    overlap_m2 += polys[i].intersection(polys[j]).area

        pct = 100 * overlap_m2 / total_area_m2 if total_area_m2 > 0 else 0.0
        daily_overlap_pct.append(pct)

    mean_pct = np.mean(daily_overlap_pct)
    std_pct  = np.std(daily_overlap_pct)
    min_pct  = np.min(daily_overlap_pct)
    max_pct  = np.max(daily_overlap_pct)

    print(f"\n{method}:")
    print(f"  Overlap/dan (%): {mean_pct:.2f} +/- {std_pct:.2f}  (min={min_pct:.2f}, max={max_pct:.2f})")

    summary.append({
        "method": method,
        "mean_overlap_pct": round(mean_pct, 3),
        "std_overlap_pct":  round(std_pct, 3),
        "min_overlap_pct":  round(min_pct, 3),
        "max_overlap_pct":  round(max_pct, 3),
    })

out = pd.DataFrame(summary)
out.to_csv(os.path.join(RESULTS_DIR, "normalized_overlap.csv"), index=False)
print(f"\nSpremljeno: {os.path.join(RESULTS_DIR, 'normalized_overlap.csv')}")
print("\nGOTOVO.")
