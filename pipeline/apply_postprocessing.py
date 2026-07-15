"""
primijeni_postprocessing.py

Primjenjuje Stage 4-5 post-processing na MCF i Voronoi rezultate
that already exist, then evaluates all methods in two variants:
  - "raw"  : samo algoritamska dodjela (Stage 1-3 / Stage 1 za MCF/Voronoi)
  - "pp"   : s post-processingom (Stage 4-5 za sve metode)

Results se zapisuju u:
  results_dir/evaluation_two_stage.csv

Run: python3 skripte/primijeni_postprocessing.py
"""
import sys
import os
import numpy as np
import pandas as pd
from shapely import wkt
from shapely.geometry import Point
import yaml
import os

with open(os.path.join(os.path.dirname(__file__), "..", "config.yaml")) as f:
    config = yaml.safe_load(f)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "common"))

from postprocessing import apply_postprocessing
from hex_utils import hex_area

RESULTS_DIR = config["data"]["results_dir"]

CONFIG = {
    "hexagon": {
        "base_lattice_km": 1.0,
        "refine_lattice_km": 0.5,
        "s_base_m": 1000.0,
        "s_max_m": 2000.0,
        "s_floor_frac": 0.75,
    },
    "refinement": {
        "alpha": 1000000.0,
        "lambda_dens": 1.0,
        "lambda_eq": 0.001,
        "max_iterations": 10,
    },
}

METHODS = {
    "Greedy k-means (raw)":        ("method1_assignments.csv", "method1_hexagons.csv", False),
    "Greedy k-means (pp)":         ("method1_assignments.csv", "method1_hexagons.csv", True),
    "Min-Cost Flow (raw)":         ("method2_mcf_baseline_assignments.csv", "method2_mcf_baseline_hexagons.csv", False),
    "Min-Cost Flow (pp)":          ("method2_mcf_baseline_assignments.csv", "method2_mcf_baseline_hexagons.csv", True),
    "Weighted Voronoi (raw)":      ("method3_voronoi_baseline_assignments.csv", "method3_voronoi_baseline_hexagons.csv", False),
    "Weighted Voronoi (pp)":       ("method3_voronoi_baseline_assignments.csv", "method3_voronoi_baseline_hexagons.csv", True),
}


def evaluate_day(day_assignments, day_hexagons_df):
    hexagons = {}
    for _, row in day_hexagons_df.iterrows():
        hexagons[row["UserID"]] = {
            "polygon": wkt.loads(row["polygon_wkt"]),
            "area_m2": row["area_m2"],
        }
    courier_ids = list(hexagons.keys())

    overlap_m2 = 0.0
    total_area_m2 = sum(h["area_m2"] for h in hexagons.values())
    for i in range(len(courier_ids)):
        for j in range(i + 1, len(courier_ids)):
            pi = hexagons[courier_ids[i]]["polygon"]
            pj = hexagons[courier_ids[j]]["polygon"]
            if pi.intersects(pj):
                overlap_m2 += pi.intersection(pj).area

    overlap_km2 = overlap_m2 / 1e6
    overlap_pct = 100 * overlap_m2 / total_area_m2 if total_area_m2 > 0 else 0.0

    a_over_n = []
    for cid in courier_ids:
        n = (day_assignments["assigned_courier"] == cid).sum()
        if n > 0:
            a_over_n.append(hexagons[cid]["area_m2"] / n)

    outside = 0
    for _, row in day_assignments.iterrows():
        cid = row["assigned_courier"]
        if pd.isna(cid) or cid not in hexagons:
            outside += 1
            continue
        if not hexagons[cid]["polygon"].contains(Point(row["x_m"], row["y_m"])):
            outside += 1

    return {
        "overlap_km2": overlap_km2,
        "overlap_pct": overlap_pct,
        "mean_a_n": np.mean(a_over_n) if a_over_n else 0,
        "std_a_n": np.std(a_over_n) if a_over_n else 0,
        "outside": outside,
        "n_parcels": len(day_assignments),
    }


print("=" * 70)
print("EVALUACIJA: RAW vs POST-PROCESSED (Stage 4-5 za sve metode)")
print("=" * 70)

summary_rows = []

for method_name, (assign_file, hex_file, apply_pp) in METHODS.items():
    print(f"\n>>> {method_name} ({'s PP' if apply_pp else 'bez PP'})...")

    df = pd.read_csv(os.path.join(RESULTS_DIR, assign_file))
    hex_df = pd.read_csv(os.path.join(RESULTS_DIR, hex_file))
    df["date"] = df["date"].astype(str)
    hex_df["date"] = hex_df["date"].astype(str)

    if apply_pp:
        df, hex_df = apply_postprocessing(
            df, hex_df,
            CONFIG["hexagon"], CONFIG["refinement"]
        )

    daily_results = []
    for date, day_hex in hex_df.groupby("date"):
        day_assign = df[df["date"] == str(date)]
        r = evaluate_day(day_assign, day_hex)
        r["date"] = date
        daily_results.append(r)

    daily_df = pd.DataFrame(daily_results)

    row = {
        "method": method_name,
        "postprocessed": apply_pp,
        "overlap_km2_per_day": round(daily_df["overlap_km2"].mean(), 2),
        "overlap_pct": round(daily_df["overlap_pct"].mean(), 2),
        "mean_a_n": round(daily_df["mean_a_n"].mean(), 0),
        "std_a_n": round(daily_df["std_a_n"].mean(), 1),
        "outside_per_day": round(daily_df["outside"].mean(), 2),
        "outside_pct": round(
            100 * daily_df["outside"].sum() / daily_df["n_parcels"].sum(), 2
        ),
        "n_days": len(daily_df),
    }
    summary_rows.append(row)
    print(f"    Overlap/dan: {row['overlap_km2_per_day']} km2 ({row['overlap_pct']}%)")
    print(f"    Outside:     {row['outside_per_day']:.1f}/dan ({row['outside_pct']}%)")
    print(f"    Mean A/N:    {row['mean_a_n']:,.0f}")

summary_df = pd.DataFrame(summary_rows)
out_path = os.path.join(RESULTS_DIR, "evaluation_two_stage.csv")
summary_df.to_csv(out_path, index=False)

print("\n" + "=" * 70)
print("FINALNA USPOREDBA — RAW vs POST-PROCESSED")
print("=" * 70)
print(f"\n{'Method':<30} {'Overlap/dan (km2)':>18} {'Overlap%':>9} {'Outside/dan':>12} {'Outside%':>9}")
print("-" * 80)
for _, row in summary_df.iterrows():
    print(f"{row['method']:<30} {row['overlap_km2_per_day']:>18.2f} "
          f"{row['overlap_pct']:>8.2f}% {row['outside_per_day']:>12.1f} "
          f"{row['outside_pct']:>8.2f}%")

print(f"\nSaved: {out_path}")
print("\nDONE.")
