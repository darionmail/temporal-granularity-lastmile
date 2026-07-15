"""
Sensitivity Analysis za:
1. Stage 5 parameters: alpha (10^4 do 10^7) i lambda_dens (0.1, 1.0, 10.0)
   - Pokrecemo Stage 5 s razlicitim parametersma na postojecim assignments
   - Usporedujemo finalni ranking metoda

2. Composite score tezine:
   - Jednake tezine (1/3, 1/3, 1/3)
   - Outside prioritet (0.25, 0.50, 0.25)
   - Overlap prioritet (0.50, 0.25, 0.25)
   - Density prioritet (0.25, 0.25, 0.50)

Cilj: pokazati da ranking ostaje stable pod razumnim varijacijama parametara.

Run: python3 sensitivity_analysis_params.py
"""
import os
import sys
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

from hex_utils import snap_cover_cap, get_neighbor_centers, hex_area

RESULTS_DIR = config["data"]["results_dir"]

# ============================================================
# PART 1: Composite Score Sensitivity (brzo, bez ponovnog racunanja)
# ============================================================

print("=" * 65)
print("SENSITIVITY ANALYSIS")
print("=" * 65)

print("\n── PART 1: Composite Score Weight Sensitivity ──\n")

# Ucitaj post-processed rezultate
pp_data = {
    "Greedy k-means":   {"overlap_pct": 61.59, "outside_pct": 0.61, "mean_a_n": 168797},
    "Min-Cost Flow":    {"overlap_pct": 59.76, "outside_pct": 1.28, "mean_a_n": 165606},
    "Weighted Voronoi": {"overlap_pct": 57.22, "outside_pct": 0.86, "mean_a_n": 174433},
}

methods = list(pp_data.keys())
df = pd.DataFrame(pp_data).T

# Normalizacija [0,1]
norm = pd.DataFrame(index=df.index)
for col in ["overlap_pct", "outside_pct", "mean_a_n"]:
    mn, mx = df[col].min(), df[col].max()
    norm[col] = (df[col] - mn) / (mx - mn) if mx > mn else 0.0

# Razlicite sheme tezina
weight_schemes = {
    "Equal (1/3, 1/3, 1/3)":          (1/3,  1/3,  1/3),
    "Outside priority (0.25, 0.50, 0.25)": (0.25, 0.50, 0.25),
    "Overlap priority (0.50, 0.25, 0.25)": (0.50, 0.25, 0.25),
    "Density priority (0.25, 0.25, 0.50)": (0.25, 0.25, 0.50),
    "Outside dominant (0.10, 0.80, 0.10)": (0.10, 0.80, 0.10),
    "Overlap dominant (0.80, 0.10, 0.10)": (0.80, 0.10, 0.10),
}

print(f"{'Scheme':<42} {'GKM':>8} {'MCF':>8} {'VOR':>8} {'Rank 1':>12}")
print("-" * 82)

ranking_results = {}
for scheme_name, (w_ov, w_out, w_dn) in weight_schemes.items():
    scores = {}
    for m in methods:
        scores[m] = w_ov * norm.loc[m, "overlap_pct"] + \
                    w_out * norm.loc[m, "outside_pct"] + \
                    w_dn  * norm.loc[m, "mean_a_n"]
    ranked = sorted(scores, key=scores.get)
    ranking_results[scheme_name] = ranked[0]
    print(f"{scheme_name:<42} {scores['Greedy k-means']:>8.3f} "
          f"{scores['Min-Cost Flow']:>8.3f} "
          f"{scores['Weighted Voronoi']:>8.3f}  {ranked[0]}")

# Provjeri consistentst
rank1_methods = set(ranking_results.values())
if len(rank1_methods) == 1:
    print(f"\n✓ Rank 1 is STABLE across all weight schemes: {list(rank1_methods)[0]}")
else:
    print(f"\n⚠ Rank 1 varies: {rank1_methods}")

# ============================================================
# PART 2: Stage 5 Alpha/Lambda Sensitivity
# ============================================================

print("\n\n── PART 2: Stage 5 Parameter Sensitivity ──")
print("Running Stage 5 with different alpha/lambda on Method 1 (Greedy k-means)\n")

def run_stage5_evaluate(df_assign, hex_df, alpha, lambda_dens,
                        lambda_eq=0.001, max_iter=5):
    """Pokrecemo Stage 5 s danim parametersma, vracamo aggregate metrike."""
    from postprocessing import stage5_refinement, stage4_reallocation

    hex_config = {
        "base_lattice_km": 1.0,
        "refine_lattice_km": 0.5,
        "s_base_m": 1000.0,
        "s_max_m": 2000.0,
        "s_floor_frac": 0.75,
    }
    refine_config = {
        "alpha": alpha,
        "lambda_dens": lambda_dens,
        "lambda_eq": lambda_eq,
        "max_iterations": max_iter,
    }

    total_overlap = 0.0
    total_area = 0.0
    total_outside = 0
    total_parcels = 0

    for date, day_hex in hex_df.groupby("date"):
        day_assign = df_assign[df_assign["date"] == str(date)]
        points_xy = day_assign[["x_m", "y_m"]].values
        assignment = day_assign["assigned_courier"].values.copy()
        active_couriers = list(day_hex["UserID"].unique())

        hexagons = {}
        for _, row in day_hex.iterrows():
            hexagons[row["UserID"]] = {
                "center": (row["center_x"], row["center_y"]),
                "s": row["side_length_m"],
                "polygon": wkt.loads(row["polygon_wkt"]),
            }

        # Stage 4 (standard)
        assignment = stage4_reallocation(points_xy, assignment, hexagons, active_couriers)

        # Stage 5 with varied params
        hexagons = stage5_refinement(points_xy, assignment, hexagons, hex_config, refine_config)

        # Evaluate
        polys = [h["polygon"] for h in hexagons.values()]
        areas = [p.area for p in polys]
        total_area += sum(areas)
        for i in range(len(polys)):
            for j in range(i+1, len(polys)):
                if polys[i].intersects(polys[j]):
                    total_overlap += polys[i].intersection(polys[j]).area

        for k, (px, py) in enumerate(points_xy):
            cid = assignment[k]
            if cid not in hexagons:
                total_outside += 1
                continue
            if not hexagons[cid]["polygon"].contains(Point(px, py)):
                total_outside += 1
        total_parcels += len(points_xy)

    overlap_pct = 100 * total_overlap / total_area if total_area > 0 else 0
    outside_pct = 100 * total_outside / total_parcels if total_parcels > 0 else 0
    return overlap_pct, outside_pct

# Ucitaj Method 1 assignments i hexagone
df_assign = pd.read_csv(os.path.join(RESULTS_DIR, "method1_assignments.csv"))
hex_df = pd.read_csv(os.path.join(RESULTS_DIR, "method1_hexagons.csv"))
df_assign["date"] = df_assign["date"].astype(str)
hex_df["date"] = hex_df["date"].astype(str)

# Test alpha values (lambda_dens fiksno = 1.0)
alpha_values = [1e4, 1e5, 1e6, 1e7]
print(f"{'Alpha':>12} {'lambda_dens':>12} {'Overlap%':>10} {'Outside%':>10}")
print("-" * 50)

alpha_results = []
for alpha in alpha_values:
    ov, out = run_stage5_evaluate(df_assign, hex_df, alpha=alpha, lambda_dens=1.0)
    print(f"{alpha:>12.0e} {'1.0':>12} {ov:>10.2f} {out:>10.2f}")
    alpha_results.append({"alpha": alpha, "lambda_dens": 1.0, "overlap_pct": ov, "outside_pct": out})

# Test lambda_dens values (alpha fiksno = 1e6)
lambda_values = [0.1, 1.0, 10.0]
print()
for lam in lambda_values:
    if lam == 1.0:
        # Already have this result
        r = [x for x in alpha_results if x["alpha"] == 1e6]
        if r:
            print(f"{'1e6':>12} {lam:>12.1f} {r[0]['overlap_pct']:>10.2f} {r[0]['outside_pct']:>10.2f}")
            continue
    ov, out = run_stage5_evaluate(df_assign, hex_df, alpha=1e6, lambda_dens=lam)
    print(f"{'1e6':>12} {lam:>12.1f} {ov:>10.2f} {out:>10.2f}")
    alpha_results.append({"alpha": 1e6, "lambda_dens": lam, "overlap_pct": ov, "outside_pct": out})

# Save rezultate
results_df = pd.DataFrame(alpha_results)
results_df.to_csv(os.path.join(RESULTS_DIR, "sensitivity_analysis.csv"), index=False)
print(f"\nSaved: {os.path.join(RESULTS_DIR, 'sensitivity_analysis.csv')}")
print("\nDONE.")
