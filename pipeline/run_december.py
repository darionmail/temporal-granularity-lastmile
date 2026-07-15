"""
Pokretanje optimizacijskih algoritama za prosinacki uzorak.

Koristi iste algoritme kao za listopad, samo s prosinackim datasetom.
Rezultati se spremaju u posebnu mapu da ne prepisuju listopadske.

Pokreni: python3 pokreni_prosinac.py
"""
import sys
import os
import yaml
import numpy as np
import pandas as pd
import time

# Dodaj skripte i common u path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "common"))

from hex_utils import snap_cover_cap, get_neighbor_centers, hex_area

# Prosinacki config - razlikuje se od listopadskog samo u input/output putanjama
CONFIG = {
    "data": {
        "input_file": config["data"].get("input_file_december", "data/december_input.csv"),
        "results_dir": "results_dir_prosinac",
    },
    "capacity": {"n_min": 60, "n_max": 80},
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

os.makedirs(CONFIG["data"]["results_dir"], exist_ok=True)

# ============================================================
# Uvezi sve potrebne funkcije iz postojecih skripti
# ============================================================

def greedy_capacitated_kmeans_day(points_xy, courier_ids, n_min, n_max, max_iter=20, seed=42):
    rng = np.random.default_rng(seed)
    n_points = len(points_xy)
    k = len(courier_ids)
    if n_points == 0:
        return np.array([]), {cid: None for cid in courier_ids}
    init_idx = rng.choice(n_points, size=min(k, n_points), replace=False)
    centroids = {courier_ids[i]: points_xy[init_idx[i]] for i in range(len(init_idx))}
    for cid in courier_ids:
        if cid not in centroids:
            centroids[cid] = points_xy[rng.integers(0, n_points)]
    assignment = np.full(n_points, -1, dtype=object)
    for iteration in range(max_iter):
        capacity_left = {cid: n_max for cid in courier_ids}
        assignment = np.full(n_points, -1, dtype=object)
        cent_array = np.array([centroids[cid] for cid in courier_ids])
        dists = np.sqrt(((points_xy[:, None, :] - cent_array[None, :, :]) ** 2).sum(axis=2))
        order = np.dstack(np.unravel_index(np.argsort(dists, axis=None), dists.shape))[0]
        for point_idx, courier_idx in order:
            if assignment[point_idx] != -1:
                continue
            cid = courier_ids[courier_idx]
            if capacity_left[cid] > 0:
                assignment[point_idx] = cid
                capacity_left[cid] -= 1
        new_centroids = {}
        for cid in courier_ids:
            mask = assignment == cid
            if mask.sum() > 0:
                new_centroids[cid] = points_xy[mask].mean(axis=0)
            else:
                new_centroids[cid] = centroids[cid]
        shift = sum(np.linalg.norm(new_centroids[cid] - centroids[cid]) for cid in courier_ids)
        centroids = new_centroids
        if shift < 1.0:
            break
    return assignment, centroids


def build_day_hexagons(points_xy, assignment, centroids, courier_ids, hex_config):
    hexagons = {}
    for cid in courier_ids:
        mask = assignment == cid
        courier_points = points_xy[mask]
        center = centroids.get(cid)
        if center is None or len(courier_points) == 0:
            continue
        x_snap, y_snap, s, poly = snap_cover_cap(
            center[0], center[1], courier_points,
            lattice_s=hex_config["base_lattice_km"] * 1000,
            s_base=hex_config["s_base_m"],
            s_max=hex_config["s_max_m"],
            s_floor_frac=hex_config["s_floor_frac"],
        )
        hexagons[cid] = {"center": (x_snap, y_snap), "s": s, "polygon": poly}
    return hexagons


def stage4_reallocation(points_xy, assignment, hexagons, courier_ids):
    from shapely.geometry import Point
    new_assignment = assignment.copy()
    for i in range(len(points_xy)):
        current_cid = assignment[i]
        if current_cid not in hexagons:
            continue
        pt = Point(points_xy[i])
        if hexagons[current_cid]["polygon"].contains(pt):
            continue
        containing = [cid for cid in hexagons if hexagons[cid]["polygon"].contains(pt)]
        if not containing:
            continue
        elif len(containing) == 1:
            new_assignment[i] = containing[0]
        else:
            dists = [np.linalg.norm(points_xy[i] - np.array(hexagons[cid]["center"])) for cid in containing]
            new_assignment[i] = containing[np.argmin(dists)]
    return new_assignment


def stage5_refinement(points_xy, assignment, hexagons, hex_config, refine_config):
    from shapely.geometry import Point
    hexagons = {cid: dict(h) for cid, h in hexagons.items()}
    alpha = refine_config["alpha"]
    lambda_dens = refine_config["lambda_dens"]
    lambda_eq = refine_config["lambda_eq"]
    refine_lattice_m = hex_config["refine_lattice_km"] * 1000
    for _ in range(refine_config["max_iterations"]):
        densities = [hex_area(h["s"]) / max((assignment==cid).sum(), 1) for cid, h in hexagons.items()]
        delta = np.median(densities)
        any_improved = False
        for cid in list(hexagons.keys()):
            own = points_xy[assignment == cid]
            n_u = len(own)
            def score(poly, s, cx, cy):
                ov = sum(poly.intersection(hexagons[v]["polygon"]).area
                         for v in hexagons if v != cid and poly.intersects(hexagons[v]["polygon"]))
                out = sum(1 for px,py in own if not poly.contains(Point(px,py)))
                a = hex_area(s)
                d = a/n_u if n_u > 0 else a
                return ov + alpha*out + lambda_dens*d + lambda_eq*(d-delta)**2
            cur_score = score(hexagons[cid]["polygon"], hexagons[cid]["s"],
                              hexagons[cid]["center"][0], hexagons[cid]["center"][1])
            best_score = cur_score
            best = None
            for cx, cy in get_neighbor_centers(hexagons[cid]["center"][0], hexagons[cid]["center"][1], refine_lattice_m):
                _, _, s, poly = snap_cover_cap(cx, cy, own,
                    lattice_s=hex_config["base_lattice_km"]*1000,
                    s_base=hex_config["s_base_m"], s_max=hex_config["s_max_m"],
                    s_floor_frac=hex_config["s_floor_frac"])
                cs = score(poly, s, cx, cy)
                if cs < best_score:
                    best_score = cs
                    best = (cx, cy, s, poly)
            if best:
                hexagons[cid]["center"] = (best[0], best[1])
                hexagons[cid]["s"] = best[2]
                hexagons[cid]["polygon"] = best[3]
                any_improved = True
        if not any_improved:
            break
    return hexagons


def run_method1(df, config):
    """Greedy k-means + hex optimizacija, po danu."""
    hex_config = config["hexagon"]
    refine_config = config["refinement"]
    n_min = config["capacity"]["n_min"]
    n_max = config["capacity"]["n_max"]

    df = df.copy()
    df["assigned_courier"] = None
    all_hex_rows = []
    infeasibility = []
    n_days = df["date"].nunique()
    day_counter = 0

    for date, day_df in df.groupby("date"):
        day_counter += 1
        points_xy = day_df[["x_m", "y_m"]].values
        active_couriers = sorted(day_df["UserID"].unique())
        k = len(active_couriers)
        n_points = len(points_xy)

        if not (k*n_min <= n_points <= k*n_max):
            reason = "premalo" if n_points < k*n_min else "previse"
            infeasibility.append({"date": date, "n_points": n_points, "k": k, "reason": reason})

        assignment, centroids = greedy_capacitated_kmeans_day(points_xy, active_couriers, n_min, n_max)
        day_hexagons = build_day_hexagons(points_xy, assignment, centroids, active_couriers, hex_config)
        assignment = stage4_reallocation(points_xy, assignment, day_hexagons, active_couriers)
        day_hexagons = stage5_refinement(points_xy, assignment, day_hexagons, hex_config, refine_config)

        df.loc[day_df.index, "assigned_courier"] = assignment
        for cid, h in day_hexagons.items():
            all_hex_rows.append({
                "date": date, "UserID": cid,
                "center_x": h["center"][0], "center_y": h["center"][1],
                "side_length_m": h["s"], "area_m2": hex_area(h["s"]),
                "polygon_wkt": h["polygon"].wkt,
            })
        if day_counter % 5 == 0 or day_counter == n_days:
            print(f"  Dan {day_counter}/{n_days} ({date}): {n_points} paketa, {k} kurira")

    return df, pd.DataFrame(all_hex_rows), pd.DataFrame(infeasibility)


def evaluate_daily(df, hex_df):
    """Pet metrika po danu, agregirano."""
    from shapely import wkt
    from shapely.geometry import Point

    daily_results = []
    for date, day_hex in hex_df.groupby("date"):
        day_assign = df[df["date"] == str(date)]
        hexagons = {}
        for _, row in day_hex.iterrows():
            hexagons[row["UserID"]] = {
                "polygon": wkt.loads(row["polygon_wkt"]),
                "area_m2": row["area_m2"],
            }
        courier_ids = list(hexagons.keys())

        overlap_m2 = 0.0
        for i in range(len(courier_ids)):
            for j in range(i+1, len(courier_ids)):
                pi = hexagons[courier_ids[i]]["polygon"]
                pj = hexagons[courier_ids[j]]["polygon"]
                if pi.intersects(pj):
                    overlap_m2 += pi.intersection(pj).area

        a_over_n = []
        for cid in courier_ids:
            n = (day_assign["assigned_courier"] == cid).sum()
            if n > 0:
                a_over_n.append(hexagons[cid]["area_m2"] / n)

        outside = 0
        for _, row in day_assign.iterrows():
            cid = row["assigned_courier"]
            if pd.isna(cid) or cid not in hexagons:
                outside += 1
                continue
            if not hexagons[cid]["polygon"].contains(Point(row["x_m"], row["y_m"])):
                outside += 1

        daily_results.append({
            "date": date,
            "overlap_km2": overlap_m2 / 1e6,
            "sum_a_n": sum(a_over_n),
            "mean_a_n": np.mean(a_over_n) if a_over_n else 0,
            "std_a_n": np.std(a_over_n) if a_over_n else 0,
            "outside": outside,
            "n_parcels": len(day_assign),
            "n_couriers": len(courier_ids),
        })

    return pd.DataFrame(daily_results)


# ============================================================
# MAIN
# ============================================================

print("=" * 60)
print("PROSINACKA ANALIZA - GREEDY K-MEANS")
print("=" * 60)

df = pd.read_csv(CONFIG["data"]["input_file"], encoding="utf-8")
df["date"] = pd.to_datetime(df["EventDatetime"]).dt.date.astype(str)

print(f"\nUkupno paketa: {len(df):,}")
print(f"Kurira: {df['UserID'].nunique()}")
print(f"Dana: {df['date'].nunique()}")

t0 = time.time()
df_out, hex_df, infeasibility_df = run_method1(df, CONFIG)
elapsed = time.time() - t0

# Spremi
df_out.to_csv(os.path.join(CONFIG["data"]["results_dir"], "method1_assignments.csv"), index=False)
hex_df.to_csv(os.path.join(CONFIG["data"]["results_dir"], "method1_hexagons.csv"), index=False)
infeasibility_df.to_csv(os.path.join(CONFIG["data"]["results_dir"], "infeasibility.csv"), index=False)

print(f"\nTrajanje: {elapsed:.0f}s")
print(f"Infeasible dana: {len(infeasibility_df)} od {df['date'].nunique()} ({100*len(infeasibility_df)/df['date'].nunique():.1f}%)")

# Evaluacija
print("\nEvaluacija...")
hex_df["date"] = hex_df["date"].astype(str)
daily = evaluate_daily(df_out, hex_df)
daily.to_csv(os.path.join(CONFIG["data"]["results_dir"], "daily_detail_method1.csv"), index=False)

print("\n" + "=" * 60)
print("REZULTATI (Greedy k-means, prosinac 2025)")
print("=" * 60)
print(f"\nOverlap/dan (km2):  {daily['overlap_km2'].mean():.2f} +/- {daily['overlap_km2'].std():.2f}")
print(f"Mean A/N (m2/pak):  {daily['mean_a_n'].mean():.0f} +/- {daily['mean_a_n'].std():.0f}")
print(f"Std A/N:            {daily['std_a_n'].mean():.1f}")
print(f"Outside/dan:        {daily['outside'].mean():.2f} +/- {daily['outside'].std():.2f}")
print(f"Outside %:          {100*daily['outside'].sum()/daily['n_parcels'].sum():.2f}%")

print("\nGOTOVO.")
