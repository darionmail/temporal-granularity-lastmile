"""
02_greedy_kmeans_hex.py (v2 - PO DANU)

Metoda 1: Capacitated k-Means with Hexagonal Optimization (Sekcija 5.1 rada)
RESTRUKTURIRANO: svih 5 stadija se izvode PO DANU (konzistentno s originalnim
radom koji je radio 5 dnevnih snapshotova), umjesto agregacije kroz cijeli mjesec.

Za svaki dan:
  Stage 1: Banded Assignment (Greedy CKM) na tockama TOG dana
  Stage 2: (preskoceno - nema stabilizacije kroz dane jer je svaki dan zaseban)
  Stage 3: Snap-Cover-Cap Hexagon Construction (na temelju centroida TOG dana)
  Stage 4: Polygon-Aware Post-Reallocation (unutar TOG dana)
  Stage 5: Overlap and Density Refinement (unutar TOG dana)

Finalni rezultat: 20 kurira x 27 dana = do 540 dnevnih heksagona.
Assignments CSV ima sve pakete s kolonom 'assigned_courier' (po danu).
Hexagons CSV ima jedan red po (kurir, dan) kombinaciji.

Pokreni: python3 02_greedy_kmeans_hex.py
"""
import sys
import os
import yaml
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "common"))
from hex_utils import snap_cover_cap, get_neighbor_centers, hex_area

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yaml")


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


# ============================================================
# STAGE 1: Daily Banded Assignment (Greedy Capacitated K-Means)
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


# ============================================================
# STAGE 3: Snap-Cover-Cap Hexagon Construction (po danu)
# ============================================================

def build_day_hexagons(day_points_xy, assignment, centroids, courier_ids, hex_config):
    hexagons = {}
    for cid in courier_ids:
        mask = assignment == cid
        courier_points = day_points_xy[mask]
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


# ============================================================
# STAGE 4: Polygon-Aware Post-Reallocation (po danu)
# ============================================================

def stage4_polygon_reallocation_day(day_points_xy, assignment, hexagons, courier_ids):
    from shapely.geometry import Point
    new_assignment = assignment.copy()

    for i in range(len(day_points_xy)):
        current_cid = assignment[i]
        if current_cid not in hexagons:
            continue
        point_x, point_y = day_points_xy[i]
        pt = Point(point_x, point_y)
        current_poly = hexagons[current_cid]["polygon"]

        if current_poly.contains(pt) or current_poly.touches(pt):
            continue

        containing = [cid for cid in hexagons if hexagons[cid]["polygon"].contains(pt)]

        if len(containing) == 0:
            continue
        elif len(containing) == 1:
            new_assignment[i] = containing[0]
        else:
            dists = [np.linalg.norm([point_x - hexagons[cid]["center"][0],
                                      point_y - hexagons[cid]["center"][1]]) for cid in containing]
            new_assignment[i] = containing[np.argmin(dists)]

    return new_assignment


# ============================================================
# STAGE 5: Overlap and Density Refinement (po danu)
# ============================================================

def compute_score(cid, candidate_poly, candidate_s, hexagons, day_points_xy, assignment, alpha, lambda_dens, lambda_eq, delta):
    from shapely.geometry import Point
    overlap_sum = 0.0
    for other_cid, other_hex in hexagons.items():
        if other_cid == cid:
            continue
        if candidate_poly.intersects(other_hex["polygon"]):
            overlap_sum += candidate_poly.intersection(other_hex["polygon"]).area

    mask = assignment == cid
    own_points = day_points_xy[mask]
    outside_count = sum(1 for px, py in own_points if not candidate_poly.contains(Point(px, py)))

    n_u = len(own_points)
    a_u = hex_area(candidate_s)
    density = a_u / n_u if n_u > 0 else a_u

    return overlap_sum + alpha * outside_count + lambda_dens * density + lambda_eq * (density - delta) ** 2


def stage5_refinement_day(day_points_xy, assignment, hexagons, hex_config, refine_config):
    hexagons = {cid: dict(h) for cid, h in hexagons.items()}
    alpha = refine_config["alpha"]
    lambda_dens = refine_config["lambda_dens"]
    lambda_eq = refine_config["lambda_eq"]
    max_iter = refine_config["max_iterations"]
    refine_lattice_m = hex_config["refine_lattice_km"] * 1000

    for iteration in range(max_iter):
        densities = []
        for cid, h in hexagons.items():
            n_u = (assignment == cid).sum()
            if n_u > 0:
                densities.append(hex_area(h["s"]) / n_u)
        delta = np.median(densities) if densities else 0.0

        any_improved = False

        for cid in list(hexagons.keys()):
            current_center = hexagons[cid]["center"]
            current_score = compute_score(
                cid, hexagons[cid]["polygon"], hexagons[cid]["s"], hexagons,
                day_points_xy, assignment, alpha, lambda_dens, lambda_eq, delta
            )

            candidates = get_neighbor_centers(current_center[0], current_center[1], refine_lattice_m)
            best_score = current_score
            best_candidate = None

            mask = assignment == cid
            own_points = day_points_xy[mask]

            for cand_x, cand_y in candidates:
                _, _, cand_s, cand_poly = snap_cover_cap(
                    cand_x, cand_y, own_points,
                    lattice_s=hex_config["base_lattice_km"] * 1000,
                    s_base=hex_config["s_base_m"],
                    s_max=hex_config["s_max_m"],
                    s_floor_frac=hex_config["s_floor_frac"],
                )
                cand_score = compute_score(
                    cid, cand_poly, cand_s, hexagons, day_points_xy, assignment,
                    alpha, lambda_dens, lambda_eq, delta
                )
                if cand_score < best_score:
                    best_score = cand_score
                    best_candidate = (cand_x, cand_y, cand_s, cand_poly)

            if best_candidate is not None:
                hexagons[cid]["center"] = (best_candidate[0], best_candidate[1])
                hexagons[cid]["s"] = best_candidate[2]
                hexagons[cid]["polygon"] = best_candidate[3]
                any_improved = True

        if not any_improved:
            break

    return hexagons


# ============================================================
# MAIN PIPELINE - PO DANU
# ============================================================

def main():
    config = load_config()
    df = pd.read_csv(config["data"]["input_file"], encoding="utf-8")
    df["date"] = pd.to_datetime(df["EventDatetime"]).dt.date

    n_min = config["capacity"]["n_min"]
    n_max = config["capacity"]["n_max"]
    hex_config = config["hexagon"]
    refine_config = config["refinement"]

    print("=" * 60)
    print("METODA 1: GREEDY CAPACITATED K-MEANS + HEX OPTIMIZACIJA")
    print("(restrukturirano - optimizacija PO DANU)")
    print("=" * 60)

    df["assigned_courier"] = None
    all_hex_rows = []
    infeasibility_log = []

    n_days = df["date"].nunique()
    day_counter = 0

    for date, day_df in df.groupby("date"):
        day_counter += 1
        points_xy = day_df[["x_m", "y_m"]].values
        active_couriers = sorted(day_df["UserID"].unique())
        n_points = len(points_xy)
        k = len(active_couriers)

        min_total_needed = k * n_min
        max_total_allowed = k * n_max
        day_feasible = min_total_needed <= n_points <= max_total_allowed
        if not day_feasible:
            reason = "premalo paketa" if n_points < min_total_needed else "previse paketa"
            infeasibility_log.append({
                "date": date, "n_points": n_points, "n_active_couriers": k,
                "min_needed": min_total_needed, "max_allowed": max_total_allowed, "reason": reason,
            })

        # Stage 1
        assignment, centroids = greedy_capacitated_kmeans_day(points_xy, active_couriers, n_min, n_max)

        for cid in active_couriers:
            n_assigned = (assignment == cid).sum()
            if n_assigned < n_min or n_assigned > n_max:
                infeasibility_log.append({
                    "date": date, "courier": cid, "n_assigned": n_assigned,
                    "n_min": n_min, "n_max": n_max, "reason": "kurir_izvan_raspona",
                })

        # Stage 3
        day_hexagons = build_day_hexagons(points_xy, assignment, centroids, active_couriers, hex_config)

        # Stage 4
        assignment = stage4_polygon_reallocation_day(points_xy, assignment, day_hexagons, active_couriers)

        # Stage 5
        day_hexagons = stage5_refinement_day(points_xy, assignment, day_hexagons, hex_config, refine_config)

        df.loc[day_df.index, "assigned_courier"] = assignment

        for cid, h in day_hexagons.items():
            all_hex_rows.append({
                "date": date, "UserID": cid,
                "center_x": h["center"][0], "center_y": h["center"][1],
                "side_length_m": h["s"], "area_m2": hex_area(h["s"]),
                "polygon_wkt": h["polygon"].wkt,
            })

        if day_counter % 5 == 0 or day_counter == n_days:
            print(f"  Dan {day_counter}/{n_days} ({date}) obradjen.")

    infeasibility_df = pd.DataFrame(infeasibility_log)
    if len(infeasibility_df) > 0:
        n_days_infeasible = infeasibility_df.loc[
            infeasibility_df["reason"].isin(["premalo paketa", "previse paketa"]), "date"
        ].nunique()
        n_courier_violations = (infeasibility_df["reason"] == "kurir_izvan_raspona").sum()
        print(f"\nUPOZORENJE: {n_days_infeasible} dana je teorijski infeasible za [{n_min},{n_max}] kapacitet.")
        print(f"UPOZORENJE: {n_courier_violations} (kurir, dan) parova zavrsilo izvan [{n_min},{n_max}] raspona.")
    else:
        print(f"\nSvi dani/kuriri su unutar [{n_min},{n_max}] kapaciteta.")

    os.makedirs(config["data"]["results_dir"], exist_ok=True)
    out_assignments = os.path.join(config["data"]["results_dir"], "method1_assignments.csv")
    df.to_csv(out_assignments, index=False)

    hex_df = pd.DataFrame(all_hex_rows)
    out_hexagons = os.path.join(config["data"]["results_dir"], "method1_hexagons.csv")
    hex_df.to_csv(out_hexagons, index=False)

    out_infeasibility = os.path.join(config["data"]["results_dir"], "method1_infeasibility_log.csv")
    infeasibility_df.to_csv(out_infeasibility, index=False)

    print(f"\nDodijeljeno {(df['assigned_courier'].notna()).sum():,} od {len(df):,} paketa.")
    print(f"Ukupno heksagona (kurir x dan): {len(hex_df):,}")
    print(f"\nRezultati spremljeni:")
    print(f"  {out_assignments}")
    print(f"  {out_hexagons}")
    print(f"  {out_infeasibility}")
    print("\nGOTOVO.")


if __name__ == "__main__":
    main()
