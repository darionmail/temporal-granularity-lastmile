"""
04_weighted_voronoi.py

Method 3: Weighted Voronoi (Power Diagram) (Section 5.3 rada)

Pocevsi od centroid-based seed pozicija, parcele se dodjeljuju prema power distance
(Euclidean distance minus weight). Weights are iteratively updated using an additive scheme
proporcionalnom kapacitetskoj devijaciji na svakom sajtu.

Tuned varijanta: smanjeni learning rate (0.1x baseline), feedback gustoce (A/N) u
azuriranju tezina, i L2 regularizacija da se sprijeci divergencija tezina.

Run:
  python3 04_weighted_voronoi.py            # baseline
  python3 04_weighted_voronoi.py --tuned    # tuned varijanta
"""
import sys
import os
import argparse
import yaml
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "common"))
from hex_utils import snap_cover_cap, hex_area

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yaml")


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def power_distance(points_xy, center, weight):
    """Power distance = euclidean_distance^2 - weight (standardna definicija za power diagrams)."""
    d2 = ((points_xy - center) ** 2).sum(axis=1)
    return d2 - weight


def assign_by_power_distance(points_xy, centers, weights, courier_ids):
    """Dodjeljuje svaku tocku kuriru s minimalnom power distance."""
    n_points = len(points_xy)
    n_couriers = len(courier_ids)

    power_dists = np.zeros((n_points, n_couriers))
    for j, cid in enumerate(courier_ids):
        power_dists[:, j] = power_distance(points_xy, centers[cid], weights[cid])

    best_courier_idx = np.argmin(power_dists, axis=1)
    assignment = np.array([courier_ids[i] for i in best_courier_idx])
    return assignment


def weighted_voronoi_iterate(points_xy, courier_ids, n_min, n_max, tuned, learning_rate,
                              l2_reg, max_iter, convergence_tol, seed=42):
    """
    Iterativni weighted Voronoi / power diagram solver.

    Pocetne pozicije: nasumicni odabir tocaka kao seedovi.
    Weights: start at 0, updated additively based on capacity deviation:
        w_u <- w_u + lr * (N_u - target_N)
    gdje je target_N = (n_min+n_max)/2 ili stvarni prosjek.

    Tuned varijanta dodaje density feedback i L2 regularizaciju: w_u <- (1-l2_reg)*w_u + ...
    """
    rng = np.random.default_rng(seed)
    n_points = len(points_xy)
    k = len(courier_ids)

    if n_points == 0:
        return np.array([]), {cid: np.array([0.0, 0.0]) for cid in courier_ids}, {cid: 0.0 for cid in courier_ids}

    init_idx = rng.choice(n_points, size=min(k, n_points), replace=False)
    centers = {courier_ids[i]: points_xy[init_idx[i]] for i in range(len(init_idx))}
    for cid in courier_ids:
        if cid not in centers:
            centers[cid] = points_xy[rng.integers(0, n_points)]

    weights = {cid: 0.0 for cid in courier_ids}
    target_n = (n_min + n_max) / 2.0

    assignment = np.array(courier_ids)[rng.integers(0, k, size=n_points)]

    for iteration in range(max_iter):
        assignment = assign_by_power_distance(points_xy, centers, weights, courier_ids)

        counts = {cid: (assignment == cid).sum() for cid in courier_ids}
        max_deviation = max(abs(counts[cid] - target_n) for cid in courier_ids)

        if max_deviation < convergence_tol:
            break

        # Update centers (centroid svake regije)
        for cid in courier_ids:
            mask = assignment == cid
            if mask.sum() > 0:
                centers[cid] = points_xy[mask].mean(axis=0)

        # Update weights
        for cid in courier_ids:
            deviation = counts[cid] - target_n

            if tuned:
                n_u = counts[cid]
                density_feedback = deviation / max(n_u, 1)
                weights[cid] = (1 - l2_reg) * weights[cid] + learning_rate * (deviation + density_feedback)
            else:
                weights[cid] = weights[cid] + learning_rate * deviation

    return assignment, centers, weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tuned", action="store_true", help="Koristi tuned varijantu (reduced LR + regularizacija)")
    args = parser.parse_args()

    config = load_config()
    df = pd.read_csv(config["data"]["input_file"], encoding="utf-8")
    df["date"] = pd.to_datetime(df["EventDatetime"]).dt.date

    n_min = config["capacity"]["n_min"]
    n_max = config["capacity"]["n_max"]
    hex_config = config["hexagon"]
    wv_config = config["weighted_voronoi"]

    all_couriers = sorted(df["UserID"].unique())

    label = "TUNED (stabilizacija)" if args.tuned else "BASELINE"
    print("=" * 60)
    print(f"METODA 3: WEIGHTED VORONOI / POWER DIAGRAM - {label}")
    print("=" * 60)

    learning_rate = wv_config["learning_rate_tuned"] if args.tuned else wv_config["learning_rate_baseline"]

    df["assigned_courier"] = None

    for date, day_df in df.groupby("date"):
        points_xy = day_df[["x_m", "y_m"]].values
        active_couriers = sorted(day_df["UserID"].unique())

        assignment, centers, weights = weighted_voronoi_iterate(
            points_xy, active_couriers, n_min, n_max,
            tuned=args.tuned,
            learning_rate=learning_rate,
            l2_reg=wv_config["l2_regularization"],
            max_iter=wv_config["max_iterations"],
            convergence_tol=wv_config["convergence_tol"],
        )

        df.loc[day_df.index, "assigned_courier"] = assignment

    print(f"\nDodijeljeno {(df['assigned_courier'].notna()).sum():,} od {len(df):,} parcels.")

    # Konstrukcija heksagona PO DANU (consistent s metodologijom rada - dnevni teritoriji)
    print("\nKonstrukcija heksagonalnih teritorija (snap-cover-cap) PO DANU...")
    all_hex_rows = []
    for date, day_df in df.groupby("date"):
        active_couriers = sorted(day_df["UserID"].unique())
        for cid in active_couriers:
            courier_points = day_df.loc[day_df["assigned_courier"] == cid, ["x_m", "y_m"]].values
            if len(courier_points) == 0:
                continue
            center = courier_points.mean(axis=0)
            x_snap, y_snap, s, poly = snap_cover_cap(
                center[0], center[1], courier_points,
                lattice_s=hex_config["base_lattice_km"] * 1000,
                s_base=hex_config["s_base_m"],
                s_max=hex_config["s_max_m"],
                s_floor_frac=hex_config["s_floor_frac"],
            )
            all_hex_rows.append({
                "date": date, "UserID": cid,
                "center_x": x_snap, "center_y": y_snap,
                "side_length_m": s, "area_m2": hex_area(s),
                "polygon_wkt": poly.wkt,
            })

    hex_df = pd.DataFrame(all_hex_rows)
    print(f"Total heksagona (kurir x dan): {len(hex_df):,}")

    suffix = "tuned" if args.tuned else "baseline"
    os.makedirs(config["data"]["results_dir"], exist_ok=True)
    out_assignments = os.path.join(config["data"]["results_dir"], f"method3_voronoi_{suffix}_assignments.csv")
    df.to_csv(out_assignments, index=False)

    out_hexagons = os.path.join(config["data"]["results_dir"], f"method3_voronoi_{suffix}_hexagons.csv")
    hex_df.to_csv(out_hexagons, index=False)

    print(f"\nResults spremljeni:")
    print(f"  {out_assignments}")
    print(f"  {out_hexagons}")
    print("\nDONE.")


if __name__ == "__main__":
    main()
