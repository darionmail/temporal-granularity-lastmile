"""
03_min_cost_flow.py

Method 2: Min-Cost Flow Assignment (Section 5.2 rada)

Parcele se dodjeljuju kuririma per day koristeci min-cost flow s kapacitetskim
ogranicenjima. Edge costovi = Euclidean distance. Tuned varijanta dodaje mekanu
kvadratnu penalizaciju za dodjele preko 4km (distance_penalty_km).

Nakon dodjele, konvertira se u heksagonalne teritorije koristeci snap-cover-cap
(ista procedura kao Method 1, Stage 3) za usporedivost.

Run:
  python3 03_min_cost_flow.py            # baseline
  python3 03_min_cost_flow.py --tuned    # tuned varijanta s distance penalty
"""
import sys
import os
import argparse
import yaml
import numpy as np
import pandas as pd
import networkx as nx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "common"))
from hex_utils import snap_cover_cap, hex_area

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yaml")

COST_SCALE = 1000  # MCF u networkx zahtijeva integer costove - skaliramo i round-amo


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def solve_day_mcf(points_xy, point_ids, courier_ids, n_min, n_max, tuned=False,
                   penalty_km=4.0, penalty_weight=50.0):
    """
    Rjesava MCF za jedan dan koristeci networkx min_cost_flow.

    Graf: source -> parcel_nodes -> courier_nodes -> sink
    - source -> parcel: capacity=1, cost=0
    - parcel -> courier: capacity=1, cost=round(distance * COST_SCALE) [+ penalty ako tuned]
    - courier -> sink: capacity=[n_min, n_max] -> modelirano kao lower/upper bound

    networkx min_cost_flow ne podrzava direktno lower bounds na lukovima u osnovnoj verziji,
    pa koristimo standardni trik: ako n_min nije strogo nuzan (soft constraint), stavljamo
    samo upper bound (n_max) i tezimo da se sve parcele dodijele (max flow = n_parcels).
    """
    G = nx.DiGraph()

    n_points = len(points_xy)
    n_couriers = len(courier_ids)

    source = "SOURCE"
    sink = "SINK"

    G.add_node(source, demand=-n_points)
    G.add_node(sink, demand=n_points)

    for i, pid in enumerate(point_ids):
        node = f"P_{pid}"
        G.add_node(node, demand=0)
        G.add_edge(source, node, capacity=1, weight=0)

    for cid in courier_ids:
        node = f"C_{cid}"
        G.add_node(node, demand=0)
        # Upper bound kapacitet couriers tog days
        G.add_edge(node, sink, capacity=n_max, weight=0)

    for i, pid in enumerate(point_ids):
        p_node = f"P_{pid}"
        for cid in courier_ids:
            c_node = f"C_{cid}"
            # Distance to courier 'center' will be computed based on
            # trenutne dodjele iz prethodne iteracije; za pocetnu MCF dodjelu koristimo
            # udaljenost do najblize tocke medju vec dodijeljenima (ili globalni centroid)
            pass  # cost se postavlja izvana, vidi solve_day_mcf_with_costs

    return G  # placeholder, stvarna logika je u funkciji ispod


def build_and_solve_mcf(points_xy, point_ids, courier_centers, n_max, tuned, penalty_km, penalty_weight):
    """
    Gradi i rjesava MCF graf gdje su kuririma vec dodijeljeni privremeni centri
    (iz prethodne iteracije ili inicijalizacije), a cost = distanca tocka<->centar couriers.
    """
    G = nx.DiGraph()
    n_points = len(points_xy)
    courier_ids = list(courier_centers.keys())

    source = "SOURCE"
    sink = "SINK"
    G.add_node(source, demand=-n_points)
    G.add_node(sink, demand=n_points)

    for i, pid in enumerate(point_ids):
        node = f"P_{pid}"
        G.add_node(node, demand=0)
        G.add_edge(source, node, capacity=1, weight=0)

    for cid in courier_ids:
        node = f"C_{cid}"
        G.add_node(node, demand=0)
        G.add_edge(node, sink, capacity=int(n_max), weight=0)

    for i, pid in enumerate(point_ids):
        p_node = f"P_{pid}"
        px, py = points_xy[i]
        for cid in courier_ids:
            c_node = f"C_{cid}"
            cx, cy = courier_centers[cid]
            dist_m = np.sqrt((px - cx) ** 2 + (py - cy) ** 2)
            cost = dist_m

            if tuned and dist_m > penalty_km * 1000:
                excess = (dist_m - penalty_km * 1000) / 1000.0  # km preko praga
                cost += penalty_weight * (excess ** 2) * 1000  # skaliraj na metre-ekvivalent

            G.add_edge(p_node, c_node, capacity=1, weight=int(round(cost)))

    try:
        flow_dict = nx.min_cost_flow(G)
    except nx.NetworkXUnfeasible:
        # Kapacitet nedovoljan za sve tocke - povecaj n_max privremeno (soft fallback)
        for cid in courier_ids:
            G[f"C_{cid}"][sink]["capacity"] = n_points  # otvori kapacitet potpuno
        flow_dict = nx.min_cost_flow(G)

    assignment = {}
    for i, pid in enumerate(point_ids):
        p_node = f"P_{pid}"
        for cid in courier_ids:
            c_node = f"C_{cid}"
            if flow_dict.get(p_node, {}).get(c_node, 0) > 0:
                assignment[pid] = cid
                break

    return assignment


def solve_day(day_df, courier_ids, n_min, n_max, tuned, penalty_km, penalty_weight, max_outer_iter=5):
    """
    Iterativni MCF: pocni s centroidima = prosjek svih tocaka podijeljen nasumicno,
    then iteratively update courier centers based on assignment (similar to Lloyd's algorithm,
    ali assignment step je MCF umjesto nearest-neighbor).
    """
    points_xy = day_df[["x_m", "y_m"]].values
    point_ids = day_df["ShipmentItemBarcode"].values

    rng = np.random.default_rng(42)
    n_points = len(points_xy)

    if n_points == 0:
        return {}

    init_idx = rng.choice(n_points, size=min(len(courier_ids), n_points), replace=False)
    courier_centers = {courier_ids[i]: points_xy[init_idx[i]] for i in range(len(init_idx))}
    for cid in courier_ids:
        if cid not in courier_centers:
            courier_centers[cid] = points_xy[rng.integers(0, n_points)]

    assignment = {}
    for outer_iter in range(max_outer_iter):
        assignment = build_and_solve_mcf(points_xy, point_ids, courier_centers, n_max, tuned, penalty_km, penalty_weight)

        # Update center based on new assignment
        new_centers = {}
        for cid in courier_ids:
            assigned_idx = [i for i, pid in enumerate(point_ids) if assignment.get(pid) == cid]
            if assigned_idx:
                new_centers[cid] = points_xy[assigned_idx].mean(axis=0)
            else:
                new_centers[cid] = courier_centers[cid]

        shift = sum(np.linalg.norm(new_centers[cid] - courier_centers[cid]) for cid in courier_ids)
        courier_centers = new_centers

        if shift < 5.0:
            break

    return assignment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tuned", action="store_true", help="Koristi tuned varijantu s distance penalty")
    args = parser.parse_args()

    config = load_config()
    df = pd.read_csv(config["data"]["input_file"], encoding="utf-8")
    df["date"] = pd.to_datetime(df["EventDatetime"]).dt.date

    n_min = config["capacity"]["n_min"]
    n_max = config["capacity"]["n_max"]
    hex_config = config["hexagon"]
    mcf_config = config["mcf"]

    all_couriers = sorted(df["UserID"].unique())

    label = "TUNED (distance penalty)" if args.tuned else "BASELINE"
    print("=" * 60)
    print(f"METODA 2: MIN-COST FLOW - {label}")
    print("=" * 60)

    df["assigned_courier"] = None

    for date, day_df in df.groupby("date"):
        active_couriers = sorted(day_df["UserID"].unique())
        assignment = solve_day(
            day_df, active_couriers, n_min, n_max,
            tuned=args.tuned,
            penalty_km=mcf_config["distance_penalty_km"],
            penalty_weight=mcf_config["penalty_weight"],
        )
        for idx, row in day_df.iterrows():
            df.loc[idx, "assigned_courier"] = assignment.get(row["ShipmentItemBarcode"])

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
    out_assignments = os.path.join(config["data"]["results_dir"], f"method2_mcf_{suffix}_assignments.csv")
    df.to_csv(out_assignments, index=False)

    out_hexagons = os.path.join(config["data"]["results_dir"], f"method2_mcf_{suffix}_hexagons.csv")
    hex_df.to_csv(out_hexagons, index=False)

    print(f"\nResults spremljeni:")
    print(f"  {out_assignments}")
    print(f"  {out_hexagons}")
    print("\nDONE.")


if __name__ == "__main__":
    main()
