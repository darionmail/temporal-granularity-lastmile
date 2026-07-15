"""
usporedi_rute.py

Za svaki (kurir, dan) uspoređuje:
1. Stvarnu rutu (redoslijed iz EventDatetime timestampova)
2. Optimalnu rutu (TSP nearest-neighbor heuristika)

Koristeći OSMnx cestovni graf za Zagreb.

Run: python3 skripte/usporedi_rute.py
"""
import os
import sys
import numpy as np
import pandas as pd
import osmnx as ox
import networkx as nx
import warnings
import yaml
import os

with open(os.path.join(os.path.dirname(__file__), "..", "config.yaml")) as f:
    config = yaml.safe_load(f)

warnings.filterwarnings("ignore")

INPUT_FILE = config["data"]["input_file"]
OUTPUT_FILE = "results_dir/route_comparison.csv"
GRAPH_CACHE = os.path.join(os.path.dirname(config["data"]["results_dir"]), "road_network_graph.graphml")

# ── 1. Ucitaj podatke ──
print("Loading data...")
df = pd.read_csv(INPUT_FILE, encoding="utf-8")
df["EventDatetime"] = pd.to_datetime(df["EventDatetime"])
df["date"] = df["EventDatetime"].dt.date
df = df.sort_values(["UserID", "date", "EventDatetime"])

print(f"  Paketa: {len(df):,}, Kurira: {df['UserID'].nunique()}, Dana: {df['date'].nunique()}")

# ── 2. OSMnx graf (cache lokalno) ──
if os.path.exists(GRAPH_CACHE):
    print("Loading cached cestovni graf...")
    G = ox.load_graphml(GRAPH_CACHE)
else:
    print("Preuzimam cestovni graf za Zagreb zapad (prvi put, par minuta)...")
    # Bbox koji pokriva sve koordinate u datasetu
    north = df["EventGeoY"].max() + 0.02
    south = df["EventGeoY"].min() - 0.02
    east  = df["EventGeoX"].max() + 0.02
    west  = df["EventGeoX"].min() - 0.02
    G = ox.graph_from_bbox((north, south, east, west), network_type="drive")
    ox.save_graphml(G, GRAPH_CACHE)
    print(f"  Graf spremljen: {G.number_of_nodes()} cvorova, {G.number_of_edges()} bridova")

print(f"Graf: {G.number_of_nodes()} cvorova, {G.number_of_edges()} bridova")

# ── 3. Helper: ukupna duljina rute po redoslijedu točaka ──
def route_length_km(G, lats, lons):
    """
    Izračunava ukupnu duljinu rute (km) za zadani redoslijed točaka.
    Koristi OSMnx shortest_path između uzastopnih točaka.
    """
    if len(lats) < 2:
        return 0.0

    nodes = ox.nearest_nodes(G, lons, lats)
    total_m = 0.0
    for i in range(len(nodes) - 1):
        try:
            path = nx.shortest_path(G, nodes[i], nodes[i+1], weight="length")
            # OSMnx 2.x: suma duljina bridova duž puta
            total_m += sum(
                G[path[j]][path[j+1]][0].get("length", 0)
                for j in range(len(path) - 1)
            )
        except nx.NetworkXNoPath:
            # Fallback: Haversine
            R = 6371000
            lat1, lon1 = np.radians(lats[i]), np.radians(lons[i])
            lat2, lon2 = np.radians(lats[i+1]), np.radians(lons[i+1])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
            total_m += 2 * 6371000 * np.arcsin(np.sqrt(a))

    return total_m / 1000.0


# ── 4. TSP nearest-neighbor heuristika ──
def nearest_neighbor_tsp(lats, lons):
    """
    Nearest-neighbor TSP heuristika.
    Počinje od prve točke (depot/distribucijsko središte),
    uvijek ide na najbližu neposjećenu točku.
    Vraća optimirani redoslijed indeksa.
    """
    n = len(lats)
    if n <= 2:
        return list(range(n))

    coords = list(zip(lats, lons))
    visited = [False] * n
    order = [0]
    visited[0] = True

    for _ in range(n - 1):
        current = order[-1]
        best_dist = float("inf")
        best_idx = -1
        lat_c, lon_c = coords[current]
        for j in range(n):
            if not visited[j]:
                # Haversine approximation (brzo, bez grafa)
                dlat = np.radians(coords[j][0] - lat_c)
                dlon = np.radians(coords[j][1] - lon_c)
                a = np.sin(dlat/2)**2 + np.cos(np.radians(lat_c)) * \
                    np.cos(np.radians(coords[j][0])) * np.sin(dlon/2)**2
                d = 2 * 6371 * np.arcsin(np.sqrt(a))
                if d < best_dist:
                    best_dist = d
                    best_idx = j
        order.append(best_idx)
        visited[best_idx] = True

    return order


# ── 5. Glavna petlja ──
print("\nComputing rute po (kurir, dan)...")
results = []

groups = list(df.groupby(["UserID", "date"]))
total = len(groups)

for idx, ((uid, date), day_df) in enumerate(groups):
    if idx % 50 == 0:
        print(f"  {idx}/{total} ({100*idx//total}%)...")

    lats = day_df["EventGeoY"].values
    lons = day_df["EventGeoX"].values
    n = len(lats)

    if n < 2:
        continue

    # Stvarna ruta (već sortirana po EventDatetime)
    actual_km = route_length_km(G, lats, lons)

    # Optimalna ruta (TSP nearest-neighbor)
    opt_order = nearest_neighbor_tsp(lats, lons)
    opt_lats = lats[opt_order]
    opt_lons = lons[opt_order]
    optimal_km = route_length_km(G, opt_lats, opt_lons)

    # Ušteda
    saving_km = actual_km - optimal_km
    saving_pct = 100 * saving_km / actual_km if actual_km > 0 else 0.0

    results.append({
        "UserID": uid,
        "date": date,
        "n_parcels": n,
        "actual_km": round(actual_km, 3),
        "optimal_km": round(optimal_km, 3),
        "saving_km": round(saving_km, 3),
        "saving_pct": round(saving_pct, 1),
    })

# ── 6. Results ──
results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_FILE, index=False)

print("\n" + "=" * 60)
print("REZULTATI USPOREDBE RUTA")
print("=" * 60)
print(f"\nNumber (kurir, dan) parova: {len(results_df)}")
print(f"\n--- Stvarna ruta (km/dan) ---")
print(results_df["actual_km"].describe().round(2))
print(f"\n--- Optimalna ruta (km/dan) ---")
print(results_df["optimal_km"].describe().round(2))
print(f"\n--- Ušteda (km/dan) ---")
print(results_df["saving_km"].describe().round(2))
print(f"\n--- Ušteda (%) ---")
print(results_df["saving_pct"].describe().round(1))
print(f"\nUkupna ušteda kroz sve (kurir,dan) parove: {results_df['saving_km'].sum():.1f} km")
print(f"Prosjecna ušteda po danu po kuriru: {results_df['saving_km'].mean():.1f} km ({results_df['saving_pct'].mean():.1f}%)")

print(f"\nSaved: {OUTPUT_FILE}")
print("\nDONE.")
