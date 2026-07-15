"""
usporedi_rute_haversine.py

Za svaki (kurir, dan) uspoređuje:
1. Stvarnu rutu (redoslijed iz EventDatetime timestampova)
2. Optimalnu rutu (TSP nearest-neighbor heuristika)

Koristi Haversine (crow-flies) udaljenosti — ne treba internet ni OSMnx graf.
Omjer stvarna/optimalna ostaje metodološki valjan za usporedbu efikasnosti.

Pokreni: python3 skripte/usporedi_rute_haversine.py
"""
import numpy as np
import pandas as pd
import yaml
import os

with open(os.path.join(os.path.dirname(__file__), "..", "config.yaml")) as f:
    config = yaml.safe_load(f)


INPUT_FILE = config["data"]["input_file"]
OUTPUT_FILE = "results_dir/route_comparison_haversine.csv"

def haversine_km(lat1, lon1, lat2, lon2):
    """Udaljenost između dvije GPS točke u km (crow-flies)."""
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def route_length_km(lats, lons):
    """Ukupna duljina rute za zadani redoslijed točaka."""
    total = 0.0
    for i in range(len(lats) - 1):
        total += haversine_km(lats[i], lons[i], lats[i+1], lons[i+1])
    return total

def nearest_neighbor_tsp(lats, lons):
    """Nearest-neighbor TSP heuristika. Vraća optimirani redoslijed indeksa."""
    n = len(lats)
    if n <= 2:
        return list(range(n))
    visited = [False] * n
    order = [0]
    visited[0] = True
    for _ in range(n - 1):
        curr = order[-1]
        best_d, best_j = float("inf"), -1
        for j in range(n):
            if not visited[j]:
                d = haversine_km(lats[curr], lons[curr], lats[j], lons[j])
                if d < best_d:
                    best_d, best_j = d, j
        order.append(best_j)
        visited[best_j] = True
    return order

# ── Ucitaj podatke ──
print("Ucitavam podatke...")
df = pd.read_csv(INPUT_FILE, encoding="utf-8")
df["EventDatetime"] = pd.to_datetime(df["EventDatetime"])
df["date"] = df["EventDatetime"].dt.date
df = df.sort_values(["UserID", "date", "EventDatetime"])
print(f"  Paketa: {len(df):,}, Kurira: {df['UserID'].nunique()}, Dana: {df['date'].nunique()}")

# ── Glavna petlja ──
print("\nRacunam rute...")
results = []
groups = list(df.groupby(["UserID", "date"]))

for idx, ((uid, date), day_df) in enumerate(groups):
    lats = day_df["EventGeoY"].values
    lons = day_df["EventGeoX"].values
    n = len(lats)
    if n < 2:
        continue

    # Stvarna ruta (sortirano po EventDatetime)
    actual_km = route_length_km(lats, lons)

    # Optimalna ruta (TSP nearest-neighbor)
    opt_order = nearest_neighbor_tsp(lats, lons)
    optimal_km = route_length_km(lats[opt_order], lons[opt_order])

    saving_km  = actual_km - optimal_km
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

# ── Rezultati ──
results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_FILE, index=False)

print("\n" + "=" * 60)
print("REZULTATI USPOREDBE RUTA (Haversine crow-flies)")
print("=" * 60)
print(f"Parova (kurir, dan): {len(results_df)}")

print("\n--- Stvarna ruta (km/dan) ---")
print(results_df["actual_km"].describe().round(2))

print("\n--- Optimalna ruta / NN-TSP (km/dan) ---")
print(results_df["optimal_km"].describe().round(2))

print("\n--- Ušteda (km/dan) ---")
print(results_df["saving_km"].describe().round(2))

print("\n--- Ušteda (%) ---")
print(results_df["saving_pct"].describe().round(1))

print(f"\nProsjecna ušteda: {results_df['saving_km'].mean():.1f} km/dan "
      f"({results_df['saving_pct'].mean():.1f}%)")
print(f"Ukupna ušteda kroz sve parove: {results_df['saving_km'].sum():.0f} km")

print(f"\nNapomena: udaljenosti su crow-flies (Haversine), ne cestovne.")
print(f"  Cestovne su tipično 1.2-1.4x vece (urban road factor).")
print(f"  Omjer stvarna/optimalna ostaje metodoloski valjan za usporedbu.")
print(f"\nSpremljeno: {OUTPUT_FILE}")
print("\nGOTOVO.")
