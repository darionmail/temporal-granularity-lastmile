"""
Visualization heksagonalnih teritorija za jedan reprezentativni dan.

Odabire dan najblizi medijalnom dnevnom volumeu (izbjegava ekstreme),
te crta teritorije + dodijeljene pakete za sve tri glavne metode
(Greedy k-means, MCF baseline, Weighted Voronoi baseline) radi vizualne usporedbe.

Sprema PNG fajlove u results/ folder.

Run: python3 vizualiziraj_teritorije.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
import yaml
import os

with open(os.path.join(os.path.dirname(__file__), "..", "config.yaml")) as f:
    config = yaml.safe_load(f)

matplotlib.use("Agg")  # bez GUI-a, samo file output
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from shapely import wkt

RESULTS_DIR = config["data"]["results_dir"]
OUT_DIR = RESULTS_DIR

METHODS = {
    "Greedy k-means": ("method1_assignments.csv", "method1_hexagons.csv"),
    "Min-Cost Flow": ("method2_mcf_baseline_assignments.csv", "method2_mcf_baseline_hexagons.csv"),
    "Weighted Voronoi": ("method3_voronoi_baseline_assignments.csv", "method3_voronoi_baseline_hexagons.csv"),
}

# Odaberi reprezentativni dan iz Method 1 (najblizi medijalnom volumeu)
m1_assign = pd.read_csv(os.path.join(RESULTS_DIR, "method1_assignments.csv"))
daily_volume = m1_assign.groupby("date").size()
median_vol = daily_volume.median()
representative_day = (daily_volume - median_vol).abs().idxmin()
print(f"Representative day: {representative_day} (volume={daily_volume[representative_day]}, median={median_vol:.0f})")

# Paleta boja za kurire (konzistentna kroz metode)
all_couriers = sorted(m1_assign["UserID"].dropna().unique())
cmap = plt.colormaps["tab20"].resampled(len(all_couriers))
courier_colors = {cid: cmap(i) for i, cid in enumerate(all_couriers)}


def plot_method(method_name, assign_file, hex_file, ax):
    assign = pd.read_csv(os.path.join(RESULTS_DIR, assign_file))
    hexes = pd.read_csv(os.path.join(RESULTS_DIR, hex_file))

    day_assign = assign[assign["date"] == representative_day]
    day_hexes = hexes[hexes["date"] == representative_day]

    # Draw hexagons
    for _, hrow in day_hexes.iterrows():
        cid = hrow["UserID"]
        poly = wkt.loads(hrow["polygon_wkt"])
        xs, ys = poly.exterior.xy
        color = courier_colors.get(cid, "gray")
        mpl_poly = MplPolygon(list(zip(xs, ys)), closed=True,
                              facecolor=color, edgecolor=color, alpha=0.18, linewidth=1.5)
        ax.add_patch(mpl_poly)

    # Crtaj pakete (tocke), obojene po kuriru
    for cid in day_assign["assigned_courier"].dropna().unique():
        pts = day_assign[day_assign["assigned_courier"] == cid]
        ax.scatter(pts["x_m"], pts["y_m"], s=6, color=courier_colors.get(cid, "gray"),
                   alpha=0.7, edgecolors="none")

    ax.set_title(f"{method_name}", fontsize=13, fontweight="bold")
    ax.set_xlabel("x (m, from center)", fontsize=9)
    ax.set_ylabel("y (m, from center)", fontsize=9)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)


# Jedna figura s tri panela (jedan po metodi)
fig, axes = plt.subplots(1, 3, figsize=(20, 7))
for ax, (method_name, (af, hf)) in zip(axes, METHODS.items()):
    plot_method(method_name, af, hf, ax)

fig.suptitle(f"Hexagonal Territories - Representative Day {representative_day} "
             f"(Zagreb West distribution area, {daily_volume[representative_day]} parcels, 20 couriers)",
             fontsize=15, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.96])

out_path = os.path.join(OUT_DIR, "teritoriji_comparison.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")

# Zasebna, veca figura samo za Greedy k-means (primarna metoda)
fig2, ax2 = plt.subplots(1, 1, figsize=(11, 10))
plot_method("Greedy k-means", *METHODS["Greedy k-means"], ax2)
ax2.set_title(f"Hexagonal Territories - Greedy k-means\n"
              f"Representative Day {representative_day}, Zagreb West distribution area "
              f"({daily_volume[representative_day]} parcels, 20 couriers)",
              fontsize=13, fontweight="bold")
out_path2 = os.path.join(OUT_DIR, "teritoriji_greedy_kmeans.png")
plt.savefig(out_path2, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path2}")

print("\nDONE.")
