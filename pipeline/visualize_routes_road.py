"""
vizualiziraj_rute_cestovne.py

Vizualizacija rezultata usporedbe stvarnih i optimalnih ruta
koristeći stvarne cestovne udaljenosti (OSMnx).

Pokreni: python3 skripte/vizualiziraj_rute_cestovne.py
"""
import numpy as np
import pandas as pd
import matplotlib
import yaml
import os

with open(os.path.join(os.path.dirname(__file__), "..", "config.yaml")) as f:
    config = yaml.safe_load(f)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

INPUT  = "results_dir/route_comparison.csv"
OUT1   = "results_dir/road_saving_histogram.png"
OUT2   = "results_dir/road_saving_by_courier.png"
OUT3   = "results_dir/road_actual_vs_optimal.png"
OUT4   = "results_dir/road_vs_haversine_comparison.png"

df = pd.read_csv(INPUT)
df["UserID"] = df["UserID"].astype(int)

# ── Figure 1: Histogram uštede (%) — cestovne ──
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(df["saving_pct"], bins=30, color="#2E5C8A", edgecolor="white", alpha=0.85)
ax.axvline(df["saving_pct"].mean(), color="#E24B4A", linewidth=2,
           linestyle="--", label=f"Mean = {df['saving_pct'].mean():.1f}%")
ax.axvline(df["saving_pct"].median(), color="#1D9E75", linewidth=2,
           linestyle=":", label=f"Median = {df['saving_pct'].median():.1f}%")
ax.axvline(0, color="black", linewidth=1, linestyle=":", alpha=0.5)
ax.axvspan(df["saving_pct"].min(), 0, alpha=0.08, color="#E24B4A")
ax.set_xlabel("Route length saving (%)\n(actual − optimal) / actual × 100", fontsize=12)
ax.set_ylabel("Number of (courier, day) pairs", fontsize=12)
ax.set_title("Distribution of Route Efficiency Gains\n"
             "Nearest-Neighbor TSP vs. Actual Delivery Sequence\n"
             "Zagreb West distribution area, October 2025 — Road Network Distances (OSMnx)", fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
note = (f"n={len(df)}  |  Mean={df['saving_pct'].mean():.1f}%  |  "
        f"Median={df['saving_pct'].median():.1f}%  |  Std={df['saving_pct'].std():.1f}%  |  "
        f"Total saving: {df['saving_km'].sum():.0f} km")
ax.text(0.5, -0.13, note, transform=ax.transAxes,
        ha="center", fontsize=10, color="#595959")
plt.tight_layout()
plt.savefig(OUT1, dpi=150, bbox_inches="tight")
plt.close()
print(f"Spremljeno: {OUT1}")

# ── Figure 2: Box plot po kuriru ──
fig, ax = plt.subplots(figsize=(14, 6))
couriers = sorted(df["UserID"].unique())
data_by_courier = [df[df["UserID"] == c]["saving_pct"].values for c in couriers]
bp = ax.boxplot(data_by_courier, patch_artist=True,
                medianprops=dict(color="#E24B4A", linewidth=2))
for patch in bp["boxes"]:
    patch.set_facecolor("#9FE1CB")
    patch.set_alpha(0.7)
ax.axhline(0, color="black", linewidth=1, linestyle=":", alpha=0.5)
ax.axhline(df["saving_pct"].mean(), color="#2E5C8A", linewidth=1.5,
           linestyle="--", alpha=0.7, label=f"Overall mean ({df['saving_pct'].mean():.1f}%)")
ax.set_xticks(range(1, len(couriers)+1))
ax.set_xticklabels([str(c) for c in couriers], rotation=45, fontsize=9)
ax.set_xlabel("Courier ID", fontsize=12)
ax.set_ylabel("Route length saving (%)", fontsize=12)
ax.set_title("Route Efficiency Gain by Courier — Road Network Distances\n"
             "Zagreb West distribution area, October 2025", fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(OUT2, dpi=150, bbox_inches="tight")
plt.close()
print(f"Spremljeno: {OUT2}")

# ── Figure 3: Scatter actual vs optimal ──
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(df["actual_km"], df["optimal_km"], alpha=0.4,
           color="#2E5C8A", s=25, edgecolors="none")
lim = max(df["actual_km"].max(), df["optimal_km"].max()) * 1.05
ax.plot([0, lim], [0, lim], "k--", linewidth=1, alpha=0.5, label="actual = optimal")
ax.set_xlabel("Actual route length (km, road network)", fontsize=12)
ax.set_ylabel("Optimal route length / NN-TSP (km, road network)", fontsize=12)
ax.set_title("Actual vs. Optimal Route Length — Road Network\n"
             "Each point = one (courier, day) pair", fontsize=13)
ax.set_xlim(0, lim)
ax.set_ylim(0, lim)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
below = (df["optimal_km"] < df["actual_km"]).sum()
above = (df["optimal_km"] >= df["actual_km"]).sum()
ax.text(0.05, 0.93,
        f"Points below diagonal: {below} ({100*below/len(df):.0f}%)\n"
        f"  → optimal < actual (NN-TSP saves distance)\n"
        f"Points above diagonal: {above} ({100*above/len(df):.0f}%)\n"
        f"  → optimal > actual (courier beats NN-TSP)",
        transform=ax.transAxes, fontsize=10, verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
plt.tight_layout()
plt.savefig(OUT3, dpi=150, bbox_inches="tight")
plt.close()
print(f"Spremljeno: {OUT3}")

# ── Figure 4: Usporedba Haversine vs Cestovne udaljenosti ──
# Ucitaj i Haversine rezultate za usporedbu
try:
    df_h = pd.read_csv("results_dir/route_comparison_haversine.csv")
    df_h["UserID"] = df_h["UserID"].astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Lijevo: distribucija uštede %
    axes[0].hist(df_h["saving_pct"], bins=25, alpha=0.6,
                 color="#E24B4A", label=f"Haversine (mean={df_h['saving_pct'].mean():.1f}%)",
                 edgecolor="white")
    axes[0].hist(df["saving_pct"], bins=25, alpha=0.6,
                 color="#2E5C8A", label=f"Road network (mean={df['saving_pct'].mean():.1f}%)",
                 edgecolor="white")
    axes[0].axvline(0, color="black", linewidth=1, linestyle=":", alpha=0.5)
    axes[0].set_xlabel("Route length saving (%)", fontsize=12)
    axes[0].set_ylabel("Number of (courier, day) pairs", fontsize=12)
    axes[0].set_title("Saving Distribution: Haversine vs. Road Network", fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Desno: scatter Haversine vs cestovne udaljenosti (actual)
    common = df.merge(df_h, on=["UserID", "date"], suffixes=("_road", "_hav"))
    axes[1].scatter(common["actual_km_hav"], common["actual_km_road"],
                    alpha=0.4, color="#534AB7", s=20, edgecolors="none")
    lim2 = max(common["actual_km_hav"].max(), common["actual_km_road"].max()) * 1.05
    axes[1].plot([0, lim2], [0, lim2*1.3], "k--", linewidth=1, alpha=0.4,
                 label="Road = 1.3 × Haversine")
    axes[1].set_xlabel("Actual route (km, Haversine)", fontsize=12)
    axes[1].set_ylabel("Actual route (km, road network)", fontsize=12)
    axes[1].set_title("Haversine vs. Road Network Distances\n(actual routes)", fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    # Izracunaj stvarni road factor
    road_factor = (common["actual_km_road"] / common["actual_km_hav"]).median()
    axes[1].text(0.05, 0.93, f"Median road factor: {road_factor:.2f}×",
                 transform=axes[1].transAxes, fontsize=11,
                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    fig.suptitle("Haversine vs. Road Network Route Analysis\n"
                 "Zagreb West distribution area, October 2025", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT4, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Spremljeno: {OUT4}")
    print(f"\nStvarni road factor (medijan): {road_factor:.2f}x")
except FileNotFoundError:
    print("Haversine rezultati nisu pronađeni, preskačem Figure 4")

print("\nGOTOVO.")
