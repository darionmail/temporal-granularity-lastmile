"""
Composite score za post-processed rezultate.
Run: python3 skripte/composite_score_pp.py
"""
import os
import numpy as np
import pandas as pd
import yaml
import os

with open(os.path.join(os.path.dirname(__file__), "..", "config.yaml")) as f:
    config = yaml.safe_load(f)


RESULTS_DIR = config["data"]["results_dir"]

df = pd.read_csv(os.path.join(RESULTS_DIR, "evaluation_two_stage.csv"))

# Odvojimo raw i pp
raw = df[~df["postprocessed"]].set_index("method")
pp  = df[df["postprocessed"]].set_index("method")

print("=" * 60)
print("COMPOSITE SCORE — POST-PROCESSED VARIJANTA")
print("=" * 60)

metrics = ["overlap_pct", "outside_pct", "mean_a_n"]
labels  = ["Overlap%", "Outside%", "Mean A/N"]

# Normalize to [0,1] within pp set
norm = pd.DataFrame(index=pp.index)
for m in metrics:
    mn, mx = pp[m].min(), pp[m].max()
    norm[m] = (pp[m] - mn) / (mx - mn) if mx > mn else 0.0

# Varijanta A: jednake tezine
norm["score_equal"] = norm[metrics].mean(axis=1)

# Varijanta B: outside prioritet
norm["score_outside"] = (
    0.25 * norm["overlap_pct"] +
    0.50 * norm["outside_pct"] +
    0.25 * norm["mean_a_n"]
)

norm["rank_equal"]   = norm["score_equal"].rank().astype(int)
norm["rank_outside"] = norm["score_outside"].rank().astype(int)

print("\n--- Normalized values (pp, 0=best) ---")
print(norm[metrics].round(3).to_string())

print("\n--- Composite Score (jednake tezine) ---")
print(norm[["score_equal","rank_equal"]].sort_values("score_equal").round(3).to_string())

print("\n--- Composite Score (outside prioritet) ---")
print(norm[["score_outside","rank_outside"]].sort_values("score_outside").round(3).to_string())

# Comparison raw vs pp composite
print("\n" + "=" * 60)
print("USPOREDBA RAW vs PP COMPOSITE SCORE (jednake tezine)")
print("=" * 60)

norm_raw = pd.DataFrame(index=raw.index)
for m in metrics:
    mn, mx = raw[m].min(), raw[m].max()
    norm_raw[m] = (raw[m] - mn) / (mx - mn) if mx > mn else 0.0
norm_raw["score_equal"] = norm_raw[metrics].mean(axis=1)
norm_raw["rank_raw"] = norm_raw["score_equal"].rank().astype(int)

for method_pp, method_raw in [
    ("Greedy k-means (pp)", "Greedy k-means (raw)"),
    ("Min-Cost Flow (pp)", "Min-Cost Flow (raw)"),
    ("Weighted Voronoi (pp)", "Weighted Voronoi (raw)"),
]:
    rank_raw = norm_raw.loc[method_raw, "rank_raw"]
    rank_pp  = norm.loc[method_pp, "rank_equal"]
    score_raw = norm_raw.loc[method_raw, "score_equal"]
    score_pp  = norm.loc[method_pp, "score_equal"]
    label = method_pp.replace(" (pp)", "")
    print(f"{label:<25}: raw rank={rank_raw}, pp rank={rank_pp}  "
          f"(score raw={score_raw:.3f} → pp={score_pp:.3f})")

print("\nDONE.")
