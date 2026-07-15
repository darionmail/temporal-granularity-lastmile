"""
Statisticki testovi s korekcijom za visestruke usporedbe.

Provodi parovne Wilcoxon signed-rank testove na dnevnim metrikama,
te primjenjuje Bonferroni i Benjamini-Hochberg (FDR) korekciju
na p-vrijednosti unutar svake metrike.

Dodatno: testira Std A/N (mjera koja se prikazuje ali nije bila testirana).

Run: python3 statisticki_testovi_korekcija.py
"""
import os
import itertools
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import yaml
import os

with open(os.path.join(os.path.dirname(__file__), "..", "config.yaml")) as f:
    config = yaml.safe_load(f)


RESULTS_DIR = config["data"]["results_dir"]

METHODS = {
    "Greedy k-means": "daily_detail_Greedy_k-means.csv",
    "Min-Cost Flow": "daily_detail_Min-Cost_Flow.csv",
    "Min-Cost Flow (tuned)": "daily_detail_Min-Cost_Flow_tuned.csv",
    "Weighted Voronoi": "daily_detail_Weighted_Voronoi.csv",
    "Weighted Voronoi (tuned)": "daily_detail_Weighted_Voronoi_tuned.csv",
}

daily = {}
for method, fname in METHODS.items():
    path = os.path.join(RESULTS_DIR, fname)
    df = pd.read_csv(path)
    daily[method] = df.set_index("date").sort_index()


def bh_fdr(pvals):
    """Benjamini-Hochberg FDR korekcija. Vraca q-vrijednosti."""
    p = np.array(pvals)
    n = len(p)
    order = np.argsort(p)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, n + 1)
    q = p * n / ranks
    # Make monotonic from largest to smallest
    sorted_q = q[order]
    for i in range(n - 2, -1, -1):
        sorted_q[i] = min(sorted_q[i], sorted_q[i + 1])
    q_out = np.empty_like(q)
    q_out[order] = sorted_q
    return np.minimum(q_out, 1.0)


methods_list = list(METHODS.keys())
n_pairs = len(list(itertools.combinations(methods_list, 2)))

print("=" * 90)
print("PAROVNI WILCOXON TESTOVI S BONFERRONI I FDR KOREKCIJOM")
print("=" * 90)
print(f"\nNumber parova metoda po metrici: {n_pairs}")
print(f"Bonferroni correction: pomnozi p s {n_pairs}, ogranici na 1.0")
print(f"FDR korekcija (Benjamini-Hochberg): primjenjena unutar svake metrike\n")

all_results = []

for metric, label in [("overlap_km2", "OVERLAP (km2/day)"),
                       ("outside", "OUTSIDE (parcels/day)"),
                       ("mean_a_n", "MEAN A/N (m2/parcel)"),
                       ("std_a_n", "STD A/N (m2/parcel) [NEW]")]:
    print(f"\n{'='*90}")
    print(f"Metrika: {label}")
    print(f"{'='*90}")

    pairs = []
    raw_ps = []
    for m_a, m_b in itertools.combinations(methods_list, 2):
        common = daily[m_a].index.intersection(daily[m_b].index)
        x = daily[m_a].loc[common, metric].values
        y = daily[m_b].loc[common, metric].values
        diff = x - y
        if np.all(diff == 0):
            p = 1.0
        else:
            try:
                stat, p = wilcoxon(x, y)
            except ValueError:
                p = float("nan")
        pairs.append((m_a, m_b))
        raw_ps.append(p)

    # Korekcije
    bonf = [min(p * n_pairs, 1.0) for p in raw_ps]
    fdr = bh_fdr(raw_ps)

    print(f"{'Method A':<28} {'Method B':<28} {'raw p':>10} {'Bonf':>10} {'FDR (q)':>10} {'Final?':>8}")
    print("-" * 90)
    for (m_a, m_b), p, bp, fp in zip(pairs, raw_ps, bonf, fdr):
        sig = "DA" if fp < 0.05 else "ne"
        print(f"{m_a:<28} {m_b:<28} {p:>10.5f} {bp:>10.5f} {fp:>10.5f} {sig:>8}")
        all_results.append({
            "metric": label, "method_a": m_a, "method_b": m_b,
            "raw_p": round(p, 5), "bonferroni_p": round(bp, 5),
            "fdr_q": round(fp, 5), "significant_after_fdr": fp < 0.05
        })

results_df = pd.DataFrame(all_results)
out_path = os.path.join(RESULTS_DIR, "statistical_tests_with_correction.csv")
results_df.to_csv(out_path, index=False)

# Summary koliko ih prezivi korekciju
print("\n" + "=" * 90)
print("SUMMARY")
print("=" * 90)
n_raw_sig = sum(1 for p in results_df["raw_p"] if p < 0.05)
n_bonf_sig = sum(1 for p in results_df["bonferroni_p"] if p < 0.05)
n_fdr_sig = results_df["significant_after_fdr"].sum()
print(f"\nTotal comparisons: {len(results_df)}")
print(f"Significant (sirovo p < 0.05): {n_raw_sig}")
print(f"Significant nakon Bonferroni: {n_bonf_sig}")
print(f"Significant nakon FDR (Benjamini-Hochberg, q < 0.05): {n_fdr_sig}")

print(f"\nSaved: {out_path}")
print("\nDONE.")
