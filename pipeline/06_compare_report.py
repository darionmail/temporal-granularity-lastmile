"""
06_compare_report.py

Generira finalni usporedni izvjestaj svih metoda, formatiran slicno Tablici 2 iz rada.
Dodaje dnevno-normalizirane vrijednosti (podijeljeno s n_days) za izravnu usporedbu
s originalnim radom koji je radio na 5-dnevnom uzorku, ne 27-dnevnom.

Run: python3 06_compare_report.py
"""
import os
import pandas as pd
import yaml
import os

with open(os.path.join(os.path.dirname(__file__), "..", "config.yaml")) as f:
    config = yaml.safe_load(f)


RESULTS_DIR = config["data"]["results_dir"]
SUMMARY_PATH = os.path.join(RESULTS_DIR, "evaluation_summary.csv")


def main():
    df = pd.read_csv(SUMMARY_PATH)

    order = ["Greedy k-means", "Min-Cost Flow", "Min-Cost Flow (tuned)",
             "Weighted Voronoi", "Weighted Voronoi (tuned)"]
    df["sort_key"] = df["method"].apply(lambda m: order.index(m) if m in order else 999)
    df = df.sort_values("sort_key").drop(columns=["sort_key"])

    # Daily-normalized values (overlap and outside are SUMS across days; A/N metrics are already averages)
    df["overlap_km2_per_day"] = df["overlap_km2"] / df["n_days"]
    df["outside_per_day"] = df["outside"] / df["n_days"]

    print("=" * 80)
    print("FINAL METHOD COMPARISON - Zagreb West distribution area, October 2025 (RAW, 27 days)")
    print("=" * 80)
    print()

    header = f"{'Algoritam':<28} {'Overlap (km2)':>14} {'Sum A/N':>12} {'Mean A/N':>10} {'Std A/N':>10} {'Outside':>9} {'Outside%':>9}"
    print(header)
    print("-" * len(header))

    for _, row in df.iterrows():
        print(f"{row['method']:<28} {row['overlap_km2']:>14.2f} {row['sum_a_n']:>12.0f} "
              f"{row['mean_a_n']:>10.0f} {row['std_a_n']:>10.1f} {row['outside']:>9d} {row['outside_pct']:>8.2f}%")

    print()
    print("=" * 80)
    print("DAILY-NORMALIZED VALUES (for comparability with the original paper)")
    print("=" * 80)
    print()

    header2 = f"{'Algoritam':<28} {'Overlap/dan (km2)':>18} {'Mean A/N':>10} {'Std A/N':>10} {'Outside/dan':>12} {'Outside%':>9}"
    print(header2)
    print("-" * len(header2))

    for _, row in df.iterrows():
        print(f"{row['method']:<28} {row['overlap_km2_per_day']:>18.2f} "
              f"{row['mean_a_n']:>10.0f} {row['std_a_n']:>10.1f} {row['outside_per_day']:>12.1f} {row['outside_pct']:>8.2f}%")

    print()
    print("=" * 80)
    print("NOTE: lower values are better for all metrics (Outside is particularly critical).")
    print("NOTE: Sum A/N is not normalized per day as it is already a daily average (A/N per day, averaged across days).")
    print("NAPOMENA: originalni rad je radio na 5 days / 10 couriers / 1000 parcels; ovaj uzorak na 27 days / 20 couriers / 27029 parcels.")
    print("=" * 80)

    print("\nNajbolja metoda po metrici (dnevno-normalizirano):")
    for col, label in [("overlap_km2_per_day", "Overlap/dan"), ("sum_a_n", "Sum A/N"),
                        ("mean_a_n", "Mean A/N"), ("std_a_n", "Std A/N"), ("outside_per_day", "Outside/dan")]:
        best_row = df.loc[df[col].idxmin()]
        print(f"  {label}: {best_row['method']} ({best_row[col]:.2f})")


if __name__ == "__main__":
    main()



if __name__ == "__main__":
    main()
