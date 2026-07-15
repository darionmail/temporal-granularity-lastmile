"""
05_evaluate.py (v2 - PO DANU)

Racuna pet evaluacijskih metrika iz rada (Sekcija 6.2), ali sada PO DANU
(jer hexagons CSV ima kolonu 'date' - svaki dan ima svoj nezavisni set heksagona).

Metrike se racunaju za SVAKI dan posebno, pa agregiraju (sum/mean kroz dane)
za usporedivost s originalnim radom koji je radio na 5-dnevnom prosjeku.

- Overlap (km^2): SUMA preklapanja kroz sve dane
- Sum A/N, Mean A/N, Std A/N: PROSJEK kroz dane (prosjecni dnevni A/N)
- Outside: SUMA broja paketa izvan dodijeljenog heksagona kroz sve dane

Pokreni: python3 05_evaluate.py <method_name> <assignments_csv> <hexagons_csv>
"""
import sys
import os
import numpy as np
import pandas as pd
from shapely import wkt
from shapely.geometry import Point


def evaluate_day(day_assignments, day_hexagons_df):
    """Racuna metrike za jedan dan. Vraca dict ili None ako nema heksagona."""
    if len(day_hexagons_df) == 0:
        return None

    hexagons = {}
    for _, row in day_hexagons_df.iterrows():
        hexagons[row["UserID"]] = {
            "polygon": wkt.loads(row["polygon_wkt"]),
            "area_m2": row["area_m2"],
        }

    courier_ids = list(hexagons.keys())

    # Overlap
    overlap_total_m2 = 0.0
    for i in range(len(courier_ids)):
        for j in range(i + 1, len(courier_ids)):
            poly_i = hexagons[courier_ids[i]]["polygon"]
            poly_j = hexagons[courier_ids[j]]["polygon"]
            if poly_i.intersects(poly_j):
                overlap_total_m2 += poly_i.intersection(poly_j).area
    overlap_km2 = overlap_total_m2 / 1_000_000.0

    # A/N per courier
    a_over_n = {}
    for cid in courier_ids:
        n_u = (day_assignments["assigned_courier"] == cid).sum()
        a_u = hexagons[cid]["area_m2"]
        a_over_n[cid] = a_u / n_u if n_u > 0 else np.nan

    a_over_n_values = np.array([v for v in a_over_n.values() if not np.isnan(v)])
    sum_a_n = a_over_n_values.sum() if len(a_over_n_values) > 0 else 0.0
    mean_a_n = a_over_n_values.mean() if len(a_over_n_values) > 0 else 0.0
    std_a_n = a_over_n_values.std() if len(a_over_n_values) > 0 else 0.0

    # Outside
    outside_count = 0
    for _, row in day_assignments.iterrows():
        cid = row["assigned_courier"]
        if pd.isna(cid) or cid not in hexagons:
            outside_count += 1
            continue
        pt = Point(row["x_m"], row["y_m"])
        if not hexagons[cid]["polygon"].contains(pt) and not hexagons[cid]["polygon"].touches(pt):
            outside_count += 1

    return {
        "overlap_km2": overlap_km2,
        "sum_a_n": sum_a_n,
        "mean_a_n": mean_a_n,
        "std_a_n": std_a_n,
        "outside": outside_count,
        "n_couriers": len(courier_ids),
        "n_parcels": len(day_assignments),
    }


def evaluate(method_name, assignments_csv, hexagons_csv):
    df = pd.read_csv(assignments_csv)
    hex_df = pd.read_csv(hexagons_csv)

    daily_results = []
    for date, day_hex in hex_df.groupby("date"):
        day_assignments = df[df["date"] == date]
        result = evaluate_day(day_assignments, day_hex)
        if result is not None:
            result["date"] = date
            daily_results.append(result)

    daily_df = pd.DataFrame(daily_results)

    results = {
        "method": method_name,
        "overlap_km2": round(daily_df["overlap_km2"].sum(), 2),
        "sum_a_n": round(daily_df["sum_a_n"].mean(), 0),
        "mean_a_n": round(daily_df["mean_a_n"].mean(), 0),
        "std_a_n": round(daily_df["std_a_n"].mean(), 1),
        "outside": int(daily_df["outside"].sum()),
        "outside_pct": round(100 * daily_df["outside"].sum() / daily_df["n_parcels"].sum(), 2),
        "n_days": len(daily_df),
    }

    return results, daily_df


def main():
    if len(sys.argv) != 4:
        print("Usage: python3 05_evaluate.py <method_name> <assignments_csv> <hexagons_csv>")
        sys.exit(1)

    method_name = sys.argv[1]
    assignments_csv = sys.argv[2]
    hexagons_csv = sys.argv[3]

    results, daily_df = evaluate(method_name, assignments_csv, hexagons_csv)

    print("=" * 60)
    print(f"EVALUACIJA: {method_name}  (agregirano kroz {results['n_days']} dana)")
    print("=" * 60)
    for k, v in results.items():
        print(f"  {k}: {v}")

    results_dir = os.path.dirname(assignments_csv)
    summary_path = os.path.join(results_dir, "evaluation_summary.csv")

    row_df = pd.DataFrame([results])
    if os.path.exists(summary_path):
        existing = pd.read_csv(summary_path)
        existing = existing[existing["method"] != method_name]
        combined = pd.concat([existing, row_df], ignore_index=True)
    else:
        combined = row_df

    combined.to_csv(summary_path, index=False)

    daily_detail_path = os.path.join(
        results_dir,
        f"daily_detail_{method_name.replace(' ', '_').replace('(', '').replace(')', '')}.csv"
    )
    daily_df.to_csv(daily_detail_path, index=False)

    print(f"\nSpremljeno u: {summary_path}")
    print(f"Dnevni detalj spremljen u: {daily_detail_path}")


if __name__ == "__main__":
    main()
