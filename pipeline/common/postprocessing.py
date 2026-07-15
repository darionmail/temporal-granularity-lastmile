"""
postprocessing.py — Zajednički Stage 4 i Stage 5 post-processing modul.

Može se primijeniti na output BILO KOJE metode dodjele parcels
(k-means, MCF, Voronoi) jer ovisi samo o assignments i hexagonima,
ne o tome kako je dodjela napravljena.

Stage 4: Polygon-Aware Post-Reallocation
  - Pakete koji su izvan svog heksagona premjestiti u heksagon koji ih sadrži
  - Ako ih sadrži više, odabrati najbliži centar

Stage 5: Overlap and Density Refinement
  - Za svaki heksagon, pretraziti 6 susjednih lattice pozicija
  - Prihvatiti pomak koji smanjuje scoring funkciju (overlap + outside penalizacija + density)

Koristiti:
  from common.postprocessing import apply_postprocessing
  df, hex_df = apply_postprocessing(df, hex_df, hex_config, refine_config)
"""
import numpy as np
import pandas as pd
from shapely import wkt
from shapely.geometry import Point

# hex_utils mora biti dostupan u sys.path
from hex_utils import snap_cover_cap, get_neighbor_centers, hex_area


def stage4_reallocation(day_points_xy, assignment, hexagons, active_couriers):
    """
    Premjesti pakete koji su izvan svog heksagona u heksagon koji ih sadrži.
    Vraća ažurirani assignment array.
    """
    new_assignment = assignment.copy()
    for i in range(len(day_points_xy)):
        current_cid = assignment[i]
        if current_cid not in hexagons:
            continue
        pt = Point(day_points_xy[i])
        if hexagons[current_cid]["polygon"].contains(pt):
            continue
        containing = [cid for cid in hexagons if hexagons[cid]["polygon"].contains(pt)]
        if not containing:
            continue
        elif len(containing) == 1:
            new_assignment[i] = containing[0]
        else:
            dists = [np.linalg.norm(day_points_xy[i] - np.array(hexagons[cid]["center"]))
                     for cid in containing]
            new_assignment[i] = containing[np.argmin(dists)]
    return new_assignment


def stage5_refinement(day_points_xy, assignment, hexagons, hex_config, refine_config):
    """
    Lokalno pretrazivanje 0.5km resetke radi smanjenja overlapa i penalizacije outsidea.
    Vraca azurirani hexagons dict.
    """
    hexagons = {cid: dict(h) for cid, h in hexagons.items()}
    alpha = refine_config["alpha"]
    lambda_dens = refine_config["lambda_dens"]
    lambda_eq = refine_config["lambda_eq"]
    refine_lattice_m = hex_config["refine_lattice_km"] * 1000

    for _ in range(refine_config["max_iterations"]):
        densities = []
        for cid, h in hexagons.items():
            n_u = (assignment == cid).sum()
            if n_u > 0:
                densities.append(hex_area(h["s"]) / n_u)
        delta = np.median(densities) if densities else 0.0

        any_improved = False
        for cid in list(hexagons.keys()):
            own_pts = day_points_xy[assignment == cid]
            n_u = len(own_pts)

            def score(poly, s):
                ov = sum(
                    poly.intersection(hexagons[v]["polygon"]).area
                    for v in hexagons if v != cid and poly.intersects(hexagons[v]["polygon"])
                )
                out = sum(1 for px, py in own_pts if not poly.contains(Point(px, py)))
                d = hex_area(s) / n_u if n_u > 0 else hex_area(s)
                return ov + alpha * out + lambda_dens * d + lambda_eq * (d - delta) ** 2

            cur_score = score(hexagons[cid]["polygon"], hexagons[cid]["s"])
            best_score = cur_score
            best = None

            for cx, cy in get_neighbor_centers(
                hexagons[cid]["center"][0], hexagons[cid]["center"][1], refine_lattice_m
            ):
                _, _, s, poly = snap_cover_cap(
                    cx, cy, own_pts,
                    lattice_s=hex_config["base_lattice_km"] * 1000,
                    s_base=hex_config["s_base_m"],
                    s_max=hex_config["s_max_m"],
                    s_floor_frac=hex_config["s_floor_frac"],
                )
                cs = score(poly, s)
                if cs < best_score:
                    best_score = cs
                    best = (cx, cy, s, poly)

            if best:
                hexagons[cid]["center"] = (best[0], best[1])
                hexagons[cid]["s"] = best[2]
                hexagons[cid]["polygon"] = best[3]
                any_improved = True

        if not any_improved:
            break

    return hexagons


def apply_postprocessing(df, hex_df, hex_config, refine_config):
    """
    Primjeni Stage 4 i Stage 5 na output bilo koje metode.

    Parameters:
      df      — assignments DataFrame s kolonama: date, x_m, y_m, assigned_courier
      hex_df  — hexagons DataFrame s kolonama: date, UserID, center_x, center_y,
                side_length_m, area_m2, polygon_wkt

    Vraca:
      df      — ažurirani assignments
      hex_df  — ažurirani hexagoni
    """
    df = df.copy()
    all_hex_rows = []

    for date, day_hex in hex_df.groupby("date"):
        day_mask = df["date"] == date
        day_df = df[day_mask]
        points_xy = day_df[["x_m", "y_m"]].values
        assignment = day_df["assigned_courier"].values.copy()
        active_couriers = list(day_hex["UserID"].unique())

        # Rekonstruiraj hexagons dict
        hexagons = {}
        for _, row in day_hex.iterrows():
            hexagons[row["UserID"]] = {
                "center": (row["center_x"], row["center_y"]),
                "s": row["side_length_m"],
                "polygon": wkt.loads(row["polygon_wkt"]),
            }

        # Stage 4
        assignment = stage4_reallocation(points_xy, assignment, hexagons, active_couriers)

        # Stage 5
        hexagons = stage5_refinement(points_xy, assignment, hexagons, hex_config, refine_config)

        # Save ažurirane assignments
        df.loc[day_mask, "assigned_courier"] = assignment

        # Save ažurirane hexagone
        for cid, h in hexagons.items():
            all_hex_rows.append({
                "date": date,
                "UserID": cid,
                "center_x": h["center"][0],
                "center_y": h["center"][1],
                "side_length_m": h["s"],
                "area_m2": hex_area(h["s"]),
                "polygon_wkt": h["polygon"].wkt,
            })

    return df, pd.DataFrame(all_hex_rows)
