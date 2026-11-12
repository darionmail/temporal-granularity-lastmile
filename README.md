# Rolling-Horizon Territory Design for Last-Mile Parcel Delivery — analysis code

Code accompanying the manuscript **“Rolling-Horizon Territory Design for Last-Mile Parcel Delivery: An Out-of-Sample Benchmark of Re-Optimisation and Minimal-Change Rebalancing”**, prepared for submission to *Algorithms* (MDPI).

The delivery data belong to a parcel operator and are not included. The expected input is a CSV with the columns
`ShipmentItemBarcode, EventDatetime, FacilityCode, UserID, EventGeoX (lon), EventGeoY (lat)`, one row per delivered parcel.

## Private configuration

The public repository uses the pseudonyms `Depot01`–`Depot04`, matching the manuscript. Two small private configuration files are intentionally not distributed:

- `facility_map_private.csv` — maps the operator's four provider facility codes to `Depot01`–`Depot04`; copy `facility_map_private_template.csv` and fill the codes.
- `depots_private.csv` — contains the four depot latitude/longitude pairs needed for depot-drive and road-network calculations; copy `depots_private_template.csv` and fill the coordinates.

The raw delivery data and the two October/December samples used for the Section 5.4 identity diagnostic are restricted and are not included. Set `PITFALL_DATA_DIR` to a directory containing `depot01_october_sample.csv` and `depot01_december_sample.csv` when reproducing that diagnostic.

## Installation

Python 3.11 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The separate computational-performance re-run reported in the manuscript used Python 3.13.18, SciPy 1.18.1 (bundled HiGHS), NumPy 2.5.3 and pandas 3.0.5 on Linux.

## Pipeline

Run the scripts from `src/`.

| Step | Script | Output | Paper section |
|---|---|---|---|
| 0 | `extract_year.py <input.csv.gz>` | `year.pkl` | 3.1 |
| 1 | `prep.py` | `deliv.pkl` (EPSG:3765 coordinates, 250 m hexagonal basic units) | 3–4.1 |
| 2 | `rq12.py` | `rq1_stability.csv`, `rq2_feasibility_all_depots.csv`, `rq2_feasibility_summary.csv` | 5.1–5.2 |
| 3 | `rq3_experiment.py <Depot>` for Depot01, Depot02, Depot03, Depot04 | `rq3i_<Depot>.csv` (metrics), `cent_*.csv.gz` (district-day loads and centroids), `plans_*.csv` (realised plan balance after rounding), `milplog_*.csv` (PEN solver status, gap, fallback) | 4.3, 5.3 |
| 3b | `rq3_experiment.py Depot01 annual` | `rq3i_Depot01_annual.csv` (plan age) | 5.3 |
| 4 | `rq3_sensitivity.py Depot01` with env vars `BU`, `TARGET`, `SEED`, `ENGINE` (`mcf`, `greedy`, `wvor`) | `sens_*.csv` | 4.6, 5.3 |
| 4b | `recompute.py` | `rq3f_*.csv`: in-band share recomputed from district loads with a `1e-6` tolerance at the band edges | 5.3 |
| 4c | `add_stem.py` (uses `depots.py`) | adds `stem_km` (2 × straight-line depot–district-centroid distance) and `total_km` to `rq3f_*.csv` | 4.4, 5.3 |
| R1 | `build_network.py <croatia.osm.pbf>` (or set `OSM_PBF`) | `roadnet.pkl`: drivable network, one-ways, speeds | 4.4 |
| R2 | `road_matrices.py` | `road_mats.pkl`: road distance / free-flow time between BUs and the depot | 4.4 |
| R2b | `road_times_on_dist.py` | replaces the time matrices in `road_mats.pkl` with free-flow driving time along each shortest-distance path (lexicographic Dijkstra: length first, then time), for mapped, uniform 25 km/h and congested speeds | 4.4, Appendix C |
| R3 | `rq3r.py <Depot>` | `rq3r_*.csv`, `assign_*.csv.gz` (BU-to-district assignment per design-day) | 5.3 |
| R4 | `route_all.py <Depot>` (uses `routing.py`) | `road_*.csv`: road km and minutes per route | 5.3 |
| R5 | `validate_road.py` | `road_validation.csv`: modelled vs actual scan-sequence routes | 4.4 |
| R6 | `road_analysis.py`, `cluster_stats_road.py R4road.csv` | `R4road.csv`, road-travel indices, paired and cluster-robust tests | 5.3, Appendix A |
| R7 | `route_all.py <Depot> uniform25`, `... congested`, `... seed1 1`, then `road_robust.py` and `sens.py` (Table C2) | travel-model sensitivity (within-cell term, speeds, heuristic seed) and road vs straight-line agreement | Appendix C |
| R8 | `breakeven.py` | break-even per retained parcel on the 872 depot-days with a continuity measure | 6.1 |
| 5 | `summarize_f.py` | `paired_tests_f.csv` (depot-level Wilcoxon, Hodges–Lehmann, rank-biserial) | 5.3 |
| 5b | `cluster_stats.py rq3f_` | `cluster_robust.csv` (calendar-week block bootstrap, weekly and monthly Wilcoxon) | 5.3, Table A1 |
| 6 | `validate_bhh.py` | `bhh_validation.csv`, `bhh_validation_all.csv` | 4.4 |
| 7 | `pitfall_otr.py` | `pitfall_otr.csv` | 5.4 |
| 8 | `../figures/fig1_study_area.py`, `fig2_tradeoff.py`, `fig3_plan_age.py`, `fig4_identity.py` | Figures 1–4 (written to `figures/output/`) | Figures 1–4 |

Figure 2 reads `R4road.csv` so that its travel axis is the road-travel index reported in the manuscript, rather than the straight-line robustness metric.

Core modules are `engine.py` (capacitated k-means with min-cost LP assignment, penalised MILP rebalancing, greedy and weighted-Voronoi engines, BHH tour estimate), `stats.py` (paired tests), and `hexlib.py` / `load.py` (hexagonal service envelopes for the Section 5.4 reproduction).

## Solver and computational-performance settings

The PEN integer programme is solved with HiGHS through SciPy with a **10 s time limit** and **0.1% relative MIP gap**, as specified in the manuscript.

A separate computational-performance re-run used the same BU ordering, parameters and random seed (42). It reproduced the PEN solver outcomes reported in Table 5c exactly for all three values of λ: **805 optimal and 111 infeasible instances out of 916 test days**, with no run reaching the 10 s limit. The largest observed PEN wall-clock time across the three λ values was 8.6 s.

Runtime of the main analysis is approximately 8 minutes per depot on one core (HiGHS via SciPy). Actual wall-clock times depend on hardware and system load.

## Random seeds

The default random seed is 42. Sensitivity runs use seeds 1–3.

## Data availability

The raw delivery records were provided by a parcel operator under a data-sharing agreement and cannot be redistributed publicly because of commercial confidentiality and data-protection obligations. The supplementary material provides a codebook, depot- and month-level aggregates, experimental parameters and synthetic example rows.

Requests for access to the raw data should be addressed to the corresponding author, who will forward them to the data provider. Access remains subject to the provider's approval and an appropriate data-use agreement.

## Reproducibility notes

Exact depot coordinates and provider facility codes are intentionally excluded from the public repository. Reproducing the road-network analysis therefore requires the private configuration files described above and a Croatia OpenStreetMap extract.

The code and public documentation are intended to reproduce the analysis once the restricted input data and private configuration have been obtained.
