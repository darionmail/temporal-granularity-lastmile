# Temporal granularity of parcel delivery territory plans — analysis code

Code for Novaković & Mršić, "How often should parcel delivery territories be redesigned? Workload balance, territory continuity and travel under fixed, rebalanced and daily plans" (prepared for submission to the *International Journal of Logistics Research and Applications*).

The delivery data belong to a parcel operator and are not included. The expected input is a CSV with the columns
`ShipmentItemBarcode, EventDatetime, FacilityCode, UserID, EventGeoX (lon), EventGeoY (lat)`, one row per delivered parcel.


## Private configuration

The public repository uses the pseudonyms `Depot01`–`Depot04`, matching the manuscript. Two small private configuration files are intentionally not distributed:

- `facility_map_private.csv` — maps the operator's four provider facility codes to `Depot01`–`Depot04`; copy `facility_map_private_template.csv` and fill the codes.
- `depots_private.csv` — contains the four depot latitude/longitude pairs needed for depot-drive and road-network calculations; copy `depots_private_template.csv` and fill the coordinates.

The raw delivery data and the two October/December samples used for the Section 5.4 identity diagnostic are restricted and are not included. Set `PITFALL_DATA_DIR` to a directory containing `depot01_october_sample.csv` and `depot01_december_sample.csv` when reproducing that diagnostic.

## Pipeline (run from `src/`)

| Step | Script | Output | Paper section |
|---|---|---|---|
| 0 | `extract_year.py <input.csv.gz>` | `year.pkl` | 3.1 |
| 1 | `prep.py` | `deliv.pkl` (EPSG:3765 coordinates, 250 m hexagonal basic units) | 3–4.1 |
| 2 | `rq12.py` | `rq1_stability.csv`, `rq2_feasibility_all_depots.csv`, `rq2_feasibility_summary.csv` | 5.1–5.2 |
| 3 | `rq3_experiment.py <Depot>` for Depot01, Depot02, Depot03, Depot04 | `rq3i_<Depot>.csv` (metrics), `cent_*.csv.gz` (district-day loads and centroids), `plans_*.csv` (realised plan balance after rounding), `milplog_*.csv` (PEN solver status, gap, fallback) | 4.3, 5.3 |
| 3b | `rq3_experiment.py Depot01 annual` | `rq3i_Depot01_annual.csv` (plan age) | 5.3 |
| 4 | `rq3_sensitivity.py Depot01` with env vars `BU`, `TARGET`, `SEED`, `ENGINE` (mcf, greedy, wvor) | `sens_*.csv` | 4.6, 5.3 |
| 4b | `recompute.py` | `rq3f_*.csv`: in-band share recomputed from district loads with a 1e-6 tolerance at the band edges | 5.3 |
| 4c | `add_stem.py` (uses `depots.py`) | adds `stem_km` (2 × straight-line depot–district-centroid distance) and `total_km` to `rq3f_*.csv` | 4.4, 5.3 |
| R1 | `build_network.py <croatia.osm.pbf>` (or set `OSM_PBF`) | `roadnet.pkl`: drivable network, one-ways, speeds | 4.4 |
| R2 | `road_matrices.py` | `road_mats.pkl`: road distance / free-flow time between BUs and the depot | 4.4 |
| R2b | `road_times_on_dist.py` | replaces the time matrices in `road_mats.pkl` with the free-flow driving time along each shortest-distance path (lexicographic Dijkstra: length first, then time), for mapped, uniform 25 km/h and congested speeds | 4.4, B |
| R3 | `rq3r.py <Depot>` (same experiment; also saves `assign_*.csv.gz`, the BU-to-district assignment per design-day). For exact reproduction under heavy CPU load use `engine_i60.py` (60 s MILP limit) | `rq3r_*.csv`, `assign_*.csv.gz` | 5.3 |
| R4 | `route_all.py <Depot>` (uses `routing.py`) | `road_*.csv`: road km and minutes per route | 5.3 |
| R5 | `validate_road.py` | `road_validation.csv`: modelled vs actual scan-sequence routes | 4.4 |
| R6 | `road_analysis.py`, `cluster_stats_road.py R4road.csv` | `R4road.csv`, road-travel indices, paired and cluster-robust tests | 5.3, Appendix A |
| R7 | `route_all.py <Depot> uniform25`, `... congested`, `... seed1 1`, then `road_robust.py` and `sens.py` (Table B2) | travel-model sensitivity (within-cell term, speeds, heuristic seed) and road vs straight-line agreement | Appendix B |
| R8 | `breakeven.py` | break-even per retained parcel on the 872 depot-days with a continuity measure | 6.1 |
| 5 | `summarize_f.py` | `paired_tests_f.csv` (depot-level Wilcoxon, Hodges–Lehmann, rank-biserial) | 5.3 |
| 5b | `cluster_stats.py rq3f_` | `cluster_robust.csv` (calendar-week block bootstrap, weekly and monthly Wilcoxon) | 5.3, Table A1 |
| 6 | `validate_bhh.py` | `bhh_validation.csv`, `bhh_validation_all.csv` | 4.4 |
| 7 | `pitfall_otr.py` | `pitfall_otr.csv` | 5.4 |
| 8 | `../figures/fig1_study_area.py`, `fig2_tradeoff.py`, `fig3_plan_age.py`, `fig4_identity.py` | Figures 1–4 (written to `figures/output/`) | Figures 1–4 |

Figure 2 reads `R4road.csv` so that its travel axis is the road-travel index reported in the manuscript (not the straight-line robustness metric).

Core modules: `engine.py` (capacitated k-means with min-cost LP assignment, penalised MILP rebalancing, greedy and weighted-Voronoi engines, BHH tour estimate), `stats.py` (paired tests), `hexlib.py` / `load.py` (hexagonal service envelopes for the Section 5.4 reproduction).

Runtime: about 8 minutes per depot on one core (HiGHS via SciPy).

Seeds: 42 by default; the sensitivity runs use seeds 1–3.
