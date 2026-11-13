# Timing re-run (R12, Algorithms MDPI)

- `timing.py` + `engine.py`: re-run of Algorithm 1 (FIX, DWS, DCS, PEN λ = 0.25/1/5 km) on aggregated BU-day loads (`timing_input_bu_loads.xz`), with wall-clock timing of each call. `engine.py` is identical to the cap_kmeans/assign_lp part of the original `src/engine.py`; logic, BU ordering and RNG sequence (seed 42) follow `src/rq3_experiment.py`.
- Run: `python3 timing.py Depot01 timing_input_bu_loads.xz` (one process per depot).
- Environment: cloud VM, Intel Xeon @ 2.8 GHz, 2 vCPU, 7.8 GB RAM, Linux, Python 3.13.18, SciPy 1.18.1 (bundled HiGHS), NumPy 2.5.3, pandas 3.0.5; two depots run in parallel.
- Check: PEN solver outcomes reproduce Table 5c exactly (207/22, 185/44, 190/39, 223/6) for all three λ; Proposition 1 flags exactly the infeasible days.
- `timing_all.csv`: one row per call (sec, n BUs, K, binary variables, solver status, Proposition 1 flag). `table3b.csv`: summary used in the paper.
