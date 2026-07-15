# Temporal Granularity Matters: A Daily Territory Design Framework for Capacity-Constrained Last-Mile Delivery

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

Analysis code for the paper:

> Novakovic, D.; Mrsic, L. *Temporal Granularity Matters: A Daily Territory Design Framework for Capacity-Constrained Last-Mile Delivery.* Logistics, 2026.

## Overview

Empirical study of temporal-spatial mismatch in last-mile delivery territory design.

Core finding: monthly aggregation causes 30-69% of parcels to fall outside their designated territory. Per-day construction reduces this to 0.61-1.28% (more than 30x improvement without any algorithmic change).

## Repository Structure

    pipeline/
      01_preprocessing.py           - Data loading and preprocessing
      02_greedy_kmeans.py           - Method 1: Capacitated k-means + hexagonal optimization
      03_min_cost_flow.py           - Method 2: Min-cost flow assignment
      04_weighted_voronoi.py        - Method 3: Weighted Voronoi / power diagram
      05_evaluate.py                - Evaluation metrics (overlap, A/N, outside%)
      06_compare_report.py          - Comparative analysis
      apply_postprocessing.py       - Stage 4-5 post-processing (algorithm-agnostic)
      statistical_tests.py          - Wilcoxon + Bonferroni + FDR + Cliff's delta
      composite_score.py            - Composite operational score
      stability_analysis.py         - Day-to-day territory stability analysis
      normalize_overlap.py          - Normalized overlap (% of total territory area)
      sensitivity_analysis.py       - Stage 5 parameter sensitivity analysis
      route_comparison_road.py      - Route efficiency vs NN-TSP (OSMnx road network)
      route_comparison_haversine.py - Route efficiency vs NN-TSP (Haversine)
      visualize_territories.py      - Territory visualizations
      visualize_routes_road.py      - Road network route visualizations
      visualize_routes_haversine.py - Haversine route visualizations
      prepare_december.py           - Seasonal validation data preparation
      run_december.py               - Run pipeline on December dataset
      common/geo_utils.py           - Geographic utilities
      common/hex_utils.py           - Hexagonal grid geometry
      common/postprocessing.py      - Shared Stage 4-5 post-processing module

## Input Data Format

CSV file with columns:

    ShipmentItemBarcode - unique parcel identifier
    EventDatetime       - delivery timestamp (YYYY-MM-DD HH:MM:SS)
    UserID              - courier identifier (anonymised)
    EventGeoX           - delivery longitude (WGS84)
    EventGeoY           - delivery latitude (WGS84)

The dataset used in the paper was obtained under a data sharing agreement and is not publicly available.

## Installation

    conda create -n phd_analiza python=3.12
    conda activate phd_analiza
    pip install -r requirements.txt

## Usage

    # Full pipeline via Snakemake
    snakemake --cores 4

    # Step by step
    python3 pipeline/01_preprocessing.py
    python3 pipeline/02_greedy_kmeans.py
    python3 pipeline/03_min_cost_flow.py
    python3 pipeline/04_weighted_voronoi.py
    python3 pipeline/apply_postprocessing.py
    python3 pipeline/05_evaluate.py
    python3 pipeline/statistical_tests.py
    python3 pipeline/composite_score.py
    python3 pipeline/stability_analysis.py

## Citation

    @article{novakovic2026temporal,
      title={Temporal Granularity Matters: A Daily Territory Design Framework
             for Capacity-Constrained Last-Mile Delivery},
      author={Novakovic, Dario and Mrsic, Leo},
      journal={Logistics},
      year={2026},
      publisher={MDPI}
    }

## License

MIT License
