# Snakefile - Teritorijalna optimizacija dostavnih ruta, Zagreb zapad HPE, listopad 2025
#
# Pokreni cijeli workflow: snakemake --cores 4
# Dry run (provjera plana bez izvrsavanja): snakemake -n
# Vizualizacija DAG-a: snakemake --dag | dot -Tpdf > workflow.pdf

RESULTS = "/podaci/data_fixed/rezultati"

rule all:
    input:
        f"{RESULTS}/evaluation_summary.csv",
        f"{RESULTS}/final_comparison_report.txt"

rule preprocessing:
    input:
        "skripte/01_preprocessing.py"
    output:
        "/podaci/data_fixed/optimizacija_preprocessed.csv"
    shell:
        "python3 skripte/01_preprocessing.py"

rule method1_kmeans_hex:
    input:
        data="/podaci/data_fixed/optimizacija_preprocessed.csv",
        script="skripte/02_greedy_kmeans_hex.py"
    output:
        assignments=f"{RESULTS}/method1_assignments.csv",
        hexagons=f"{RESULTS}/method1_hexagons.csv"
    shell:
        "python3 skripte/02_greedy_kmeans_hex.py"

rule method2_mcf_baseline:
    input:
        data="/podaci/data_fixed/optimizacija_preprocessed.csv",
        script="skripte/03_min_cost_flow.py"
    output:
        assignments=f"{RESULTS}/method2_mcf_baseline_assignments.csv",
        hexagons=f"{RESULTS}/method2_mcf_baseline_hexagons.csv"
    shell:
        "python3 skripte/03_min_cost_flow.py"

rule method2_mcf_tuned:
    input:
        data="/podaci/data_fixed/optimizacija_preprocessed.csv",
        script="skripte/03_min_cost_flow.py"
    output:
        assignments=f"{RESULTS}/method2_mcf_tuned_assignments.csv",
        hexagons=f"{RESULTS}/method2_mcf_tuned_hexagons.csv"
    shell:
        "python3 skripte/03_min_cost_flow.py --tuned"

rule method3_voronoi_baseline:
    input:
        data="/podaci/data_fixed/optimizacija_preprocessed.csv",
        script="skripte/04_weighted_voronoi.py"
    output:
        assignments=f"{RESULTS}/method3_voronoi_baseline_assignments.csv",
        hexagons=f"{RESULTS}/method3_voronoi_baseline_hexagons.csv"
    shell:
        "python3 skripte/04_weighted_voronoi.py"

rule method3_voronoi_tuned:
    input:
        data="/podaci/data_fixed/optimizacija_preprocessed.csv",
        script="skripte/04_weighted_voronoi.py"
    output:
        assignments=f"{RESULTS}/method3_voronoi_tuned_assignments.csv",
        hexagons=f"{RESULTS}/method3_voronoi_tuned_hexagons.csv"
    shell:
        "python3 skripte/04_weighted_voronoi.py --tuned"

rule evaluate_all:
    input:
        m1_a=f"{RESULTS}/method1_assignments.csv",
        m1_h=f"{RESULTS}/method1_hexagons.csv",
        m2b_a=f"{RESULTS}/method2_mcf_baseline_assignments.csv",
        m2b_h=f"{RESULTS}/method2_mcf_baseline_hexagons.csv",
        m2t_a=f"{RESULTS}/method2_mcf_tuned_assignments.csv",
        m2t_h=f"{RESULTS}/method2_mcf_tuned_hexagons.csv",
        m3b_a=f"{RESULTS}/method3_voronoi_baseline_assignments.csv",
        m3b_h=f"{RESULTS}/method3_voronoi_baseline_hexagons.csv",
        m3t_a=f"{RESULTS}/method3_voronoi_tuned_assignments.csv",
        m3t_h=f"{RESULTS}/method3_voronoi_tuned_hexagons.csv",
        script="skripte/05_evaluate.py"
    output:
        f"{RESULTS}/evaluation_summary.csv"
    shell:
        """
        python3 skripte/05_evaluate.py "Greedy k-means" {input.m1_a} {input.m1_h}
        python3 skripte/05_evaluate.py "Min-Cost Flow" {input.m2b_a} {input.m2b_h}
        python3 skripte/05_evaluate.py "Min-Cost Flow (tuned)" {input.m2t_a} {input.m2t_h}
        python3 skripte/05_evaluate.py "Weighted Voronoi" {input.m3b_a} {input.m3b_h}
        python3 skripte/05_evaluate.py "Weighted Voronoi (tuned)" {input.m3t_a} {input.m3t_h}
        """

rule final_report:
    input:
        f"{RESULTS}/evaluation_summary.csv"
    output:
        f"{RESULTS}/final_comparison_report.txt"
    shell:
        "python3 skripte/06_compare_report.py > {output}"
