#!/usr/bin/env python
"""
run.py
=======
Command-line entry point for:

    OPTIMIZATION OF SCC WITH RECYCLED COARSE AGGREGATE AND QUARRY DUST
    AS PARTIAL REPLACEMENT USING MACHINE LEARNING TECHNIQUES
    (HYBRID ANN + GA-OPTIMIZED SVR MODELS)

Trains a hybrid stacking model (ANN + Genetic-Algorithm-tuned SVR,
combined through a linear meta-learner) to predict the compressive
strength of Self-Compacting Concrete (SCC) mixes, for both the
Quarry Dust (QD) and Recycled Coarse Aggregate (RCA) datasets.

All saved artifacts are CSV (metrics, per-sample predictions, GA
convergence history) or PNG (plots) -- no binary model files
(joblib/pickle) are written to disk. See src/train_pipeline.py.

USAGE
-----
    # Train on both datasets (default), full run
    python run.py

    # Train on a single dataset
    python run.py --dataset qd
    python run.py --dataset rca

    # Fast smoke-test run (fewer epochs / GA generations)
    python run.py --dataset both --quick

    # Interactive prediction (trains an in-memory model, then prompts you)
    python run.py --predict --dataset qd
    python run.py --predict --dataset qd --quick   # faster, less accurate

    # Reproducibility / tuning knobs
    python run.py --dataset both --seed 42
"""

import argparse
import sys

from src import config
from src.predict_cli import run_interactive_prediction
from src.train_pipeline import run_pipeline
from src.visualize import plot_dataset_comparison, write_dataset_comparison_csv


def parse_args():
    parser = argparse.ArgumentParser(
        description="Hybrid ANN + GA-SVR compressive strength predictor for SCC "
                    "with RCA / Quarry Dust partial replacement."
    )
    parser.add_argument(
        "--dataset", choices=["qd", "rca", "both"], default="both",
        help="Which dataset to run the pipeline on (default: both).",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Run a fast, reduced-size version (fewer ANN epochs and GA generations) "
             "for smoke-testing the pipeline.",
    )
    parser.add_argument(
        "--seed", type=int, default=config.RANDOM_SEED,
        help=f"Random seed for reproducibility (default: {config.RANDOM_SEED}).",
    )
    parser.add_argument(
        "--predict", action="store_true",
        help="Train an in-memory model for --dataset, then run an interactive "
             "CLI prediction (no files are read from or written to disk for "
             "this mode besides the usual metrics/plots).",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress GA per-generation progress output.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.predict:
        if args.dataset == "both":
            print("Please choose a single --dataset (qd or rca) for prediction.")
            sys.exit(1)
        result = run_pipeline(
            args.dataset, quick=args.quick, seed=args.seed, verbose=not args.quiet
        )
        run_interactive_prediction(result["bundle"])
        return

    dataset_keys = ["qd", "rca"] if args.dataset == "both" else [args.dataset]

    results = {}
    for key in dataset_keys:
        results[key] = run_pipeline(
            key, quick=args.quick, seed=args.seed, verbose=not args.quiet
        )

    if len(results) > 1:
        comparison_results = {
            k: {"test_metrics": v["test_metrics"]} for k, v in results.items()
        }
        comp_png = plot_dataset_comparison(comparison_results, config.PLOT_DIR)
        comp_csv_path = f"{config.METRICS_DIR}/dataset_comparison.csv"
        write_dataset_comparison_csv(comparison_results, comp_csv_path)
        print(f"\nSaved comparison plot -> {comp_png}")
        print(f"Saved comparison CSV  -> {comp_csv_path}")

    print("\nAll done. See outputs/metrics (*.csv) and outputs/plots (*.png).")


if __name__ == "__main__":
    main()
