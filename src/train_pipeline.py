"""
train_pipeline.py
-------------------
End-to-end pipeline for a single dataset:
  1. Load data
  2. Normalize + split (70/15/15)
  3. Train ANN
  4. GA-optimize SVR hyperparameters on the validation set
  5. Train the tuned SVR
  6. Train a linear meta-learner stacking ANN + SVR predictions
  7. Evaluate on train / val / test sets
  8. Save metrics, predictions, and GA history as CSV, and all plots as PNG

No model-serialization files (e.g. joblib/pickle) are written. Trained
model objects are returned in-memory (in the `bundle` key of the result
dict) so callers -- including `run.py --predict` -- can use them
directly within the same process. This keeps every artifact on disk in
an open, human-readable format (CSV/PNG) rather than a binary blob.
"""

import csv
import os
import time

from src import config, models, visualize
from src.data_loader import load_dataset
from src.evaluate import compute_metrics
from src.preprocessing import normalize_and_split


def _write_metrics_csv(path, dataset_name, n_samples, C, epsilon, gamma,
                        runtime_seconds, metrics_by_split):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "dataset", "n_samples", "svr_C", "svr_epsilon", "svr_gamma",
            "runtime_seconds", "split", "rmse", "mae", "r2", "mape",
        ])
        for split_name, m in metrics_by_split.items():
            writer.writerow([
                dataset_name, n_samples, f"{C:.6f}", f"{epsilon:.6f}", f"{gamma:.6f}",
                runtime_seconds, split_name, f"{m.rmse:.6f}", f"{m.mae:.6f}",
                f"{m.r2:.6f}", f"{m.mape:.6f}",
            ])


def _write_predictions_csv(path, predictions_by_split):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "sample_index", "actual_mpa", "predicted_mpa",
                          "abs_error_mpa", "abs_pct_error"])
        for split_name, (y_true, y_pred) in predictions_by_split.items():
            for i, (yt, yp) in enumerate(zip(y_true, y_pred)):
                abs_err = abs(yt - yp)
                pct_err = (abs_err / yt * 100) if yt != 0 else float("nan")
                writer.writerow([split_name, i, f"{yt:.4f}", f"{yp:.4f}",
                                  f"{abs_err:.4f}", f"{pct_err:.4f}"])


def _write_ga_history_csv(path, history):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["generation", "best_validation_r2"])
        for gen, neg_r2 in enumerate(history, start=1):
            writer.writerow([gen, f"{-neg_r2:.6f}"])


def run_pipeline(dataset_key: str, quick: bool = False, seed: int = config.RANDOM_SEED,
                  verbose: bool = True) -> dict:
    t0 = time.time()
    dataset_label = dataset_key.upper()
    dataset_name = config.DATASETS[dataset_key]["name"]
    print(f"\n{'=' * 70}\nRunning hybrid ANN + GA-SVR pipeline on: {dataset_name}\n{'=' * 70}")

    # 1. Load
    df = load_dataset(dataset_key)
    print(f"[1/7] Loaded {len(df)} samples, {len(config.FEATURE_COLUMNS)} features.")

    # 2. Normalize + split
    split = normalize_and_split(df, seed=seed)
    print(f"[2/7] Split -> train={len(split.X_train)}, val={len(split.X_val)}, "
          f"test={len(split.X_test)}")

    # 3. Train ANN
    max_epochs = config.ANN_MAX_EPOCHS_QUICK if quick else config.ANN_MAX_EPOCHS
    ann = models.train_ann(split.X_train, split.y_train, max_epochs=max_epochs, seed=seed)
    print(f"[3/7] ANN trained ({max_epochs} max epochs, "
          f"{config.ANN_HIDDEN_LAYER_SIZE[0]} hidden neurons).")

    # 4. GA-optimize SVR hyperparameters
    pop_size = config.GA_POP_SIZE_QUICK if quick else config.GA_POP_SIZE
    generations = config.GA_GENERATIONS_QUICK if quick else config.GA_GENERATIONS
    print(f"[4/7] Optimizing SVR hyperparameters with GA "
          f"(pop={pop_size}, generations={generations})...")
    ga_result = models.optimize_svr_hyperparameters(
        split.X_train, split.y_train, split.X_val, split.y_val,
        pop_size=pop_size, generations=generations, seed=seed, verbose=verbose,
    )
    C, epsilon, gamma = ga_result.best_solution
    print(f"      Best hyperparameters -> C={C:.4f}, epsilon={epsilon:.4f}, gamma={gamma:.4f}")
    print(f"      Best validation R^2 = {-ga_result.best_fitness:.4f}")

    # 5. Train tuned SVR
    svr = models.train_svr(split.X_train, split.y_train, C, epsilon, gamma)
    print("[5/7] Final SVR trained with GA-optimized hyperparameters.")

    # 6. Meta-learner (stacking)
    meta = models.train_meta_learner(svr, ann, split.X_train, split.y_train)
    print("[6/7] Linear meta-learner trained on [SVR, ANN] stacked predictions.")

    # 7. Evaluate
    def _eval(X, y_norm, split_name):
        y_pred_norm = models.predict_hybrid(svr, ann, meta, X)
        y_true_mpa = split.y_scaler.denormalize(y_norm.reshape(-1, 1)).ravel()
        y_pred_mpa = split.y_scaler.denormalize(y_pred_norm.reshape(-1, 1)).ravel()
        metrics = compute_metrics(y_true_mpa, y_pred_mpa)
        print(f"      {split_name:5s} -> {metrics}")
        return metrics, y_true_mpa, y_pred_mpa

    print("[7/7] Evaluation:")
    train_metrics, y_train_true, y_train_pred = _eval(split.X_train, split.y_train, "train")
    val_metrics, y_val_true, y_val_pred = _eval(split.X_val, split.y_val, "val")
    test_metrics, y_test_true, y_test_pred = _eval(split.X_test, split.y_test, "test")

    runtime_seconds = round(time.time() - t0, 2)

    # --- In-memory model bundle (no file persistence) ---
    bundle = {
        "svr": svr, "ann": ann, "meta": meta,
        "x_scaler": split.X_scaler, "y_scaler": split.y_scaler,
        "svr_hyperparams": {"C": float(C), "epsilon": float(epsilon), "gamma": float(gamma)},
        "feature_columns": config.FEATURE_COLUMNS,
        "dataset_key": dataset_key,
    }

    # --- Save metrics as CSV ---
    metrics_path = os.path.join(config.METRICS_DIR, f"metrics_{dataset_key}.csv")
    _write_metrics_csv(
        metrics_path, dataset_name, len(df), C, epsilon, gamma, runtime_seconds,
        {"train": train_metrics, "val": val_metrics, "test": test_metrics},
    )

    # --- Save per-sample predictions as CSV ---
    predictions_path = os.path.join(config.METRICS_DIR, f"predictions_{dataset_key}.csv")
    _write_predictions_csv(
        predictions_path,
        {
            "train": (y_train_true, y_train_pred),
            "val": (y_val_true, y_val_pred),
            "test": (y_test_true, y_test_pred),
        },
    )

    # --- Save GA convergence history as CSV ---
    ga_history_path = os.path.join(config.METRICS_DIR, f"ga_history_{dataset_key}.csv")
    _write_ga_history_csv(ga_history_path, ga_result.history)

    # --- Plots (PNG) ---
    visualize.plot_ann_convergence(models.ann_loss_curve(ann), dataset_label, config.PLOT_DIR)
    visualize.plot_ga_convergence(ga_result.history, dataset_label, config.PLOT_DIR)
    visualize.plot_actual_vs_predicted(y_test_true, y_test_pred, dataset_label, config.PLOT_DIR)
    visualize.plot_error_percent(y_test_true, y_test_pred, dataset_label, config.PLOT_DIR)

    print(f"\nSaved metrics     -> {metrics_path}")
    print(f"Saved predictions -> {predictions_path}")
    print(f"Saved GA history  -> {ga_history_path}")
    print(f"Saved plots       -> {config.PLOT_DIR}")
    print(f"Pipeline runtime  -> {runtime_seconds}s")

    return {
        "dataset_key": dataset_key,
        "metrics_path": metrics_path,
        "predictions_path": predictions_path,
        "ga_history_path": ga_history_path,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "ga_result": ga_result,
        "bundle": bundle,
    }
