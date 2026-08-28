"""
visualize.py
-------------
Plotting utilities: ANN convergence, GA convergence, actual-vs-predicted
scatter with regression line and 90% empirical confidence interval, and
absolute percentage error bars. Mirrors the plots produced by the
original MATLAB script.
"""

import os

import matplotlib
matplotlib.use("Agg")  # headless backend, safe for scripts/CI
import matplotlib.pyplot as plt
import numpy as np


def plot_ann_convergence(loss_curve, dataset_label: str, out_dir: str):
    if not loss_curve:
        return None
    plt.figure(figsize=(7, 5))
    plt.plot(range(1, len(loss_curve) + 1), loss_curve, linewidth=2)
    plt.title(f"ANN Convergence: Epochs vs Loss ({dataset_label})")
    plt.xlabel("Epochs")
    plt.ylabel("Training Loss")
    plt.grid(True, alpha=0.3)
    path = os.path.join(out_dir, f"ann_convergence_{dataset_label}.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_ga_convergence(history, dataset_label: str, out_dir: str):
    plt.figure(figsize=(7, 5))
    plt.plot(range(1, len(history) + 1), [-h for h in history], linewidth=2, color="darkorange")
    plt.title(f"GA Convergence: Generation vs Best Validation R^2 ({dataset_label})")
    plt.xlabel("Generation")
    plt.ylabel("Best Validation R^2")
    plt.grid(True, alpha=0.3)
    path = os.path.join(out_dir, f"ga_convergence_{dataset_label}.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_actual_vs_predicted(y_true_mpa, y_pred_mpa, dataset_label: str, out_dir: str,
                              model_name: str = "Hybrid"):
    y_true_mpa = np.asarray(y_true_mpa).ravel()
    y_pred_mpa = np.asarray(y_pred_mpa).ravel()

    plt.figure(figsize=(7, 6))
    plt.scatter(y_true_mpa, y_pred_mpa, alpha=0.7, label="Predicted vs Actual", edgecolor="k")

    p = np.polyfit(y_true_mpa, y_pred_mpa, 1)
    x_fit = np.linspace(y_true_mpa.min(), y_true_mpa.max(), 100)
    y_fit = np.polyval(p, x_fit)
    plt.plot(x_fit, y_fit, "r-", linewidth=2, label="Regression Line")

    residuals = np.sort(y_true_mpa - y_pred_mpa)
    lower_idx = max(0, int(round(0.05 * len(residuals))) - 1)
    upper_idx = min(len(residuals) - 1, int(round(0.95 * len(residuals))) - 1)
    lower_bound = y_fit + residuals[lower_idx]
    upper_bound = y_fit + residuals[upper_idx]
    plt.plot(x_fit, lower_bound, "--g", linewidth=1, label="90% Lower CI")
    plt.plot(x_fit, upper_bound, "--g", linewidth=1, label="90% Upper CI")

    # 1:1 reference line
    lims = [min(x_fit.min(), y_pred_mpa.min()), max(x_fit.max(), y_pred_mpa.max())]
    plt.plot(lims, lims, "k:", linewidth=1, label="1:1 Line")

    plt.xlabel("Actual Compressive Strength (MPa)")
    plt.ylabel("Predicted Compressive Strength (MPa)")
    plt.title(f"{model_name}: Actual vs Predicted ({dataset_label}), 90% CI")
    plt.legend(loc="upper left", fontsize=8)
    plt.grid(True, alpha=0.3)

    path = os.path.join(out_dir, f"actual_vs_predicted_{dataset_label}.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_error_percent(y_true_mpa, y_pred_mpa, dataset_label: str, out_dir: str):
    y_true_mpa = np.asarray(y_true_mpa).ravel()
    y_pred_mpa = np.asarray(y_pred_mpa).ravel()
    nonzero = y_true_mpa != 0
    error_percent = np.abs((y_true_mpa[nonzero] - y_pred_mpa[nonzero]) / y_true_mpa[nonzero]) * 100

    plt.figure(figsize=(8, 5))
    plt.bar(range(len(error_percent)), error_percent, color="steelblue")
    plt.axhline(np.mean(error_percent), color="red", linestyle="--",
                label=f"Mean = {np.mean(error_percent):.2f}%")
    plt.title(f"Absolute Percentage Error per Test Sample ({dataset_label})")
    plt.xlabel("Test Sample Index")
    plt.ylabel("Absolute Percentage Error (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    path = os.path.join(out_dir, f"error_percent_{dataset_label}.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def write_dataset_comparison_csv(results: dict, out_path: str):
    """Write a CSV comparing test RMSE / MAE / R^2 / MAPE across datasets."""
    import csv
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "rmse", "mae", "r2", "mape"])
        for key, res in results.items():
            m = res["test_metrics"]
            writer.writerow([key, f"{m.rmse:.6f}", f"{m.mae:.6f}", f"{m.r2:.6f}", f"{m.mape:.6f}"])
    return out_path


def plot_dataset_comparison(results: dict, out_dir: str):
    """Bar chart comparing test RMSE / MAE / R^2 across datasets."""
    labels = list(results.keys())
    rmse_vals = [results[k]["test_metrics"].rmse for k in labels]
    mae_vals = [results[k]["test_metrics"].mae for k in labels]
    r2_vals = [results[k]["test_metrics"].r2 for k in labels]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, vals, title, color in zip(
        axes, [rmse_vals, mae_vals, r2_vals],
        ["Test RMSE (MPa)", "Test MAE (MPa)", "Test R^2"],
        ["indianred", "goldenrod", "seagreen"],
    ):
        ax.bar(labels, vals, color=color)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Hybrid Model Performance: QD vs RCA Datasets")
    path = os.path.join(out_dir, "dataset_comparison.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path
