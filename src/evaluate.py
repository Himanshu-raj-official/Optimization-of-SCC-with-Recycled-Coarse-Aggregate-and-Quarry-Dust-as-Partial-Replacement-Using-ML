"""
evaluate.py
------------
Regression evaluation metrics: RMSE, MAE, R^2, MAPE.
"""

from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class Metrics:
    rmse: float
    mae: float
    r2: float
    mape: float

    def as_dict(self):
        return asdict(self)

    def __str__(self):
        return (f"RMSE: {self.rmse:.4f} | MAE: {self.mae:.4f} | "
                f"R^2: {self.r2:.4f} | MAPE: {self.mape:.2f}%")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    nonzero = y_true != 0
    mape = float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100) \
        if nonzero.any() else float("nan")

    return Metrics(rmse=rmse, mae=mae, r2=r2, mape=mape)
