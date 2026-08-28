"""
preprocessing.py
-----------------
Min-max normalization and train/validation/test splitting,
mirroring the logic of the original MATLAB script
(dividerand-style 70/15/15 split) but using scikit-learn utilities
for reproducibility.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src import config


@dataclass
class MinMaxParams:
    """Stores min/max vectors so new (e.g. user-entered) samples can be
    normalized/denormalized identically to the training data."""
    min_vals: np.ndarray
    max_vals: np.ndarray

    def normalize(self, X: np.ndarray) -> np.ndarray:
        denom = np.where(self.max_vals - self.min_vals == 0, 1, self.max_vals - self.min_vals)
        return (X - self.min_vals) / denom

    def denormalize(self, X_norm: np.ndarray) -> np.ndarray:
        return X_norm * (self.max_vals - self.min_vals) + self.min_vals


@dataclass
class SplitData:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    X_scaler: MinMaxParams
    y_scaler: MinMaxParams


def normalize_and_split(df: pd.DataFrame, seed: int = config.RANDOM_SEED) -> SplitData:
    """
    Min-max normalize features/target to [0, 1] and split into
    train (70%) / validation (15%) / test (15%) sets.
    """
    X = df[config.FEATURE_COLUMNS].to_numpy(dtype=float)
    y = df[config.TARGET_COLUMN].to_numpy(dtype=float).reshape(-1, 1)

    x_scaler = MinMaxParams(min_vals=X.min(axis=0), max_vals=X.max(axis=0))
    y_scaler = MinMaxParams(min_vals=y.min(axis=0), max_vals=y.max(axis=0))

    X_norm = x_scaler.normalize(X)
    y_norm = y_scaler.normalize(y).ravel()

    # First split off the test set, then split remainder into train/val
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_norm, y_norm, test_size=config.TEST_RATIO, random_state=seed
    )
    val_fraction_of_temp = config.VAL_RATIO / (config.TRAIN_RATIO + config.VAL_RATIO)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_fraction_of_temp, random_state=seed
    )

    return SplitData(
        X_train=X_train, X_val=X_val, X_test=X_test,
        y_train=y_train, y_val=y_val, y_test=y_test,
        X_scaler=x_scaler, y_scaler=y_scaler,
    )
