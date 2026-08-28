"""
data_loader.py
---------------
Loads the SCC (Self-Compacting Concrete) mix-design datasets
(Quarry Dust and Recycled Coarse Aggregate) and returns clean,
consistently-named pandas DataFrames.
"""

import pandas as pd

from src import config


def load_dataset(dataset_key: str) -> pd.DataFrame:
    """
    Load a single dataset by key ('qd' or 'rca').

    Parameters
    ----------
    dataset_key : str
        One of 'qd', 'rca'.

    Returns
    -------
    pd.DataFrame with columns = config.ALL_COLUMNS
    """
    if dataset_key not in config.DATASETS:
        raise ValueError(
            f"Unknown dataset '{dataset_key}'. Choose from {list(config.DATASETS)}."
        )

    meta = config.DATASETS[dataset_key]
    header_arg = 0 if meta["has_header"] else None

    df = pd.read_csv(meta["path"], header=header_arg)

    if df.shape[1] != len(config.ALL_COLUMNS):
        raise ValueError(
            f"Dataset '{dataset_key}' has {df.shape[1]} columns, "
            f"expected {len(config.ALL_COLUMNS)}."
        )

    df.columns = config.ALL_COLUMNS

    # Coerce to numeric in case of stray strings / formatting
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    n_before = len(df)
    df = df.dropna().reset_index(drop=True)
    n_after = len(df)
    if n_after < n_before:
        print(f"[data_loader] Dropped {n_before - n_after} rows with missing values "
              f"from '{dataset_key}' dataset.")

    return df


def load_all_datasets() -> dict:
    """Load both QD and RCA datasets. Returns {'qd': df, 'rca': df}."""
    return {key: load_dataset(key) for key in config.DATASETS}


if __name__ == "__main__":
    for key in config.DATASETS:
        d = load_dataset(key)
        print(f"\n{config.DATASETS[key]['name']}: {d.shape[0]} rows x {d.shape[1]} cols")
        print(d.describe().T[["min", "max", "mean"]])
