"""
predict_cli.py
----------------
Interactive command-line prediction, replacing MATLAB's `inputdlg`
pop-up boxes. Works entirely in-memory: it trains (or reuses) a hybrid
model bundle for the requested dataset and prompts the user for the 10
mix-design inputs, then reports the predicted compressive strength.

No model file is read from or written to disk (see train_pipeline.py):
prediction always operates on the in-memory `bundle` produced by
`run_pipeline()` within the same process.
"""

from src import config, models

PROMPTS = [
    ("Cement (kg/m3)", "Cement"),
    ("Fine Aggregate (kg/m3)", "FineAggregate"),
    ("Coarse Aggregate (kg/m3)", "CoarseAggregate"),
    ("Water (L)", "Water"),
    ("Water-to-Binder Ratio (w/b)", "w_b_ratio"),
    ("Fly Ash (kg/m3)", "FlyAsh"),
    ("GGBS (kg/m3)", "GGBS"),
    ("Silica Fume, SF (kg/m3)", "SilicaFume"),
    ("Recycled Concrete Aggregate, RCA (kg/m3)", "RCA"),
    ("Curing Age (Days)", "Days"),
]


def predict_from_values(bundle: dict, values: list) -> float:
    """Predict compressive strength (MPa) from a list of 10 raw inputs,
    using an in-memory model bundle returned by `run_pipeline()`."""
    import numpy as np

    X = np.array(values, dtype=float).reshape(1, -1)
    X_norm = bundle["x_scaler"].normalize(X)
    y_pred_norm = models.predict_hybrid(bundle["svr"], bundle["ann"], bundle["meta"], X_norm)
    y_pred = bundle["y_scaler"].denormalize(y_pred_norm.reshape(-1, 1)).ravel()[0]
    return float(y_pred)


def run_interactive_prediction(bundle: dict):
    dataset_key = bundle["dataset_key"]
    print(f"\n--- Predict Compressive Strength ({config.DATASETS[dataset_key]['name']}) ---")
    values = []
    for label, _ in PROMPTS:
        while True:
            raw = input(f"Enter {label}: ").strip()
            try:
                values.append(float(raw))
                break
            except ValueError:
                print("  Please enter a numeric value.")

    strength = predict_from_values(bundle, values)
    print(f"\nPredicted Compressive Strength: {strength:.2f} MPa\n")
    return strength
