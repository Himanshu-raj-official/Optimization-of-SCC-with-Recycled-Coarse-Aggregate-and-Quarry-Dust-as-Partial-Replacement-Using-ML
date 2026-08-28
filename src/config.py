"""
config.py
---------
Central configuration: file paths, column names, and default
hyperparameters for the ANN + GA-SVR hybrid pipeline.
"""

import os

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
METRICS_DIR = os.path.join(OUTPUT_DIR, "metrics")

# NOTE: no model-serialization directory. This project never writes
# binary model files (joblib/pickle); every saved artifact is either a
# CSV (metrics, predictions, GA history) or a PNG (plots). Trained
# scikit-learn model objects only ever live in memory for the duration
# of a single `run.py` invocation.
for _d in (OUTPUT_DIR, PLOT_DIR, METRICS_DIR):
    os.makedirs(_d, exist_ok=True)

DATASETS = {
    "qd": {
        "name": "Quarry Dust (QD)",
        "path": os.path.join(DATA_DIR, "concrete_QD_data.csv"),
        "has_header": False,
    },
    "rca": {
        "name": "Recycled Coarse Aggregate (RCA)",
        "path": os.path.join(DATA_DIR, "concrete_RCA_data.csv"),
        "has_header": True,
    },
}

# ----------------------------------------------------------------------
# Feature / target columns
# ----------------------------------------------------------------------
# Both datasets share the same 10 mix-design inputs + 1 target, matching
# the original MATLAB `inputdlg` prompt order.
FEATURE_COLUMNS = [
    "Cement",
    "FineAggregate",
    "CoarseAggregate",
    "Water",
    "w_b_ratio",
    "FlyAsh",
    "GGBS",
    "SilicaFume",
    "RCA",
    "Days",
]
TARGET_COLUMN = "CompressiveStrength"
ALL_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]

# ----------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------
RANDOM_SEED = 28

# ----------------------------------------------------------------------
# Train / val / test split ratios
# ----------------------------------------------------------------------
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ----------------------------------------------------------------------
# ANN (MLPRegressor) defaults
# ----------------------------------------------------------------------
ANN_HIDDEN_LAYER_SIZE = (10,)
ANN_MAX_EPOCHS = 500          # full run
ANN_MAX_EPOCHS_QUICK = 50     # --quick run
ANN_LEARNING_RATE_INIT = 0.01

# ----------------------------------------------------------------------
# GA-optimized SVR search space: [C, epsilon, gamma]
# ----------------------------------------------------------------------
SVR_BOUNDS_LOW = [1.0, 0.01, 0.01]
SVR_BOUNDS_HIGH = [50.0, 1.0, 5.0]

GA_POP_SIZE = 20
GA_GENERATIONS = 50
GA_POP_SIZE_QUICK = 10
GA_GENERATIONS_QUICK = 8
GA_CROSSOVER_RATE = 0.8
GA_MUTATION_RATE = 0.2
GA_ELITE_FRACTION = 0.1
