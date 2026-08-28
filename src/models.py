"""
models.py
----------
Defines the three learners used in the hybrid stacking pipeline:

1. ANN            -> sklearn.neural_network.MLPRegressor
                      (Python analogue of MATLAB's `fitnet`)
2. GA-SVR         -> sklearn.svm.SVR (RBF kernel) with hyperparameters
                      [C, epsilon, gamma] tuned by a Genetic Algorithm
                      (Python analogue of MATLAB's `fitrsvm` + `ga`)
3. Meta-learner   -> sklearn.linear_model.LinearRegression, stacking the
                      ANN and SVR predictions (Python analogue of
                      MATLAB's `fitrlinear`)
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

from src import config
from src.genetic_algorithm import GeneticAlgorithm, GAResult


# ----------------------------------------------------------------------
# 1. ANN
# ----------------------------------------------------------------------
def build_ann(max_epochs: int, seed: int = config.RANDOM_SEED) -> MLPRegressor:
    # solver='adam' is used (instead of 'lbfgs') so a per-epoch loss_curve_
    # is always available for the convergence plot, matching the original
    # MATLAB "Epochs vs Error" figure.
    return MLPRegressor(
        hidden_layer_sizes=config.ANN_HIDDEN_LAYER_SIZE,
        activation="tanh",
        solver="adam",
        learning_rate_init=config.ANN_LEARNING_RATE_INIT,
        max_iter=max_epochs,
        random_state=seed,
    )


def train_ann(X_train, y_train, max_epochs: int, seed: int = config.RANDOM_SEED):
    ann = build_ann(max_epochs, seed)
    ann.fit(X_train, y_train)
    return ann


def ann_loss_curve(ann: MLPRegressor):
    """Returns per-iteration training loss, if available (adam/sgd solvers)."""
    return getattr(ann, "loss_curve_", None)


# ----------------------------------------------------------------------
# 2. GA-optimized SVR
# ----------------------------------------------------------------------
def build_svr(C: float, epsilon: float, gamma: float) -> SVR:
    return SVR(kernel="rbf", C=C, epsilon=epsilon, gamma=gamma)


def _svr_fitness(hyperparams, X_train, y_train, X_val, y_val) -> float:
    """Fitness = negative R^2 on the validation set (GA minimizes)."""
    C, epsilon, gamma = hyperparams
    model = build_svr(C, epsilon, gamma)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    r2 = r2_score(y_val, y_pred)
    return -r2


def optimize_svr_hyperparameters(
    X_train, y_train, X_val, y_val,
    pop_size: int, generations: int, seed: int = config.RANDOM_SEED,
    verbose: bool = True,
) -> GAResult:
    ga = GeneticAlgorithm(
        bounds_low=config.SVR_BOUNDS_LOW,
        bounds_high=config.SVR_BOUNDS_HIGH,
        pop_size=pop_size,
        generations=generations,
        crossover_rate=config.GA_CROSSOVER_RATE,
        mutation_rate=config.GA_MUTATION_RATE,
        elite_fraction=config.GA_ELITE_FRACTION,
        seed=seed,
        verbose=verbose,
    )
    fitness_fn = lambda h: _svr_fitness(h, X_train, y_train, X_val, y_val)
    return ga.optimize(fitness_fn)


def train_svr(X_train, y_train, C, epsilon, gamma) -> SVR:
    model = build_svr(C, epsilon, gamma)
    model.fit(X_train, y_train)
    return model


# ----------------------------------------------------------------------
# 3. Meta-learner (stacking)
# ----------------------------------------------------------------------
def build_meta_features(svr_model: SVR, ann_model: MLPRegressor, X) -> np.ndarray:
    svr_pred = svr_model.predict(X).reshape(-1, 1)
    ann_pred = ann_model.predict(X).reshape(-1, 1)
    return np.hstack([svr_pred, ann_pred])


def train_meta_learner(svr_model, ann_model, X_train, y_train) -> LinearRegression:
    meta_X = build_meta_features(svr_model, ann_model, X_train)
    meta = LinearRegression()
    meta.fit(meta_X, y_train)
    return meta


def predict_hybrid(svr_model, ann_model, meta_model, X) -> np.ndarray:
    meta_X = build_meta_features(svr_model, ann_model, X)
    return meta_model.predict(meta_X)
