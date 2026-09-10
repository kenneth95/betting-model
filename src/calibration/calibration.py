"""
Model pipelines and calibration/classification diagnostics, shared by
train_models.py, backtest.py, and walk_forward.py.

CALIBRATION METHODOLOGY AND LEAKAGE PREVENTION (see walk_forward.py for
where this is actually invoked inside the walk-forward loop):
Calibration is fitted with sklearn's CalibratedClassifierCV(cv=k) using
ONLY the training window for a given fold (every match strictly before
the test season). Internally this refits the base pipeline on k-1 folds
and calibrates on the held-out fold, repeating k times and averaging --
so both the base model and the calibration map are fully blind to the
test season, exactly like an uncalibrated model. Random k-fold splits
*within* the training window do not leak information relative to the
test season, since every training match predates the test season
regardless of which fold it lands in; the only ordering that matters
(train strictly before test) is preserved.
"""

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.utils import config


# =============================================================================
# MODEL PIPELINES
# =============================================================================

def make_logistic_pipeline():
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    random_state=config.RANDOM_STATE,
                ),
            ),
        ]
    )


def make_hgb_pipeline():
    """HistGradientBoostingClassifier natively handles missing values and
    doesn't need feature scaling, so no imputer/scaler stage is needed.
    max_iter/max_depth are conservative, non-tuned values -- these were
    not selected by looking at any test-season result. Walk-forward
    testing found this underperforms plain logistic regression at this
    dataset's size (~3,400 rows in the largest training fold): see
    FINDINGS.md. Kept available for when more data (more seasons/leagues)
    makes a higher-capacity model worth revisiting.
    """
    return Pipeline(
        steps=[
            (
                "model",
                HistGradientBoostingClassifier(
                    max_iter=150,
                    max_depth=3,
                    learning_rate=0.05,
                    random_state=config.RANDOM_STATE,
                ),
            ),
        ]
    )


PIPELINE_FACTORIES = {
    "logistic": make_logistic_pipeline,
    "hgb": make_hgb_pipeline,
}


# =============================================================================
# CLASSIFICATION METRICS
# =============================================================================

def multiclass_brier(y_true, prob):
    actual = (
        pd.get_dummies(y_true)
        .reindex(columns=[0, 1, 2], fill_value=0)
        .values
    )
    return float(np.mean(np.sum((prob - actual) ** 2, axis=1)))


def classification_metrics(y_true, prob):
    pred = np.argmax(prob, axis=1)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "logloss": log_loss(y_true, prob, labels=[0, 1, 2]),
        "brier": multiclass_brier(y_true, prob),
    }


# =============================================================================
# CALIBRATION DIAGNOSTICS
# =============================================================================

def calibration_bins(outcome_table, n_bins=10):
    """Predicted-vs-observed probability, pooled over ALL THREE outcomes
    of every match (not just the outcome that was bet), so this is not
    subject to the same selection effects an EV-bucket or bet-level
    calibration table would have.
    """
    table = outcome_table.copy()
    bin_edges = np.linspace(0, 1, n_bins + 1)
    table["bin"] = pd.cut(table["probability"], bins=bin_edges, include_lowest=True)

    rows = []
    for b, g in table.groupby("bin", observed=True):
        if len(g) == 0:
            continue
        rows.append({
            "bin": str(b),
            "n": len(g),
            "predicted_prob": g["probability"].mean(),
            "observed_freq": g["won"].mean(),
        })
    return pd.DataFrame(rows)


def expected_calibration_error(bin_table):
    if bin_table.empty:
        return np.nan
    total = bin_table["n"].sum()
    weighted_gap = (
        bin_table["n"] * (bin_table["predicted_prob"] - bin_table["observed_freq"]).abs()
    ).sum()
    return float(weighted_gap / total)


def safe_cv_folds(y_train, max_folds=5):
    """CalibratedClassifierCV needs at least 2 examples of the rarest
    class per fold; shrink the fold count automatically for very small
    early walk-forward training windows instead of erroring out."""
    n_per_class = y_train.value_counts().min()
    return min(max_folds, max(2, n_per_class))
