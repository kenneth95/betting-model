import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    log_loss,
)

from src.utils import config as cfg


# =============================================================================
# CONFIG
# =============================================================================
#
# NOTE: TRAIN_END/VALIDATION_END below define the single train/validation/test
# split used ONLY by this script (for the original 3-model comparison). This
# is a quick, single-split sanity check — it is NOT the primary evaluation
# anymore. See src/walk_forward.py for the expanding-window evaluation that
# should be used for any real conclusions about model quality or ROI.
#
# Feature lists and RANDOM_STATE come from config.py so they can never again
# silently drift from what backtest.py / walk_forward.py use (this previously
# happened: backtest.py's training window did not match this script's).

INPUT_FILE = cfg.MODEL_DATA_PATH

TRAIN_END = "2024-05-31"
VALIDATION_END = "2025-05-31"

RANDOM_STATE = cfg.RANDOM_STATE


# =============================================================================
# LOAD DATA
# =============================================================================

print("=" * 80)
print("EPL MODEL TRAINING")
print("=" * 80)

df = pd.read_csv(INPUT_FILE)

df["Date"] = pd.to_datetime(df["Date"])

print(f"\nMatches loaded: {len(df):,}")

# Target:
# H = 0
# D = 1
# A = 2

if "target" not in df.columns:
    raise ValueError("Target column not found.")

print("\nTarget distribution:")
print(df["target"].value_counts().sort_index())


# =============================================================================
# TIME SPLIT
# =============================================================================

train = df[df["Date"] <= TRAIN_END].copy()

validation = df[
    (df["Date"] > TRAIN_END)
    & (df["Date"] <= VALIDATION_END)
].copy()

test = df[df["Date"] > VALIDATION_END].copy()


print("\n" + "=" * 80)
print("CHRONOLOGICAL SPLIT")
print("=" * 80)

print(f"Training:   {len(train):,} matches")
print(f"Validation: {len(validation):,} matches")
print(f"Test:       {len(test):,} matches")

print("\nDate ranges:")

for name, data in [
    ("Training", train),
    ("Validation", validation),
    ("Test", test),
]:
    if len(data) > 0:
        print(
            f"{name:<12}: "
            f"{data['Date'].min().date()} -> "
            f"{data['Date'].max().date()}"
        )


# =============================================================================
# FEATURE GROUPS
# =============================================================================

# Team/form and market feature lists now live in config.py (single source
# of truth shared with backtest.py and walk_forward.py).
TEAM_FEATURES = cfg.TEAM_FEATURES
MARKET_FEATURES = cfg.MARKET_FEATURES


# =============================================================================
# CHECK FEATURES
# =============================================================================

print("\n" + "=" * 80)
print("FEATURE CHECK")
print("=" * 80)

for group_name, features in [
    ("TEAM", TEAM_FEATURES),
    ("MARKET", MARKET_FEATURES),
]:

    missing = [f for f in features if f not in df.columns]

    if missing:
        print(f"\n{group_name} features missing:")

        for feature in missing:
            print(f"  MISSING: {feature}")

        raise ValueError(
            f"{group_name} feature columns are missing."
        )

    print(
        f"{group_name} features: "
        f"{len(features)} available"
    )


# =============================================================================
# PREPARE MODEL DATA
# =============================================================================

y_train = train["target"]
y_validation = validation["target"]
y_test = test["target"]


# =============================================================================
# MODEL FUNCTION
# =============================================================================

def create_model(features):

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                features,
            )
        ]
    )

    model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    return model


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_model(name, model, features):

    print("\n" + "=" * 80)
    print(f"{name}")
    print("=" * 80)

    X_train = train[features]
    X_validation = validation[features]
    X_test = test[features]

    print(f"Features: {len(features)}")

    # Train
    model.fit(
        X_train,
        y_train,
    )

    # Probabilities
    validation_prob = model.predict_proba(
        X_validation
    )

    test_prob = model.predict_proba(
        X_test
    )

    # Predictions
    validation_pred = model.predict(
        X_validation
    )

    test_pred = model.predict(
        X_test
    )

    # =========================================================================
    # ACCURACY
    # =========================================================================

    validation_accuracy = accuracy_score(
        y_validation,
        validation_pred,
    )

    test_accuracy = accuracy_score(
        y_test,
        test_pred,
    )

    # =========================================================================
    # LOG LOSS
    # =========================================================================

    validation_logloss = log_loss(
        y_validation,
        validation_prob,
    )

    test_logloss = log_loss(
        y_test,
        test_prob,
    )

    # =========================================================================
    # MULTICLASS BRIER SCORE
    # =========================================================================

    validation_actual = (
        pd.get_dummies(
            y_validation
        )
        .reindex(
            columns=[0, 1, 2],
            fill_value=0,
        )
        .values
    )

    test_actual = (
        pd.get_dummies(
            y_test
        )
        .reindex(
            columns=[0, 1, 2],
            fill_value=0,
        )
        .values
    )

    validation_brier = np.mean(
        np.sum(
            (
                validation_prob
                - validation_actual
            ) ** 2,
            axis=1,
        )
    )

    test_brier = np.mean(
        np.sum(
            (
                test_prob
                - test_actual
            ) ** 2,
            axis=1,
        )
    )

    # =========================================================================
    # PRINT RESULTS
    # =========================================================================

    print("\nVALIDATION")
    print(
        f"Accuracy: {validation_accuracy:.4f}"
    )
    print(
        f"Log Loss: {validation_logloss:.4f}"
    )
    print(
        f"Brier:    {validation_brier:.4f}"
    )

    print("\nTEST")
    print(
        f"Accuracy: {test_accuracy:.4f}"
    )
    print(
        f"Log Loss: {test_logloss:.4f}"
    )
    print(
        f"Brier:    {test_brier:.4f}"
    )

    return {
        "name": name,

        "validation_accuracy":
            validation_accuracy,

        "validation_logloss":
            validation_logloss,

        "validation_brier":
            validation_brier,

        "test_accuracy":
            test_accuracy,

        "test_logloss":
            test_logloss,

        "test_brier":
            test_brier,

        "model": model,
    }


# =============================================================================
# TRAIN THREE MODELS
# =============================================================================

results = []


# =============================================================================
# MODEL A — TEAM / FORM ONLY
# =============================================================================

results.append(
    evaluate_model(
        "MODEL A — TEAM / FORM ONLY",
        create_model(TEAM_FEATURES),
        TEAM_FEATURES,
    )
)


# =============================================================================
# MODEL B — MARKET ONLY
# =============================================================================

results.append(
    evaluate_model(
        "MODEL B — MARKET ONLY",
        create_model(MARKET_FEATURES),
        MARKET_FEATURES,
    )
)


# =============================================================================
# MODEL C — TEAM + MARKET
# =============================================================================

COMBINED_FEATURES = (
    TEAM_FEATURES
    + MARKET_FEATURES
)

results.append(
    evaluate_model(
        "MODEL C — TEAM + MARKET",
        create_model(COMBINED_FEATURES),
        COMBINED_FEATURES,
    )
)


# =============================================================================
# COMPARISON
# =============================================================================

print("\n" + "=" * 80)
print("MODEL COMPARISON")
print("=" * 80)

comparison = pd.DataFrame(
    [
        {
            "Model": r["name"],

            "Validation Accuracy":
                r["validation_accuracy"],

            "Validation LogLoss":
                r["validation_logloss"],

            "Validation Brier":
                r["validation_brier"],

            "Test Accuracy":
                r["test_accuracy"],

            "Test LogLoss":
                r["test_logloss"],

            "Test Brier":
                r["test_brier"],
        }

        for r in results
    ]
)

print(
    comparison.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# =============================================================================
# SAVE RESULTS
# =============================================================================

cfg.MODELS_DIR.mkdir(
    exist_ok=True
)

comparison.to_csv(
    cfg.MODELS_DIR / "model_comparison.csv",
    index=False,
)

print("\nResults saved:")
print("models/model_comparison.csv")

print("\n" + "=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)
