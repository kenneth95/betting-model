import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    log_loss,
)


# =============================================================================
# CONFIG
# =============================================================================

INPUT_FILE = Path("data/processed/epl_model_data.csv")

TRAIN_END = "2024-05-31"
VALIDATION_END = "2025-05-31"

RANDOM_STATE = 42


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

# Team/form features created in Step 7
TEAM_FEATURES = [
    "home_form_points_5",
    "away_form_points_5",

    "home_goals_for_5_overall",
    "away_goals_for_5_overall",

    "home_goals_against_5_overall",
    "away_goals_against_5_overall",

    "home_wins_5",
    "away_wins_5",

    "home_draws_5",
    "away_draws_5",

    "home_losses_5",
    "away_losses_5",

    "home_points_5",
    "away_points_5",

    "home_goals_for_5",
    "away_goals_for_5",

    "home_goals_against_5",
    "away_goals_against_5",

    "form_points_diff",
    "goals_for_diff",
    "goals_against_diff",
    "wins_diff",

    "home_away_points_diff",
    "home_away_goals_for_diff",
    "home_away_goals_against_diff",
]


# Market features
MARKET_FEATURES = [
    "market_prob_home",
    "market_prob_draw",
    "market_prob_away",
    "market_overround",

    "b365_prob_home",
    "b365_prob_draw",
    "b365_prob_away",
    "b365_overround",

    "b365_market_diff_home",
    "b365_market_diff_draw",
    "b365_market_diff_away",

    "market_range_home",
    "market_range_draw",
    "market_range_away",

    "market_source_avg",
]


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

Path("models").mkdir(
    exist_ok=True
)

comparison.to_csv(
    "models/model_comparison.csv",
    index=False,
)

print("\nResults saved:")
print("models/model_comparison.csv")

print("\n" + "=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)
