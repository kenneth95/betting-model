from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# =============================================================================
# CONFIG
# =============================================================================

DATA_PATH = Path("data/processed/epl_model_data.csv")
OUTPUT_PATH = Path("models/backtest_results.csv")
DETAIL_OUTPUT_PATH = Path("models/backtest_model_c_5pct_ev.csv")

RANDOM_STATE = 42
STAKE = 1.00

EV_THRESHOLDS = [
    0.00,
    0.02,
    0.05,
    0.10,
    0.15,
    0.20,
]

MIN_PROBABILITY = 0.00


# =============================================================================
# FEATURES
# =============================================================================

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
# MODEL
# =============================================================================

def make_pipeline():

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                )
            ),
        ]
    )


# =============================================================================
# METRICS
# =============================================================================

def calculate_drawdown(profits):

    if len(profits) == 0:
        return 0.0

    cumulative = profits.cumsum()
    peak = cumulative.cummax()

    drawdown = cumulative - peak

    return drawdown.min()


def calculate_metrics(bets):

    if len(bets) == 0:

        return {
            "bets": 0,
            "wins": 0,
            "win_rate": 0.0,
            "stake": 0.0,
            "profit": 0.0,
            "roi": 0.0,
            "avg_odds": 0.0,
            "avg_ev": 0.0,
            "max_drawdown": 0.0,
        }

    total_stake = bets["stake"].sum()
    total_profit = bets["profit"].sum()

    wins = bets["won"].sum()

    win_rate = wins / len(bets)

    roi = (
        total_profit / total_stake
        if total_stake > 0
        else 0
    )

    return {
        "bets": len(bets),
        "wins": int(wins),
        "win_rate": win_rate,
        "stake": total_stake,
        "profit": total_profit,
        "roi": roi,
        "avg_odds": bets["odds"].mean(),
        "avg_ev": bets["ev"].mean(),
        "max_drawdown": calculate_drawdown(
            bets["profit"]
        ),
    }


# =============================================================================
# LOAD DATA
# =============================================================================

print("=" * 80)
print("EPL BETTING BACKTEST")
print("=" * 80)

print()

df = pd.read_csv(DATA_PATH)

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values("Date").reset_index(drop=True)

print(f"Matches loaded: {len(df):,}")


# =============================================================================
# REQUIRED COLUMN CHECK
# =============================================================================

print()
print("=" * 80)
print("REQUIRED COLUMN CHECK")
print("=" * 80)

required_columns = [
    "Date",
    "HomeTeam",
    "AwayTeam",
    "FTR",
    "target",
    "B365H",
    "B365D",
    "B365A",
]

for col in required_columns:

    if col not in df.columns:
        raise ValueError(
            f"Missing required column: {col}"
        )

    print(f"{col}: OK")


# =============================================================================
# BOOKMAKER ODDS CHECK
# =============================================================================

print()
print("=" * 80)
print("BOOKMAKER ODDS CHECK")
print("=" * 80)

for col in ["B365H", "B365D", "B365A"]:

    print(
        f"{col}: "
        f"{df[col].notna().sum():,} available"
    )


# =============================================================================
# CHRONOLOGICAL SPLIT
# =============================================================================
#
# IMPORTANT:
#
# epl_model_data.csv does not contain a Season column.
#
# Therefore we use the exact date boundaries used by train_models.py:
#
# Training:
#   2016/17 through 2023/24
#
# Validation:
#   2024/25
#
# Test:
#   2025/26
#
# The backtest uses ONLY the 2025/26 test season.
#
# =============================================================================

TRAIN_END = pd.Timestamp("2024-05-19")

TEST_START = pd.Timestamp("2025-08-15")

train_df = df[
    df["Date"] <= TRAIN_END
].copy()

test_df = df[
    df["Date"] >= TEST_START
].copy()

print()
print("=" * 80)
print("BACKTEST SPLIT")
print("=" * 80)

print(
    f"Training matches: {len(train_df):,}"
)

print(
    f"Test matches:     {len(test_df):,}"
)

print()

print(
    f"Training dates: "
    f"{train_df['Date'].min().date()} -> "
    f"{train_df['Date'].max().date()}"
)

print(
    f"Test dates:     "
    f"{test_df['Date'].min().date()} -> "
    f"{test_df['Date'].max().date()}"
)

# Safety checks
if len(train_df) != 3040:

    raise ValueError(
        f"Expected 3,040 training matches, "
        f"found {len(train_df):,}"
    )

if len(test_df) != 380:

    raise ValueError(
        f"Expected 380 test matches, "
        f"found {len(test_df):,}"
    )

print()
print("Split validation: OK")


# =============================================================================
# FEATURE CHECK
# =============================================================================

print()
print("=" * 80)
print("FEATURE CHECK")
print("=" * 80)

for col in TEAM_FEATURES + MARKET_FEATURES:

    if col not in df.columns:

        raise ValueError(
            f"Missing feature: {col}"
        )

print(
    f"TEAM features:   {len(TEAM_FEATURES)} available"
)

print(
    f"MARKET features: {len(MARKET_FEATURES)} available"
)


# =============================================================================
# TARGET
# =============================================================================

X_train_team = train_df[TEAM_FEATURES]
X_test_team = test_df[TEAM_FEATURES]

X_train_market = train_df[MARKET_FEATURES]
X_test_market = test_df[MARKET_FEATURES]

y_train = train_df["target"]


# =============================================================================
# MODEL B — MARKET ONLY
# =============================================================================

print()
print("=" * 80)
print("MODEL B — MARKET ONLY")
print("=" * 80)

market_model = make_pipeline()

market_model.fit(
    X_train_market,
    y_train
)

market_probabilities = market_model.predict_proba(
    X_test_market
)

print("Training complete.")
print("Test probabilities generated.")


# =============================================================================
# MODEL C — TEAM + MARKET
# =============================================================================

print()
print("=" * 80)
print("MODEL C — TEAM + MARKET")
print("=" * 80)

combined_features = (
    TEAM_FEATURES +
    MARKET_FEATURES
)

X_train_combined = train_df[
    combined_features
]

X_test_combined = test_df[
    combined_features
]

combined_model = make_pipeline()

combined_model.fit(
    X_train_combined,
    y_train
)

combined_probabilities = combined_model.predict_proba(
    X_test_combined
)

print("Training complete.")
print("Test probabilities generated.")


# =============================================================================
# CREATE PREDICTIONS
# =============================================================================

predictions = test_df[
    [
        "Date",
        "HomeTeam",
        "AwayTeam",
        "FTR",
        "target",
        "B365H",
        "B365D",
        "B365A",
    ]
].copy()


# Model B
predictions["market_model_home_prob"] = (
    market_probabilities[:, 0]
)

predictions["market_model_draw_prob"] = (
    market_probabilities[:, 1]
)

predictions["market_model_away_prob"] = (
    market_probabilities[:, 2]
)


# Model C
predictions["combined_model_home_prob"] = (
    combined_probabilities[:, 0]
)

predictions["combined_model_draw_prob"] = (
    combined_probabilities[:, 1]
)

predictions["combined_model_away_prob"] = (
    combined_probabilities[:, 2]
)


# =============================================================================
# BACKTEST
# =============================================================================

def run_backtest(
    predictions,
    model_name,
    home_prob_col,
    draw_prob_col,
    away_prob_col,
    ev_threshold,
):

    bets = []

    for _, row in predictions.iterrows():

        outcomes = [
            {
                "selection": "H",
                "probability": row[home_prob_col],
                "odds": row["B365H"],
            },
            {
                "selection": "D",
                "probability": row[draw_prob_col],
                "odds": row["B365D"],
            },
            {
                "selection": "A",
                "probability": row[away_prob_col],
                "odds": row["B365A"],
            },
        ]

        valid = []

        for outcome in outcomes:

            probability = outcome["probability"]
            odds = outcome["odds"]

            if pd.isna(probability):
                continue

            if pd.isna(odds):
                continue

            if odds <= 1:
                continue

            if probability < MIN_PROBABILITY:
                continue

            ev = (
                probability * odds
            ) - 1

            outcome["ev"] = ev

            valid.append(outcome)

        if not valid:
            continue

        # Highest-EV selection
        best = max(
            valid,
            key=lambda x: x["ev"]
        )

        # Require minimum EV
        if best["ev"] < ev_threshold:
            continue

        won = (
            best["selection"] ==
            row["FTR"]
        )

        if won:

            profit = (
                STAKE *
                (best["odds"] - 1)
            )

        else:

            profit = -STAKE

        bets.append(
            {
                "date": row["Date"],
                "home_team": row["HomeTeam"],
                "away_team": row["AwayTeam"],
                "actual_result": row["FTR"],

                "model": model_name,

                "selection": best["selection"],

                "probability": best["probability"],

                "fair_odds": (
                    1 /
                    best["probability"]
                ),

                "odds": best["odds"],

                "ev": best["ev"],

                "stake": STAKE,

                "won": int(won),

                "profit": profit,
            }
        )

    return pd.DataFrame(bets)


# =============================================================================
# RUN ALL EV THRESHOLDS
# =============================================================================

all_results = []

for threshold in EV_THRESHOLDS:

    print()
    print("=" * 80)
    print(
        f"EV THRESHOLD: {threshold:.0%}"
    )
    print("=" * 80)

    # -------------------------------------------------------------------------
    # MODEL B
    # -------------------------------------------------------------------------

    market_bets = run_backtest(
        predictions,
        "MODEL B — MARKET ONLY",
        "market_model_home_prob",
        "market_model_draw_prob",
        "market_model_away_prob",
        threshold,
    )

    market_metrics = calculate_metrics(
        market_bets
    )

    print()
    print("MODEL B — MARKET ONLY")

    print(
        f"Bets:          {market_metrics['bets']}"
    )

    print(
        f"Wins:          {market_metrics['wins']}"
    )

    print(
        f"Win rate:      "
        f"{market_metrics['win_rate']:.2%}"
    )

    print(
        f"Total stake:   "
        f"${market_metrics['stake']:.2f}"
    )

    print(
        f"Profit:        "
        f"${market_metrics['profit']:.2f}"
    )

    print(
        f"ROI:           "
        f"{market_metrics['roi']:.2%}"
    )

    print(
        f"Average odds:  "
        f"{market_metrics['avg_odds']:.3f}"
    )

    print(
        f"Average EV:    "
        f"{market_metrics['avg_ev']:.2%}"
    )

    print(
        f"Max drawdown:  "
        f"${market_metrics['max_drawdown']:.2f}"
    )

    all_results.append(
        {
            "model": "MODEL B — MARKET ONLY",
            "ev_threshold": threshold,
            **market_metrics,
        }
    )

    # -------------------------------------------------------------------------
    # MODEL C
    # -------------------------------------------------------------------------

    combined_bets = run_backtest(
        predictions,
        "MODEL C — TEAM + MARKET",
        "combined_model_home_prob",
        "combined_model_draw_prob",
        "combined_model_away_prob",
        threshold,
    )

    combined_metrics = calculate_metrics(
        combined_bets
    )

    print()
    print("MODEL C — TEAM + MARKET")

    print(
        f"Bets:          {combined_metrics['bets']}"
    )

    print(
        f"Wins:          {combined_metrics['wins']}"
    )

    print(
        f"Win rate:      "
        f"{combined_metrics['win_rate']:.2%}"
    )

    print(
        f"Total stake:   "
        f"${combined_metrics['stake']:.2f}"
    )

    print(
        f"Profit:        "
        f"${combined_metrics['profit']:.2f}"
    )

    print(
        f"ROI:           "
        f"{combined_metrics['roi']:.2%}"
    )

    print(
        f"Average odds:  "
        f"{combined_metrics['avg_odds']:.3f}"
    )

    print(
        f"Average EV:    "
        f"{combined_metrics['avg_ev']:.2%}"
    )

    print(
        f"Max drawdown:  "
        f"${combined_metrics['max_drawdown']:.2f}"
    )

    all_results.append(
        {
            "model": "MODEL C — TEAM + MARKET",
            "ev_threshold": threshold,
            **combined_metrics,
        }
    )


# =============================================================================
# SUMMARY
# =============================================================================

results_df = pd.DataFrame(
    all_results
)

print()
print("=" * 80)
print("BACKTEST SUMMARY")
print("=" * 80)

summary_columns = [
    "model",
    "ev_threshold",
    "bets",
    "wins",
    "win_rate",
    "profit",
    "roi",
    "avg_odds",
    "avg_ev",
    "max_drawdown",
]

summary = results_df[
    summary_columns
].copy()

print(
    summary.to_string(
        index=False,
        formatters={
            "ev_threshold": (
                "{:.0%}".format
            ),
            "win_rate": (
                "{:.2%}".format
            ),
            "profit": (
                "${:.2f}".format
            ),
            "roi": (
                "{:.2%}".format
            ),
            "avg_odds": (
                "{:.3f}".format
            ),
            "avg_ev": (
                "{:.2%}".format
            ),
            "max_drawdown": (
                "${:.2f}".format
            ),
        },
    )
)


# =============================================================================
# SAVE SUMMARY
# =============================================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)


# =============================================================================
# SAVE DETAILED 5% EV MODEL C BETS
# =============================================================================

detail_bets = run_backtest(
    predictions,
    "MODEL C — TEAM + MARKET",
    "combined_model_home_prob",
    "combined_model_draw_prob",
    "combined_model_away_prob",
    0.05,
)

detail_bets.to_csv(
    DETAIL_OUTPUT_PATH,
    index=False
)


# =============================================================================
# SAVE ALL TEST PREDICTIONS
# =============================================================================

prediction_path = Path(
    "models/test_predictions_2025_26.csv"
)

predictions.to_csv(
    prediction_path,
    index=False
)


# =============================================================================
# COMPLETE
# =============================================================================

print()
print("=" * 80)
print("BACKTEST COMPLETE")
print("=" * 80)

print()

print(
    f"Summary saved: "
    f"{OUTPUT_PATH}"
)

print(
    f"Detailed 5% EV bets saved: "
    f"{DETAIL_OUTPUT_PATH}"
)

print(
    f"Test predictions saved: "
    f"{prediction_path}"
)

print()

print("Files created:")
print("  1. models/backtest_results.csv")
print("  2. models/backtest_model_c_5pct_ev.csv")
print("  3. models/test_predictions_2025_26.csv")
