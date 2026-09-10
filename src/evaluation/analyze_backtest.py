from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import config

# =============================================================================
# CONFIG
# =============================================================================

INPUT_PATH = config.MODELS_DIR / "test_predictions_2025_26.csv"

OUTPUT_DIR = config.MODELS_DIR

BET_DETAIL_PATH = config.MODELS_DIR / "backtest_model_c_5pct_ev.csv"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =============================================================================
# LOAD DATA
# =============================================================================

print("=" * 80)
print("EPL BACKTEST ANALYSIS")
print("=" * 80)

print()

predictions = pd.read_csv(INPUT_PATH)

predictions["date"] = pd.to_datetime(
    predictions["Date"]
)

print(
    f"Test predictions loaded: "
    f"{len(predictions):,}"
)


# =============================================================================
# LOAD DETAILED BETS
# =============================================================================

bets_5 = pd.read_csv(
    BET_DETAIL_PATH
)

bets_5["date"] = pd.to_datetime(
    bets_5["date"]
)

print(
    f"5% EV bets loaded: "
    f"{len(bets_5):,}"
)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def profit_from_bet(row):

    if row["won"] == 1:
        return row["odds"] - 1

    return -1


def wilson_interval(
    wins,
    n,
    z=1.96
):

    if n == 0:
        return 0.0, 0.0

    p = wins / n

    denominator = (
        1 +
        z**2 / n
    )

    centre = (
        p +
        z**2 / (2 * n)
    ) / denominator

    margin = (
        z *
        np.sqrt(
            (
                p * (1 - p) / n
            )
            +
            (
                z**2 / (4 * n**2)
            )
        )
    ) / denominator

    return (
        max(0, centre - margin),
        min(1, centre + margin)
    )


def summarize_bets(
    data,
    group_column=None
):

    if len(data) == 0:
        return pd.DataFrame()

    if group_column is None:

        grouped = [
            ("ALL", data)
        ]

    else:

        grouped = data.groupby(
            group_column,
            dropna=False
        )

    rows = []

    for group_name, group in grouped:

        n = len(group)

        wins = int(
            group["won"].sum()
        )

        stake = group["stake"].sum()

        profit = group["profit"].sum()

        roi = (
            profit / stake
            if stake > 0
            else 0
        )

        win_rate = (
            wins / n
            if n > 0
            else 0
        )

        lower, upper = wilson_interval(
            wins,
            n
        )

        rows.append(
            {
                "group": group_name,
                "bets": n,
                "wins": wins,
                "win_rate": win_rate,
                "win_rate_lower_95": lower,
                "win_rate_upper_95": upper,
                "profit": profit,
                "roi": roi,
                "avg_odds": group["odds"].mean(),
                "avg_ev": group["ev"].mean(),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# 1. ODDS BUCKETS
# =============================================================================

print()
print("=" * 80)
print("1. ROI BY ODDS RANGE — MODEL C, 5% EV")
print("=" * 80)

odds_bins = [
    1.00,
    1.50,
    2.00,
    3.00,
    5.00,
    10.00,
    100.00,
]

odds_labels = [
    "1.00–1.49",
    "1.50–1.99",
    "2.00–2.99",
    "3.00–4.99",
    "5.00–9.99",
    "10.00+",
]

bets_5["odds_range"] = pd.cut(
    bets_5["odds"],
    bins=odds_bins,
    labels=odds_labels,
    right=False
)

odds_analysis = summarize_bets(
    bets_5,
    "odds_range"
)

print(
    odds_analysis.to_string(
        index=False,
        formatters={
            "win_rate": "{:.2%}".format,
            "win_rate_lower_95": "{:.2%}".format,
            "win_rate_upper_95": "{:.2%}".format,
            "profit": "${:.2f}".format,
            "roi": "{:.2%}".format,
            "avg_odds": "{:.3f}".format,
            "avg_ev": "{:.2%}".format,
        }
    )
)

odds_analysis.to_csv(
    OUTPUT_DIR /
    "analysis_odds_ranges.csv",
    index=False
)


# =============================================================================
# 2. HOME / DRAW / AWAY
# =============================================================================

print()
print("=" * 80)
print("2. PERFORMANCE BY SELECTION")
print("=" * 80)

selection_analysis = summarize_bets(
    bets_5,
    "selection"
)

print(
    selection_analysis.to_string(
        index=False,
        formatters={
            "win_rate": "{:.2%}".format,
            "win_rate_lower_95": "{:.2%}".format,
            "win_rate_upper_95": "{:.2%}".format,
            "profit": "${:.2f}".format,
            "roi": "{:.2%}".format,
            "avg_odds": "{:.3f}".format,
            "avg_ev": "{:.2%}".format,
        }
    )
)

selection_analysis.to_csv(
    OUTPUT_DIR /
    "analysis_selection.csv",
    index=False
)


# =============================================================================
# 3. MONTHLY PERFORMANCE
# =============================================================================

print()
print("=" * 80)
print("3. MONTHLY PERFORMANCE")
print("=" * 80)

bets_5["month"] = (
    bets_5["date"]
    .dt.to_period("M")
    .astype(str)
)

monthly_analysis = summarize_bets(
    bets_5,
    "month"
)

print(
    monthly_analysis.to_string(
        index=False,
        formatters={
            "win_rate": "{:.2%}".format,
            "win_rate_lower_95": "{:.2%}".format,
            "win_rate_upper_95": "{:.2%}".format,
            "profit": "${:.2f}".format,
            "roi": "{:.2%}".format,
            "avg_odds": "{:.3f}".format,
            "avg_ev": "{:.2%}".format,
        }
    )
)

monthly_analysis.to_csv(
    OUTPUT_DIR /
    "analysis_monthly.csv",
    index=False
)


# =============================================================================
# 4. EV BUCKETS
# =============================================================================

print()
print("=" * 80)
print("4. PERFORMANCE BY PREDICTED EV")
print("=" * 80)

ev_bins = [
    0.05,
    0.10,
    0.15,
    0.20,
    0.30,
    1.00,
]

ev_labels = [
    "5–9.9%",
    "10–14.9%",
    "15–19.9%",
    "20–29.9%",
    "30%+",
]

bets_5["ev_range"] = pd.cut(
    bets_5["ev"],
    bins=ev_bins,
    labels=ev_labels,
    right=False,
    include_lowest=True
)

ev_analysis = summarize_bets(
    bets_5,
    "ev_range"
)

print(
    ev_analysis.to_string(
        index=False,
        formatters={
            "win_rate": "{:.2%}".format,
            "win_rate_lower_95": "{:.2%}".format,
            "win_rate_upper_95": "{:.2%}".format,
            "profit": "${:.2f}".format,
            "roi": "{:.2%}".format,
            "avg_odds": "{:.3f}".format,
            "avg_ev": "{:.2%}".format,
        }
    )
)

ev_analysis.to_csv(
    OUTPUT_DIR /
    "analysis_ev_ranges.csv",
    index=False
)


# =============================================================================
# 5. CALIBRATION ANALYSIS
# =============================================================================

print()
print("=" * 80)
print("5. PROBABILITY CALIBRATION")
print("=" * 80)

# -------------------------------------------------------------------------
# For each bet, probability represents the model's probability of the
# selected outcome.
# -------------------------------------------------------------------------

prob_bins = [
    0.00,
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    1.00,
]

prob_labels = [
    "0–9.9%",
    "10–19.9%",
    "20–29.9%",
    "30–39.9%",
    "40–49.9%",
    "50–59.9%",
    "60–69.9%",
    "70–79.9%",
    "80–89.9%",
    "90–100%",
]

bets_5["probability_range"] = pd.cut(
    bets_5["probability"],
    bins=prob_bins,
    labels=prob_labels,
    right=False,
    include_lowest=True
)

calibration_rows = []

for group_name, group in bets_5.groupby(
    "probability_range",
    observed=False
):

    if len(group) == 0:
        continue

    calibration_rows.append(
        {
            "probability_range": group_name,
            "bets": len(group),
            "average_predicted_probability":
                group["probability"].mean(),
            "actual_win_rate":
                group["won"].mean(),
            "profit":
                group["profit"].sum(),
            "roi":
                group["profit"].sum()
                /
                group["stake"].sum(),
        }
    )

calibration = pd.DataFrame(
    calibration_rows
)

print(
    calibration.to_string(
        index=False,
        formatters={
            "average_predicted_probability":
                "{:.2%}".format,
            "actual_win_rate":
                "{:.2%}".format,
            "profit":
                "${:.2f}".format,
            "roi":
                "{:.2%}".format,
        }
    )
)

calibration.to_csv(
    OUTPUT_DIR /
    "analysis_calibration.csv",
    index=False
)


# =============================================================================
# 6. CUMULATIVE PROFIT
# =============================================================================

print()
print("=" * 80)
print("6. CUMULATIVE PROFIT")
print("=" * 80)

bets_5 = bets_5.sort_values(
    "date"
).reset_index(drop=True)

bets_5["cumulative_profit"] = (
    bets_5["profit"].cumsum()
)

bets_5["peak_profit"] = (
    bets_5["cumulative_profit"]
    .cummax()
)

bets_5["drawdown"] = (
    bets_5["cumulative_profit"]
    -
    bets_5["peak_profit"]
)

max_drawdown = (
    bets_5["drawdown"].min()
)

final_profit = (
    bets_5["cumulative_profit"].iloc[-1]
    if len(bets_5) > 0
    else 0
)

print(
    f"Final cumulative profit: "
    f"${final_profit:.2f}"
)

print(
    f"Maximum drawdown: "
    f"${max_drawdown:.2f}"
)

bets_5.to_csv(
    OUTPUT_DIR /
    "analysis_5pct_ev_bets.csv",
    index=False
)


# =============================================================================
# 7. MODEL C VS MARKET MODEL
# =============================================================================

print()
print("=" * 80)
print("7. MODEL COMPARISON")
print("=" * 80)

# Reconstruct Model B 5% EV bets from the saved predictions.
#
# The saved prediction file contains both models' probabilities.

def generate_bets(
    predictions,
    model_name,
    home_col,
    draw_col,
    away_col,
    threshold=0.05
):

    results = []

    for _, row in predictions.iterrows():

        outcomes = [
            (
                "H",
                row[home_col],
                row["B365H"]
            ),
            (
                "D",
                row[draw_col],
                row["B365D"]
            ),
            (
                "A",
                row[away_col],
                row["B365A"]
            ),
        ]

        candidates = []

        for selection, probability, odds in outcomes:

            if pd.isna(probability):
                continue

            if pd.isna(odds):
                continue

            if odds <= 1:
                continue

            ev = (
                probability * odds
            ) - 1

            candidates.append(
                {
                    "selection": selection,
                    "probability": probability,
                    "odds": odds,
                    "ev": ev,
                }
            )

        if not candidates:
            continue

        best = max(
            candidates,
            key=lambda x: x["ev"]
        )

        if best["ev"] < threshold:
            continue

        won = (
            best["selection"] ==
            row["FTR"]
        )

        profit = (
            best["odds"] - 1
            if won
            else -1
        )

        results.append(
            {
                "model": model_name,
                "date": row["Date"],
                "selection": best["selection"],
                "probability":
                    best["probability"],
                "odds": best["odds"],
                "ev": best["ev"],
                "won": int(won),
                "profit": profit,
                "stake": 1.0,
            }
        )

    return pd.DataFrame(results)


market_bets = generate_bets(
    predictions,
    "MODEL B — MARKET ONLY",
    "market_model_home_prob",
    "market_model_draw_prob",
    "market_model_away_prob",
    threshold=0.05
)

combined_bets = generate_bets(
    predictions,
    "MODEL C — TEAM + MARKET",
    "combined_model_home_prob",
    "combined_model_draw_prob",
    "combined_model_away_prob",
    threshold=0.05
)

comparison_rows = []

for name, data in [
    ("MODEL B — MARKET ONLY", market_bets),
    ("MODEL C — TEAM + MARKET", combined_bets),
]:

    if len(data) == 0:
        continue

    profit = data["profit"].sum()
    stake = data["stake"].sum()

    comparison_rows.append(
        {
            "model": name,
            "bets": len(data),
            "wins": data["won"].sum(),
            "win_rate":
                data["won"].mean(),
            "profit": profit,
            "roi":
                profit / stake,
            "avg_odds":
                data["odds"].mean(),
            "avg_ev":
                data["ev"].mean(),
        }
    )

comparison = pd.DataFrame(
    comparison_rows
)

print(
    comparison.to_string(
        index=False,
        formatters={
            "win_rate": "{:.2%}".format,
            "profit": "${:.2f}".format,
            "roi": "{:.2%}".format,
            "avg_odds": "{:.3f}".format,
            "avg_ev": "{:.2%}".format,
        }
    )
)

comparison.to_csv(
    OUTPUT_DIR /
    "analysis_model_comparison_5pct.csv",
    index=False
)


# =============================================================================
# FINAL SUMMARY
# =============================================================================

print()
print("=" * 80)
print("KEY RESULTS")
print("=" * 80)

print()

if len(bets_5) > 0:

    print(
        f"MODEL C — 5% EV"
    )

    print(
        f"Number of bets: "
        f"{len(bets_5)}"
    )

    print(
        f"Wins: "
        f"{bets_5['won'].sum()}"
    )

    print(
        f"Win rate: "
        f"{bets_5['won'].mean():.2%}"
    )

    print(
        f"Profit: "
        f"${bets_5['profit'].sum():.2f}"
    )

    print(
        f"ROI: "
        f"{bets_5['profit'].sum() / bets_5['stake'].sum():.2%}"
    )

    print(
        f"Average odds: "
        f"{bets_5['odds'].mean():.3f}"
    )

    print(
        f"Average predicted EV: "
        f"{bets_5['ev'].mean():.2%}"
    )

    print(
        f"Maximum drawdown: "
        f"${max_drawdown:.2f}"
    )

else:

    print(
        "No 5% EV bets available."
    )


print()
print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

print()

print("Files created:")

print(
    "  models/analysis_odds_ranges.csv"
)

print(
    "  models/analysis_selection.csv"
)

print(
    "  models/analysis_monthly.csv"
)

print(
    "  models/analysis_ev_ranges.csv"
)

print(
    "  models/analysis_calibration.csv"
)

print(
    "  models/analysis_5pct_ev_bets.csv"
)

print(
    "  models/analysis_model_comparison_5pct.csv"
)
