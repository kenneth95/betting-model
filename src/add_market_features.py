from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path("data/processed/epl_features.csv")
OUTPUT_FILE = Path("data/processed/epl_model_data.csv")

# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("ADDING MARKET / ODDS FEATURES")
print("=" * 80)

df = pd.read_csv(INPUT_FILE)

print(f"\nMatches loaded: {len(df):,}")

# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    "AvgH",
    "AvgD",
    "AvgA",
    "B365H",
    "B365D",
    "B365A"
]

print("\nChecking odds columns...")

for column in required_columns:

    if column not in df.columns:
        raise ValueError(
            f"Required odds column missing: {column}"
        )

    available = df[column].notna().sum()

    print(
        f"  OK  {column:<8} "
        f"{available:,} available "
        f"({available / len(df) * 100:.1f}%)"
    )

# ============================================================
# DETERMINE MARKET SOURCE
# ============================================================

print("\nDetermining market probability source...")

# AvgH/D/A are available for 2,660 matches.
#
# When all three are available, use the average market.
#
# When they are unavailable, use Bet365 as the fallback.
#
# This allows us to retain all 3,800 matches.

avg_available = (
    df["AvgH"].notna()
    & df["AvgD"].notna()
    & df["AvgA"].notna()
)

df["market_source_avg"] = avg_available.astype(int)

print(
    f"Average market available: "
    f"{avg_available.sum():,}"
)

print(
    f"Bet365 fallback required: "
    f"{(~avg_available).sum():,}"
)

# ============================================================
# SELECT ODDS
# ============================================================

# Start with Bet365 odds.

df["market_odds_home"] = df["B365H"]
df["market_odds_draw"] = df["B365D"]
df["market_odds_away"] = df["B365A"]

# Replace with average market odds where all
# three Avg odds are available.

df.loc[
    avg_available,
    "market_odds_home"
] = df.loc[
    avg_available,
    "AvgH"
]

df.loc[
    avg_available,
    "market_odds_draw"
] = df.loc[
    avg_available,
    "AvgD"
]

df.loc[
    avg_available,
    "market_odds_away"
] = df.loc[
    avg_available,
    "AvgA"
]

# ============================================================
# RAW IMPLIED PROBABILITIES
# ============================================================

print("\nCalculating implied probabilities...")

df["market_raw_prob_home"] = (
    1 / df["market_odds_home"]
)

df["market_raw_prob_draw"] = (
    1 / df["market_odds_draw"]
)

df["market_raw_prob_away"] = (
    1 / df["market_odds_away"]
)

# ============================================================
# MARKET OVERROUND
# ============================================================

df["market_overround"] = (
    df["market_raw_prob_home"]
    + df["market_raw_prob_draw"]
    + df["market_raw_prob_away"]
)

# ============================================================
# NORMALIZED MARKET PROBABILITIES
# ============================================================

df["market_prob_home"] = (
    df["market_raw_prob_home"]
    / df["market_overround"]
)

df["market_prob_draw"] = (
    df["market_raw_prob_draw"]
    / df["market_overround"]
)

df["market_prob_away"] = (
    df["market_raw_prob_away"]
    / df["market_overround"]
)

# ============================================================
# BET365 PROBABILITIES
# ============================================================

print("Calculating Bet365 probabilities...")

df["b365_raw_prob_home"] = (
    1 / df["B365H"]
)

df["b365_raw_prob_draw"] = (
    1 / df["B365D"]
)

df["b365_raw_prob_away"] = (
    1 / df["B365A"]
)

df["b365_overround"] = (
    df["b365_raw_prob_home"]
    + df["b365_raw_prob_draw"]
    + df["b365_raw_prob_away"]
)

df["b365_prob_home"] = (
    df["b365_raw_prob_home"]
    / df["b365_overround"]
)

df["b365_prob_draw"] = (
    df["b365_raw_prob_draw"]
    / df["b365_overround"]
)

df["b365_prob_away"] = (
    df["b365_raw_prob_away"]
    / df["b365_overround"]
)

# ============================================================
# MARKET DISAGREEMENT
# ============================================================

print("Calculating market disagreement features...")

df["b365_market_diff_home"] = (
    df["b365_prob_home"]
    - df["market_prob_home"]
)

df["b365_market_diff_draw"] = (
    df["b365_prob_draw"]
    - df["market_prob_draw"]
)

df["b365_market_diff_away"] = (
    df["b365_prob_away"]
    - df["market_prob_away"]
)

# ============================================================
# MARKET PRICE RANGE
# ============================================================

print("Calculating market price ranges...")

if all(
    column in df.columns
    for column in [
        "MaxH",
        "MaxD",
        "MaxA"
    ]
):

    df["market_range_home"] = (
        df["MaxH"] - df["market_odds_home"]
    )

    df["market_range_draw"] = (
        df["MaxD"] - df["market_odds_draw"]
    )

    df["market_range_away"] = (
        df["MaxA"] - df["market_odds_away"]
    )

# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 80)
print("MARKET FEATURE VALIDATION")
print("=" * 80)

# ------------------------------------------------------------
# ROW COUNT
# ------------------------------------------------------------

print(f"\nRows: {len(df):,}")

if len(df) != 3800:
    raise ValueError(
        f"Expected 3,800 rows, got {len(df):,}"
    )

print("Row count: OK")

# ------------------------------------------------------------
# MATCH UNIQUENESS
# ------------------------------------------------------------

unique_matches = df["match_id"].nunique()

print(
    f"Unique matches: {unique_matches:,}"
)

if unique_matches != 3800:
    raise ValueError(
        "Match ID uniqueness check failed."
    )

print("Match uniqueness: OK")

# ------------------------------------------------------------
# MARKET PROBABILITY VALIDATION
# ------------------------------------------------------------

print("\nChecking market probabilities...")

probability_columns = [
    "market_prob_home",
    "market_prob_draw",
    "market_prob_away"
]

market_sum = df[
    probability_columns
].sum(axis=1)

valid_probability_rows = (
    df[probability_columns]
    .notna()
    .all(axis=1)
)

valid_count = valid_probability_rows.sum()

print(
    f"Valid probability rows: "
    f"{valid_count:,} / {len(df):,}"
)

if valid_count > 0:

    max_error = (
        market_sum[valid_probability_rows] - 1
    ).abs().max()

    print(
        f"Maximum probability sum error: "
        f"{max_error:.10f}"
    )

    if max_error > 0.000001:
        raise ValueError(
            "Market probabilities do not sum to 1."
        )

    print(
        "Market probability normalization: OK"
    )

# ------------------------------------------------------------
# PROBABILITY RANGE CHECK
# ------------------------------------------------------------

print("\nChecking probability ranges...")

for column in probability_columns:

    minimum = df[column].min()
    maximum = df[column].max()

    print(
        f"  {column}: "
        f"{minimum:.4f} - {maximum:.4f}"
    )

    if minimum < 0 or maximum > 1:
        raise ValueError(
            f"Invalid probability range in {column}"
        )

print("Probability range: OK")

# ------------------------------------------------------------
# OVERROUND
# ------------------------------------------------------------

print("\nMarket overround summary:")

print(
    df["market_overround"]
    .describe()
    .to_string()
)

market_margin = (
    df["market_overround"] - 1
) * 100

print(
    f"\nAverage market margin: "
    f"{market_margin.mean():.2f}%"
)

print(
    f"Median market margin: "
    f"{market_margin.median():.2f}%"
)

# ------------------------------------------------------------
# MARKET SOURCE
# ------------------------------------------------------------

print("\nMarket source:")

print(
    f"Average market: "
    f"{(df['market_source_avg'] == 1).sum():,}"
)

print(
    f"Bet365 fallback: "
    f"{(df['market_source_avg'] == 0).sum():,}"
)

# ------------------------------------------------------------
# MISSING MARKET FEATURES
# ------------------------------------------------------------

print("\nMissing market features:")

market_features = [
    "market_odds_home",
    "market_odds_draw",
    "market_odds_away",
    "market_raw_prob_home",
    "market_raw_prob_draw",
    "market_raw_prob_away",
    "market_overround",
    "market_prob_home",
    "market_prob_draw",
    "market_prob_away",
    "b365_prob_home",
    "b365_prob_draw",
    "b365_prob_away"
]

missing = (
    df[market_features]
    .isna()
    .sum()
)

missing = missing[missing > 0]

if len(missing) == 0:
    print("  None")
else:
    print(
        missing.to_string()
    )

# ============================================================
# SAMPLE
# ============================================================

print("\nSample market features:")

sample_columns = [
    "Date",
    "HomeTeam",
    "AwayTeam",

    "market_odds_home",
    "market_odds_draw",
    "market_odds_away",

    "market_prob_home",
    "market_prob_draw",
    "market_prob_away",

    "market_overround",
    "market_source_avg"
]

print(
    df[
        sample_columns
    ]
    .head(10)
    .to_string(index=False)
)

# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 80)
print("MODEL DATASET CREATED")
print("=" * 80)

print(
    f"File:    {OUTPUT_FILE}"
)

print(
    f"Rows:    {len(df):,}"
)

print(
    f"Columns: {len(df.columns):,}"
)

print("\nDONE")