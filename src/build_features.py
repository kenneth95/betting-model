from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path("data/processed/epl_master.csv")
OUTPUT_DIR = Path("data/processed")
OUTPUT_FILE = OUTPUT_DIR / "epl_features.csv"

FORM_WINDOW = 5

# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("BUILDING PRE-MATCH FEATURES")
print("=" * 80)

df = pd.read_csv(INPUT_FILE)

# Parse dates
df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce"
)

# Safety check
if df["Date"].isna().any():
    raise ValueError(
        f"Found {df['Date'].isna().sum()} rows with invalid dates."
    )

# Sort chronologically
df = df.sort_values(
    ["Date", "match_id"]
).reset_index(drop=True)

print(f"\nMatches loaded: {len(df):,}")

# ============================================================
# TARGET
# ============================================================

print("\nCreating target...")

df["target"] = df["FTR"].map({
    "H": 0,
    "D": 1,
    "A": 2
})

if df["target"].isna().any():
    raise ValueError(
        f"Found {df['target'].isna().sum()} rows with invalid FTR values."
    )

# ============================================================
# CREATE TEAM HISTORY
# ============================================================

print("Creating team history...")

# ------------------------------------------------------------
# HOME MATCHES
# ------------------------------------------------------------

home = df[
    [
        "Date",
        "match_id",
        "HomeTeam",
        "AwayTeam",
        "FTHG",
        "FTAG",
        "FTR"
    ]
].copy()

home["Team"] = home["HomeTeam"]
home["Venue"] = "Home"

home["GoalsFor"] = home["FTHG"]
home["GoalsAgainst"] = home["FTAG"]

home["Win"] = (
    home["FTR"] == "H"
).astype(int)

home["Draw"] = (
    home["FTR"] == "D"
).astype(int)

home["Loss"] = (
    home["FTR"] == "A"
).astype(int)

home["Points"] = np.select(
    [
        home["FTR"] == "H",
        home["FTR"] == "D"
    ],
    [
        3,
        1
    ],
    default=0
)

# ------------------------------------------------------------
# AWAY MATCHES
# ------------------------------------------------------------

away = df[
    [
        "Date",
        "match_id",
        "HomeTeam",
        "AwayTeam",
        "FTHG",
        "FTAG",
        "FTR"
    ]
].copy()

away["Team"] = away["AwayTeam"]
away["Venue"] = "Away"

away["GoalsFor"] = away["FTAG"]
away["GoalsAgainst"] = away["FTHG"]

away["Win"] = (
    away["FTR"] == "A"
).astype(int)

away["Draw"] = (
    away["FTR"] == "D"
).astype(int)

away["Loss"] = (
    away["FTR"] == "H"
).astype(int)

away["Points"] = np.select(
    [
        away["FTR"] == "A",
        away["FTR"] == "D"
    ],
    [
        3,
        1
    ],
    default=0
)

# ------------------------------------------------------------
# COMBINE
# ------------------------------------------------------------

team_history = pd.concat(
    [
        home[
            [
                "Date",
                "match_id",
                "Team",
                "Venue",
                "GoalsFor",
                "GoalsAgainst",
                "Points",
                "Win",
                "Draw",
                "Loss"
            ]
        ],
        away[
            [
                "Date",
                "match_id",
                "Team",
                "Venue",
                "GoalsFor",
                "GoalsAgainst",
                "Points",
                "Win",
                "Draw",
                "Loss"
            ]
        ]
    ],
    ignore_index=True
)

team_history = team_history.sort_values(
    ["Team", "Date", "match_id"]
).reset_index(drop=True)

# ============================================================
# OVERALL TEAM FORM
# ============================================================

print("Calculating overall team form...")

grouped = team_history.groupby("Team")

# IMPORTANT:
# shift(1) means the current match is excluded.
# Therefore these features only use previous matches.

team_history["form_points_5"] = (
    grouped["Points"]
    .transform(
        lambda x:
        x.shift(1)
        .rolling(
            FORM_WINDOW,
            min_periods=1
        )
        .mean()
    )
)

team_history["goals_for_5"] = (
    grouped["GoalsFor"]
    .transform(
        lambda x:
        x.shift(1)
        .rolling(
            FORM_WINDOW,
            min_periods=1
        )
        .mean()
    )
)

team_history["goals_against_5"] = (
    grouped["GoalsAgainst"]
    .transform(
        lambda x:
        x.shift(1)
        .rolling(
            FORM_WINDOW,
            min_periods=1
        )
        .mean()
    )
)

team_history["wins_5"] = (
    grouped["Win"]
    .transform(
        lambda x:
        x.shift(1)
        .rolling(
            FORM_WINDOW,
            min_periods=1
        )
        .sum()
    )
)

team_history["draws_5"] = (
    grouped["Draw"]
    .transform(
        lambda x:
        x.shift(1)
        .rolling(
            FORM_WINDOW,
            min_periods=1
        )
        .sum()
    )
)

team_history["losses_5"] = (
    grouped["Loss"]
    .transform(
        lambda x:
        x.shift(1)
        .rolling(
            FORM_WINDOW,
            min_periods=1
        )
        .sum()
    )
)

# ============================================================
# HOME-SPECIFIC FORM
# ============================================================

print("Calculating home-specific form...")

home_history = team_history[
    team_history["Venue"] == "Home"
].copy()

home_group = home_history.groupby("Team")

home_history["home_points_5"] = (
    home_group["Points"]
    .transform(
        lambda x:
        x.shift(1)
        .rolling(
            FORM_WINDOW,
            min_periods=1
        )
        .mean()
    )
)

home_history["home_goals_for_5"] = (
    home_group["GoalsFor"]
    .transform(
        lambda x:
        x.shift(1)
        .rolling(
            FORM_WINDOW,
            min_periods=1
        )
        .mean()
    )
)

home_history["home_goals_against_5"] = (
    home_group["GoalsAgainst"]
    .transform(
        lambda x:
        x.shift(1)
        .rolling(
            FORM_WINDOW,
            min_periods=1
        )
        .mean()
    )
)

# ============================================================
# AWAY-SPECIFIC FORM
# ============================================================

print("Calculating away-specific form...")

away_history = team_history[
    team_history["Venue"] == "Away"
].copy()

away_group = away_history.groupby("Team")

away_history["away_points_5"] = (
    away_group["Points"]
    .transform(
        lambda x:
        x.shift(1)
        .rolling(
            FORM_WINDOW,
            min_periods=1
        )
        .mean()
    )
)

away_history["away_goals_for_5"] = (
    away_group["GoalsFor"]
    .transform(
        lambda x:
        x.shift(1)
        .rolling(
            FORM_WINDOW,
            min_periods=1
        )
        .mean()
    )
)

away_history["away_goals_against_5"] = (
    away_group["GoalsAgainst"]
    .transform(
        lambda x:
        x.shift(1)
        .rolling(
            FORM_WINDOW,
            min_periods=1
        )
        .mean()
    )
)

# ============================================================
# BUILD HOME FEATURE TABLE
# ============================================================

print("Building home feature table...")

home_features = home_history[
    [
        "Date",
        "match_id",
        "Team",

        # Overall form
        "form_points_5",
        "goals_for_5",
        "goals_against_5",
        "wins_5",
        "draws_5",
        "losses_5",

        # Home-specific form
        "home_points_5",
        "home_goals_for_5",
        "home_goals_against_5"
    ]
].copy()

home_features = home_features.rename(
    columns={
        "Team": "HomeTeam",

        "form_points_5":
            "home_form_points_5",

        "goals_for_5":
            "home_goals_for_5_overall",

        "goals_against_5":
            "home_goals_against_5_overall",

        "wins_5":
            "home_wins_5",

        "draws_5":
            "home_draws_5",

        "losses_5":
            "home_losses_5"
    }
)

# ============================================================
# BUILD AWAY FEATURE TABLE
# ============================================================

print("Building away feature table...")

away_features = away_history[
    [
        "Date",
        "match_id",
        "Team",

        # Overall form
        "form_points_5",
        "goals_for_5",
        "goals_against_5",
        "wins_5",
        "draws_5",
        "losses_5",

        # Away-specific form
        "away_points_5",
        "away_goals_for_5",
        "away_goals_against_5"
    ]
].copy()

away_features = away_features.rename(
    columns={
        "Team": "AwayTeam",

        "form_points_5":
            "away_form_points_5",

        "goals_for_5":
            "away_goals_for_5_overall",

        "goals_against_5":
            "away_goals_against_5_overall",

        "wins_5":
            "away_wins_5",

        "draws_5":
            "away_draws_5",

        "losses_5":
            "away_losses_5"
    }
)

# ============================================================
# MERGE FEATURES
# ============================================================

print("Merging features...")

# Each team contributes exactly one feature row
# to each match.
#
# We use match_id + Date + team to guarantee
# exact match alignment.

df = df.merge(
    home_features,
    on=[
        "match_id",
        "Date",
        "HomeTeam"
    ],
    how="left",
    validate="one_to_one"
)

df = df.merge(
    away_features,
    on=[
        "match_id",
        "Date",
        "AwayTeam"
    ],
    how="left",
    validate="one_to_one"
)

# ============================================================
# CREATE FINAL FEATURE NAMES
# ============================================================

# Use the venue-specific rolling goals as the primary
# home/away goals-for features.

df["home_goals_for_5"] = df[
    "home_goals_for_5"
]

df["away_goals_for_5"] = df[
    "away_goals_for_5"
]

# ============================================================
# DIFFERENCE FEATURES
# ============================================================

print("Creating difference features...")

df["form_points_diff"] = (
    df["home_form_points_5"]
    - df["away_form_points_5"]
)

df["goals_for_diff"] = (
    df["home_goals_for_5"]
    - df["away_goals_for_5"]
)

df["goals_against_diff"] = (
    df["home_goals_against_5_overall"]
    - df["away_goals_against_5_overall"]
)

df["wins_diff"] = (
    df["home_wins_5"]
    - df["away_wins_5"]
)

df["home_away_points_diff"] = (
    df["home_points_5"]
    - df["away_points_5"]
)

df["home_away_goals_for_diff"] = (
    df["home_goals_for_5"]
    - df["away_goals_for_5"]
)

df["home_away_goals_against_diff"] = (
    df["home_goals_against_5"]
    - df["away_goals_against_5"]
)

# ============================================================
# FINAL FEATURE LIST
# ============================================================

feature_columns = [

    # --------------------------------------------------------
    # Overall recent form
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Venue-specific form
    # --------------------------------------------------------

    "home_points_5",
    "away_points_5",

    "home_goals_for_5",
    "away_goals_for_5",

    "home_goals_against_5",
    "away_goals_against_5",

    # --------------------------------------------------------
    # Difference features
    # --------------------------------------------------------

    "form_points_diff",
    "goals_for_diff",
    "goals_against_diff",
    "wins_diff",

    "home_away_points_diff",
    "home_away_goals_for_diff",
    "home_away_goals_against_diff"
]

# Remove accidental duplicates
feature_columns = list(
    dict.fromkeys(feature_columns)
)

# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 80)
print("FEATURE VALIDATION")
print("=" * 80)

print(f"Rows:    {len(df):,}")
print(f"Columns: {len(df.columns):,}")

# ------------------------------------------------------------
# ROW COUNT
# ------------------------------------------------------------

if len(df) != 3800:
    raise ValueError(
        f"Expected 3,800 rows but got {len(df):,}."
    )

print("\nRow count: OK")

# ------------------------------------------------------------
# MATCH ID UNIQUENESS
# ------------------------------------------------------------

unique_matches = df["match_id"].nunique()

print(f"Unique matches: {unique_matches:,}")

if unique_matches != 3800:
    raise ValueError(
        "Duplicate or missing match IDs detected."
    )

print("Match uniqueness: OK")

# ------------------------------------------------------------
# FEATURE COLUMNS
# ------------------------------------------------------------

print("\nChecking feature columns...")

missing_features = [
    column
    for column in feature_columns
    if column not in df.columns
]

if missing_features:
    print("\nMISSING FEATURES:")

    for column in missing_features:
        print(f"  MISSING  {column}")

    raise ValueError(
        "One or more required feature columns are missing."
    )

for column in feature_columns:
    print(f"  OK  {column}")

# ------------------------------------------------------------
# MISSING VALUES
# ------------------------------------------------------------

print("\nMissing values in features:")

missing = (
    df[feature_columns]
    .isna()
    .sum()
)

missing = missing[missing > 0]

if len(missing) == 0:
    print("  None")
else:
    print(missing.to_string())

# ------------------------------------------------------------
# TARGET
# ------------------------------------------------------------

print("\nTarget distribution:")

target_distribution = (
    df["target"]
    .value_counts()
    .sort_index()
)

print(target_distribution)

expected_targets = {
    0: 1696,
    1: 883,
    2: 1221
}

actual_targets = target_distribution.to_dict()

if actual_targets != expected_targets:
    print("\nWARNING: Target distribution differs from expected dataset totals.")
else:
    print("Target distribution: OK")

# ------------------------------------------------------------
# CHECK FOR DUPLICATE ROWS
# ------------------------------------------------------------

print("\nChecking duplicate matches...")

duplicate_matches = df[
    df["match_id"].duplicated(
        keep=False
    )
]

if len(duplicate_matches) > 0:
    raise ValueError(
        f"Found {len(duplicate_matches)} duplicated match rows."
    )

print("Duplicate check: OK")

# ------------------------------------------------------------
# CHECK FIRST MATCHES
# ------------------------------------------------------------

print("\nFirst 10 matches and features:")

preview_columns = [
    "Date",
    "HomeTeam",
    "AwayTeam",
    "target",

    "home_form_points_5",
    "away_form_points_5",

    "home_goals_for_5",
    "away_goals_for_5",

    "home_points_5",
    "away_points_5"
]

print(
    df[
        preview_columns
    ]
    .head(10)
    .to_string(index=False)
)

# ============================================================
# SAVE
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 80)
print("FEATURE DATASET CREATED")
print("=" * 80)

print(f"File:    {OUTPUT_FILE}")
print(f"Rows:    {len(df):,}")
print(f"Columns: {len(df.columns):,}")
print(f"Features: {len(feature_columns):,}")

print("\nDONE")