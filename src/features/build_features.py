from pathlib import Path
import pandas as pd
import numpy as np

from src.utils import config

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = config.MASTER_PATH
OUTPUT_DIR = config.PROCESSED_DIR
OUTPUT_FILE = config.FEATURES_PATH

FORM_WINDOW = 5


# ============================================================
# REUSABLE COMPUTATION
# ============================================================
#
# Extracted so src/export/push_predictions.py can compute pre-match team
# form for a not-yet-played fixture using the SAME logic as training,
# rather than a second hand-written copy that could silently drift from
# this one (the exact class of bug the original audit found with
# duplicated feature lists).
#
# Safe to call with rows whose FTR/FTHG/FTAG are NaN (future fixtures):
# every rolling feature uses shift(1), so a row's own result is never
# used for its own features. An unplayed fixture does contribute a
# Win=0/Draw=0/Loss=0/Points=0 entry to that team's history table, but
# only a row chronologically AFTER it could ever read that entry, and by
# construction there is no such row for a fixture that hasn't been
# played yet -- so this doesn't corrupt anything.

def compute_team_form_features(df: pd.DataFrame, form_window: int = FORM_WINDOW) -> pd.DataFrame:
    """Given a chronologically-sorted match dataframe with columns
    Date, match_id, HomeTeam, AwayTeam, FTHG, FTAG, FTR (FTR/FTHG/FTAG
    may be NaN for future fixtures), return df with the TEAM_FEATURES
    columns (see config.py) merged in.
    """
    df = df.sort_values(["Date", "match_id"]).reset_index(drop=True)

    # ------------------------------------------------------------
    # TEAM HISTORY (home + away appearances, one row per team per match)
    # ------------------------------------------------------------

    home = df[["Date", "match_id", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]].copy()
    home["Team"] = home["HomeTeam"]
    home["Venue"] = "Home"
    home["GoalsFor"] = home["FTHG"]
    home["GoalsAgainst"] = home["FTAG"]
    home["Win"] = (home["FTR"] == "H").astype(int)
    home["Draw"] = (home["FTR"] == "D").astype(int)
    home["Loss"] = (home["FTR"] == "A").astype(int)
    home["Points"] = np.select([home["FTR"] == "H", home["FTR"] == "D"], [3, 1], default=0)

    away = df[["Date", "match_id", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]].copy()
    away["Team"] = away["AwayTeam"]
    away["Venue"] = "Away"
    away["GoalsFor"] = away["FTAG"]
    away["GoalsAgainst"] = away["FTHG"]
    away["Win"] = (away["FTR"] == "A").astype(int)
    away["Draw"] = (away["FTR"] == "D").astype(int)
    away["Loss"] = (away["FTR"] == "H").astype(int)
    away["Points"] = np.select([away["FTR"] == "A", away["FTR"] == "D"], [3, 1], default=0)

    cols = ["Date", "match_id", "Team", "Venue", "GoalsFor", "GoalsAgainst", "Points", "Win", "Draw", "Loss"]
    team_history = pd.concat([home[cols], away[cols]], ignore_index=True)
    team_history = team_history.sort_values(["Team", "Date", "match_id"]).reset_index(drop=True)

    # ------------------------------------------------------------
    # OVERALL TEAM FORM (shift(1) excludes the current match)
    # ------------------------------------------------------------

    grouped = team_history.groupby("Team")

    def rolling_mean(col):
        return grouped[col].transform(
            lambda x: x.shift(1).rolling(form_window, min_periods=1).mean()
        )

    def rolling_sum(col):
        return grouped[col].transform(
            lambda x: x.shift(1).rolling(form_window, min_periods=1).sum()
        )

    team_history["form_points_5"] = rolling_mean("Points")
    team_history["goals_for_5"] = rolling_mean("GoalsFor")
    team_history["goals_against_5"] = rolling_mean("GoalsAgainst")
    team_history["wins_5"] = rolling_sum("Win")
    team_history["draws_5"] = rolling_sum("Draw")
    team_history["losses_5"] = rolling_sum("Loss")

    # ------------------------------------------------------------
    # HOME-SPECIFIC FORM
    # ------------------------------------------------------------

    home_history = team_history[team_history["Venue"] == "Home"].copy()
    home_group = home_history.groupby("Team")
    home_history["home_points_5"] = home_group["Points"].transform(
        lambda x: x.shift(1).rolling(form_window, min_periods=1).mean()
    )
    home_history["home_goals_for_5"] = home_group["GoalsFor"].transform(
        lambda x: x.shift(1).rolling(form_window, min_periods=1).mean()
    )
    home_history["home_goals_against_5"] = home_group["GoalsAgainst"].transform(
        lambda x: x.shift(1).rolling(form_window, min_periods=1).mean()
    )

    # ------------------------------------------------------------
    # AWAY-SPECIFIC FORM
    # ------------------------------------------------------------

    away_history = team_history[team_history["Venue"] == "Away"].copy()
    away_group = away_history.groupby("Team")
    away_history["away_points_5"] = away_group["Points"].transform(
        lambda x: x.shift(1).rolling(form_window, min_periods=1).mean()
    )
    away_history["away_goals_for_5"] = away_group["GoalsFor"].transform(
        lambda x: x.shift(1).rolling(form_window, min_periods=1).mean()
    )
    away_history["away_goals_against_5"] = away_group["GoalsAgainst"].transform(
        lambda x: x.shift(1).rolling(form_window, min_periods=1).mean()
    )

    # ------------------------------------------------------------
    # FEATURE TABLES + MERGE
    # ------------------------------------------------------------

    home_features = home_history[[
        "Date", "match_id", "Team",
        "form_points_5", "goals_for_5", "goals_against_5", "wins_5", "draws_5", "losses_5",
        "home_points_5", "home_goals_for_5", "home_goals_against_5",
    ]].rename(columns={
        "Team": "HomeTeam",
        "form_points_5": "home_form_points_5",
        "goals_for_5": "home_goals_for_5_overall",
        "goals_against_5": "home_goals_against_5_overall",
        "wins_5": "home_wins_5",
        "draws_5": "home_draws_5",
        "losses_5": "home_losses_5",
    })

    away_features = away_history[[
        "Date", "match_id", "Team",
        "form_points_5", "goals_for_5", "goals_against_5", "wins_5", "draws_5", "losses_5",
        "away_points_5", "away_goals_for_5", "away_goals_against_5",
    ]].rename(columns={
        "Team": "AwayTeam",
        "form_points_5": "away_form_points_5",
        "goals_for_5": "away_goals_for_5_overall",
        "goals_against_5": "away_goals_against_5_overall",
        "wins_5": "away_wins_5",
        "draws_5": "away_draws_5",
        "losses_5": "away_losses_5",
    })

    df = df.merge(home_features, on=["match_id", "Date", "HomeTeam"], how="left", validate="one_to_one")
    df = df.merge(away_features, on=["match_id", "Date", "AwayTeam"], how="left", validate="one_to_one")

    # ------------------------------------------------------------
    # DIFFERENCE FEATURES
    # ------------------------------------------------------------

    df["form_points_diff"] = df["home_form_points_5"] - df["away_form_points_5"]
    df["goals_for_diff"] = df["home_goals_for_5"] - df["away_goals_for_5"]
    df["goals_against_diff"] = df["home_goals_against_5_overall"] - df["away_goals_against_5_overall"]
    df["wins_diff"] = df["home_wins_5"] - df["away_wins_5"]
    df["home_away_points_diff"] = df["home_points_5"] - df["away_points_5"]
    df["home_away_goals_for_diff"] = df["home_goals_for_5"] - df["away_goals_for_5"]
    df["home_away_goals_against_diff"] = df["home_goals_against_5"] - df["away_goals_against_5"]

    return df


# ============================================================
# SCRIPT ENTRY POINT (unchanged behaviour, run with:
#   python -m src.features.build_features)
# ============================================================

def main():
    print("=" * 80)
    print("BUILDING PRE-MATCH FEATURES")
    print("=" * 80)

    df = pd.read_csv(INPUT_FILE)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    if df["Date"].isna().any():
        raise ValueError(f"Found {df['Date'].isna().sum()} rows with invalid dates.")

    df = df.sort_values(["Date", "match_id"]).reset_index(drop=True)
    print(f"\nMatches loaded: {len(df):,}")

    print("\nCreating target...")
    df["target"] = df["FTR"].map({"H": 0, "D": 1, "A": 2})
    if df["target"].isna().any():
        raise ValueError(f"Found {df['target'].isna().sum()} rows with invalid FTR values.")

    print("Computing team form features...")
    df = compute_team_form_features(df)

    feature_columns = list(dict.fromkeys(config.TEAM_FEATURES))

    # ------------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------------

    print("\n" + "=" * 80)
    print("FEATURE VALIDATION")
    print("=" * 80)
    print(f"Rows:    {len(df):,}")
    print(f"Columns: {len(df.columns):,}")

    if len(df) != 3800:
        raise ValueError(f"Expected 3,800 rows but got {len(df):,}.")
    print("\nRow count: OK")

    unique_matches = df["match_id"].nunique()
    print(f"Unique matches: {unique_matches:,}")
    if unique_matches != 3800:
        raise ValueError("Duplicate or missing match IDs detected.")
    print("Match uniqueness: OK")

    print("\nChecking feature columns...")
    missing_features = [c for c in feature_columns if c not in df.columns]
    if missing_features:
        print("\nMISSING FEATURES:")
        for c in missing_features:
            print(f"  MISSING  {c}")
        raise ValueError("One or more required feature columns are missing.")
    for c in feature_columns:
        print(f"  OK  {c}")

    print("\nMissing values in features:")
    missing = df[feature_columns].isna().sum()
    missing = missing[missing > 0]
    print("  None" if len(missing) == 0 else missing.to_string())

    print("\nTarget distribution:")
    target_distribution = df["target"].value_counts().sort_index()
    print(target_distribution)

    expected_targets = {0: 1696, 1: 883, 2: 1221}
    if target_distribution.to_dict() != expected_targets:
        print("\nWARNING: Target distribution differs from expected dataset totals.")
    else:
        print("Target distribution: OK")

    print("\nChecking duplicate matches...")
    duplicate_matches = df[df["match_id"].duplicated(keep=False)]
    if len(duplicate_matches) > 0:
        raise ValueError(f"Found {len(duplicate_matches)} duplicated match rows.")
    print("Duplicate check: OK")

    print("\nFirst 10 matches and features:")
    preview_columns = [
        "Date", "HomeTeam", "AwayTeam", "target",
        "home_form_points_5", "away_form_points_5",
        "home_goals_for_5", "away_goals_for_5",
        "home_points_5", "away_points_5",
    ]
    print(df[preview_columns].head(10).to_string(index=False))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "=" * 80)
    print("FEATURE DATASET CREATED")
    print("=" * 80)
    print(f"File:    {OUTPUT_FILE}")
    print(f"Rows:    {len(df):,}")
    print(f"Columns: {len(df.columns):,}")
    print(f"Features: {len(feature_columns):,}")
    print("\nDONE")


if __name__ == "__main__":
    main()
