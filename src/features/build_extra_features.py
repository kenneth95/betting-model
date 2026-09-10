"""
Builds two new pre-match feature groups that were not in the original
pipeline, requested as part of the P2 feature-engineering pass:

  1. Elo ratings (elo_home, elo_away, elo_diff)
  2. Rest / fixture-congestion (home_days_rest, away_days_rest, rest_diff,
     home_matches_14d, away_matches_14d, congestion_diff)

LEAKAGE SAFETY (read this before trusting the output):

Both feature groups are, like the existing rolling-form features in
build_features.py, computed as a single chronological pass over the
*entire* match history. This is safe because each row's Elo/rest value
depends only on that team's own matches strictly BEFORE this one,
regardless of which season or fold it later ends up in during
walk-forward evaluation:

  - Elo: the rating used for a match is captured *before* that match's
    result is applied to the running rating. The update (using that
    match's actual result) happens only after the pre-match feature is
    recorded, so a team's Elo going into match N never reflects match N
    or anything after it.
  - Rest days / congestion: computed from the strictly-previous match
    date(s) for that team via pandas' `closed='left'` time-rolling
    window, which explicitly excludes the current row.

Because of this, precomputing these features once over the full
timeline (rather than recomputing them separately inside every
walk-forward fold) does NOT leak test-season information into training
folds -- it's mathematically identical to recomputing them per fold,
just faster. This mirrors the design already used for the rolling
team-form features in build_features.py.

What IS a modelling choice here (not tuned against any test result):
  - K-factor = 20, home advantage = 100 Elo points, initial rating = 1500,
    season regression-to-mean factor = 0.75. These are standard
    textbook values for football Elo systems (e.g. the World Football
    Elo Ratings), not fitted to this dataset.
"""

import math

import pandas as pd

from src.utils import config

# =============================================================================
# CONFIG
# =============================================================================

INPUT_FILE = config.MODEL_DATA_PATH
OUTPUT_FILE = config.MODEL_DATA_EXTRA_PATH

INITIAL_ELO = 1500.0
HOME_ADVANTAGE = 100.0
K_FACTOR = 20.0
SEASON_REGRESSION = 0.75  # fraction of distance-from-mean retained across a close season

REST_WINDOW_DAYS = "14D"


# =============================================================================
# LOAD DATA
# =============================================================================

print("=" * 80)
print("BUILDING ELO + REST/CONGESTION FEATURES")
print("=" * 80)

df = pd.read_csv(INPUT_FILE)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Date", "match_id"]).reset_index(drop=True)

print(f"\nMatches loaded: {len(df):,}")


# =============================================================================
# ELO RATINGS (single chronological pass, pre-match values only)
# =============================================================================

print("\nCalculating Elo ratings...")


def mov_multiplier(goal_diff):
    """Simplified margin-of-victory multiplier (standard textbook form,
    not fitted to this dataset). 1.0 for a draw."""
    if goal_diff == 0:
        return 1.0
    return math.log(goal_diff + 1) + 1.0


ratings = {}
current_season = None
elo_rows = []

for row in df.itertuples():
    if current_season is None:
        current_season = row.season
    elif row.season != current_season:
        # New season: regress every known team's rating partway back to
        # the mean, reflecting squad changes / off-season uncertainty.
        for team in ratings:
            ratings[team] = INITIAL_ELO + SEASON_REGRESSION * (ratings[team] - INITIAL_ELO)
        current_season = row.season

    home, away = row.HomeTeam, row.AwayTeam
    elo_home_pre = ratings.get(home, INITIAL_ELO)
    elo_away_pre = ratings.get(away, INITIAL_ELO)

    elo_rows.append({
        "match_id": row.match_id,
        "elo_home": elo_home_pre,
        "elo_away": elo_away_pre,
    })

    # --- update AFTER recording the pre-match feature ---
    diff = (elo_home_pre + HOME_ADVANTAGE) - elo_away_pre
    expected_home = 1.0 / (1.0 + 10 ** (-diff / 400.0))

    if row.FTR == "H":
        actual_home = 1.0
    elif row.FTR == "D":
        actual_home = 0.5
    else:
        actual_home = 0.0

    goal_diff = abs(row.FTHG - row.FTAG)
    mult = mov_multiplier(goal_diff) * K_FACTOR

    change = mult * (actual_home - expected_home)
    ratings[home] = elo_home_pre + change
    ratings[away] = elo_away_pre - change

elo_df = pd.DataFrame(elo_rows)
elo_df["elo_diff"] = elo_df["elo_home"] - elo_df["elo_away"]

print(f"Elo features built for {len(elo_df):,} matches.")
print(f"Final rating range: {min(ratings.values()):.0f} - {max(ratings.values()):.0f}")


# =============================================================================
# REST DAYS / FIXTURE CONGESTION
# =============================================================================

print("\nCalculating rest days and fixture congestion...")

home_appearances = df[["Date", "match_id", "HomeTeam"]].rename(columns={"HomeTeam": "Team"})
away_appearances = df[["Date", "match_id", "AwayTeam"]].rename(columns={"AwayTeam": "Team"})

appearances = pd.concat([home_appearances, away_appearances], ignore_index=True)
appearances = appearances.sort_values(["Team", "Date", "match_id"]).reset_index(drop=True)

# Days since this team's previous match (any venue). First appearance -> NaN.
appearances["days_rest"] = (
    appearances.groupby("Team")["Date"].diff().dt.days
)

# Matches played by this team in the 14 days strictly BEFORE this match
# (closed='left' excludes the current row itself).
congestion_parts = []
for team, group in appearances.groupby("Team"):
    g = group.set_index("Date").sort_index()
    counts = g["match_id"].rolling(REST_WINDOW_DAYS, closed="left").count()
    g = g.reset_index()
    g["matches_14d"] = counts.values
    congestion_parts.append(g[["match_id", "Team", "days_rest", "matches_14d"]])

appearances = pd.concat(congestion_parts, ignore_index=True)

# A count of 0 recent matches is a real, informative value (e.g. the
# first fixture of a team's season), not missing data -- fill it in
# explicitly so the imputer doesn't have to guess at it later.
appearances["matches_14d"] = appearances["matches_14d"].fillna(0)

home_rest = appearances.rename(columns={
    "Team": "HomeTeam", "days_rest": "home_days_rest", "matches_14d": "home_matches_14d",
})
away_rest = appearances.rename(columns={
    "Team": "AwayTeam", "days_rest": "away_days_rest", "matches_14d": "away_matches_14d",
})

df = df.merge(home_rest[["match_id", "HomeTeam", "home_days_rest", "home_matches_14d"]],
              on=["match_id", "HomeTeam"], how="left", validate="one_to_one")
df = df.merge(away_rest[["match_id", "AwayTeam", "away_days_rest", "away_matches_14d"]],
              on=["match_id", "AwayTeam"], how="left", validate="one_to_one")

df["rest_diff"] = df["home_days_rest"] - df["away_days_rest"]
df["congestion_diff"] = df["home_matches_14d"] - df["away_matches_14d"]
df = df.copy()  # de-fragment after several sequential column assignments


# =============================================================================
# MERGE ELO
# =============================================================================

df = df.merge(elo_df, on="match_id", how="left", validate="one_to_one")


# =============================================================================
# VALIDATION
# =============================================================================

print("\n" + "=" * 80)
print("VALIDATION")
print("=" * 80)

if len(df) != 3800:
    raise ValueError(f"Expected 3,800 rows, got {len(df):,}")
print("Row count: OK")

if df["match_id"].nunique() != 3800:
    raise ValueError("Duplicate/missing match IDs after merge.")
print("Match uniqueness: OK")

new_features = [
    "elo_home", "elo_away", "elo_diff",
    "home_days_rest", "away_days_rest", "rest_diff",
    "home_matches_14d", "away_matches_14d", "congestion_diff",
]

missing = df[new_features].isna().sum()
missing = missing[missing > 0]
print("\nMissing values (expected: a small number of first-ever appearances "
      "for days_rest; those rows have no previous match to measure from):")
print(missing.to_string() if len(missing) else "  None")

# Sanity check: Elo diff should correlate with the market's own view of the
# match (not identical, but same direction) -- purely a coherence check,
# not a leakage check.
corr = df[["elo_diff", "market_prob_home"]].corr().iloc[0, 1]
print(f"\nCorrelation(elo_diff, market_prob_home): {corr:.3f} (sanity check, expect positive)")

print("\nSample rows:")
print(
    df[["Date", "HomeTeam", "AwayTeam", "elo_home", "elo_away", "elo_diff",
        "home_days_rest", "away_days_rest"]]
    .tail(10)
    .to_string(index=False)
)

# =============================================================================
# SAVE
# =============================================================================

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)
print(f"File: {OUTPUT_FILE}")
print(f"Rows: {len(df):,}  Columns: {len(df.columns):,}")
