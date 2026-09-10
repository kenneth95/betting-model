"""
Trains the production model on ALL available historical data (unlike
walk_forward.py, which deliberately holds seasons out for evaluation --
see FINDINGS.md) and pushes probabilities for upcoming fixtures into
Supabase for the Next.js app to read.

Run from the project root with:
    python -m src.export.push_predictions

Requires:
  - data/upcoming/fixtures.csv filled in (copy fixtures_template.csv and
    edit it -- see that file for the expected columns).
  - Environment variables SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY set
    (the service_role key, NOT the anon key -- this script bypasses RLS
    to write matches/predictions, which is exactly why it must never run
    in the browser or use the anon key).
  - `pip install supabase` (not otherwise a dependency of the research
    pipeline).

WHAT THIS DOES NOT DO (by design, see FINDINGS.md and README.md):
  - Does not fetch live odds. You supply the odds for each fixture in
    fixtures.csv by hand for now -- that's a real integration to scope
    separately, not something to bolt on here.
  - Does not compute EV/edge/fair-odds. Those depend on odds a user
    enters in the app itself and are computed there (lib/valueEngine.ts),
    not precomputed and stored.
  - Does not use Elo/rest-days/HistGB. Per config.PRODUCTION_FEATURES,
    the shipped model is Team + Market + sigmoid calibration -- the only
    variant that showed a real calibration improvement in walk-forward
    testing without needing more data than this dataset has.

FEATURE-COMPUTATION SAFETY: team-form features for each fixture are
computed by appending the fixture as a new chronological row onto the
FULL historical match history and re-running the exact same
compute_team_form_features() used in training (src/features/build_features.py).
This guarantees the fixture's features are defined identically to every
training row, rather than a second hand-written "current form" formula
that could quietly drift from it.
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from src.calibration.calibration import PIPELINE_FACTORIES, safe_cv_folds
from src.features.add_market_features import compute_market_features
from src.features.build_features import compute_team_form_features
from src.utils import config

TARGET_TO_LABEL = {0: "H", 1: "D", 2: "A"}


# =============================================================================
# LOAD + VALIDATE FIXTURES
# =============================================================================

def load_fixtures() -> pd.DataFrame:
    if not config.UPCOMING_FIXTURES_PATH.exists():
        raise FileNotFoundError(
            f"No fixtures file at {config.UPCOMING_FIXTURES_PATH}.\n"
            f"Copy data/upcoming/fixtures_template.csv to that path and fill "
            f"it in with the fixtures you want predictions for."
        )

    fixtures = pd.read_csv(config.UPCOMING_FIXTURES_PATH)

    required = ["home_team", "away_team", "kickoff_utc", "B365H", "B365D", "B365A"]
    missing = [c for c in required if c not in fixtures.columns]
    if missing:
        raise ValueError(f"fixtures.csv is missing required columns: {missing}")

    for optional in ["AvgH", "AvgD", "AvgA"]:
        if optional not in fixtures.columns:
            fixtures[optional] = np.nan

    fixtures["kickoff_utc"] = pd.to_datetime(fixtures["kickoff_utc"], utc=True)

    if fixtures[["B365H", "B365D", "B365A"]].isna().any().any():
        raise ValueError("Every fixture needs B365H/B365D/B365A odds -- found missing values.")

    return fixtures


def validate_team_names(fixtures: pd.DataFrame, known_teams: set[str]) -> None:
    used = set(fixtures["home_team"]) | set(fixtures["away_team"])
    unknown = used - known_teams
    if unknown:
        raise ValueError(
            f"Unrecognized team name(s) not seen in historical data: {sorted(unknown)}.\n"
            f"Check spelling against the historical dataset's team names "
            f"(promoted teams need at least a manual name check against how "
            f"football-data.co.uk spells them, e.g. \"Nott'm Forest\")."
        )


def season_for_date(d: pd.Timestamp) -> str:
    """EPL seasons run Aug -> May; use July 1 as the season-year boundary."""
    start_year = d.year if d.month >= 7 else d.year - 1
    return f"{start_year}/{str(start_year + 1)[-2:]}"


# =============================================================================
# FEATURE COMPUTATION FOR FIXTURES
# =============================================================================

def build_fixture_base_rows(fixtures: pd.DataFrame) -> pd.DataFrame:
    """One row per fixture, matching the raw schema compute_team_form_features
    expects (Date, match_id, HomeTeam, AwayTeam, FTHG, FTAG, FTR -- the
    latter three NaN since the fixture hasn't been played)."""
    rows = []
    for _, r in fixtures.iterrows():
        date_only = pd.Timestamp(r["kickoff_utc"].date())
        match_id = f"{date_only.date()}_{r['home_team']}_{r['away_team']}"
        rows.append({
            "Date": date_only,
            "match_id": match_id,
            "HomeTeam": r["home_team"],
            "AwayTeam": r["away_team"],
            "FTHG": np.nan,
            "FTAG": np.nan,
            "FTR": np.nan,
            "B365H": r["B365H"],
            "B365D": r["B365D"],
            "B365A": r["B365A"],
            "AvgH": r["AvgH"],
            "AvgD": r["AvgD"],
            "AvgA": r["AvgA"],
            "kickoff_utc": r["kickoff_utc"],
            "season": season_for_date(date_only),
        })
    return pd.DataFrame(rows)


def compute_fixture_features(fixture_rows: pd.DataFrame) -> pd.DataFrame:
    """Returns fixture_rows with every column in config.PRODUCTION_FEATURES
    populated, using the identical functions training uses."""

    # --- Team form: needs the full historical sequence, so append the
    # fixture(s) onto real history and recompute chronologically. ---
    historical_base = pd.read_csv(config.MASTER_PATH)
    historical_base["Date"] = pd.to_datetime(historical_base["Date"])
    historical_base = historical_base[
        ["Date", "match_id", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
    ]

    combined_base = pd.concat(
        [historical_base, fixture_rows[["Date", "match_id", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]]],
        ignore_index=True,
    )
    combined_with_form = compute_team_form_features(combined_base)
    fixture_team_features = combined_with_form[
        combined_with_form["match_id"].isin(fixture_rows["match_id"])
    ].copy()

    # --- Market features: purely row-wise, no historical context needed. ---
    fixture_with_market = compute_market_features(fixture_rows.copy())

    # market_range_* only get computed when MaxH/D/A are present (they
    # aren't, for a fixture -- we don't have a multi-bookmaker spread
    # yet). Fill them in as NaN explicitly rather than letting a later
    # column-selection KeyError stand in for "missing data" -- the
    # median imputer in the production pipeline handles NaN correctly
    # for exactly this reason.
    for col in config.MARKET_FEATURES:
        if col not in fixture_with_market.columns:
            fixture_with_market[col] = np.nan

    result = fixture_team_features.merge(
        fixture_with_market[["match_id"] + config.MARKET_FEATURES + ["kickoff_utc", "season"]],
        on="match_id",
        how="left",
        validate="one_to_one",
    )

    missing = result[config.PRODUCTION_FEATURES].isna().sum()
    missing = missing[missing > 0]
    if len(missing):
        print("WARNING: missing feature values for at least one fixture (will be "
              "median-imputed like any other missing value):")
        print(missing.to_string())

    return result


# =============================================================================
# TRAIN THE PRODUCTION MODEL (on ALL historical data -- see module docstring)
# =============================================================================

def train_production_model():
    historical = pd.read_csv(config.MODEL_DATA_PATH)
    X = historical[config.PRODUCTION_FEATURES]
    y = historical["target"]

    print(f"Training production model on {len(historical):,} historical matches "
          f"({config.PRODUCTION_MODEL_NAME})...")

    model = CalibratedClassifierCV(
        PIPELINE_FACTORIES["logistic"](), method="sigmoid", cv=safe_cv_folds(y)
    )
    model.fit(X, y)
    return model


# =============================================================================
# SUPABASE
# =============================================================================

def get_supabase_client():
    try:
        from supabase import create_client
    except ImportError as exc:
        raise ImportError("Run `pip install supabase` first.") from exc

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        raise EnvironmentError(
            "Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY environment "
            "variables before running this script (service_role key, not "
            "the anon key -- this script needs to bypass RLS)."
        )
    return create_client(url, key)


def upsert_teams(client, team_names: set[str]) -> dict[str, str]:
    for name in sorted(team_names):
        client.table("teams").upsert({"name": name}, on_conflict="name").execute()
    resp = client.table("teams").select("id,name").in_("name", list(team_names)).execute()
    return {row["name"]: row["id"] for row in resp.data}


def upsert_model_version(client, walk_forward_metrics: dict | None) -> str:
    # Only one model_version should be is_active at a time -- the app
    # always reads whichever one that is.
    client.table("model_versions").update({"is_active": False}).neq(
        "name", config.PRODUCTION_MODEL_NAME
    ).execute()

    payload = {
        "name": config.PRODUCTION_MODEL_NAME,
        "description": "Team + Market logistic regression, sigmoid-calibrated, "
                        "trained on all available historical seasons.",
        "is_active": True,
    }
    if walk_forward_metrics:
        payload.update({
            "walk_forward_logloss": walk_forward_metrics.get("logloss"),
            "walk_forward_brier": walk_forward_metrics.get("brier"),
            "walk_forward_ece": walk_forward_metrics.get("ece"),
        })

    resp = client.table("model_versions").upsert(payload, on_conflict="name").execute()
    return resp.data[0]["id"]


def upsert_matches_and_predictions(client, fixture_features: pd.DataFrame, probs, model_version_id: str, team_id_map: dict):
    for i, row in fixture_features.reset_index(drop=True).iterrows():
        match_payload = {
            "season": row["season"],
            "home_team_id": team_id_map[row["HomeTeam"]],
            "away_team_id": team_id_map[row["AwayTeam"]],
            "kickoff_utc": row["kickoff_utc"].isoformat(),
            "status": "scheduled",
        }
        match_resp = client.table("matches").upsert(
            match_payload, on_conflict="home_team_id,away_team_id,kickoff_utc"
        ).execute()
        match_id = match_resp.data[0]["id"]

        client.table("predictions").upsert(
            {
                "match_id": match_id,
                "model_version_id": model_version_id,
                "prob_home": float(probs[i, 0]),
                "prob_draw": float(probs[i, 1]),
                "prob_away": float(probs[i, 2]),
            },
            on_conflict="match_id,model_version_id",
        ).execute()

        print(f"  {row['HomeTeam']} vs {row['AwayTeam']}: "
              f"H={probs[i, 0]:.1%} D={probs[i, 1]:.1%} A={probs[i, 2]:.1%}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("PUSHING PREDICTIONS TO SUPABASE")
    print("=" * 80)

    fixtures = load_fixtures()
    print(f"\nFixtures loaded: {len(fixtures)}")

    historical_teams = set(pd.read_csv(config.MASTER_PATH)["HomeTeam"]) | set(
        pd.read_csv(config.MASTER_PATH)["AwayTeam"]
    )
    validate_team_names(fixtures, historical_teams)

    fixture_rows = build_fixture_base_rows(fixtures)
    fixture_features = compute_fixture_features(fixture_rows)

    model = train_production_model()
    probs = model.predict_proba(fixture_features[config.PRODUCTION_FEATURES])

    print("\nPredictions:")
    for i, row in fixture_features.reset_index(drop=True).iterrows():
        print(f"  {row['HomeTeam']} vs {row['AwayTeam']}: "
              f"H={probs[i, 0]:.1%} D={probs[i, 1]:.1%} A={probs[i, 2]:.1%}")

    client = get_supabase_client()

    team_names = set(fixture_features["HomeTeam"]) | set(fixture_features["AwayTeam"])
    team_id_map = upsert_teams(client, team_names)

    model_version_id = upsert_model_version(client, walk_forward_metrics=None)

    print("\nWriting matches + predictions...")
    upsert_matches_and_predictions(client, fixture_features, probs, model_version_id, team_id_map)

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
