"""
Shared configuration for the EPL betting research pipeline.

This module exists to fix a specific problem found during the code audit:
TEAM_FEATURES, MARKET_FEATURES, and split-date constants were previously
copy-pasted independently into train_models.py and backtest.py. They had
already drifted apart (backtest.py silently trained on one fewer season
than train_models.py validated on). Every script that needs these values
should import them from here instead of redefining them.
"""

from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

MODELS_DIR = ROOT_DIR / "models"
WALK_FORWARD_DIR = MODELS_DIR / "walk_forward"
PLOTS_DIR = WALK_FORWARD_DIR / "plots"
UPCOMING_DIR = DATA_DIR / "upcoming"
UPCOMING_FIXTURES_PATH = UPCOMING_DIR / "fixtures.csv"

# The model actually shipped to production. Per FINDINGS.md: sigmoid
# calibration measurably improved calibration (ECE) with no cost to
# LogLoss/Brier; Elo/rest-days and HistGB did not improve on this, so
# they stay out of the production feature set even though the code for
# them is kept (see src/features/build_extra_features.py). If that
# changes (e.g. a future re-evaluation with more data), update these two
# constants -- everything in src/export/ reads from here, not a
# hardcoded copy.
PRODUCTION_MODEL_NAME = "team_market_sigmoid_v1"
PRODUCTION_FEATURES = None  # set below, after COMBINED_FEATURES exists

MASTER_PATH = PROCESSED_DIR / "epl_master.csv"
FEATURES_PATH = PROCESSED_DIR / "epl_features.csv"
MODEL_DATA_PATH = PROCESSED_DIR / "epl_model_data.csv"
MODEL_DATA_EXTRA_PATH = PROCESSED_DIR / "epl_model_data_extra.csv"

# =============================================================================
# REPRODUCIBILITY / BETTING CONSTANTS
# =============================================================================

RANDOM_STATE = 42
STAKE = 1.00

# Pre-specified EV thresholds. These are fixed in advance and used
# identically across every walk-forward season. Do NOT add/remove
# thresholds after looking at results for specific seasons.
EV_THRESHOLDS = [0.00, 0.02, 0.05, 0.10, 0.15]

BOOTSTRAP_ITERATIONS = 5000

# =============================================================================
# FEATURE GROUPS
# =============================================================================
# These must exactly match the columns produced by build_features.py and
# add_market_features.py. Post-match columns (FTHG, FTAG, FTR, HS, AS,
# HST, AST, HF, AF, HC, AC, HY, AY, HR, AR, and all half-time fields) must
# NEVER be added to either list.

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
    # NOTE ON b365_market_diff_*: verified empirically (not just from
    # provider docs) that AvgH/D/A carries no more predictive information
    # than B365H/D/A alone (LogLoss 0.9677 vs 0.9689, Brier 0.5745 vs
    # 0.5749 on the 2,660 matches where both exist) -- so this is a
    # genuine cross-bookmaker-disagreement signal, not a disguised
    # closing-line/leakage channel. Avg's ~1pt lower margin is just the
    # mechanical effect of averaging several books, not later information.
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

COMBINED_FEATURES = TEAM_FEATURES + MARKET_FEATURES

# New in the P2 feature-engineering pass (see build_extra_features.py).
# Elo captures longer-horizon team strength than the 5-match rolling form
# already in TEAM_FEATURES; rest/congestion captures schedule effects
# neither TEAM_FEATURES nor MARKET_FEATURES touch at all. Both groups
# were leakage-audited in build_extra_features.py's own docstring before
# being added here.
ELO_REST_FEATURES = [
    "elo_home",
    "elo_away",
    "elo_diff",
    "home_days_rest",
    "away_days_rest",
    "rest_diff",
    "home_matches_14d",
    "away_matches_14d",
    "congestion_diff",
]

EXTENDED_FEATURES = COMBINED_FEATURES + ELO_REST_FEATURES

PRODUCTION_FEATURES = COMBINED_FEATURES

# Columns that must NEVER appear in a feature list (post-match information).
# Used only as a defensive assertion in scripts that build feature lists.
FORBIDDEN_LEAKAGE_COLUMNS = [
    "FTHG", "FTAG", "FTR", "HTHG", "HTAG", "HTR",
    "HS", "AS", "HST", "AST", "HF", "AF",
    "HC", "AC", "HY", "AY", "HR", "AR",
]


def assert_no_leakage(feature_list):
    """Raise if any forbidden post-match column is present in a feature list."""
    leaked = [f for f in feature_list if f in FORBIDDEN_LEAKAGE_COLUMNS]
    if leaked:
        raise ValueError(f"Leakage columns found in feature list: {leaked}")
