"""
Core betting/value-engine logic: fair odds, EV, bet selection strategies,
betting metrics, and season-block bootstrap significance testing.

This is the module referenced in section 15 of the original brief as the
"core value engine" that the eventual SaaS API/frontend will consume. It
is deliberately kept independent of any specific model or evaluation
script (walk_forward.py, backtest.py, and any future API layer all call
into this module rather than reimplementing the math).

CONCEPTS (kept distinct on purpose, per section 15/27 of the brief):
  probability          - the model's estimated chance of an outcome
  implied_probability  - 1 / bookmaker odds (what the bookmaker's price implies)
  fair_odds            - 1 / probability (what odds *should* be if probability is correct)
  edge                 - probability - implied_probability
  ev                   - probability * odds - 1  (expected value of a unit stake)
"""

import numpy as np
import pandas as pd

from src.utils import config


# =============================================================================
# FAIR ODDS / EV / EDGE (the core value-engine math -- see module docstring)
# =============================================================================

def implied_probability(odds):
    return 1.0 / odds


def fair_odds(probability):
    return 1.0 / probability


def expected_value(probability, odds):
    return probability * odds - 1.0


def edge(probability, odds):
    return probability - implied_probability(odds)


# =============================================================================
# OUTCOME TABLE (one row per possible H/D/A bet, per match, per model)
# =============================================================================

OUTCOME_LABELS = ["H", "D", "A"]
ODDS_COLUMNS = {"H": "B365H", "D": "B365D", "A": "B365A"}


def build_outcome_table(test_season, model_name, test_df, prob):
    """Long-form table: one row per (match, outcome) with EV pre-computed.

    This is deliberately threshold-agnostic and selection-strategy-agnostic:
    EV is computed once per outcome, and betting strategies/thresholds are
    applied afterward by filtering this table. That keeps different
    selection strategies (e.g. argmax vs all-qualifying) perfectly
    comparable, since they are built from identical underlying EV
    estimates.
    """
    records = []

    for i, (_, row) in enumerate(test_df.iterrows()):
        for j, label in enumerate(OUTCOME_LABELS):
            probability = prob[i, j]
            odds = row[ODDS_COLUMNS[label]]

            if pd.isna(probability) or pd.isna(odds) or odds <= 1:
                continue

            records.append(
                {
                    "test_season": test_season,
                    "model": model_name,
                    "match_key": i,
                    "date": row["Date"],
                    "home_team": row["HomeTeam"],
                    "away_team": row["AwayTeam"],
                    "selection": label,
                    "probability": probability,
                    "odds": odds,
                    "implied_probability": implied_probability(odds),
                    "edge": edge(probability, odds),
                    "ev": expected_value(probability, odds),
                    "won": int(row["FTR"] == label),
                }
            )

    return pd.DataFrame.from_records(records)


# =============================================================================
# BET SELECTION STRATEGIES
# =============================================================================

def select_bets(outcome_table, strategy, threshold):
    """
    strategy:
      "argmax"          - bet only the single highest-EV qualifying outcome
                           per match (the original backtest.py behaviour).
      "all_qualifying"  - bet every outcome whose EV clears the threshold,
                           even if a match produces 0, 1, 2 or 3 bets.

    Comparing these two was the audit's check for the "optimizer's curse":
    if argmax showed a much larger ROI than all_qualifying at the same
    threshold, that would suggest the apparent edge is partly an artifact
    of always picking the extreme of several noisy EV estimates rather
    than a stable signal. Checked directly in walk-forward testing: it is
    NOT the dominant effect in this dataset (the two strategies track
    closely) -- the low ROI/high variance turned out to be the bigger
    story. Both strategies are kept here for anyone re-checking that.
    """
    qualifying = outcome_table[outcome_table["ev"] >= threshold]

    if strategy == "all_qualifying":
        return qualifying

    if strategy == "argmax":
        if qualifying.empty:
            return qualifying
        idx = qualifying.groupby(
            ["test_season", "model", "match_key"]
        )["ev"].idxmax()
        return qualifying.loc[idx]

    raise ValueError(f"Unknown strategy: {strategy}")


# =============================================================================
# BETTING METRICS
# =============================================================================

def calculate_drawdown(profits):
    if len(profits) == 0:
        return 0.0
    cumulative = profits.cumsum()
    peak = cumulative.cummax()
    return float((cumulative - peak).min())


def bet_metrics(bets, stake=None):
    stake = config.STAKE if stake is None else stake

    if len(bets) == 0:
        return {
            "bets": 0, "wins": 0, "win_rate": 0.0, "stake": 0.0,
            "profit": 0.0, "roi": 0.0, "avg_odds": np.nan,
            "avg_ev": np.nan, "max_drawdown": 0.0,
        }

    profit = np.where(bets["won"] == 1, stake * (bets["odds"] - 1), -stake)
    total_stake = stake * len(bets)
    total_profit = profit.sum()

    return {
        "bets": len(bets),
        "wins": int(bets["won"].sum()),
        "win_rate": bets["won"].mean(),
        "stake": total_stake,
        "profit": total_profit,
        "roi": total_profit / total_stake if total_stake > 0 else 0.0,
        "avg_odds": bets["odds"].mean(),
        "avg_ev": bets["ev"].mean(),
        "max_drawdown": calculate_drawdown(pd.Series(profit)),
    }


# =============================================================================
# SEASON-BLOCK BOOTSTRAP SIGNIFICANCE TESTING
# =============================================================================

def season_block_bootstrap(bets_by_season, n_boot, seed, stake=None):
    """Resample whole seasons with replacement, not individual bets.

    Individual bets within a season are not independent (shared team form,
    shared market conditions in a given window), so bootstrapping at the
    bet level would understate uncertainty. Resampling at the season level
    is the more conservative, defensible choice given only a handful of
    independent season-units are ever available for a single league.

    Implemented with plain numpy arrays (rather than repeated pandas
    concatenation inside the loop) so it stays fast even when this is
    called for many model x strategy x threshold combinations.
    """
    stake = config.STAKE if stake is None else stake
    seasons = list(bets_by_season.keys())

    if len(seasons) == 0:
        return {"roi_ci_low": np.nan, "roi_ci_high": np.nan, "p_roi_le_0": np.nan}

    profit_sums = np.zeros(len(seasons))
    stake_sums = np.zeros(len(seasons))

    for i, s in enumerate(seasons):
        b = bets_by_season[s]
        if len(b) == 0:
            continue
        won = b["won"].to_numpy()
        odds = b["odds"].to_numpy()
        profit = np.where(won == 1, stake * (odds - 1), -stake)
        profit_sums[i] = profit.sum()
        stake_sums[i] = stake * len(b)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(seasons), size=(n_boot, len(seasons)))

    boot_profit = profit_sums[idx].sum(axis=1)
    boot_stake = stake_sums[idx].sum(axis=1)

    with np.errstate(invalid="ignore", divide="ignore"):
        roi_samples = np.where(boot_stake > 0, boot_profit / boot_stake, np.nan)

    valid = roi_samples[~np.isnan(roi_samples)]

    if len(valid) == 0:
        return {"roi_ci_low": np.nan, "roi_ci_high": np.nan, "p_roi_le_0": np.nan}

    return {
        "roi_ci_low": float(np.percentile(valid, 2.5)),
        "roi_ci_high": float(np.percentile(valid, 97.5)),
        # two-sided-style p-value for "ROI is indistinguishable from 0"
        "p_roi_le_0": float((valid <= 0).mean()),
    }


# =============================================================================
# MARKET MARGIN (section 16 of the brief)
# =============================================================================

def normalize_market_probabilities(odds_h, odds_d, odds_a):
    """Given raw 1X2 odds, return (normalized_probs, overround)."""
    raw = np.column_stack([
        implied_probability(odds_h),
        implied_probability(odds_d),
        implied_probability(odds_a),
    ])
    overround = raw.sum(axis=1) - 1.0
    normalized = raw / raw.sum(axis=1, keepdims=True)
    return normalized, overround
