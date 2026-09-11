"""
Walk-forward backtest.

Refits the model periodically using only data available before each
match, predicts that match, and scores two things:
  1. Raw prediction quality (log loss) - is the model well-calibrated?
  2. What you actually care about - if you'd only bet the matches the
     model flagged as "value" against a given bookmaker's closing odds,
     would that have made money?

This is league-agnostic: pass in whatever league's dataframe you loaded
with data_loader.load_league(), it works unchanged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .poisson_model import DixonColesModel
from .odds import devig_1x2


def backtest(
    matches: pd.DataFrame,
    bookmaker_prefix: str = "B365",
    min_train_matches: int = 300,
    edge_threshold: float = 0.05,
    refit_every_n_matches: int = 10,
    **model_kwargs,
) -> pd.DataFrame:
    matches = matches.sort_values("Date").reset_index(drop=True)
    odds_cols = [f"{bookmaker_prefix}{s}" for s in ("H", "D", "A")]
    if not all(c in matches.columns for c in odds_cols):
        raise ValueError(f"No {bookmaker_prefix} odds columns found in this dataframe")

    rows = []
    model = None
    since_refit = 0

    for i in range(min_train_matches, len(matches)):
        row = matches.iloc[i]
        if row[odds_cols].isna().any():
            continue

        if model is None or since_refit >= refit_every_n_matches:
            train = matches.iloc[:i]
            model = DixonColesModel(**model_kwargs).fit(train, as_of=row["Date"])
            since_refit = 0
        since_refit += 1

        pred = model.predict_match(row["HomeTeam"], row["AwayTeam"])
        implied, overround = devig_1x2(row[odds_cols[0]], row[odds_cols[1]], row[odds_cols[2]])

        model_probs = [pred["prob_home"], pred["prob_draw"], pred["prob_away"]]
        actual_idx = {"H": 0, "D": 1, "A": 2}[row["FTR"]]

        edges = [model_probs[k] - implied[k] for k in range(3)]
        best_side = int(np.argmax(edges))
        flagged_value = edges[best_side] >= edge_threshold

        log_loss = -np.log(max(model_probs[actual_idx], 1e-10))

        pnl = None
        if flagged_value:
            odds_for_side = row[odds_cols[best_side]]
            pnl = (odds_for_side - 1) if best_side == actual_idx else -1

        rows.append({
            "date": row["Date"],
            "home": row["HomeTeam"],
            "away": row["AwayTeam"],
            "actual": row["FTR"],
            "model_prob_home": model_probs[0],
            "model_prob_draw": model_probs[1],
            "model_prob_away": model_probs[2],
            "implied_prob_home": implied[0],
            "implied_prob_draw": implied[1],
            "implied_prob_away": implied[2],
            "log_loss": log_loss,
            "flagged_value": flagged_value,
            "flagged_side": ["H", "D", "A"][best_side] if flagged_value else None,
            "edge": edges[best_side],
            "pnl": pnl,
            "home_known": pred["home_team_known"],
            "away_known": pred["away_team_known"],
        })

    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> dict:
    value_bets = results[results["flagged_value"]]
    return {
        "n_matches_evaluated": len(results),
        "avg_log_loss": results["log_loss"].mean() if len(results) else None,
        "n_value_bets_flagged": len(value_bets),
        "value_bet_win_rate": float((value_bets["pnl"] > 0).mean()) if len(value_bets) else None,
        "value_bet_avg_roi_per_bet": float(value_bets["pnl"].mean()) if len(value_bets) else None,
        "value_bet_total_pnl_in_units": float(value_bets["pnl"].sum()) if len(value_bets) else None,
    }
