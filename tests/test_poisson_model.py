import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from betting_model.poisson_model import DixonColesModel


def _toy_matches():
    # A small round-robin so every team has some history to fit on.
    teams = ["A", "B", "C", "D"]
    rows = []
    date = pd.Timestamp("2023-08-01")
    for h in teams:
        for a in teams:
            if h != a:
                rows.append({"Date": date, "HomeTeam": h, "AwayTeam": a, "FTHG": 1, "FTAG": 1})
                date += pd.Timedelta(days=3)
    return pd.DataFrame(rows)


def test_fit_produces_ratings_for_every_team():
    model = DixonColesModel().fit(_toy_matches())
    assert set(model.teams_) == {"A", "B", "C", "D"}
    for t in model.teams_:
        assert t in model.attack_
        assert t in model.defense_


def test_predict_probabilities_sum_to_one():
    model = DixonColesModel().fit(_toy_matches())
    pred = model.predict_match("A", "B")
    total = pred["prob_home"] + pred["prob_draw"] + pred["prob_away"]
    assert abs(total - 1.0) < 1e-6


def test_unseen_team_uses_fallback_prior_not_a_crash():
    model = DixonColesModel().fit(_toy_matches())
    pred = model.predict_match("A", "NeverSeenTeam")
    assert pred["home_team_known"] is True
    assert pred["away_team_known"] is False
    # Should still produce valid probabilities, not NaN or a crash.
    assert 0 <= pred["prob_home"] <= 1
    assert 0 <= pred["prob_away"] <= 1


def test_save_and_load_round_trip(tmp_path):
    model = DixonColesModel().fit(_toy_matches())
    original_pred = model.predict_match("A", "B")

    save_path = tmp_path / "model.json"
    model.save(save_path)
    loaded = DixonColesModel.load(save_path)
    loaded_pred = loaded.predict_match("A", "B")

    assert abs(original_pred["prob_home"] - loaded_pred["prob_home"]) < 1e-9
    assert abs(original_pred["prob_draw"] - loaded_pred["prob_draw"]) < 1e-9
