"""
Generates fake football-data.co.uk-shaped season CSVs with realistic
promotion/relegation churn, purely to sanity-check that the loader,
model, and backtest run end-to-end without errors before you point them
at your real 7 years of EPL data.
"""
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

N_TEAMS = 20
N_SEASONS = 7
PROMOTED_PER_SEASON = 3

all_teams = [f"Team_{i:02d}" for i in range(N_TEAMS + N_SEASONS * PROMOTED_PER_SEASON)]
pool_idx = N_TEAMS
current_division = list(all_teams[:N_TEAMS])
true_strength = {t: rng.normal(0, 1) for t in all_teams}

frames = []
season_start = pd.Timestamp("2018-08-10")

for season in range(N_SEASONS):
    fixtures = []
    for home in current_division:
        for away in current_division:
            if home != away:
                fixtures.append((home, away))
    rng.shuffle(fixtures)

    date = season_start + pd.DateOffset(years=season)
    for i, (home, away) in enumerate(fixtures):
        date += pd.Timedelta(days=rng.integers(1, 4))
        lam_home = np.exp(0.35 + true_strength[home] - true_strength[away])
        lam_away = np.exp(true_strength[away] - true_strength[home])
        fthg = rng.poisson(lam_home)
        ftag = rng.poisson(lam_away)
        ftr = "H" if fthg > ftag else ("A" if ftag > fthg else "D")

        # crude synthetic "bookmaker" odds derived from true strength + margin,
        # just so the backtest has something to compare against
        true_p_home = 1 / (1 + np.exp(-(true_strength[home] - true_strength[away] + 0.2)))
        true_p_away = 1 / (1 + np.exp(-(true_strength[away] - true_strength[home] - 0.2)))
        true_p_draw = max(0.01, 1 - true_p_home - true_p_away)
        s = true_p_home + true_p_draw + true_p_away
        p_home, p_draw, p_away = true_p_home / s, true_p_draw / s, true_p_away / s
        margin = 1.07
        b365h = round(margin / p_home, 2)
        b365d = round(margin / p_draw, 2)
        b365a = round(margin / p_away, 2)

        frames.append({
            "Date": date.strftime("%d/%m/%Y"),
            "HomeTeam": home, "AwayTeam": away,
            "FTHG": fthg, "FTAG": ftag, "FTR": ftr,
            "B365H": b365h, "B365D": b365d, "B365A": b365a,
        })

    season_df = pd.DataFrame(frames[-len(fixtures):])
    pts = {t: 0 for t in current_division}
    for _, r in season_df.iterrows():
        if r["FTR"] == "H":
            pts[r["HomeTeam"]] += 3
        elif r["FTR"] == "A":
            pts[r["AwayTeam"]] += 3
        else:
            pts[r["HomeTeam"]] += 1
            pts[r["AwayTeam"]] += 1

    relegated = sorted(pts, key=pts.get)[:PROMOTED_PER_SEASON]
    promoted = all_teams[pool_idx: pool_idx + PROMOTED_PER_SEASON]
    pool_idx += PROMOTED_PER_SEASON
    current_division = [t for t in current_division if t not in relegated] + promoted

full = pd.DataFrame(frames)
out_dir = Path(__file__).resolve().parents[1] / "data" / "raw" / "epl_test"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "synthetic_league.csv"
full.to_csv(out_path, index=False)
print(f"Wrote {len(full)} matches across {N_SEASONS} seasons to {out_path}")
print(f"Teams that ever appeared: {full['HomeTeam'].nunique()}")
print("\nAdd this to config/leagues.yaml to test against it:")
print("  - code: epl_test\n    name: EPL (synthetic test data)\n    data_folder: data/raw/epl_test")
