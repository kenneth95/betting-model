"""
Fit (or re-load) a league's model and optionally predict a fixture.

Usage:
    python scripts/fit_and_predict.py --league epl
    python scripts/fit_and_predict.py --league epl --home Arsenal --away Chelsea
    python scripts/fit_and_predict.py --league epl --load-only --home Arsenal --away Chelsea
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from betting_model import get_league, load_league, DixonColesModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--league", required=True, help="League code from config/leagues.yaml, e.g. epl")
    parser.add_argument("--home", help="Home team name for a sample prediction")
    parser.add_argument("--away", help="Away team name for a sample prediction")
    parser.add_argument("--load-only", action="store_true",
                         help="Skip refitting, just load the last saved model for this league")
    args = parser.parse_args()

    league = get_league(args.league)
    print(f"League: {league.name} ({league.code})")

    if args.load_only:
        model = DixonColesModel.load(league.model_path)
        print(f"Loaded saved model from {league.model_path}, fit as of {model.fit_reference_date_}")
    else:
        matches = load_league(str(league.data_path))
        print(f"Loaded {len(matches)} matches, {matches['HomeTeam'].nunique()} teams ever seen")
        model = DixonColesModel(**league.model_overrides).fit(matches)
        model.save(league.model_path)
        print(f"Fit complete, saved to {league.model_path}")

    if args.home and args.away:
        pred = model.predict_match(args.home, args.away)
        print("\nPrediction:")
        for k in ("home_team", "away_team", "prob_home", "prob_draw", "prob_away",
                  "home_team_known", "away_team_known"):
            print(f"  {k}: {pred[k]}")


if __name__ == "__main__":
    main()
