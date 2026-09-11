"""
Usage:
    python scripts/run_backtest.py --league epl
    python scripts/run_backtest.py --league epl --edge-threshold 0.03 --refit-every 10
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from betting_model import get_league, load_league, available_bookmaker, backtest, summarize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--league", required=True)
    parser.add_argument("--min-train-matches", type=int, default=400)
    parser.add_argument("--edge-threshold", type=float, default=0.05)
    parser.add_argument("--refit-every", type=int, default=20)
    args = parser.parse_args()

    league = get_league(args.league)
    matches = load_league(str(league.data_path))
    bookmaker = league.bookmaker_prefix or available_bookmaker(matches)
    print(f"League: {league.name} | bookmaker: {bookmaker} | matches: {len(matches)}")

    results = backtest(
        matches,
        bookmaker_prefix=bookmaker,
        min_train_matches=args.min_train_matches,
        edge_threshold=args.edge_threshold,
        refit_every_n_matches=args.refit_every,
        **league.model_overrides,
    )
    summary = summarize(results)
    print("\nBacktest summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    out_path = Path(__file__).resolve().parents[1] / "models" / league.code / "backtest_results.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path, index=False)
    print(f"\nFull per-match results saved to {out_path}")


if __name__ == "__main__":
    main()
