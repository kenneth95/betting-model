"""
Walk-forward (expanding window) evaluation for the EPL betting model.

Run from the project root with:
    python -m src.evaluation.walk_forward

Replaces the single train/test split used in backtest.py with a proper
expanding-window evaluation: for each season from 2017/18 onward, the
model is trained on every season strictly before it and evaluated only
on that one held-out season. No season is ever used for both training
and testing, and no threshold or feature choice made here has been
tuned by looking at these results (thresholds come from config.py and
were fixed before this script was run).

For each fold this script evaluates ten probability sources:

  1. Market baseline                    - normalized bookmaker consensus
                                           probability, no fitting at all.
  2. Market logistic                    - logistic regression, MARKET_FEATURES.
  3. Market logistic + sigmoid          - (2) + Platt/sigmoid calibration.
  4. Market logistic + isotonic         - (2) + isotonic calibration.
  5. Team + Market                      - logistic regression, TEAM_FEATURES +
                                           MARKET_FEATURES.
  6. Team + Market + sigmoid            - (5) + sigmoid calibration.
  7. Team + Market + isotonic           - (5) + isotonic calibration.
  8. Team + Market + Elo/Rest           - (5) + Elo + rest/congestion features.
  9. Team + Market + Elo/Rest + sigmoid - (8) + sigmoid calibration.
 10. HistGB + Elo/Rest [+ sigmoid]      - HistGradientBoosting on the same
                                           feature set as (8), raw and calibrated.

See FINDINGS.md for what each of these actually found; the short version:
(6) is currently the best-calibrated model produced, (8)-(10) did not
improve on it, and none of them show a betting edge that survives
statistical testing.

For betting, two outcome-selection strategies are compared (see
src/betting/engine.py select_bets() docstring for why).

Outputs (all under models/walk_forward/):
  season_metrics.csv                    - per-season, per-model Accuracy/LogLoss/Brier
  season_betting.csv                    - per-season, per-model, per-strategy, per-threshold betting metrics
  aggregate_betting.csv                 - pooled betting metrics + block-bootstrap CI/p-value
  aggregate_classification_metrics.csv  - match-weighted LogLoss/Brier/accuracy per model
  calibration_table.csv                 - pooled predicted-vs-observed probability by bin, per model
  calibration_ece.csv                   - Expected Calibration Error per model
  odds_bucket_pooled.csv                - pooled ROI by odds bucket
  all_outcomes.csv                      - every individual outcome-level record
  plots/calibration_reliability.png     - reliability curve, raw vs sigmoid vs isotonic
"""

import warnings

import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV

from src.betting.engine import bet_metrics, build_outcome_table, season_block_bootstrap, select_bets
from src.calibration.calibration import (
    PIPELINE_FACTORIES,
    calibration_bins,
    classification_metrics,
    expected_calibration_error,
    safe_cv_folds,
)
from src.utils import config

# market_range_home/draw/away (derived from MaxH/D/A) are only present in
# football-data.co.uk's files from 2019/20 onward; for the 2017/18 and
# 2018/19 test folds the entire training window predates that, so the
# column is 100% missing and SimpleImputer correctly drops it for that
# fold only (not a bug -- there is nothing to impute). This is noted in
# FINDINGS.md rather than silently ignored; the warning itself is
# suppressed here purely to keep console output readable.
warnings.filterwarnings(
    "ignore",
    message="Skipping features without any observed values",
    category=UserWarning,
)

STRATEGIES = ["argmax", "all_qualifying"]

MODEL_SPECS = {
    # name: (feature_list_or_None, calibration_method_or_None, estimator_kind)
    "Market baseline": None,  # no fitting; use market_prob_* directly
    "Market logistic": (config.MARKET_FEATURES, None, "logistic"),
    "Market logistic + sigmoid": (config.MARKET_FEATURES, "sigmoid", "logistic"),
    "Market logistic + isotonic": (config.MARKET_FEATURES, "isotonic", "logistic"),
    "Team + Market": (config.COMBINED_FEATURES, None, "logistic"),
    "Team + Market + sigmoid": (config.COMBINED_FEATURES, "sigmoid", "logistic"),
    "Team + Market + isotonic": (config.COMBINED_FEATURES, "isotonic", "logistic"),
    "Team + Market + Elo/Rest": (config.EXTENDED_FEATURES, None, "logistic"),
    "Team + Market + Elo/Rest + sigmoid": (config.EXTENDED_FEATURES, "sigmoid", "logistic"),
    "HistGB + Elo/Rest": (config.EXTENDED_FEATURES, None, "hgb"),
    "HistGB + Elo/Rest + sigmoid": (config.EXTENDED_FEATURES, "sigmoid", "hgb"),
}


def load_data():
    df = pd.read_csv(config.MODEL_DATA_EXTRA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Date", "match_id"]).reset_index(drop=True)

    required_cols = ["Date", "HomeTeam", "AwayTeam", "FTR", "target",
                      "B365H", "B365D", "B365A", "season"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    return df


def fit_predict(model_name, spec, train_df, test_df, y_train):
    if spec is None:
        # Market baseline: already-normalized bookmaker consensus.
        return test_df[["market_prob_home", "market_prob_draw", "market_prob_away"]].values

    features, calib_method, estimator_kind = spec
    make_pipeline = PIPELINE_FACTORIES[estimator_kind]

    if calib_method is None:
        pipeline = make_pipeline()
        pipeline.fit(train_df[features], y_train)
        return pipeline.predict_proba(test_df[features])

    # cv splits the TRAINING window only (all dates strictly before the
    # test season); base model and calibrator are both blind to the test
    # season. See calibration.py module docstring.
    calibrated = CalibratedClassifierCV(
        make_pipeline(), method=calib_method, cv=safe_cv_folds(y_train)
    )
    calibrated.fit(train_df[features], y_train)
    return calibrated.predict_proba(test_df[features])


def run_walk_forward(df, season_order):
    season_metrics_rows = []
    outcome_tables = []

    for i in range(1, len(season_order)):
        train_seasons = season_order[:i]
        test_season = season_order[i]

        train_df = df[df["season"].isin(train_seasons)].copy()
        test_df = df[df["season"] == test_season].copy()

        print("\n" + "-" * 80)
        print(f"TEST SEASON: {test_season}  "
              f"(trained on {train_seasons[0]} .. {train_seasons[-1]}, "
              f"{len(train_df):,} matches)")
        print("-" * 80)

        y_train = train_df["target"]
        y_test = test_df["target"]

        for model_name, spec in MODEL_SPECS.items():
            prob = fit_predict(model_name, spec, train_df, test_df, y_train)

            cmetrics = classification_metrics(y_test, prob)
            cmetrics.update({"test_season": test_season, "model": model_name})
            season_metrics_rows.append(cmetrics)

            print(
                f"  {model_name:<36} "
                f"Acc={cmetrics['accuracy']:.4f}  "
                f"LogLoss={cmetrics['logloss']:.4f}  "
                f"Brier={cmetrics['brier']:.4f}"
            )

            outcome_tables.append(build_outcome_table(test_season, model_name, test_df, prob))

    return pd.DataFrame(season_metrics_rows), pd.concat(outcome_tables, ignore_index=True)


def aggregate_classification(season_metrics_df, season_sizes):
    rows = []
    for model_name, g in season_metrics_df.groupby("model"):
        weights = g["test_season"].map(season_sizes)
        rows.append({
            "model": model_name,
            "logloss": np.average(g["logloss"], weights=weights),
            "brier": np.average(g["brier"], weights=weights),
            "accuracy": np.average(g["accuracy"], weights=weights),
        })
    return pd.DataFrame(rows)


def compute_calibration(all_outcomes):
    calibration_rows = []
    ece_rows = []
    for model_name, g in all_outcomes.groupby("model"):
        bins = calibration_bins(g, n_bins=10)
        bins["model"] = model_name
        calibration_rows.append(bins)
        ece_rows.append({"model": model_name, "ece": expected_calibration_error(bins)})
    return pd.concat(calibration_rows, ignore_index=True), pd.DataFrame(ece_rows).sort_values("ece")


def save_reliability_plot(calibration_table):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")

        for label in ["Team + Market", "Team + Market + sigmoid", "Team + Market + isotonic"]:
            sub = calibration_table[calibration_table["model"] == label].sort_values("predicted_prob")
            if sub.empty:
                continue
            ax.plot(sub["predicted_prob"], sub["observed_freq"], marker="o", label=label)

        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Observed frequency")
        ax.set_title("Reliability curve — Team + Market (walk-forward, pooled)")
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        fig.tight_layout()
        fig.savefig(config.PLOTS_DIR / "calibration_reliability.png", dpi=150)
        plt.close(fig)
        print(f"\nReliability plot saved: {config.PLOTS_DIR / 'calibration_reliability.png'}")
    except Exception as exc:  # pragma: no cover - plotting is best-effort
        print(f"\n(Skipped reliability plot: {exc})")


def compute_season_betting(all_outcomes):
    rows = []
    for (season, model), group in all_outcomes.groupby(["test_season", "model"]):
        for strategy in STRATEGIES:
            for threshold in config.EV_THRESHOLDS:
                bets = select_bets(group, strategy, threshold)
                m = bet_metrics(bets)
                m.update({"test_season": season, "model": model,
                           "strategy": strategy, "ev_threshold": threshold})
                rows.append(m)
    return pd.DataFrame(rows)


def compute_aggregate_betting(all_outcomes):
    rows = []
    for (model,), group_m in all_outcomes.groupby(["model"]):
        for strategy in STRATEGIES:
            for threshold in config.EV_THRESHOLDS:
                bets_by_season = {
                    season: select_bets(group_s, strategy, threshold)
                    for season, group_s in group_m.groupby("test_season")
                }
                pooled_bets = pd.concat(bets_by_season.values(), ignore_index=True) \
                    if bets_by_season else pd.DataFrame()

                m = bet_metrics(pooled_bets)
                ci = season_block_bootstrap(
                    bets_by_season, n_boot=config.BOOTSTRAP_ITERATIONS, seed=config.RANDOM_STATE,
                )
                m.update(ci)
                m.update({"model": model, "strategy": strategy, "ev_threshold": threshold})
                rows.append(m)
    return pd.DataFrame(rows)


def compute_odds_bucket_analysis(all_outcomes, threshold=0.05):
    odds_bins = [1.0, 1.5, 2.0, 3.0, 5.0, 10.0, np.inf]
    odds_labels = ["1.00-1.49", "1.50-1.99", "2.00-2.99", "3.00-4.99", "5.00-9.99", "10+"]

    rows = []
    for model in MODEL_SPECS:
        for strategy in STRATEGIES:
            model_outcomes = all_outcomes[all_outcomes["model"] == model]
            bets_by_season = {
                season: select_bets(g, strategy, threshold)
                for season, g in model_outcomes.groupby("test_season")
            }
            pooled = pd.concat(bets_by_season.values(), ignore_index=True) \
                if bets_by_season else pd.DataFrame()
            if pooled.empty:
                continue
            pooled = pooled.copy()
            pooled["odds_bucket"] = pd.cut(pooled["odds"], bins=odds_bins, labels=odds_labels, right=False)
            for bucket, bgroup in pooled.groupby("odds_bucket", observed=True):
                bm = bet_metrics(bgroup)
                bm.update({"model": model, "strategy": strategy, "odds_bucket": bucket})
                rows.append(bm)
    return pd.DataFrame(rows)


def main():
    config.assert_no_leakage(config.TEAM_FEATURES)
    config.assert_no_leakage(config.MARKET_FEATURES)
    config.assert_no_leakage(config.ELO_REST_FEATURES)

    print("=" * 80)
    print("WALK-FORWARD EVALUATION")
    print("=" * 80)

    df = load_data()
    season_order = df.groupby("season")["Date"].min().sort_values().index.tolist()
    print(f"\nSeasons found (chronological): {season_order}")

    season_metrics_df, all_outcomes = run_walk_forward(df, season_order)

    print("\n" + "=" * 80)
    print("AGGREGATE OUT-OF-SAMPLE CLASSIFICATION METRICS (match-weighted mean across seasons)")
    print("=" * 80)
    season_sizes = df.groupby("season").size()
    agg_class_df = aggregate_classification(season_metrics_df, season_sizes)
    print(agg_class_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n" + "=" * 80)
    print("CALIBRATION: EXPECTED CALIBRATION ERROR (pooled over all 3 outcomes, all seasons)")
    print("=" * 80)
    print(
        "Lower is better. Computed over every H/D/A probability estimate for\n"
        "every match (not just the outcome that was bet), so this is not\n"
        "distorted by which bets happened to qualify at a given EV threshold.\n"
    )
    calibration_table, ece_df = compute_calibration(all_outcomes)
    print(ece_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    save_reliability_plot(calibration_table)

    print("\n" + "=" * 80)
    print("PER-SEASON BETTING RESULTS")
    print("=" * 80)
    season_betting = compute_season_betting(all_outcomes)

    print("\n" + "=" * 80)
    print("AGGREGATE OUT-OF-SAMPLE BETTING RESULTS (pooled across all walk-forward seasons)")
    print("=" * 80)
    aggregate_betting = compute_aggregate_betting(all_outcomes)

    display_cols = ["model", "strategy", "ev_threshold", "bets", "win_rate",
                     "roi", "roi_ci_low", "roi_ci_high", "p_roi_le_0"]
    print(
        aggregate_betting[display_cols]
        .sort_values(["model", "strategy", "ev_threshold"])
        .to_string(
            index=False,
            formatters={
                "ev_threshold": "{:.0%}".format,
                "win_rate": "{:.2%}".format,
                "roi": "{:.2%}".format,
                "roi_ci_low": "{:.2%}".format,
                "roi_ci_high": "{:.2%}".format,
                "p_roi_le_0": "{:.3f}".format,
            },
        )
    )

    print("\n" + "=" * 80)
    print("SELECTION-STRATEGY COMPARISON (argmax vs all_qualifying)")
    print("=" * 80)
    pivot = aggregate_betting.pivot_table(
        index=["model", "ev_threshold"], columns="strategy", values=["bets", "roi"],
    )
    print(pivot.to_string(float_format=lambda x: f"{x:.3f}"))

    print("\n" + "=" * 80)
    print("POOLED ROI BY ODDS BUCKET (walk-forward, all seasons combined, 5% EV threshold)")
    print("=" * 80)
    odds_bucket_pooled = compute_odds_bucket_analysis(all_outcomes, threshold=0.05)
    print(
        odds_bucket_pooled[["model", "strategy", "odds_bucket", "bets", "win_rate", "roi"]]
        .to_string(index=False, formatters={"win_rate": "{:.2%}".format, "roi": "{:.2%}".format})
    )

    # -------------------------------------------------------------------
    # SAVE OUTPUTS
    # -------------------------------------------------------------------
    config.WALK_FORWARD_DIR.mkdir(parents=True, exist_ok=True)

    outputs = {
        "season_metrics.csv": season_metrics_df,
        "season_betting.csv": season_betting,
        "aggregate_betting.csv": aggregate_betting,
        "odds_bucket_pooled.csv": odds_bucket_pooled,
        "all_outcomes.csv": all_outcomes,
        "calibration_table.csv": calibration_table,
        "calibration_ece.csv": ece_df,
        "aggregate_classification_metrics.csv": agg_class_df,
    }
    for name, frame in outputs.items():
        frame.to_csv(config.WALK_FORWARD_DIR / name, index=False)

    print("\n" + "=" * 80)
    print("FILES WRITTEN")
    print("=" * 80)
    for name in outputs:
        print(f"  {config.WALK_FORWARD_DIR / name}")


if __name__ == "__main__":
    main()
