# EPL Betting Intelligence — Research Engine

Quantitative research pipeline for EPL match probabilities, market
comparison, and betting-value evaluation. See **FINDINGS.md** for what
this pipeline has found so far and why several early results turned
out not to hold up — read that before trusting any number in here.

## Project layout

```
src/
  data/         raw file ingestion, deduplication, inspection
  features/     pre-match feature engineering (form, market, Elo, rest)
  models/       single-split model comparison (quick sanity check)
  calibration/  model pipelines + calibration/classification diagnostics
  betting/      fair-odds/EV math, bet selection, betting metrics, bootstrap
  evaluation/   walk-forward engine (primary), single-split backtest (legacy)
  export/       pushes production predictions to the Supabase-backed app
  utils/        config.py — single source of truth for paths/features/dates
data/
  raw/          original football-data.co.uk season files
  processed/    epl_master.csv → epl_features.csv → epl_model_data.csv → epl_model_data_extra.csv
  upcoming/     fixtures_template.csv (copy to fixtures.csv and fill in odds by hand)
models/         saved metrics, predictions, and the walk_forward/ output directory
```

Every script reads its paths and feature lists from `src/utils/config.py`
— nothing is hardcoded or duplicated across scripts. If you need to
change a date boundary, a feature list, an EV threshold, or the random
seed, change it there once.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install pandas numpy scikit-learn scipy matplotlib
pip install supabase  # only needed for src/export/push_predictions.py
```

## Running the pipeline

Run every command from the project root (the directory containing
`src/`), using `-m` so the package imports resolve correctly:

```bash
# 1. Data ingestion (only needed if data/raw/ changes)
python -m src.data.create_master
python -m src.data.check_duplicates      # optional sanity check

# 2. Feature engineering
python -m src.features.build_features
python -m src.features.add_market_features
python -m src.features.build_extra_features   # Elo + rest/congestion

# 3. Quick single-split model comparison (NOT the primary evaluation —
#    see FINDINGS.md for why a single test season is unreliable)
python -m src.models.train_models
python -m src.evaluation.backtest
python -m src.evaluation.analyze_backtest

# 4. PRIMARY EVALUATION — walk-forward, all 9 out-of-sample seasons,
#    calibration, and betting significance testing
python -m src.evaluation.walk_forward
```

`walk_forward.py` is the one to trust. It writes its full output to
`models/walk_forward/` (CSVs + a reliability plot under `plots/`).

## Pushing predictions to the app

```bash
cp data/upcoming/fixtures_template.csv data/upcoming/fixtures.csv
# edit fixtures.csv: home_team, away_team, kickoff_utc, B365H, B365D, B365A
# (AvgH/D/A optional — leave blank if you don't have a multi-bookmaker
# average yet, it'll fall back to B365 like the historical dataset does)

export SUPABASE_URL=...              # from Supabase project settings > API
export SUPABASE_SERVICE_ROLE_KEY=... # service_role key, NOT anon — never expose this to the app/browser

python -m src.export.push_predictions
```

This trains the production model (Team + Market, sigmoid-calibrated —
`config.PRODUCTION_MODEL_NAME`/`PRODUCTION_FEATURES`) on **all**
historical data, computes features for each fixture in
`fixtures.csv` using the exact same functions training uses
(`compute_team_form_features`, `compute_market_features` — not a second
hand-written copy), and upserts `teams` / `matches` / `model_versions` /
`predictions` into Supabase. Safe to re-run: matches and predictions are
upserted on their natural keys, not re-inserted.

Team names in `fixtures.csv` must exactly match how they appear in the
historical data (the script validates this and lists any it doesn't
recognize) — check spelling for promoted teams especially.

## Reproducibility

- All randomness (model fitting, bootstrap resampling) is seeded via
  `config.RANDOM_STATE = 42`.
- EV thresholds (`config.EV_THRESHOLDS`) are fixed in `config.py` and
  were never adjusted after seeing a result — see FINDINGS.md's
  "what was and wasn't tuned against test data" section for the exact
  boundary of what counts as pre-specified here.
- Every feature list has a `config.assert_no_leakage()` check run at
  the start of `walk_forward.py` that raises if a post-match column
  (`FTHG`, `HS`, etc.) is ever accidentally included.
- Re-running the full pipeline end-to-end with the same input data
  reproduces the same row/column counts and metrics reported in
  FINDINGS.md exactly (verified while building this).
