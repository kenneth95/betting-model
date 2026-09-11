# Betting model

League-agnostic Dixon-Coles Poisson model, structured so adding a new
league is a config change, not a code change.

## Folder layout

```
betting_model/
├── config/leagues.yaml     # <- add a league here
├── data/raw/<code>/        # <- drop that league's CSVs here
├── models/<code>/          # fitted ratings + backtest results, generated
├── src/betting_model/      # core package, never references a league name
├── scripts/                # CLIs: fit_and_predict.py, run_backtest.py
└── tests/                  # pytest
```

## Adding a new league

1. Create `data/raw/<code>/` and drop that league's football-data.co.uk
   season CSVs in it (rename them so multiple seasons don't overwrite -
   football-data.co.uk names every EPL file `E0.csv`, every La Liga file
   `SP1.csv`, etc., so add a season suffix yourself).
2. Add an entry to `config/leagues.yaml`:
   ```yaml
   - code: la_liga
     name: La Liga
     data_folder: data/raw/la_liga
     api_sports_league_id: 140
     bookmaker_prefix: null
     model_overrides: {}
   ```
3. Run it:
   ```
   python scripts/fit_and_predict.py --league la_liga
   python scripts/run_backtest.py --league la_liga
   ```

Nothing in `src/betting_model/` changes. `model_overrides` lets you tune
hyperparameters per league if needed (e.g. a smaller league with fewer
matches per season might want more shrinkage — higher `l2_penalty`).

## Day-to-day commands

```bash
# One-time / occasional: fit a league's model and save its ratings
python scripts/fit_and_predict.py --league epl

# Cheap: predict a fixture from the last saved fit, no refitting
python scripts/fit_and_predict.py --league epl --load-only --home Arsenal --away Chelsea

# Validate the model before trusting it: walk-forward backtest
python scripts/run_backtest.py --league epl
```

## Setup

```bash
pip install -r requirements.txt
python scripts/make_synthetic_data.py   # optional: generates a synthetic
                                          # "epl_test" league (already
                                          # registered in leagues.yaml) so
                                          # you can verify everything runs
                                          # before touching real data
python -m pytest tests/                  # run the test suite
```

## Why fit/predict are separated

`DixonColesModel.fit()` runs an optimization over the whole match history
and takes a few seconds. You don't want that happening on every user
request. The pattern is:

- **Fit weekly** (or whenever you pull new results) via
  `fit_and_predict.py`, which saves ratings to `models/<code>/latest.json`.
- **Predict on demand** by loading that saved file (`DixonColesModel.load()`
  or `--load-only`), which is near-instant — no re-optimization.

This is also the shape you'll want once this plugs into the Supabase/Vercel
pipeline: the weekly job fits and writes ratings, your app's prediction
logic just reads them.

## Caveats carried over from the single-league version

- The synthetic `epl_test` league exists purely to verify the code runs —
  its placeholder odds are not realistic, so its backtest numbers are
  meaningless. Delete it once you trust your real backtests.
- See `src/betting_model/poisson_model.py`'s docstring for the full
  explanation of how promotion/relegation is handled (time decay,
  shrinkage, and the fallback prior for teams with no history at all).
