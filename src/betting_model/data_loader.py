"""
Load and combine football-data.co.uk style season CSVs into a single
league-agnostic match dataframe.

This works for any league without changes because it only relies on the
column names football-data.co.uk uses consistently across every league
they publish (EPL, La Liga, Bundesliga, etc.): Date, HomeTeam, AwayTeam,
FTHG, FTAG, FTR, plus a handful of bookmaker odds columns that vary by
season/league but follow the same naming pattern.
"""
from __future__ import annotations

import glob
import os
from typing import Optional

import pandas as pd

REQUIRED_COLUMNS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]

# Bookmaker odds column prefixes football-data.co.uk has used over the years.
# Not every prefix exists in every file - we just keep whichever are present.
BOOKMAKER_PREFIXES = ["B365", "BF", "BW", "IW", "PS", "WH", "VC", "1XB", "Max", "Avg"]


def _parse_date(series: pd.Series) -> pd.Series:
    # football-data.co.uk has used both dd/mm/yy and dd/mm/yyyy over time.
    return pd.to_datetime(series, dayfirst=True, format="mixed", errors="coerce")


def load_season_csv(path: str) -> pd.DataFrame:
    """Load one season file for one league."""
    df = pd.read_csv(path, encoding="latin1")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing expected columns: {missing}")

    df["Date"] = _parse_date(df["Date"])
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    df["FTHG"] = df["FTHG"].astype(int)
    df["FTAG"] = df["FTAG"].astype(int)
    df["HomeTeam"] = df["HomeTeam"].str.strip()
    df["AwayTeam"] = df["AwayTeam"].str.strip()

    keep_cols = list(REQUIRED_COLUMNS)
    for prefix in BOOKMAKER_PREFIXES:
        for suffix in ("H", "D", "A"):
            col = f"{prefix}{suffix}"
            if col in df.columns:
                keep_cols.append(col)

    return df[keep_cols].sort_values("Date").reset_index(drop=True)


def load_league(folder: str, pattern: str = "*.csv") -> pd.DataFrame:
    """
    Load and concatenate every season file for one league from `folder`.

    Point this at a folder containing however many seasons you've
    downloaded for a single league (e.g. all of EPL's season files -
    football-data.co.uk names them E0.csv per season, so put each
    season's file in its own dated subfolder, or rename them before
    dropping them in one folder, e.g. E0_2018.csv, E0_2019.csv, ...).

    Works unchanged for any other league's files, since the column
    names are the same across football-data.co.uk's whole catalogue.
    """
    paths = sorted(glob.glob(os.path.join(folder, pattern)))
    if not paths:
        raise FileNotFoundError(f"No CSV files found in {folder} matching {pattern}")

    frames = [load_season_csv(p) for p in paths]
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["Date", "HomeTeam", "AwayTeam"])
    return combined.sort_values("Date").reset_index(drop=True)


def available_bookmaker(df: pd.DataFrame, preferred: Optional[list[str]] = None) -> Optional[str]:
    """
    Return the first bookmaker prefix (checked in `preferred` order) that
    has complete H/D/A odds columns in this dataframe. Falls back to None
    if none of the preferred bookmakers are present - check the average
    market column ('Avg') manually in that case.
    """
    preferred = preferred or ["B365", "BF", "PS", "Avg"]
    for prefix in preferred:
        cols = [f"{prefix}{s}" for s in ("H", "D", "A")]
        if all(c in df.columns for c in cols):
            return prefix
    return None
