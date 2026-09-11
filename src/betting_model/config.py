"""
League registry. Every league your app supports gets one entry in
config/leagues.yaml - adding a new league means adding an entry here and
dropping its data in data/raw/<code>/, not writing any new code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # betting_model/


@dataclass
class League:
    code: str                              # short id, e.g. "epl" - used for folder names, CLI args
    name: str                              # display name, e.g. "English Premier League"
    data_folder: str                       # relative to project root, e.g. "data/raw/epl"
    api_sports_league_id: Optional[int] = None   # for the live weekly pipeline, not the historical backtest
    bookmaker_prefix: Optional[str] = None       # force a specific bookmaker; None = auto-detect
    model_overrides: dict[str, Any] = field(default_factory=dict)  # DixonColesModel kwargs, e.g. {"xi": 0.002}

    @property
    def data_path(self) -> Path:
        return PROJECT_ROOT / self.data_folder

    @property
    def model_path(self) -> Path:
        return PROJECT_ROOT / "models" / self.code / "latest.json"


def load_leagues(config_path: Optional[Path] = None) -> dict[str, League]:
    config_path = config_path or (PROJECT_ROOT / "config" / "leagues.yaml")
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    leagues = {}
    for entry in raw["leagues"]:
        league = League(
            code=entry["code"],
            name=entry["name"],
            data_folder=entry["data_folder"],
            api_sports_league_id=entry.get("api_sports_league_id"),
            bookmaker_prefix=entry.get("bookmaker_prefix"),
            model_overrides=entry.get("model_overrides", {}),
        )
        leagues[league.code] = league
    return leagues


def get_league(code: str, config_path: Optional[Path] = None) -> League:
    leagues = load_leagues(config_path)
    if code not in leagues:
        available = ", ".join(sorted(leagues))
        raise KeyError(f"No league '{code}' in config/leagues.yaml. Available: {available}")
    return leagues[code]
