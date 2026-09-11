"""
Convert bookmaker decimal odds into de-vigged (fair) probabilities.

Bookmaker odds always imply a total probability above 100% - that excess
is their margin ("overround"). We strip it out proportionally so the
three implied probabilities sum to exactly 1, which makes them directly
comparable to the model's own probabilities.
"""
from __future__ import annotations


def devig_1x2(odds_home: float, odds_draw: float, odds_away: float) -> tuple[list[float], float]:
    raw = [1 / odds_home, 1 / odds_draw, 1 / odds_away]
    overround = sum(raw)
    fair = [p / overround for p in raw]
    return fair, overround
