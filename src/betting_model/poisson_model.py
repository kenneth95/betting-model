"""
League-agnostic Dixon-Coles style time-weighted Poisson model.

Fit it on any league's match history (a dataframe with Date, HomeTeam,
AwayTeam, FTHG, FTAG) and it estimates one attack rating and one defense
rating per team, plus a shared home-advantage term and a low-score
correlation term (rho, from Dixon & Coles 1997). Nothing here is
hardcoded to a specific league or set of team names, so the same class
works unchanged for EPL, La Liga, KPL, or any other league in the same
CSV shape - just call .fit() on that league's dataframe.

Promotion and relegation are handled three ways:
  1. Time-decay weighting: recent matches count more than old ones, so a
     team's rating reflects its current form/level, not a division it
     played in years ago. Old-season noise fades out on its own.
  2. Shrinkage regularization: teams with few weighted matches in the
     fitting window (freshly promoted sides, mid-season) get pulled
     toward league-average ratings instead of an overconfident estimate
     built from a handful of games.
  3. An explicit fallback prior for teams with zero history in the
     dataset at all (never played in this league before): they're
     assigned the average rating of the bottom quartile of known teams,
     discounted further, standing in for "a typical newly promoted side."
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson


@dataclass
class DixonColesModel:
    xi: float = 0.0018                # time decay rate per day (~1yr half-life)
    l2_penalty: float = 0.15          # shrinkage strength toward league average
    max_goals: int = 10               # scoreline grid size for match probabilities
    promoted_discount: float = 0.85   # attack multiplier applied to unseen-team fallback

    teams_: list[str] = field(default_factory=list, init=False)
    attack_: dict[str, float] = field(default_factory=dict, init=False)
    defense_: dict[str, float] = field(default_factory=dict, init=False)
    home_advantage_: float = field(default=0.0, init=False)
    rho_: float = field(default=0.0, init=False)
    fallback_attack_: float = field(default=0.0, init=False)
    fallback_defense_: float = field(default=0.0, init=False)
    fit_reference_date_: Optional[pd.Timestamp] = field(default=None, init=False)
    effective_matches_: dict[str, float] = field(default_factory=dict, init=False)

    # ---------- fitting ----------

    def fit(self, matches: pd.DataFrame, as_of: Optional[pd.Timestamp] = None) -> "DixonColesModel":
        """
        matches: dataframe with Date, HomeTeam, AwayTeam, FTHG, FTAG.
        as_of: date to weight time-decay from. Defaults to the latest
               match date in `matches`. During backtesting, pass the
               actual matchday date explicitly so the model only ever
               sees data that would genuinely have been available then.
        """
        matches = matches.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]).copy()
        as_of = as_of or matches["Date"].max()
        self.fit_reference_date_ = as_of

        days_ago = (as_of - matches["Date"]).dt.days.clip(lower=0)
        weights = np.exp(-self.xi * days_ago)

        self.teams_ = sorted(set(matches["HomeTeam"]) | set(matches["AwayTeam"]))
        n = len(self.teams_)
        idx = {t: i for i, t in enumerate(self.teams_)}

        home_idx = matches["HomeTeam"].map(idx).to_numpy()
        away_idx = matches["AwayTeam"].map(idx).to_numpy()
        hg = matches["FTHG"].to_numpy()
        ag = matches["FTAG"].to_numpy()
        w = weights.to_numpy()

        effective = (
            pd.concat([
                pd.DataFrame({"team": matches["HomeTeam"], "w": weights}),
                pd.DataFrame({"team": matches["AwayTeam"], "w": weights}),
            ]).groupby("team")["w"].sum()
        )
        self.effective_matches_ = effective.to_dict()

        # x = [attack_0..attack_{n-2}, defense_0..defense_{n-1}, home_adv, rho]
        # attack_{n-1} is implied as -sum(other attacks), which fixes the
        # sum-to-zero identifiability constraint without a solver constraint.
        x0 = np.zeros((n - 1) + n + 2)

        def unpack(x):
            attack = np.empty(n)
            attack[:-1] = x[: n - 1]
            attack[-1] = -attack[:-1].sum()
            defense = x[n - 1 : n - 1 + n]
            home_adv = x[-2]
            rho = x[-1]
            return attack, defense, home_adv, rho

        def neg_log_likelihood(x):
            attack, defense, home_adv, rho = unpack(x)
            lam_home = np.exp(attack[home_idx] + defense[away_idx] + home_adv)
            lam_away = np.exp(attack[away_idx] + defense[home_idx])

            ll = (
                poisson.logpmf(hg, lam_home)
                + poisson.logpmf(ag, lam_away)
                + np.log(np.clip(self._tau(hg, ag, lam_home, lam_away, rho), 1e-10, None))
            )
            weighted_ll = np.sum(w * ll)
            reg = self.l2_penalty * (np.sum(attack**2) + np.sum(defense**2))
            return -weighted_ll + reg

        result = minimize(neg_log_likelihood, x0, method="L-BFGS-B")
        attack, defense, home_adv, rho = unpack(result.x)

        self.attack_ = dict(zip(self.teams_, attack))
        self.defense_ = dict(zip(self.teams_, defense))
        self.home_advantage_ = float(home_adv)
        self.rho_ = float(np.clip(rho, -0.2, 0.2))  # rho should stay small; clip guards against runaway fits

        ranked = sorted(self.attack_.items(), key=lambda kv: kv[1])
        bottom_quartile = ranked[: max(1, n // 4)]
        bottom_teams = [t for t, _ in bottom_quartile]
        self.fallback_attack_ = float(np.mean([self.attack_[t] for t in bottom_teams])) * self.promoted_discount
        self.fallback_defense_ = float(np.mean([self.defense_[t] for t in bottom_teams]))

        return self

    @staticmethod
    def _tau(hg, ag, lam_home, lam_away, rho):
        """Dixon-Coles low-score correction, applied only to 0-0/1-0/0-1/1-1."""
        hg = np.atleast_1d(hg)
        ag = np.atleast_1d(ag)
        lam_home = np.atleast_1d(lam_home)
        lam_away = np.atleast_1d(lam_away)
        tau = np.ones_like(hg, dtype=float)
        m00 = (hg == 0) & (ag == 0)
        m10 = (hg == 1) & (ag == 0)
        m01 = (hg == 0) & (ag == 1)
        m11 = (hg == 1) & (ag == 1)
        tau[m00] = 1 - lam_home[m00] * lam_away[m00] * rho
        tau[m10] = 1 + lam_away[m10] * rho
        tau[m01] = 1 + lam_home[m01] * rho
        tau[m11] = 1 - rho
        return tau

    # ---------- ratings lookup with promotion/relegation fallback ----------

    def _rating(self, team: str) -> tuple[float, float]:
        if team in self.attack_:
            return self.attack_[team], self.defense_[team]
        return self.fallback_attack_, self.fallback_defense_

    def is_known_team(self, team: str) -> bool:
        return team in self.attack_

    def effective_sample_size(self, team: str) -> float:
        """Time-decay-weighted match count for a team - low values mean an uncertain rating."""
        return self.effective_matches_.get(team, 0.0)

    # ---------- persistence ----------
    # Fitting is the slow, occasional step (run weekly). Predicting is cheap
    # and should happen against saved ratings, not a live refit - save()
    # after fitting, load() wherever you actually generate predictions.

    def to_dict(self) -> dict:
        return {
            "hyperparameters": {
                "xi": self.xi,
                "l2_penalty": self.l2_penalty,
                "max_goals": self.max_goals,
                "promoted_discount": self.promoted_discount,
            },
            "attack": self.attack_,
            "defense": self.defense_,
            "home_advantage": self.home_advantage_,
            "rho": self.rho_,
            "fallback_attack": self.fallback_attack_,
            "fallback_defense": self.fallback_defense_,
            "effective_matches": self.effective_matches_,
            "fit_reference_date": str(self.fit_reference_date_) if self.fit_reference_date_ is not None else None,
        }

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "DixonColesModel":
        with open(path) as f:
            data = json.load(f)

        model = cls(**data["hyperparameters"])
        model.attack_ = data["attack"]
        model.defense_ = data["defense"]
        model.teams_ = sorted(model.attack_.keys())
        model.home_advantage_ = data["home_advantage"]
        model.rho_ = data["rho"]
        model.fallback_attack_ = data["fallback_attack"]
        model.fallback_defense_ = data["fallback_defense"]
        model.effective_matches_ = data["effective_matches"]
        model.fit_reference_date_ = (
            pd.Timestamp(data["fit_reference_date"]) if data["fit_reference_date"] else None
        )
        return model

    # ---------- prediction ----------

    def predict_match(self, home_team: str, away_team: str) -> dict:
        a_home, d_home = self._rating(home_team)
        a_away, d_away = self._rating(away_team)

        lam_home = float(np.exp(a_home + d_away + self.home_advantage_))
        lam_away = float(np.exp(a_away + d_home))

        g = self.max_goals + 1
        home_pmf = poisson.pmf(np.arange(g), lam_home)
        away_pmf = poisson.pmf(np.arange(g), lam_away)
        score_matrix = np.outer(home_pmf, away_pmf)

        for hgv in (0, 1):
            for agv in (0, 1):
                score_matrix[hgv, agv] *= self._tau(hgv, agv, lam_home, lam_away, self.rho_)[0]
        score_matrix /= score_matrix.sum()

        prob_home = float(np.tril(score_matrix, -1).sum())
        prob_draw = float(np.trace(score_matrix))
        prob_away = float(np.triu(score_matrix, 1).sum())

        return {
            "home_team": home_team,
            "away_team": away_team,
            "lambda_home": lam_home,
            "lambda_away": lam_away,
            "prob_home": prob_home,
            "prob_draw": prob_draw,
            "prob_away": prob_away,
            "home_team_known": self.is_known_team(home_team),
            "away_team_known": self.is_known_team(away_team),
            "score_matrix": score_matrix,
        }
