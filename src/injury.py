"""Injury-proneness modeling.

Two signals are blended into a single 0-1 "injury proneness" score per
player-season:

1. A trailing games-missed rate (mean over the prior `window` seasons, not
   including the current one -- using the current season's own injuries to
   predict itself would leak the outcome we ultimately care about for team
   risk going into a playoff push).
2. An age-adjusted risk curve: injury risk is roughly flat through a
   player's late 20s and then climbs, consistent with the well documented
   aging-curve effect in sports-injury literature.

Team-level risk then weights each player's score by their on-court impact
(minutes x usage), since a durability concern for a bench piece matters far
less than one for a 35-minutes/game usage-heavy star.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_trailing_injury_rate(player_seasons: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """Attach `trailing_missed_rate`: mean games-missed rate over the prior
    `window` seasons for that player (excludes the current season)."""
    df = player_seasons.sort_values(["player_id", "year"]).copy()
    df["missed_rate"] = df["games_missed"] / 82
    df["trailing_missed_rate"] = (
        df.groupby("player_id")["missed_rate"]
          .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    )
    # Rookies/first tracked season: no prior history, fall back to this season's rate.
    df["trailing_missed_rate"] = df["trailing_missed_rate"].fillna(df["missed_rate"])
    return df


def age_injury_risk_curve(age: pd.Series) -> pd.Series:
    """Injury risk contribution from age alone: negligible before ~29,
    then accelerates -- a simple superlinear curve standing in for the
    well-known aging-injury relationship."""
    return np.clip(0.012 * np.maximum(age - 29, 0) ** 1.5, 0, 0.55)


def player_injury_proneness(player_seasons: pd.DataFrame, window: int = 3,
                             history_weight: float = 0.7) -> pd.DataFrame:
    """Attach `injury_proneness_score` (0-1) blending trailing history and age."""
    df = add_trailing_injury_rate(player_seasons, window=window)
    age_risk = age_injury_risk_curve(df["age"])
    score = history_weight * df["trailing_missed_rate"] + (1 - history_weight) * age_risk
    df["injury_proneness_score"] = np.clip(score, 0, 1)
    return df


def team_injury_risk(player_seasons_scored: pd.DataFrame, min_minutes: float = 10.0) -> pd.DataFrame:
    """Team-season injury risk: rotation players' proneness scores weighted
    by their (minutes x usage) on-court impact, so losing a high-usage
    starter to injury risk weighs more than a fringe bench player's.
    """
    rotation = player_seasons_scored[player_seasons_scored.minutes_per_game >= min_minutes].copy()
    rotation["impact_weight"] = (rotation.minutes_per_game * rotation.usage_rate).clip(lower=0.01)

    def _weighted(group: pd.DataFrame) -> float:
        return float(np.average(group.injury_proneness_score, weights=group.impact_weight))

    out = (
        rotation.groupby(["season", "year", "team"])
        .apply(_weighted, include_groups=False)
        .rename("team_injury_risk")
        .reset_index()
    )
    return out
