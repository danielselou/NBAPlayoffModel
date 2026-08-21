"""Playoff performance delta modeling ("playoff riser" vs "empty stats" faller).

Some players' games elevate in the postseason (tighter rotations, more
half-court defensive attention rewards real skills); others see their
regular-season numbers evaporate against playoff-caliber defenses. This
module quantifies that tendency per player from their playoff-vs-regular-
-season stat deltas, then aggregates it to a team-level "playoff readiness"
signal, weighted so a star's history matters far more than a bench player's.

`career_trailing_playoff_delta` is deliberately causal: it only averages a
player's *prior* postseasons, never the season being predicted, so it's
safe to use as an input feature for forecasting an upcoming playoff run
without leaking the answer.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_playoff_deltas(player_seasons: pd.DataFrame) -> pd.DataFrame:
    """Attach per-season playoff-vs-regular-season deltas where available."""
    df = player_seasons.copy()
    has_playoff_data = df["playoff_pts_per36"].notna()
    df["pts_per36_delta"] = np.where(has_playoff_data, df["playoff_pts_per36"] - df["pts_per36"], np.nan)
    df["ts_pct_delta"] = np.where(has_playoff_data, df["playoff_ts_pct"] - df["ts_pct"], np.nan)
    return df


def career_trailing_playoff_delta(player_seasons_with_deltas: pd.DataFrame) -> pd.DataFrame:
    """Attach `trailing_playoff_delta`: a player's average playoff scoring
    delta over all *prior* postseason appearances (causal -- excludes the
    current season). NaN until a player has at least one prior playoff trip.
    """
    df = player_seasons_with_deltas.sort_values(["player_id", "year"]).copy()

    def _trailing_mean(group: pd.DataFrame) -> pd.Series:
        prior_deltas = group["pts_per36_delta"].shift(1)
        return prior_deltas.expanding(min_periods=1).mean()

    df["trailing_playoff_delta"] = (
        df.groupby("player_id", group_keys=False)[["pts_per36_delta"]]
        .apply(_trailing_mean)
    )
    return df


def team_playoff_dropoff_score(player_seasons_scored: pd.DataFrame, min_minutes: float = 10.0,
                                neutral_fill: float = 0.0) -> pd.DataFrame:
    """Team-season "playoff readiness" score: rotation players' trailing
    playoff deltas weighted by (minutes x usage). Players with no playoff
    track record yet contribute a neutral (league-average, ~0) value rather
    than being dropped, so young playoff teams aren't scored on an empty
    subset of veterans.
    """
    rotation = player_seasons_scored[player_seasons_scored.minutes_per_game >= min_minutes].copy()
    rotation["impact_weight"] = (rotation.minutes_per_game * rotation.usage_rate).clip(lower=0.01)
    rotation["trailing_playoff_delta"] = rotation["trailing_playoff_delta"].fillna(neutral_fill)

    def _weighted(group: pd.DataFrame) -> float:
        return float(np.average(group.trailing_playoff_delta, weights=group.impact_weight))

    out = (
        rotation.groupby(["season", "year", "team"])
        .apply(_weighted, include_groups=False)
        .rename("team_playoff_dropoff_score")
        .reset_index()
    )
    return out


def build_playoff_dropoff_features(player_seasons: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience pipeline: returns (player-level table, team-season table)."""
    with_deltas = add_playoff_deltas(player_seasons)
    with_trailing = career_trailing_playoff_delta(with_deltas)
    team_scores = team_playoff_dropoff_score(with_trailing)
    return with_trailing, team_scores
