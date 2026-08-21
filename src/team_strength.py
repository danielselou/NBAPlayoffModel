"""Composite team-strength rating: on-court skill + off-court context.

Brings together every signal computed elsewhere in the pipeline --
Pythagorean/SRS on-court rating (features.py), injury risk (injury.py),
playoff-dropoff tendency (playoff_dropoff.py) -- plus a handful of
"outside the box score" factors that public models (538's, Basketball-
Reference's SRS write-ups, etc.) treat as real playoff-relevant context:
roster continuity (chemistry/systems familiarity), coaching stability,
and home-court/travel effects. Each component is z-scored within its own
season before combining, so no single stat's raw scale dominates and every
team is judged against its own era's league context (again the "Era Ball"
adjustment).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import era_adjusted_zscore

# A handful of teams with a documented, measurable home-court physiological
# edge from altitude (Denver, Utah) -- visiting teams' shooting/energy dips
# are a frequently cited real effect, used here as a small static bonus.
ALTITUDE_TEAMS = {"DEN", "UTA"}

# Rough, static "big market" flag: proxies free-agency appeal / off-court
# financial flexibility that can matter for sustained team-building, even
# though it isn't wired into how the synthetic labels themselves were
# generated -- the ML model is left to discover how much (if any) weight it
# actually deserves rather than having that assumption baked in by hand.
BIG_MARKET_TEAMS = {"LAL", "LAC", "GSW", "BOS", "NYK", "BKN", "CHI", "MIA", "DAL", "HOU"}

DEFAULT_WEIGHTS = {
    "srs_z": 0.35,
    "net_rating_era_z": 0.20,
    "team_injury_risk_z": -0.15,
    "team_playoff_dropoff_score_z": 0.15,
    "roster_continuity_z": 0.08,
    "coach_stability_z": 0.05,
    "home_court_index_z": 0.02,
}


def add_offcourt_factors(team_seasons: pd.DataFrame) -> pd.DataFrame:
    """Attach continuity/coaching/home-court/market context columns."""
    df = team_seasons.copy()
    df["roster_continuity"] = df["roster_continuity"].fillna(df.groupby("year")["roster_continuity"].transform("mean"))
    df["roster_continuity"] = df["roster_continuity"].fillna(df["roster_continuity"].mean())
    df["coach_stability_index"] = np.clip(df["coach_tenure_years"] / 10.0, 0, 1)
    df["home_court_index"] = df["team"].isin(ALTITUDE_TEAMS).astype(float)
    df["big_market_flag"] = df["team"].isin(BIG_MARKET_TEAMS).astype(float)
    return df


def build_composite_ratings(team_seasons_advanced: pd.DataFrame, team_injury: pd.DataFrame,
                             team_dropoff: pd.DataFrame, weights: dict | None = None) -> pd.DataFrame:
    """Merge every team-season signal and compute one composite rating.

    Expects `team_seasons_advanced` to already carry `srs` and `net_rating`
    (from features.add_advanced_team_columns) and off-court columns (from
    add_offcourt_factors).
    """
    weights = weights or DEFAULT_WEIGHTS
    merged = (
        team_seasons_advanced
        .merge(team_injury, on=["season", "year", "team"], how="left")
        .merge(team_dropoff, on=["season", "year", "team"], how="left")
    )
    merged["team_injury_risk"] = merged["team_injury_risk"].fillna(merged["team_injury_risk"].mean())
    merged["team_playoff_dropoff_score"] = merged["team_playoff_dropoff_score"].fillna(0.0)

    for raw_col in ("srs", "net_rating", "team_injury_risk", "team_playoff_dropoff_score",
                    "roster_continuity", "coach_stability_index", "home_court_index"):
        merged[f"{raw_col}_z"] = era_adjusted_zscore(merged, raw_col, season_col="season").fillna(0.0)

    # Match the naming used in DEFAULT_WEIGHTS.
    merged["net_rating_era_z"] = merged["net_rating_z"]
    merged["srs_z"] = merged["srs_z"]
    merged["coach_stability_z"] = merged["coach_stability_index_z"]

    composite = np.zeros(len(merged))
    for col, w in weights.items():
        composite += w * merged[col].values
    merged["composite_team_rating"] = composite
    return merged
