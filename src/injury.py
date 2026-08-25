"""Injury-proneness modeling.

Two signals are blended into a single 0-1 "injury proneness" score per
player-season:

1. A trailing games-missed rate (mean over the prior `window` seasons, not
   including the current one -- using the current season's own injuries to
   predict itself would leak the outcome we ultimately care about for team
   risk going into a playoff push).
2. An age/workload-adjusted risk curve, fit with `lifelines.CoxPHFitter` --
   the standard survival-analysis tool real sports-injury research uses for
   "games survived before a significant injury" modeling -- on the league's
   own age/usage-rate-vs-injury history, rather than a hand-picked curve
   shape. It's fit once on the full league (a league-wide, actuarial-table-
   style relationship, not per player or per prediction target), so it
   carries no more same-season leakage risk than a fixed formula would; if
   `lifelines` isn't installed or the fit doesn't converge, a hand-tuned
   superlinear age curve (negligible before ~29, then accelerating, per the
   well documented real aging-injury relationship) is used instead so the
   pipeline never hard-depends on the optional library.

Team-level risk then weights each player's score by their on-court impact
(minutes x usage), since a durability concern for a bench piece matters far
less than one for a 35-minutes/game usage-heavy star.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Missing at least this many games in a season is treated as a "significant
# injury event" for the survival-analysis fit below (vs. ordinary rest
# days/minor dings folded into normal load management).
SIGNIFICANT_INJURY_GAMES = 5


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
    """Fallback injury risk contribution from age alone (used only if
    `lifelines` is unavailable or the Cox fit fails): negligible before
    ~29, then accelerates -- a simple superlinear curve standing in for
    the well documented aging-injury relationship."""
    return np.clip(0.012 * np.maximum(age - 29, 0) ** 1.5, 0, 0.55)


def fit_age_workload_hazard(player_seasons: pd.DataFrame):
    """Fit a Cox proportional-hazards model (`lifelines`) relating age and
    workload (usage rate) to in-season injury risk -- the textbook real-
    world technique for "time/games until injury" analysis in sports-
    medicine research. "Duration" is games survived before a significant
    injury (or a full healthy 82, right-censored, if none occurred);
    covariates are age and usage_rate. Returns the fitted model, or None
    if `lifelines` isn't installed or the fit doesn't converge (callers
    fall back to `age_injury_risk_curve`).
    """
    try:
        from lifelines import CoxPHFitter
        from lifelines.exceptions import ConvergenceError
    except ImportError:
        return None

    df = player_seasons[["age", "usage_rate", "games_missed"]].dropna().copy()
    df = df[df["age"].between(18, 45)]
    if len(df) < 200:  # not enough data for a stable fit
        return None

    event = (df["games_missed"] >= SIGNIFICANT_INJURY_GAMES).astype(int)
    duration = np.where(event == 1, np.clip(82 - df["games_missed"], 1, 82), 82)
    cox_df = pd.DataFrame({
        "duration": duration, "event": event.values,
        "age": df["age"].values, "usage_rate": df["usage_rate"].values,
    })

    cph = CoxPHFitter()
    try:
        cph.fit(cox_df, duration_col="duration", event_col="event")
    except (ConvergenceError, Exception):
        return None
    return cph


def cox_hazard_risk(player_seasons: pd.DataFrame, cph) -> pd.Series:
    """0-1 injury-risk score from the fitted Cox model's partial hazard
    (higher hazard = higher relative risk), rank-normalized across this
    league so it combines cleanly with the trailing-rate signal regardless
    of the hazard ratio's raw (unitless) scale."""
    hazard = cph.predict_partial_hazard(player_seasons[["age", "usage_rate"]])
    return hazard.rank(pct=True).clip(0, 1)


def player_injury_proneness(player_seasons: pd.DataFrame, window: int = 3,
                             history_weight: float = 0.7) -> pd.DataFrame:
    """Attach `injury_proneness_score` (0-1) blending trailing history with
    an age/workload risk component -- Cox-model-fit via `lifelines` when
    available, else the hand-tuned age curve."""
    df = add_trailing_injury_rate(player_seasons, window=window)
    cph = fit_age_workload_hazard(df)
    if cph is not None:
        age_risk = cox_hazard_risk(df, cph)
    else:
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
