"""Reusable statistical feature engineering.

Formulas here follow standard public basketball-analytics definitions (Dean
Oliver's Four Factors, Basketball-Reference's Pythagorean win expectation and
Simple Rating System) so they work the same whether the underlying box
scores came from `nba_api` or the synthetic generator. The "Era Ball"
adjustment is the `era_adjusted_zscore` function: instead of comparing a raw
counting/rate stat across seasons where league-wide pace and 3-point volume
changed enormously, every value is expressed as a z-score relative to that
season's own league mean/std, so a 38% 3-point rate in 2003 and a 38% rate in
2023 are judged relative to their own eras rather than face value.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def effective_fg_pct(fgm: pd.Series, fg3m: pd.Series, fga: pd.Series) -> pd.Series:
    """eFG% = (FGM + 0.5*3PM) / FGA -- credits the extra value of a made three."""
    return (fgm + 0.5 * fg3m) / fga.replace(0, np.nan)


def true_shooting_pct(pts: pd.Series, fga: pd.Series, fta: pd.Series) -> pd.Series:
    """TS% = PTS / (2*(FGA + 0.44*FTA)) -- shooting efficiency incl. free throws."""
    denom = 2 * (fga + 0.44 * fta)
    return pts / denom.replace(0, np.nan)


def four_factors(df: pd.DataFrame, *, fgm="fgm", fga="fga", fg3m="fg3m", fta="fta",
                  ftm="ftm", tov="tov", orb="orb", opp_drb="opp_drb") -> pd.DataFrame:
    """Dean Oliver's Four Factors (offensive side): eFG%, TOV%, ORB%, FT Rate.

    Weights (historically ~0.4 / 0.25 / 0.2 / 0.15) are the widely cited
    approximation of each factor's contribution to scoring-margin variance.
    """
    efg = effective_fg_pct(df[fgm], df[fg3m], df[fga])
    poss_est = df[fga] + 0.44 * df[fta] + df[tov]
    tov_pct = df[tov] / poss_est.replace(0, np.nan)
    orb_pct = df[orb] / (df[orb] + df[opp_drb]).replace(0, np.nan)
    ft_rate = df[ftm] / df[fga].replace(0, np.nan)
    out = pd.DataFrame({"efg_pct": efg, "tov_pct": tov_pct, "orb_pct": orb_pct, "ft_rate": ft_rate})
    out["four_factors_score"] = (
        0.40 * out.efg_pct - 0.25 * out.tov_pct + 0.20 * out.orb_pct + 0.15 * out.ft_rate
    )
    return out


def per_36(stat: pd.Series, minutes: pd.Series) -> pd.Series:
    """Convert a per-game counting stat to a per-36-minute rate."""
    return stat / minutes.replace(0, np.nan) * 36


def possessions_per_100(stat: pd.Series, pace: pd.Series) -> pd.Series:
    """Pace-adjust a per-game stat to a per-100-possessions rate."""
    return stat / pace.replace(0, np.nan) * 100


def era_adjusted_zscore(df: pd.DataFrame, value_col: str, season_col: str = "season") -> pd.Series:
    """Z-score `value_col` within each season -- the "Era Ball" adjustment.

    Lets a 2003 player's 3PA rate be judged against 2003's league context
    instead of being penalized/inflated relative to a different era's norms.
    """
    grp = df.groupby(season_col)[value_col]
    mean = grp.transform("mean")
    std = grp.transform("std").replace(0, np.nan)
    return (df[value_col] - mean) / std


def pythagorean_win_pct(pts_for: pd.Series, pts_against: pd.Series, exponent: float = 13.91) -> pd.Series:
    """Basketball-Reference's Pythagorean win expectation from point differential."""
    pf_exp = pts_for ** exponent
    pa_exp = pts_against ** exponent
    return pf_exp / (pf_exp + pa_exp)


def simple_rating_system(team_df: pd.DataFrame, season_col: str = "season",
                          team_col: str = "team", mov_col: str = "net_rating",
                          max_iter: int = 100, tol: float = 1e-6) -> pd.Series:
    """Iterative Simple Rating System (SRS): margin-of-victory adjusted for
    strength of schedule, computed independently within each season.

    Since we don't retain a full game-by-game schedule, opponent strength is
    approximated using each team's conference as its effective "schedule" (a
    reasonable proxy: teams play the bulk of their games within-conference).
    SRS = MOV + SOS, solved by repeatedly replacing each team's SOS term with
    the average rating of its (conference) opponents until convergence.
    """
    result = pd.Series(index=team_df.index, dtype=float)
    for season, season_idx in team_df.groupby(season_col).groups.items():
        sub = team_df.loc[season_idx]
        mov = dict(zip(sub[team_col], sub[mov_col]))
        srs = dict(mov)  # start SRS estimate at raw MOV
        for _ in range(max_iter):
            new_srs = {}
            for _, row in sub.iterrows():
                conf_peers = sub[(sub["conference"] == row["conference"]) & (sub[team_col] != row[team_col])]
                sos = np.mean([srs[t] for t in conf_peers[team_col]]) if len(conf_peers) else 0.0
                new_srs[row[team_col]] = mov[row[team_col]] + sos * 0.15  # damped SOS term
            delta = max(abs(new_srs[t] - srs[t]) for t in srs)
            srs = new_srs
            if delta < tol:
                break
        result.loc[season_idx] = sub[team_col].map(srs).values
    return result


def add_advanced_team_columns(team_seasons: pd.DataFrame) -> pd.DataFrame:
    """Attach Pythagorean win% and SRS to a team-season dataframe in place-safe fashion."""
    df = team_seasons.copy()
    df["pyth_win_pct"] = pythagorean_win_pct(df["pts_for"], df["pts_against"])
    df["srs"] = simple_rating_system(df)
    for col in ("net_rating", "avg_three_att_rate", "avg_ts_pct", "pace"):
        if col in df.columns:
            df[f"{col}_era_z"] = era_adjusted_zscore(df, col)
    return df
