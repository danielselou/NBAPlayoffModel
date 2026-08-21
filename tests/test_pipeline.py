"""Sanity tests for the NBA playoff prediction pipeline.

These aren't meant to certify predictive accuracy (impossible to guarantee
on synthetic data) -- they check that every stage produces internally
consistent, correctly-bounded output and that the modeled causal chain
(skill -> ratings -> wins -> seeding -> playoff results) actually holds
together, so a future refactor can't silently break the generator's
correlations without a test catching it.
"""
import numpy as np
import pandas as pd
import pytest

from src.data_collection import generate_synthetic_league
from src.features import (
    add_advanced_team_columns, effective_fg_pct, era_adjusted_zscore,
    pythagorean_win_pct, true_shooting_pct,
)
from src.injury import player_injury_proneness, team_injury_risk
from src.model import build_ml_dataset, train_and_evaluate
from src.playoff_dropoff import build_playoff_dropoff_features
from src.simulate import monte_carlo_team_outcomes, simulate_series, win_probability
from src.team_strength import add_offcourt_factors, build_composite_ratings


@pytest.fixture(scope="module")
def league():
    return generate_synthetic_league(2015, 2024, seed=7)


@pytest.fixture(scope="module")
def composite(league):
    team_seasons = add_advanced_team_columns(league["team_seasons"])
    team_seasons = add_offcourt_factors(team_seasons)
    player_seasons = player_injury_proneness(league["player_seasons"])
    team_inj = team_injury_risk(player_seasons)
    player_seasons, team_drop = build_playoff_dropoff_features(player_seasons)
    return build_composite_ratings(team_seasons, team_inj, team_drop)


# --------------------------------------------------------------------------
# data_collection
# --------------------------------------------------------------------------

def test_league_shapes(league):
    assert len(league["team_seasons"]) == 30 * 10
    assert len(league["playoff_results"]) == 30 * 10
    assert len(league["player_seasons"]) > 0


def test_games_played_bounded(league):
    gp = league["player_seasons"]["games_played"]
    assert gp.between(0, 82).all()


def test_wins_bounded_and_consistent(league):
    ts = league["team_seasons"]
    assert ts["wins"].between(0, 82).all()
    assert (ts["wins"] + ts["losses"] == 82).all()


def test_exactly_one_champion_per_season(league):
    champs_per_season = league["playoff_results"].groupby("season")["champion"].sum()
    assert (champs_per_season == 1).all()


def test_eight_playoff_teams_per_conference(league):
    made = league["playoff_results"][league["playoff_results"].made_playoffs]
    counts = made.groupby(["season", "conference"]).size()
    assert (counts == 8).all()


def test_rounds_won_range(league):
    assert league["playoff_results"]["rounds_won"].between(0, 4).all()


def test_seed_strength_predicts_playoff_success(league):
    """Higher seeds should advance further and win titles more often -- this
    is the calibration check that catches the home-court-dominates-skill bug."""
    made = league["playoff_results"][league["playoff_results"].made_playoffs]
    avg_rounds_by_seed = made.groupby("seed")["rounds_won"].mean()
    assert avg_rounds_by_seed.loc[1] > avg_rounds_by_seed.loc[8]
    # Monotonically non-increasing on average from seed 1 to seed 4 (the top
    # tier); wildcard-seed noise makes a strict global ordering too brittle.
    assert avg_rounds_by_seed.loc[1] >= avg_rounds_by_seed.loc[4]


# --------------------------------------------------------------------------
# features
# --------------------------------------------------------------------------

def test_effective_fg_pct_known_values():
    fgm = pd.Series([10.0])
    fg3m = pd.Series([4.0])
    fga = pd.Series([20.0])
    # eFG% = (10 + 0.5*4) / 20 = 0.60
    assert effective_fg_pct(fgm, fg3m, fga).iloc[0] == pytest.approx(0.60)


def test_true_shooting_pct_known_values():
    pts = pd.Series([30.0])
    fga = pd.Series([20.0])
    fta = pd.Series([10.0])
    # TS% = 30 / (2*(20 + 0.44*10)) = 30 / 48.8
    assert true_shooting_pct(pts, fga, fta).iloc[0] == pytest.approx(30 / 48.8)


def test_pythagorean_win_pct_bounded():
    pf = pd.Series([100.0, 120.0, 90.0])
    pa = pd.Series([100.0, 100.0, 110.0])
    result = pythagorean_win_pct(pf, pa)
    assert result.between(0, 1).all()
    assert result.iloc[0] == pytest.approx(0.5)  # equal points -> 50%


def test_era_adjusted_zscore_mean_zero_per_group():
    df = pd.DataFrame({"season": ["A"] * 4 + ["B"] * 4, "val": [1, 2, 3, 4, 10, 20, 30, 40]})
    z = era_adjusted_zscore(df, "val", "season")
    grouped_mean = z.groupby(df["season"]).mean()
    assert grouped_mean.abs().max() < 1e-9


def test_advanced_team_columns_srs_tracks_net_rating(composite):
    corr = np.corrcoef(composite["srs"], composite["net_rating"])[0, 1]
    assert corr > 0.9


def test_pyth_win_pct_tracks_actual_win_pct(composite):
    corr = np.corrcoef(composite["pyth_win_pct"], composite["win_pct"])[0, 1]
    assert corr > 0.85


# --------------------------------------------------------------------------
# injury
# --------------------------------------------------------------------------

def test_injury_proneness_score_bounded(league):
    scored = player_injury_proneness(league["player_seasons"])
    assert scored["injury_proneness_score"].between(0, 1).all()


def test_team_injury_risk_bounded(league):
    scored = player_injury_proneness(league["player_seasons"])
    team_risk = team_injury_risk(scored)
    assert team_risk["team_injury_risk"].between(0, 1).all()
    assert len(team_risk) > 0


# --------------------------------------------------------------------------
# playoff_dropoff
# --------------------------------------------------------------------------

def test_playoff_deltas_only_for_playoff_teams(league):
    player_df, _ = build_playoff_dropoff_features(league["player_seasons"])
    non_playoff = player_df[player_df["rounds_won"].isna()]
    assert non_playoff["pts_per36_delta"].isna().all()


def test_trailing_playoff_delta_is_causal(league):
    """A player's very first career playoff appearance must have no trailing
    (prior-history) delta yet -- otherwise the feature would be leaking the
    current season's own outcome into itself."""
    player_df, _ = build_playoff_dropoff_features(league["player_seasons"])
    # NOTE: GroupBy.first() would independently backfill each column from the
    # first *non-NaN* row per group, misaligning year/delta/trailing across
    # different rows -- use head(1) on presorted data for a true "first row".
    first_appearances = (
        player_df[player_df["pts_per36_delta"].notna()]
        .sort_values(["player_id", "year"])
        .groupby("player_id")
        .head(1)
    )
    assert first_appearances["trailing_playoff_delta"].isna().all()


# --------------------------------------------------------------------------
# team_strength
# --------------------------------------------------------------------------

def test_composite_rating_no_nans(composite):
    assert composite["composite_team_rating"].notna().all()


def test_composite_rating_correlates_with_wins(composite):
    corr = np.corrcoef(composite["composite_team_rating"], composite["win_pct"])[0, 1]
    assert corr > 0.5


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------

def test_model_beats_baseline(league, composite):
    ml_df = build_ml_dataset(composite, league["playoff_results"])
    trained = train_and_evaluate(ml_df, test_years=(2023, 2024), seed=7)
    assert trained.metrics["made_playoffs"]["accuracy"] > 0.5
    assert trained.metrics["made_playoffs"]["roc_auc"] > 0.7


# --------------------------------------------------------------------------
# simulate
# --------------------------------------------------------------------------

def test_win_probability_bounded_and_monotonic():
    p_even = win_probability(0, 0, a_is_home=False, home_advantage=0)
    assert p_even == pytest.approx(0.5)
    p_favored = win_probability(10, 0, a_is_home=False, home_advantage=0)
    assert 0 < p_favored < 1
    assert p_favored > p_even


def test_simulate_series_returns_valid_winner():
    rng = np.random.default_rng(0)
    winner = simulate_series(5.0, -5.0, higher_seed_is_a=True, rng=rng)
    assert winner in ("A", "B")


def test_monte_carlo_probabilities_bounded():
    east = {i: 8.0 - i for i in range(1, 9)}
    west = {i: 8.0 - i for i in range(1, 9)}
    seed_to_team = {
        "East": {i: f"E{i}" for i in range(1, 9)},
        "West": {i: f"W{i}" for i in range(1, 9)},
    }
    result = monte_carlo_team_outcomes({"East": east, "West": west}, seed_to_team, n_sims=500, seed=0)
    assert result.champion.between(0, 1).all()
    assert result.champion.sum() == pytest.approx(1.0, abs=0.01)
    assert result.loc["E1", "champion"] > result.loc["E8", "champion"]
