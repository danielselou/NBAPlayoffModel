"""End-to-end NBA regular-season & playoff prediction pipeline.

Run with `python main.py`. Produces, in output/:
  - feature_importance.csv / .png  -- what actually drives playoff success
  - team_predictions_<year>.csv    -- per-team ML predictions for the latest season
  - monte_carlo_odds_<year>.csv    -- bracket-simulation round-by-round odds
  - trained_models.joblib          -- the fitted ensemble, reloadable via joblib

Pipeline stages (see each module's docstring for the methodology):
  1. src.data_collection  -- real nba_api data (best-effort) / synthetic league
  2. src.features         -- Four Factors, SRS, Pythagorean win%, era z-scores
  3. src.injury           -- player + team injury-proneness scoring
  4. src.playoff_dropoff  -- playoff riser/faller tendency scoring
  5. src.team_strength    -- composite rating incl. continuity/coaching/home-court
  6. src.model            -- ensemble ML training + evaluation
  7. src.simulate         -- Monte Carlo full-bracket championship odds
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import END_SEASON_YEAR, N_MONTE_CARLO_SIMS, N_ROUND_CLASSES, OUTPUT_DIR, RANDOM_SEED, START_SEASON_YEAR
from src.data_collection import generate_synthetic_league, season_label
from src.features import add_advanced_team_columns
from src.injury import player_injury_proneness, team_injury_risk
from src.model import build_ml_dataset, feature_importance_table, train_and_evaluate
from src.playoff_dropoff import build_playoff_dropoff_features
from src.simulate import monte_carlo_team_outcomes
from src.team_strength import add_offcourt_factors, build_composite_ratings


def run_pipeline(start_year: int = START_SEASON_YEAR, end_year: int = END_SEASON_YEAR,
                  test_years: tuple[int, ...] = (2022, 2023, 2024), seed: int = RANDOM_SEED) -> dict:
    print(f"[1/7] Generating league data for {season_label(start_year)} .. {season_label(end_year)}...")
    league = generate_synthetic_league(start_year, end_year, seed=seed)

    print("[2/7] Engineering advanced team stats (Four Factors, SRS, Pythagorean, era z-scores)...")
    team_seasons = add_advanced_team_columns(league["team_seasons"])
    team_seasons = add_offcourt_factors(team_seasons)

    print("[3/7] Scoring player injury proneness...")
    player_seasons = player_injury_proneness(league["player_seasons"])
    team_inj = team_injury_risk(player_seasons)

    print("[4/7] Scoring playoff riser/faller tendencies...")
    player_seasons, team_drop = build_playoff_dropoff_features(player_seasons)

    print("[5/7] Building composite team ratings (on-court + off-court factors)...")
    composite = build_composite_ratings(team_seasons, team_inj, team_drop)

    print("[6/7] Training ensemble models (LogisticRegression + RandomForest + XGBoost)...")
    ml_df = build_ml_dataset(composite, league["playoff_results"])
    trained = train_and_evaluate(ml_df, test_years=test_years, seed=seed)
    print(json.dumps(trained.metrics, indent=2))

    importance = feature_importance_table(trained)
    importance.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)
    _plot_feature_importance(importance)
    trained.save()

    print(f"[7/7] Projecting {season_label(end_year)} playoffs (ML + Monte Carlo bracket simulation)...")
    predictions = _predict_latest_season(trained, ml_df, end_year)
    predictions.to_csv(OUTPUT_DIR / f"team_predictions_{end_year}.csv", index=False)

    mc_odds = _run_monte_carlo(composite, end_year, seed=seed)
    mc_odds.to_csv(OUTPUT_DIR / f"monte_carlo_odds_{end_year}.csv")

    print(f"\nDone. Outputs written to {OUTPUT_DIR}/")
    return {
        "league": league, "composite": composite, "ml_dataset": ml_df, "trained": trained,
        "feature_importance": importance, "predictions": predictions, "monte_carlo": mc_odds,
    }


def _predict_latest_season(trained, ml_df: pd.DataFrame, year: int) -> pd.DataFrame:
    latest = ml_df[ml_df.year == year].copy()
    latest["made_playoffs_prob"] = trained.predict_playoff_probability(latest)
    rounds_proba = trained.predict_rounds_won_proba(latest)
    latest["expected_rounds_won"] = rounds_proba @ np.arange(N_ROUND_CLASSES)
    latest["champion_prob_model"] = trained.predict_champion_probability(latest)
    cols = ["season", "team", "conference", "wins", "made_playoffs", "rounds_won", "champion",
            "made_playoffs_prob", "expected_rounds_won", "champion_prob_model"]
    return latest[cols].sort_values("champion_prob_model", ascending=False).reset_index(drop=True)


def _run_monte_carlo(composite: pd.DataFrame, year: int, seed: int) -> pd.DataFrame:
    """Seed the top 8 teams per conference by wins and simulate the bracket
    using each team's net rating adjusted for playoff-dropoff tendency and
    injury risk -- independent of (and a cross-check against) the ML model.
    `composite` already carries team_playoff_dropoff_score and
    team_injury_risk from build_composite_ratings.
    """
    df = composite[composite.year == year].copy()
    df["sim_rating"] = (
        df["net_rating"] + df["team_playoff_dropoff_score"] * 2.0 - df["team_injury_risk"] * 10.0
    )

    ratings_by_conf, seed_to_team = {}, {}
    for conf in ("East", "West"):
        conf_df = df[df.conference == conf].sort_values("wins", ascending=False).reset_index(drop=True)
        seeds, ratings = {}, {}
        for i, row in conf_df.iterrows():
            s = i + 1
            seeds[s] = row.team
            if s <= 8:
                ratings[s] = row.sim_rating
        ratings_by_conf[conf] = ratings
        seed_to_team[conf] = seeds

    return monte_carlo_team_outcomes(ratings_by_conf, seed_to_team, n_sims=N_MONTE_CARLO_SIMS, seed=seed)


def _plot_feature_importance(importance: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ordered = importance.sort_values("avg_importance")
    ax.barh(ordered["feature"], ordered["avg_importance"], color="#1f77b4")
    ax.set_xlabel("Average importance (RandomForest + XGBoost)")
    ax.set_title("What predicts making the playoffs?")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "feature_importance.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    run_pipeline()
