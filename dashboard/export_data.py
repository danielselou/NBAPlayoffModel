"""Build a self-contained JSON payload for the visual dashboard covering
*every* modeled season, historical and future, for a season-picker UI:

- Historical seasons (year <= PRESENT_YEAR) get the generator's actual
  outcome (revealed standings + bracket) *and* a genuinely out-of-sample
  "walk-forward" prediction -- a model retrained on only the strictly-
  earlier seasons -- so the dashboard can show a real model-vs-actual
  comparison, not a prediction that already saw the answer.
- Future seasons (year > PRESENT_YEAR) get *only* the model's projection:
  the generator still has to mechanically simulate a season to keep team
  strength evolving consistently year to year, but the bracket winner at
  each node is chosen by re-simulating that specific matchup from the two
  teams' ratings (a "projected favorite"), never revealed from the
  generator's own internal outcome. An explicit, decaying confidence
  rating (not a measured accuracy) is attached, reflecting compounding
  real-world uncertainty the further out the projection reaches.

Walk-forward retraining (one fresh model per season, on an expanding
window of only earlier seasons) is what makes *both* of these honest:
a historical comparison that isn't circular, and a future projection built
by the exact same mechanism, just with no answer key to check against yet.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from config import (
    ERA_BANDS, MAX_MODELED_YEAR, MIN_TRAIN_SEASONS, PRESENT_YEAR, RANDOM_SEED, START_SEASON_YEAR,
)
from dashboard.player_names import jersey_number, player_name
from dashboard.portrait import generate_portrait_data_uri, generate_real_player_card
from dashboard.real_history import REAL_CHAMPION, REAL_MVP, real_mvp_image_data_uri
from dashboard.real_mvp_prediction import build_prediction as build_real_mvp_prediction
from dashboard.team_meta import TEAM_META
from src.data_collection import generate_synthetic_league, season_label
from src.era import era_big_man_weight, era_guard_weight, era_name
from src.features import add_advanced_team_columns, era_adjusted_zscore
from src.injury import player_injury_proneness, team_injury_risk
from src.model import (
    build_ml_dataset, future_confidence, train_and_evaluate, walk_forward_predictions,
    year_accuracy_summary,
)
from src.playoff_dropoff import build_playoff_dropoff_features
from src.simulate import simulate_series
from src.team_strength import add_offcourt_factors, build_composite_ratings

ROUND1_PAIRS = [(1, 8), (4, 5), (3, 6), (2, 7)]
WALK_FORWARD_N_ESTIMATORS = 180
SERIES_SIM_N = 6000
MVP_MIN_MINUTES = 28.0
# MVP formula is 85% box-score stats, 15% "outside the box score" -- team
# success (contender narrative), market size, and a modest narrative-buzz
# term standing in for the real, well-documented (if unquantifiable) role
# storylines play in actual MVP voting. Media/market is real MVP-voting
# behavior, kept deliberately a minority weight, not the deciding factor.
MVP_STATS_WEIGHT_TOTAL = 0.85
MVP_MEDIA_WEIGHT_TOTAL = 0.15
MVP_STATS_BASE_WEIGHTS = {  # pre-era-adjustment, pre-normalization
    "pts_per36_z": 0.36, "ast_per36_z": 0.16, "reb_per36_z": 0.16,
    "stocks_per36_z": 0.10, "ts_pct_z": 0.22,
}
MVP_MEDIA_WEIGHTS = {"team_win_pct": 0.08, "big_market_flag": 0.04, "buzz_z": 0.03}


def _narrative_buzz(player_id: int, year: int) -> float:
    """Deterministic stand-in for real, unquantifiable MVP-voter narrative
    factors (trade rumors, contract-year performances, injury-comeback
    storylines) -- a small, seeded pseudo-random value, not a real signal."""
    seed = (int(player_id) * 1_000_003 + int(year)) % (2**31)
    return float(np.random.default_rng(seed).normal())


def _series_win_prob(rating_winner: float, rating_loser: float, winner_is_higher_seed: bool,
                      rng: np.random.Generator, n_sims: int = SERIES_SIM_N) -> float:
    """Re-simulate a matchup with a *known* winner; return how often the
    model's own simulation agrees (used for historical, revealed series)."""
    wins = sum(
        simulate_series(rating_winner, rating_loser, higher_seed_is_a=winner_is_higher_seed, rng=rng) == "A"
        for _ in range(n_sims)
    )
    return wins / n_sims


def _projected_series(rating_a: float, rating_b: float, a_is_higher_seed: bool,
                       rng: np.random.Generator, n_sims: int = SERIES_SIM_N) -> tuple[str, float]:
    """Simulate a matchup with *no* known winner yet; return (winner, prob)
    for whichever side wins the majority of simulated series (used for
    future, not-yet-played series)."""
    a_wins = sum(
        simulate_series(rating_a, rating_b, higher_seed_is_a=a_is_higher_seed, rng=rng) == "A"
        for _ in range(n_sims)
    )
    prob_a = a_wins / n_sims
    return ("A", prob_a) if prob_a >= 0.5 else ("B", 1 - prob_a)


def _standings_for_year(composite_df: pd.DataFrame, wf_year_df: pd.DataFrame, conf: str, year: int) -> list[dict]:
    conf_df = composite_df[(composite_df.year == year) & (composite_df.conference == conf)].copy()
    conf_df = conf_df.sort_values("wins", ascending=False).reset_index(drop=True)
    leader_wins, leader_losses = conf_df.loc[0, "wins"], conf_df.loc[0, "losses"]
    prob_by_team = dict(zip(wf_year_df.team, wf_year_df.made_playoffs_prob))

    rows = []
    for i, row in conf_df.iterrows():
        gb = ((leader_wins - row.wins) + (row.losses - leader_losses)) / 2
        rows.append({
            "seed": i + 1, "team": row.team, "wins": int(row.wins), "losses": int(row.losses),
            "win_pct": round(float(row.win_pct), 3), "gb": round(gb, 1),
            "rating": round(float(row.net_rating), 1),
            "playoff_prob": round(float(prob_by_team.get(row.team, 0.0)), 3),
        })
    return rows


def _actual_node(team_a: str, team_b: str, seed_a: int, seed_b: int, min_rounds: int,
                  rounds_won: dict, sim_rating: dict, rng) -> dict:
    a_won = rounds_won[team_a] >= min_rounds
    winner, loser = (team_a, team_b) if a_won else (team_b, team_a)
    winner_is_higher = (winner == team_a) == (seed_a < seed_b)
    prob = _series_win_prob(sim_rating[winner], sim_rating[loser], winner_is_higher, rng)
    return {"teamA": team_a, "teamB": team_b, "seedA": seed_a, "seedB": seed_b,
            "winner": winner, "winner_prob": round(prob, 3), "projected": False}


def _projected_node(team_a: str, team_b: str, seed_a: int, seed_b: int, sim_rating: dict, rng) -> dict:
    a_is_higher = seed_a < seed_b
    which, prob = _projected_series(sim_rating[team_a], sim_rating[team_b], a_is_higher, rng)
    winner = team_a if which == "A" else team_b
    return {"teamA": team_a, "teamB": team_b, "seedA": seed_a, "seedB": seed_b,
            "winner": winner, "winner_prob": round(prob, 3), "projected": True}


def _conference_bracket(year: int, conf: str, composite_df: pd.DataFrame, playoff_results_df: pd.DataFrame,
                         sim_rating: dict, rng, historical: bool) -> dict:
    conf_df = composite_df[(composite_df.year == year) & (composite_df.conference == conf)]
    conf_df = conf_df.sort_values("wins", ascending=False).reset_index(drop=True)
    team_by_seed = {i + 1: row.team for i, row in conf_df.iterrows() if i < 8}

    if historical:
        conf_pr = playoff_results_df[(playoff_results_df.year == year) & (playoff_results_df.conference == conf)]
        rounds_won = dict(zip(conf_pr.team, conf_pr.rounds_won))

        def node(a, b, sa, sb, min_r):
            return _actual_node(a, b, sa, sb, min_r, rounds_won, sim_rating, rng)
    else:
        def node(a, b, sa, sb, min_r):
            return _projected_node(a, b, sa, sb, sim_rating, rng)

    round1, r1_winner_seed = [], {}
    for high, low in ROUND1_PAIRS:
        n = node(team_by_seed[high], team_by_seed[low], high, low, 1)
        round1.append(n)
        r1_winner_seed[high] = (n["winner"], high if n["winner"] == team_by_seed[high] else low)

    w18, seed18 = r1_winner_seed[1]
    w45, seed45 = r1_winner_seed[4]
    w36, seed36 = r1_winner_seed[3]
    w27, seed27 = r1_winner_seed[2]
    semis = [node(w18, w45, seed18, seed45, 2), node(w36, w27, seed36, seed27, 2)]

    seedA = semis[0]["seedA"] if semis[0]["winner"] == semis[0]["teamA"] else semis[0]["seedB"]
    seedB = semis[1]["seedA"] if semis[1]["winner"] == semis[1]["teamA"] else semis[1]["seedB"]
    conf_finals = node(semis[0]["winner"], semis[1]["winner"], seedA, seedB, 3)

    return {"round1": round1, "semis": semis, "conf_finals": conf_finals, "champion": conf_finals["winner"]}


def _finals(east_champ: str, west_champ: str, sim_rating: dict, rng, historical: bool,
            actual_champion: str | None = None) -> dict:
    if historical:
        winner = actual_champion
        loser = west_champ if winner == east_champ else east_champ
        winner_is_higher = sim_rating[winner] >= sim_rating[loser]
        prob = _series_win_prob(sim_rating[winner], sim_rating[loser], winner_is_higher, rng)
        return {"teamEast": east_champ, "teamWest": west_champ, "winner": winner,
                "winner_prob": round(prob, 3), "projected": False}
    east_is_higher = sim_rating[east_champ] >= sim_rating[west_champ]
    which, prob = _projected_series(sim_rating[east_champ], sim_rating[west_champ], east_is_higher, rng)
    winner = east_champ if which == "A" else west_champ
    return {"teamEast": east_champ, "teamWest": west_champ, "winner": winner,
            "winner_prob": round(prob, 3), "projected": True}


def _comparison_rows(standings: dict, wf_year_df: pd.DataFrame, playoff_results_year: pd.DataFrame) -> list[dict]:
    seed_by_team = {row["team"]: row["seed"] for conf_rows in standings.values() for row in conf_rows}
    actual = playoff_results_year.set_index("team")
    rows = []
    for _, r in wf_year_df.iterrows():
        a = actual.loc[r.team]
        rows.append({
            "team": r.team, "seed": seed_by_team.get(r.team),
            "predicted_playoff_prob": round(float(r.made_playoffs_prob), 3),
            "actual_made_playoffs": bool(a.made_playoffs),
            "predicted_expected_rounds": round(float(r.expected_rounds_won), 2),
            "actual_rounds_won": int(a.rounds_won),
            "predicted_champion_prob": round(float(r.champion_prob), 4),
            "actual_champion": bool(a.champion),
        })
    return rows


def _era_stats_weights(year: int) -> dict[str, float]:
    """Era-adjusted, normalized-to-0.85 stats sub-weights: rebounding/stocks
    matter relatively more in big-man-favorable eras, efficiency (heavily
    3PT/spacing-driven) relatively more in guard/perimeter-favorable eras."""
    big_wt, guard_wt = era_big_man_weight(year), era_guard_weight(year)
    raw = dict(MVP_STATS_BASE_WEIGHTS)
    raw["reb_per36_z"] *= big_wt
    raw["stocks_per36_z"] *= big_wt
    raw["ts_pct_z"] *= guard_wt
    total = sum(raw.values())
    return {k: (v / total) * MVP_STATS_WEIGHT_TOTAL for k, v in raw.items()}


def _build_mvp_pool(player_seasons: pd.DataFrame, composite: pd.DataFrame) -> pd.DataFrame:
    """Score every rotation-caliber player-season on an 85/15 stats-vs-media
    MVP formula. The stats side reuses `era_adjusted_zscore` (the same
    function used everywhere else in the pipeline) on the box-score
    categories real MVP voting weighs, era-adjusted so a dominant rebounder
    in 1985 and a dominant three-point shooter in 2024 are each judged by
    their own era's standards. The media side is real MVP-voting behavior
    (team success, market size, narrative) kept a deliberate minority
    weight. Purely descriptive -- uses that season's own stats, same as
    real MVP voting -- so there's no look-ahead concern the way there is
    for the model's predictive features elsewhere in this project.
    """
    pool = player_seasons[player_seasons.minutes_per_game >= MVP_MIN_MINUTES].copy()
    pool["stocks_per36"] = pool.stl_per36 + pool.blk_per36
    for raw_col in ("pts_per36", "ast_per36", "reb_per36", "stocks_per36", "ts_pct"):
        pool[f"{raw_col}_z"] = era_adjusted_zscore(pool, raw_col, season_col="season").fillna(0.0)

    win_pct_by_team_year = composite.set_index(["year", "team"])["win_pct"]
    market_by_team_year = composite.set_index(["year", "team"])["big_market_flag"]
    idx = pool.set_index(["year", "team"]).index
    pool["team_win_pct"] = idx.map(win_pct_by_team_year).fillna(0.5)
    pool["big_market_flag"] = idx.map(market_by_team_year).fillna(0.0)
    pool["buzz_z"] = [
        _narrative_buzz(pid, yr) for pid, yr in zip(pool.player_id, pool.year)
    ]

    stats_score = pd.Series(0.0, index=pool.index)
    media_score = pd.Series(0.0, index=pool.index)
    for year, group_idx in pool.groupby("year").groups.items():
        stats_w = _era_stats_weights(year)
        for col, w in stats_w.items():
            stats_score.loc[group_idx] += pool.loc[group_idx, col] * w
        for col, w in MVP_MEDIA_WEIGHTS.items():
            media_score.loc[group_idx] += pool.loc[group_idx, col] * w

    pool["stats_score"] = stats_score
    pool["media_score"] = media_score
    pool["mvp_score"] = stats_score + media_score
    return pool


def _select_mvp(mvp_pool: pd.DataFrame, year: int) -> dict | None:
    year_pool = mvp_pool[mvp_pool.year == year]
    if year_pool.empty:
        return None
    row = year_pool.loc[year_pool.mvp_score.idxmax()]
    pid = int(row.player_id)
    team_color = TEAM_META.get(row.team, {}).get("color", "#999999")
    num = jersey_number(pid)
    total = row.stats_score + row.media_score
    raw_share = float(row.stats_score / total) if total else MVP_STATS_WEIGHT_TOTAL
    stats_share = min(max(raw_share, 0.0), 1.0)
    return {
        "player_id": pid, "name": player_name(pid), "team": row.team, "position": row.position,
        "jersey_number": num, "age": int(row.age), "era": era_name(year),
        "stats": {
            "pts_per36": round(float(row.pts_per36), 1), "ast_per36": round(float(row.ast_per36), 1),
            "reb_per36": round(float(row.reb_per36), 1), "stl_per36": round(float(row.stl_per36), 1),
            "blk_per36": round(float(row.blk_per36), 1), "ts_pct": round(float(row.ts_pct), 3),
        },
        "team_win_pct": round(float(row.team_win_pct), 3),
        "score_breakdown": {
            "stats_pct": round(max(stats_share, 0.0) * 100, 1),
            "media_pct": round(max(1 - stats_share, 0.0) * 100, 1),
        },
        "portrait": generate_portrait_data_uri(pid, team_color, row.position, num),
    }


def export(output_path: Path = Path(__file__).parent / "data.json") -> dict:
    t0 = time.time()
    print(f"Generating league {START_SEASON_YEAR}-{MAX_MODELED_YEAR} (history through {PRESENT_YEAR}, "
          f"projected through {MAX_MODELED_YEAR})...")
    league = generate_synthetic_league(START_SEASON_YEAR, MAX_MODELED_YEAR, seed=RANDOM_SEED)

    print("Engineering features...")
    team_seasons = add_offcourt_factors(add_advanced_team_columns(league["team_seasons"]))
    player_seasons = player_injury_proneness(league["player_seasons"])
    team_inj = team_injury_risk(player_seasons)
    player_seasons, team_drop = build_playoff_dropoff_features(player_seasons)
    composite = build_composite_ratings(team_seasons, team_inj, team_drop)
    ml_df = build_ml_dataset(composite, league["playoff_results"])

    print("Selecting season MVPs and generating portrait cards...")
    mvp_pool = _build_mvp_pool(player_seasons, composite)

    print("Training headline model (fixed held-out test years, for footer metrics)...")
    historical_ml_df = ml_df[ml_df.year <= PRESENT_YEAR]
    headline = train_and_evaluate(historical_ml_df, test_years=(PRESENT_YEAR - 2, PRESENT_YEAR - 1, PRESENT_YEAR))

    print("Running walk-forward retrospective/projection (one retrain per season)...")
    wf = walk_forward_predictions(ml_df, min_train_seasons=MIN_TRAIN_SEASONS, seed=RANDOM_SEED,
                                   n_estimators=WALK_FORWARD_N_ESTIMATORS, max_year=MAX_MODELED_YEAR)

    hist_acc = year_accuracy_summary(wf[wf.year <= PRESENT_YEAR])
    baseline_accuracy = float(hist_acc.accuracy_score.mean())
    acc_by_year = hist_acc.set_index("year").to_dict("index")

    min_year = START_SEASON_YEAR + MIN_TRAIN_SEASONS
    rng = np.random.default_rng(RANDOM_SEED)
    playoff_results = league["playoff_results"]

    years_payload = {}
    for year in sorted(wf.year.unique()):
        year = int(year)
        historical = bool(year <= PRESENT_YEAR)
        wf_year_df = wf[wf.year == year]

        standings = {
            "East": _standings_for_year(composite, wf_year_df, "East", year),
            "West": _standings_for_year(composite, wf_year_df, "West", year),
        }

        sub = composite[composite.year == year]
        sim_rating = dict(zip(
            sub["team"],
            sub["net_rating"] + sub["team_playoff_dropoff_score"] * 2.0 - sub["team_injury_risk"] * 10.0,
        ))

        east_b = _conference_bracket(year, "East", composite, playoff_results, sim_rating, rng, historical)
        west_b = _conference_bracket(year, "West", composite, playoff_results, sim_rating, rng, historical)

        actual_champion = None
        if historical:
            champ_row = playoff_results[(playoff_results.year == year) & (playoff_results.champion)]
            actual_champion = champ_row.team.iloc[0] if len(champ_row) else None
        finals = _finals(east_b["champion"], west_b["champion"], sim_rating, rng, historical, actual_champion)

        if historical and year in acc_by_year:
            a = acc_by_year[year]
            accuracy = {
                "type": "measured", "score": a["accuracy_score"],
                "playoff_hit_rate": a["playoff_hit_rate"], "rounds_closeness": a["rounds_closeness"],
                "champion_prob_assigned": a["champion_prob_assigned_to_actual"],
                "champion_predicted": a["champion_predicted"], "champion_correct": bool(a["champion_correct"]),
            }
        else:
            accuracy = {"type": "estimated", "score": round(future_confidence(year, baseline_accuracy), 3)}

        payload = {
            "year": int(year), "season": season_label(year), "historical": historical,
            "era": era_name(year),
            "standings": standings,
            "bracket": {"East": east_b, "West": west_b, "finals": finals},
            "champion": finals["winner"],
            "accuracy": accuracy,
        }
        if historical:
            pr_year = playoff_results[playoff_results.year == year]
            payload["comparison"] = _comparison_rows(standings, wf_year_df, pr_year)

        mvp = _select_mvp(mvp_pool, year)
        if mvp:
            payload["mvp"] = mvp

        if year in REAL_MVP:
            real_mvp_info = REAL_MVP[year]
            mvp_image = real_mvp_image_data_uri(year)
            payload["real_history"] = {
                "mvp": real_mvp_info, "champion": REAL_CHAMPION.get(year),
                "mvp_image": mvp_image,
                "mvp_card": None if mvp_image else generate_real_player_card(
                    real_mvp_info["name"], real_mvp_info["team"],
                    real_mvp_info["position"], real_mvp_info["number"],
                ),
            }

        years_payload[str(int(year))] = payload

    teams_used = {row["team"] for y in years_payload.values() for conf_rows in y["standings"].values() for row in conf_rows}
    teams_meta = {t: TEAM_META[t] for t in teams_used}

    hist_acc["era"] = hist_acc["year"].map(era_name)
    era_accuracy = [
        {"era": band_name, "accuracy": round(float(hist_acc.loc[hist_acc.era == band_name, "accuracy_score"].mean()), 3),
         "seasons": int((hist_acc.era == band_name).sum())}
        for _, _, band_name in ERA_BANDS
        if (hist_acc.era == band_name).any()
    ]

    real_mvp_prediction = build_real_mvp_prediction()
    for candidate in real_mvp_prediction["candidates"]:
        candidate["card"] = generate_real_player_card(
            candidate["name"], candidate["team"], candidate["position"].split("/")[0], candidate["number"],
        )

    result = {
        "present_year": PRESENT_YEAR, "min_year": min_year, "max_year": MAX_MODELED_YEAR,
        "baseline_accuracy": round(baseline_accuracy, 3),
        "metrics": headline.metrics,
        "era_accuracy": era_accuracy,
        "real_mvp_prediction": real_mvp_prediction,
        "teams": teams_meta,
        "years": years_payload,
    }

    output_path.write_text(json.dumps(result))
    print(f"Wrote {output_path} ({output_path.stat().st_size} bytes) in {time.time() - t0:.1f}s total")
    return result


if __name__ == "__main__":
    export()
