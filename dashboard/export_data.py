"""Run the full prediction pipeline and export a self-contained JSON payload
for the visual dashboard: conference standings (official-format seed/W/L/PCT/GB)
and a reconstructed playoff bracket (actual series outcomes + model-estimated
series-win probabilities), for the most recent modeled season.

The bracket is *reconstructed* rather than recorded live during simulation:
since NBA brackets never re-seed, the round1 pairings (1v8, 4v5, 3v6, 2v7) and
each later round's pairings are fully determined by seed, so the winner at
each node can be read off deterministically from each team's `rounds_won`.
Series-win probabilities are then estimated independently by re-simulating
each actual matchup thousands of times from the two teams' ratings, so the
percentages shown are the model's real estimate for that specific series, not
a placeholder.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from config import RANDOM_SEED
from dashboard.team_meta import TEAM_META
from main import run_pipeline
from src.simulate import simulate_series

ROUND1_PAIRS = [(1, 8), (4, 5), (3, 6), (2, 7)]


def _series_win_prob(rating_winner: float, rating_loser: float, winner_is_higher_seed: bool,
                      rng: np.random.Generator, n_sims: int = 6000) -> float:
    """Re-simulate the actual matchup n_sims times; return the fraction the
    known winner takes, as the model's estimated confidence in that outcome."""
    wins = sum(
        simulate_series(rating_winner, rating_loser, higher_seed_is_a=winner_is_higher_seed, rng=rng) == "A"
        for _ in range(n_sims)
    )
    return wins / n_sims


def _build_standings(composite_df, predictions_df, conf: str, year: int) -> list[dict]:
    conf_df = composite_df[(composite_df.year == year) & (composite_df.conference == conf)].copy()
    conf_df = conf_df.sort_values("wins", ascending=False).reset_index(drop=True)
    leader_wins, leader_losses = conf_df.loc[0, "wins"], conf_df.loc[0, "losses"]
    prob_by_team = dict(zip(predictions_df.team, predictions_df.made_playoffs_prob))

    rows = []
    for i, row in conf_df.iterrows():
        gb = ((leader_wins - row.wins) + (row.losses - leader_losses)) / 2
        rows.append({
            "seed": i + 1,
            "team": row.team,
            "wins": int(row.wins),
            "losses": int(row.losses),
            "win_pct": round(float(row.win_pct), 3),
            "gb": round(gb, 1),
            "rating": round(float(row.net_rating), 1),
            "playoff_prob": round(float(prob_by_team.get(row.team, 0.0)), 3),
        })
    return rows


def _build_conference_bracket(playoff_results_df, composite_df, year: int, conf: str,
                               sim_rating: dict, rng: np.random.Generator) -> dict:
    conf_pr = playoff_results_df[(playoff_results_df.year == year) & (playoff_results_df.conference == conf)]
    team_by_seed = dict(zip(conf_pr.seed, conf_pr.team))
    rounds_won = dict(zip(conf_pr.team, conf_pr.rounds_won))

    def _node(team_a: str, team_b: str, seed_a: int, seed_b: int, min_rounds_to_win: int) -> dict:
        a_won = rounds_won[team_a] >= min_rounds_to_win
        winner, loser = (team_a, team_b) if a_won else (team_b, team_a)
        winner_is_higher = (team_a == winner and seed_a < seed_b) or (team_b == winner and seed_b < seed_a)
        prob = _series_win_prob(sim_rating[winner], sim_rating[loser], winner_is_higher, rng)
        return {"teamA": team_a, "teamB": team_b, "seedA": seed_a, "seedB": seed_b,
                "winner": winner, "winner_prob": round(prob, 3)}

    round1 = []
    r1_winner_seed = {}
    for high, low in ROUND1_PAIRS:
        node = _node(team_by_seed[high], team_by_seed[low], high, low, 1)
        node["seedA"], node["seedB"] = high, low
        round1.append(node)
        r1_winner_seed[high] = (node["winner"], high if node["winner"] == team_by_seed[high] else low)

    # Semis: (w(1,8) vs w(4,5)), (w(3,6) vs w(2,7)) -- mirrors src.simulate.simulate_conference_bracket
    w18, seed18 = r1_winner_seed[1]
    w45, seed45 = r1_winner_seed[4]
    w36, seed36 = r1_winner_seed[3]
    w27, seed27 = r1_winner_seed[2]
    semis = [
        _node(w18, w45, seed18, seed45, 2),
        _node(w36, w27, seed36, seed27, 2),
    ]

    sfin = semis[0]
    wsA, seedA = sfin["winner"], sfin["seedA"] if sfin["winner"] == sfin["teamA"] else sfin["seedB"]
    sfin2 = semis[1]
    wsB, seedB = sfin2["winner"], sfin2["seedA"] if sfin2["winner"] == sfin2["teamA"] else sfin2["seedB"]
    conf_finals = _node(wsA, wsB, seedA, seedB, 3)

    return {"round1": round1, "semis": semis, "conf_finals": conf_finals,
            "champion": conf_finals["winner"]}


def _finals_node(east_champ: str, west_champ: str, champion: str, sim_rating: dict,
                  rng: np.random.Generator) -> dict:
    loser = west_champ if champion == east_champ else east_champ
    winner_is_higher = sim_rating[champion] >= sim_rating[loser]
    prob = _series_win_prob(sim_rating[champion], sim_rating[loser], winner_is_higher, rng)
    return {"teamEast": east_champ, "teamWest": west_champ, "winner": champion, "winner_prob": round(prob, 3)}


def export(output_path: Path = Path(__file__).parent / "data.json") -> dict:
    result = run_pipeline()
    composite, predictions, league = result["composite"], result["predictions"], result["league"]
    year = int(composite["year"].max())
    season_label = composite.loc[composite.year == year, "season"].iloc[0]

    standings = {
        "East": _build_standings(composite, predictions, "East", year),
        "West": _build_standings(composite, predictions, "West", year),
    }

    sim_rating = dict(zip(
        composite[composite.year == year]["team"],
        composite[composite.year == year]["net_rating"]
        + composite[composite.year == year]["team_playoff_dropoff_score"] * 2.0
        - composite[composite.year == year]["team_injury_risk"] * 10.0,
    ))

    rng = np.random.default_rng(RANDOM_SEED)
    playoff_results = league["playoff_results"]
    east_bracket = _build_conference_bracket(playoff_results, composite, year, "East", sim_rating, rng)
    west_bracket = _build_conference_bracket(playoff_results, composite, year, "West", sim_rating, rng)
    finals = _finals_node(east_bracket["champion"], west_bracket["champion"],
                           playoff_results.loc[(playoff_results.year == year) & (playoff_results.champion), "team"].iloc[0],
                           sim_rating, rng)

    teams_used = {row["team"] for conf_rows in standings.values() for row in conf_rows}
    teams_meta = {t: TEAM_META[t] for t in teams_used}

    payload = {
        "season": season_label,
        "standings": standings,
        "bracket": {"East": east_bracket, "West": west_bracket, "finals": finals},
        "teams": teams_meta,
        "metrics": result["trained"].metrics,
    }

    output_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {output_path} ({output_path.stat().st_size} bytes)")
    return payload


if __name__ == "__main__":
    export()
