"""Monte Carlo playoff simulation engine.

Win probabilities come from a log5 / Elo-style logistic function of the rating
gap between two teams, the same family of formula used by public rating
systems (538's RAPTOR/CARM-Elo, Basketball-Reference's SRS-based series odds).
Series and full-bracket simulation are pure functions so they can be reused
both to (a) generate internally-consistent historical labels for the
synthetic league and (b) run live "what happens this postseason" projections.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import HOME_COURT_ADVANTAGE_PTS, N_MONTE_CARLO_SIMS, WIN_PROB_SCALE

# Real NBA best-of-7 home patterns, one entry per possible game (2-2-1-1-1).
# True = home team is the higher seed / series "home" team for that game.
_SERIES_HOME_PATTERN = [True, True, False, False, True, False, True]


def win_probability(rating_a: float, rating_b: float, a_is_home: bool = False,
                     home_advantage: float = HOME_COURT_ADVANTAGE_PTS,
                     scale: float = WIN_PROB_SCALE) -> float:
    """P(team A beats team B in a single game), log5-style logistic over
    real net-rating point differentials (not abstract Elo units)."""
    gap = rating_a - rating_b
    if a_is_home:
        gap += home_advantage
    else:
        gap -= home_advantage
    return 1.0 / (1.0 + 10 ** (-gap / scale))


def simulate_series(rating_a: float, rating_b: float, higher_seed_is_a: bool,
                     rng: np.random.Generator, best_of: int = 7) -> str:
    """Simulate one best-of-N series. Returns 'A' or 'B' for the winner.

    The higher seed gets home-court advantage and the historical 2-2-1-1-1
    home pattern (or 1-1-1-1-1... alternating for other best_of values).
    """
    wins_needed = best_of // 2 + 1
    a_wins = b_wins = 0
    pattern = _SERIES_HOME_PATTERN if best_of == 7 else [True, False] * best_of
    for game_idx in range(best_of):
        home_is_a = pattern[game_idx] if higher_seed_is_a else not pattern[game_idx]
        p_a = win_probability(rating_a, rating_b, a_is_home=home_is_a)
        if rng.random() < p_a:
            a_wins += 1
        else:
            b_wins += 1
        if a_wins == wins_needed or b_wins == wins_needed:
            break
    return "A" if a_wins > b_wins else "B"


def simulate_conference_bracket(seed_ratings: dict[int, float], rng: np.random.Generator) -> dict:
    """Simulate a standard 8-team single-conference bracket (no re-seeding).

    seed_ratings: {seed_number(1-8): rating}. Returns dict describing each
    round's winner seed and the conference champion seed.
    """
    matchups_r1 = [(1, 8), (4, 5), (3, 6), (2, 7)]
    r1_winners = []
    for high, low in matchups_r1:
        winner = high if simulate_series(seed_ratings[high], seed_ratings[low],
                                          higher_seed_is_a=True, rng=rng) == "A" else low
        r1_winners.append(winner)

    # Bracket integrity: (1/8 winner) vs (4/5 winner), (3/6 winner) vs (2/7 winner)
    semi_pairs = [(r1_winners[0], r1_winners[1]), (r1_winners[3], r1_winners[2])]
    semi_winners = []
    for s1, s2 in semi_pairs:
        higher = min(s1, s2)
        lower = max(s1, s2)
        winner = higher if simulate_series(seed_ratings[higher], seed_ratings[lower],
                                            higher_seed_is_a=True, rng=rng) == "A" else lower
        semi_winners.append(winner)

    higher, lower = min(semi_winners), max(semi_winners)
    conf_champ = higher if simulate_series(seed_ratings[higher], seed_ratings[lower],
                                            higher_seed_is_a=True, rng=rng) == "A" else lower

    return {
        "round_of_16_winners": r1_winners,   # conf quarterfinals
        "conf_semi_winners": semi_winners,
        "conf_champion_seed": conf_champ,
    }


def simulate_full_playoffs(east_ratings: dict[int, float], west_ratings: dict[int, float],
                            rng: np.random.Generator) -> dict:
    """Simulate an entire postseason (both conferences + Finals)."""
    east = simulate_conference_bracket(east_ratings, rng)
    west = simulate_conference_bracket(west_ratings, rng)

    e_champ_rating = east_ratings[east["conf_champion_seed"]]
    w_champ_rating = west_ratings[west["conf_champion_seed"]]
    # Finals: no true "home" seed advantage rule beyond better regular-season record;
    # approximate higher-rated team as the nominal higher seed for home pattern purposes.
    east_is_higher = e_champ_rating >= w_champ_rating
    winner = simulate_series(e_champ_rating, w_champ_rating, higher_seed_is_a=east_is_higher, rng=rng)
    champion_conf = "East" if (winner == "A") == east_is_higher else "West"

    return {"east": east, "west": west, "champion_conference": champion_conf}


def monte_carlo_team_outcomes(team_ratings_by_conf_seed: dict[str, dict[int, float]],
                               seed_to_team: dict[str, dict[int, str]],
                               n_sims: int = N_MONTE_CARLO_SIMS,
                               seed: int | None = None) -> pd.DataFrame:
    """Run N full-bracket simulations and tabulate each team's advancement odds.

    team_ratings_by_conf_seed: {"East": {seed: rating}, "West": {seed: rating}}
    seed_to_team: {"East": {seed: team_abbrev}, "West": {seed: team_abbrev}}
    Returns a DataFrame indexed by team with probability columns for each round.
    """
    rng = np.random.default_rng(seed)
    all_teams = list(seed_to_team["East"].values()) + list(seed_to_team["West"].values())
    counters = {team: {"made_r1_win": 0, "made_conf_semis": 0, "made_conf_finals": 0,
                        "made_finals": 0, "champion": 0} for team in all_teams}

    for _ in range(n_sims):
        result = simulate_full_playoffs(team_ratings_by_conf_seed["East"],
                                         team_ratings_by_conf_seed["West"], rng)
        for conf in ("east", "west"):
            conf_name = "East" if conf == "east" else "West"
            r = result[conf]
            for seed_num in r["round_of_16_winners"]:
                counters[seed_to_team[conf_name][seed_num]]["made_r1_win"] += 1
            for seed_num in r["conf_semi_winners"]:
                counters[seed_to_team[conf_name][seed_num]]["made_conf_semis"] += 1
            champ_seed = r["conf_champion_seed"]
            counters[seed_to_team[conf_name][champ_seed]]["made_conf_finals"] += 1
            counters[seed_to_team[conf_name][champ_seed]]["made_finals"] += 1

        champ_conf = result["champion_conference"]
        champ_seed = result[champ_conf.lower()]["conf_champion_seed"]
        counters[seed_to_team[champ_conf][champ_seed]]["champion"] += 1

    df = pd.DataFrame(counters).T
    return (df / n_sims).sort_values("champion", ascending=False)
