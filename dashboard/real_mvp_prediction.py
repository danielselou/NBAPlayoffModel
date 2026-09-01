"""A real-world NBA MVP prediction, computed the same way the rest of this
project predicts anything: real inputs, z-scored against the candidate
pool, combined into a weighted composite -- the same "data -> composite
score -> rank" shape `src/team_strength.py` uses for team ratings. It is
NOT the same thing as (a) the model's simulated standings/champion in the
synthetic league (always a fictional team) or (b)
dashboard/real_history.py's factual record of past, already-decided real
awards.

Two things live in this module:

1. `build_prediction()` -- a live prediction for the **2025-26** season
   (see CANDIDATES/PREDICTING_SEASON below).
2. `build_backtest()` -- the same formula run on ~10 real, already-decided
   past seasons (BACKTEST_SEASONS below), checked against the real
   historical winner (dashboard/real_history.py's REAL_MVP), so you can
   see how often this exact formula would have picked the real MVP.

Why the underlying numbers are hand-entered rather than fetched: the rest
of this project's "team rankings" run on a synthetic league (see
src/data_collection.py) because bulk real NBA box scores aren't reachable
from this environment (no network access -- confirmed directly, see
src/data_collection.py:fetch_real_season). There is no live real-per-
player-data feed anywhere in this codebase. What follows instead is each
candidate's own real, publicly documented statistics, recalled from
Claude's training data rather than pulled live. Treat every number below
as an approximate recollection of public record, not an authoritative,
freshly sourced statistic; verify exact figures against Basketball-
Reference or nba.com before citing formally. Once real per-player data is
reachable, swap the hand-entered numbers for a live query and this
module's formula runs unchanged.

Real-world caveat: the actual MVP award is decided by a media panel (not
fans) after the season ends -- there's no official *MVP* fan vote to
model (real fan voting exists for the All-Star Game, a different award).
"recognition" below is a real, verifiable proxy (career MVP awards +
All-Star selections *entering* the season being scored -- never including
that season's own result, so the formula can't see the answer before
scoring it), not a cited poll or vote count.
"""
from __future__ import annotations

import numpy as np

KNOWLEDGE_CUTOFF = "January 2026"
PREDICTING_SEASON = "2025-26"
STATS_SEASON = "2024-25"  # the real season the live-prediction numbers below are drawn from

# Composite weights across the three *data-derived* components computed
# below: performance dominates, team success and real recognition history
# make up the rest.
WEIGHTS = {"performance": 0.55, "team_success": 0.20, "recognition": 0.25}

# Sub-weights for the performance composite, applied to z-scores of real
# per-game stats -- scoring heaviest, then efficiency, then playmaking/
# rebounding/stocks.
PERFORMANCE_WEIGHTS = {"pts": 0.40, "ts_pct": 0.18, "ast": 0.16, "reb": 0.16, "stocks": 0.10}


def _zscore(values: list[float]) -> np.ndarray:
    arr = np.array(values, dtype=float)
    std = arr.std()
    return (arr - arr.mean()) / std if std > 1e-9 else np.zeros_like(arr)


def _score(candidates: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Shared scoring core for both the live prediction and the backtest:
    weighted z-scored composite -> logistic squash (same log5/logistic-
    scale idea src/simulate.py uses for win probabilities, rather than a
    hard min-max stretch, so the spread reflects how far apart the
    composites actually are). Returns (blended_z, pct), aligned to
    `candidates`' order."""
    stocks = [c["real_stats"]["stl"] + c["real_stats"]["blk"] for c in candidates]
    perf_z = (
        PERFORMANCE_WEIGHTS["pts"] * _zscore([c["real_stats"]["pts"] for c in candidates])
        + PERFORMANCE_WEIGHTS["ts_pct"] * _zscore([c["real_stats"]["ts_pct"] for c in candidates])
        + PERFORMANCE_WEIGHTS["ast"] * _zscore([c["real_stats"]["ast"] for c in candidates])
        + PERFORMANCE_WEIGHTS["reb"] * _zscore([c["real_stats"]["reb"] for c in candidates])
        + PERFORMANCE_WEIGHTS["stocks"] * _zscore(stocks)
    )
    team_z = _zscore([c["team_win_pct"] for c in candidates])
    recognition_raw = [c["prior_mvp_awards"] * 3 + c["all_star_selections"] * 0.5 for c in candidates]
    recognition_z = _zscore(recognition_raw)

    blended_z = WEIGHTS["performance"] * perf_z + WEIGHTS["team_success"] * team_z + WEIGHTS["recognition"] * recognition_z
    pct = 100.0 / (1.0 + np.exp(-1.2 * blended_z))
    return blended_z, pct


def _ranked_rows(candidates: list[dict], blended_z: np.ndarray, pct: np.ndarray) -> list[dict]:
    order = np.argsort(-blended_z)
    rows = []
    for rank_idx in order:
        c = candidates[rank_idx]
        row = {
            "name": c["name"], "team": c["team"], "position": c["position"], "number": c["number"],
            "real_stats": c["real_stats"], "team_win_pct": c["team_win_pct"],
            "prior_mvp_awards": c["prior_mvp_awards"], "all_star_selections": c["all_star_selections"],
            "total_pct": round(float(pct[rank_idx]), 1),
        }
        if "blurb" in c:
            row["blurb"] = c["blurb"]
        rows.append(row)
    return rows


# Real per-game averages from each candidate's most recently *completed*
# real NBA season (2024-25) -- approximate recollections of well-
# documented public stats, not a live box-score fetch (see module
# docstring). team_win_pct is that same real team's 2024-25 record.
# prior_mvp_awards and all_star_selections are real, well-established
# career counts through 2024-25 -- the "recognition" data point below.
CANDIDATES = [
    {
        "name": "Shai Gilgeous-Alexander", "team": "Oklahoma City Thunder", "position": "PG", "number": 2,
        "real_stats": {"pts": 32.7, "reb": 5.0, "ast": 6.4, "stl": 1.7, "blk": 1.0, "ts_pct": 0.637},
        "team_win_pct": 0.829, "prior_mvp_awards": 1, "all_star_selections": 4,
        "blurb": "Won the real 2024-25 MVP and Finals MVP outright, anchoring the league's best "
                 "young contender with elite scoring efficiency -- a repeat bid for 2025-26 with "
                 "no obvious drop-off signal as of this knowledge cutoff.",
    },
    {
        "name": "Luka Doncic", "team": "Los Angeles Lakers", "position": "PG/SG", "number": 77,
        "real_stats": {"pts": 28.2, "reb": 8.2, "ast": 7.7, "stl": 1.4, "blk": 0.5, "ts_pct": 0.556},
        "team_win_pct": 0.610, "prior_mvp_awards": 0, "all_star_selections": 5,
        "blurb": "The blockbuster trade to the Lakers put him in the league's biggest market "
                 "alongside LeBron James, and his usage/production profile remains among the "
                 "sport's best -- as much a media storyline as a stat line.",
    },
    {
        "name": "Nikola Jokic", "team": "Denver Nuggets", "position": "C", "number": 15,
        "real_stats": {"pts": 29.6, "reb": 12.7, "ast": 10.2, "stl": 1.8, "blk": 0.6, "ts_pct": 0.660},
        "team_win_pct": 0.610, "prior_mvp_awards": 3, "all_star_selections": 7,
        "blurb": "Sustained, historically efficient all-around production (a 3x real MVP); the "
                 "passing/scoring/rebounding combination has no real precedent, though voter "
                 "fatigue and a slightly lower media profile have cost him before.",
    },
    {
        "name": "Anthony Edwards", "team": "Minnesota Timberwolves", "position": "SG", "number": 1,
        "real_stats": {"pts": 27.6, "reb": 5.7, "ast": 4.5, "stl": 1.2, "blk": 0.6, "ts_pct": 0.573},
        "team_win_pct": 0.598, "prior_mvp_awards": 0, "all_star_selections": 2,
        "blurb": "The league's clearest \"face of the next generation\" storyline -- highlight-"
                 "driven popularity and real two-way production growth -- but a thinner real "
                 "award history than the multi-time-MVP names on this list.",
    },
    {
        "name": "Giannis Antetokounmpo", "team": "Milwaukee Bucks", "position": "PF/C", "number": 34,
        "real_stats": {"pts": 30.4, "reb": 11.9, "ast": 6.5, "stl": 1.2, "blk": 1.0, "ts_pct": 0.611},
        "team_win_pct": 0.585, "prior_mvp_awards": 2, "all_star_selections": 7,
        "blurb": "Still a top-handful two-way force by real, recent production; the case is "
                 "capped mainly by Milwaukee's less certain contender status relative to the "
                 "other real teams on this list.",
    },
    {
        "name": "Victor Wembanyama", "team": "San Antonio Spurs", "position": "C", "number": 1,
        "real_stats": {"pts": 24.3, "reb": 11.0, "ast": 3.7, "stl": 1.1, "blk": 3.7, "ts_pct": 0.590},
        "team_win_pct": 0.415, "prior_mvp_awards": 0, "all_star_selections": 1,
        "blurb": "The single biggest hype/media narrative in the league -- a generational "
                 "defensive/two-way talent -- but a real, documented mid-2024-25 blood-clot "
                 "diagnosis cut that season short, and a non-playoff Spurs team caps the "
                 "team-success side of the real formula below.",
    },
    {
        "name": "Stephen Curry", "team": "Golden State Warriors", "position": "PG", "number": 30,
        "real_stats": {"pts": 24.5, "reb": 4.4, "ast": 6.0, "stl": 0.9, "blk": 0.2, "ts_pct": 0.624},
        "team_win_pct": 0.585, "prior_mvp_awards": 2, "all_star_selections": 10,
        "blurb": "Still elite, and arguably the single most culturally influential player of his "
                 "generation (the real recognition side of the formula reflects that), but real "
                 "per-game scoring output has settled below his outright-MVP peak.",
    },
    {
        "name": "Jayson Tatum", "team": "Boston Celtics", "position": "SF", "number": 0,
        "real_stats": {"pts": 26.8, "reb": 8.7, "ast": 6.0, "stl": 1.0, "blk": 0.6, "ts_pct": 0.567},
        "team_win_pct": 0.744, "prior_mvp_awards": 0, "all_star_selections": 6,
        "blurb": "A perennial contender's best player on real per-game production and team win%, "
                 "but a serious real-world Achilles injury near the end of the 2024-25 playoffs "
                 "is a genuine, material 2025-26 availability question this formula doesn't see.",
    },
]


def build_prediction() -> dict:
    blended_z, pct = _score(CANDIDATES)
    rows = _ranked_rows(CANDIDATES, blended_z, pct)
    return {
        "knowledge_cutoff": KNOWLEDGE_CUTOFF,
        "predicting_season": PREDICTING_SEASON,
        "stats_season": STATS_SEASON,
        "weights": WEIGHTS,
        "candidates": rows,
        "predicted_mvp": rows[0]["name"],
    }


# --------------------------------------------------------------------------
# Historical backtest: the *same* formula above, run on ~10 real, already-
# decided seasons' own actual final numbers, checked against the real
# historical winner (dashboard/real_history.py's REAL_MVP). This answers a
# different question than build_prediction() does: not "using last
# season's stats, who will win next" (build_prediction's honest, forward-
# looking constraint), but "given this season's real, final numbers, does
# the formula's ranking rule correctly identify the real result" -- the
# standard way to sanity-check a hand-built scoring rubric against history
# it was never fit to. Team is each candidate's real team *that season*
# (some played elsewhere in other years); prior_mvp_awards/
# all_star_selections count only awards/selections from *before* that
# season, so the formula never sees a season's own outcome while scoring
# it, keeping the same causal discipline as the rest of this pipeline
# (see e.g. src/injury.py's trailing-rate window or
# src/playoff_dropoff.py's prior-postseasons-only rule).
# --------------------------------------------------------------------------
BACKTEST_SEASONS: dict[int, list[dict]] = {
    2011: [
        {"name": "Derrick Rose", "team": "Chicago Bulls", "position": "PG", "number": 1,
         "real_stats": {"pts": 25.0, "reb": 4.1, "ast": 7.7, "stl": 1.0, "blk": 0.6, "ts_pct": 0.550},
         "team_win_pct": 0.756, "prior_mvp_awards": 0, "all_star_selections": 1},
        {"name": "Dwight Howard", "team": "Orlando Magic", "position": "C", "number": 12,
         "real_stats": {"pts": 22.9, "reb": 14.1, "ast": 1.9, "stl": 1.0, "blk": 2.4, "ts_pct": 0.575},
         "team_win_pct": 0.634, "prior_mvp_awards": 0, "all_star_selections": 4},
        {"name": "LeBron James", "team": "Miami Heat", "position": "SF", "number": 6,
         "real_stats": {"pts": 26.7, "reb": 7.5, "ast": 7.0, "stl": 1.6, "blk": 0.6, "ts_pct": 0.594},
         "team_win_pct": 0.707, "prior_mvp_awards": 2, "all_star_selections": 6},
    ],
    2013: [
        {"name": "LeBron James", "team": "Miami Heat", "position": "SF", "number": 6,
         "real_stats": {"pts": 26.8, "reb": 8.0, "ast": 7.3, "stl": 1.7, "blk": 0.9, "ts_pct": 0.640},
         "team_win_pct": 0.805, "prior_mvp_awards": 3, "all_star_selections": 8},
        {"name": "Kevin Durant", "team": "Oklahoma City Thunder", "position": "SF", "number": 35,
         "real_stats": {"pts": 28.1, "reb": 7.9, "ast": 4.6, "stl": 1.4, "blk": 1.3, "ts_pct": 0.647},
         "team_win_pct": 0.756, "prior_mvp_awards": 0, "all_star_selections": 4},
        {"name": "Chris Paul", "team": "LA Clippers", "position": "PG", "number": 3,
         "real_stats": {"pts": 16.9, "reb": 3.7, "ast": 9.7, "stl": 2.4, "blk": 0.2, "ts_pct": 0.575},
         "team_win_pct": 0.695, "prior_mvp_awards": 0, "all_star_selections": 6},
    ],
    2015: [
        {"name": "Stephen Curry", "team": "Golden State Warriors", "position": "PG", "number": 30,
         "real_stats": {"pts": 23.8, "reb": 4.3, "ast": 7.7, "stl": 2.0, "blk": 0.2, "ts_pct": 0.638},
         "team_win_pct": 0.805, "prior_mvp_awards": 0, "all_star_selections": 2},
        {"name": "James Harden", "team": "Houston Rockets", "position": "SG", "number": 13,
         "real_stats": {"pts": 27.4, "reb": 5.7, "ast": 7.0, "stl": 1.9, "blk": 0.7, "ts_pct": 0.605},
         "team_win_pct": 0.695, "prior_mvp_awards": 0, "all_star_selections": 2},
        {"name": "LeBron James", "team": "Cleveland Cavaliers", "position": "SF", "number": 23,
         "real_stats": {"pts": 25.3, "reb": 6.0, "ast": 7.4, "stl": 1.6, "blk": 0.7, "ts_pct": 0.577},
         "team_win_pct": 0.622, "prior_mvp_awards": 4, "all_star_selections": 10},
    ],
    2017: [
        {"name": "Russell Westbrook", "team": "Oklahoma City Thunder", "position": "PG", "number": 0,
         "real_stats": {"pts": 31.6, "reb": 10.7, "ast": 10.4, "stl": 1.6, "blk": 0.4, "ts_pct": 0.554},
         "team_win_pct": 0.585, "prior_mvp_awards": 0, "all_star_selections": 6},
        {"name": "James Harden", "team": "Houston Rockets", "position": "SG", "number": 13,
         "real_stats": {"pts": 29.1, "reb": 8.1, "ast": 11.2, "stl": 1.5, "blk": 0.5, "ts_pct": 0.613},
         "team_win_pct": 0.671, "prior_mvp_awards": 0, "all_star_selections": 4},
        {"name": "Kawhi Leonard", "team": "San Antonio Spurs", "position": "SF", "number": 2,
         "real_stats": {"pts": 25.5, "reb": 5.8, "ast": 3.5, "stl": 1.8, "blk": 0.7, "ts_pct": 0.610},
         "team_win_pct": 0.744, "prior_mvp_awards": 0, "all_star_selections": 2},
    ],
    2018: [
        {"name": "James Harden", "team": "Houston Rockets", "position": "SG", "number": 13,
         "real_stats": {"pts": 30.4, "reb": 5.4, "ast": 8.8, "stl": 1.8, "blk": 0.6, "ts_pct": 0.619},
         "team_win_pct": 0.793, "prior_mvp_awards": 0, "all_star_selections": 5},
        {"name": "LeBron James", "team": "Cleveland Cavaliers", "position": "SF", "number": 23,
         "real_stats": {"pts": 27.5, "reb": 8.6, "ast": 9.1, "stl": 1.4, "blk": 0.9, "ts_pct": 0.621},
         "team_win_pct": 0.610, "prior_mvp_awards": 4, "all_star_selections": 13},
        {"name": "Anthony Davis", "team": "New Orleans Pelicans", "position": "PF/C", "number": 23,
         "real_stats": {"pts": 28.1, "reb": 11.1, "ast": 2.3, "stl": 1.5, "blk": 2.6, "ts_pct": 0.601},
         "team_win_pct": 0.585, "prior_mvp_awards": 0, "all_star_selections": 4},
    ],
    2019: [
        {"name": "Giannis Antetokounmpo", "team": "Milwaukee Bucks", "position": "PF", "number": 34,
         "real_stats": {"pts": 27.7, "reb": 12.5, "ast": 5.9, "stl": 1.3, "blk": 1.5, "ts_pct": 0.644},
         "team_win_pct": 0.732, "prior_mvp_awards": 0, "all_star_selections": 3},
        {"name": "James Harden", "team": "Houston Rockets", "position": "SG", "number": 13,
         "real_stats": {"pts": 36.1, "reb": 6.6, "ast": 7.5, "stl": 2.0, "blk": 0.7, "ts_pct": 0.616},
         "team_win_pct": 0.646, "prior_mvp_awards": 1, "all_star_selections": 6},
        {"name": "Paul George", "team": "Oklahoma City Thunder", "position": "SF", "number": 13,
         "real_stats": {"pts": 28.0, "reb": 8.2, "ast": 4.1, "stl": 2.2, "blk": 0.4, "ts_pct": 0.587},
         "team_win_pct": 0.585, "prior_mvp_awards": 0, "all_star_selections": 5},
    ],
    2021: [
        {"name": "Nikola Jokic", "team": "Denver Nuggets", "position": "C", "number": 15,
         "real_stats": {"pts": 26.4, "reb": 10.8, "ast": 8.3, "stl": 1.3, "blk": 0.7, "ts_pct": 0.649},
         "team_win_pct": 0.653, "prior_mvp_awards": 0, "all_star_selections": 2},
        {"name": "Joel Embiid", "team": "Philadelphia 76ers", "position": "C", "number": 21,
         "real_stats": {"pts": 28.5, "reb": 10.6, "ast": 2.8, "stl": 1.0, "blk": 1.4, "ts_pct": 0.633},
         "team_win_pct": 0.681, "prior_mvp_awards": 0, "all_star_selections": 3},
        {"name": "Stephen Curry", "team": "Golden State Warriors", "position": "PG", "number": 30,
         "real_stats": {"pts": 32.0, "reb": 5.5, "ast": 5.8, "stl": 1.2, "blk": 0.1, "ts_pct": 0.653},
         "team_win_pct": 0.542, "prior_mvp_awards": 2, "all_star_selections": 6},
    ],
    2023: [
        {"name": "Joel Embiid", "team": "Philadelphia 76ers", "position": "C", "number": 21,
         "real_stats": {"pts": 33.1, "reb": 10.2, "ast": 4.2, "stl": 1.0, "blk": 1.7, "ts_pct": 0.652},
         "team_win_pct": 0.646, "prior_mvp_awards": 0, "all_star_selections": 6},
        {"name": "Nikola Jokic", "team": "Denver Nuggets", "position": "C", "number": 15,
         "real_stats": {"pts": 24.5, "reb": 11.8, "ast": 9.8, "stl": 1.3, "blk": 0.7, "ts_pct": 0.702},
         "team_win_pct": 0.646, "prior_mvp_awards": 2, "all_star_selections": 5},
        {"name": "Giannis Antetokounmpo", "team": "Milwaukee Bucks", "position": "PF", "number": 34,
         "real_stats": {"pts": 31.1, "reb": 11.8, "ast": 5.7, "stl": 0.8, "blk": 0.8, "ts_pct": 0.633},
         "team_win_pct": 0.707, "prior_mvp_awards": 2, "all_star_selections": 6},
    ],
    2024: [
        {"name": "Nikola Jokic", "team": "Denver Nuggets", "position": "C", "number": 15,
         "real_stats": {"pts": 26.4, "reb": 12.4, "ast": 9.0, "stl": 1.4, "blk": 0.9, "ts_pct": 0.651},
         "team_win_pct": 0.695, "prior_mvp_awards": 2, "all_star_selections": 6},
        {"name": "Shai Gilgeous-Alexander", "team": "Oklahoma City Thunder", "position": "PG", "number": 2,
         "real_stats": {"pts": 30.1, "reb": 5.5, "ast": 6.2, "stl": 2.0, "blk": 0.9, "ts_pct": 0.635},
         "team_win_pct": 0.695, "prior_mvp_awards": 0, "all_star_selections": 1},
        {"name": "Luka Doncic", "team": "Dallas Mavericks", "position": "PG", "number": 77,
         "real_stats": {"pts": 33.9, "reb": 9.2, "ast": 9.8, "stl": 1.4, "blk": 0.5, "ts_pct": 0.617},
         "team_win_pct": 0.610, "prior_mvp_awards": 0, "all_star_selections": 4},
    ],
    2025: [
        {"name": "Shai Gilgeous-Alexander", "team": "Oklahoma City Thunder", "position": "PG", "number": 2,
         "real_stats": {"pts": 32.7, "reb": 5.0, "ast": 6.4, "stl": 1.7, "blk": 1.0, "ts_pct": 0.637},
         "team_win_pct": 0.829, "prior_mvp_awards": 0, "all_star_selections": 2},
        {"name": "Nikola Jokic", "team": "Denver Nuggets", "position": "C", "number": 15,
         "real_stats": {"pts": 29.6, "reb": 12.7, "ast": 10.2, "stl": 1.8, "blk": 0.6, "ts_pct": 0.660},
         "team_win_pct": 0.610, "prior_mvp_awards": 3, "all_star_selections": 6},
        {"name": "Giannis Antetokounmpo", "team": "Milwaukee Bucks", "position": "PF", "number": 34,
         "real_stats": {"pts": 30.4, "reb": 11.9, "ast": 6.5, "stl": 1.2, "blk": 1.0, "ts_pct": 0.611},
         "team_win_pct": 0.585, "prior_mvp_awards": 2, "all_star_selections": 7},
    ],
}


def build_backtest() -> dict:
    from dashboard.real_history import REAL_MVP
    from src.data_collection import season_label

    seasons = []
    correct = 0
    for year in sorted(BACKTEST_SEASONS):
        candidates = BACKTEST_SEASONS[year]
        blended_z, pct = _score(candidates)
        rows = _ranked_rows(candidates, blended_z, pct)
        actual_name = REAL_MVP[year]["name"]
        predicted_name = rows[0]["name"]
        is_correct = predicted_name == actual_name
        correct += int(is_correct)
        seasons.append({
            "year": year, "season": season_label(year), "candidates": rows,
            "actual_mvp": actual_name, "predicted_mvp": predicted_name, "correct": is_correct,
        })
    return {
        "seasons": seasons,
        "n_seasons": len(seasons),
        "n_correct": correct,
        "accuracy": round(correct / len(seasons), 3) if seasons else 0.0,
    }
