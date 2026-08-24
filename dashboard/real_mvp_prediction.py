"""A real-world NBA MVP prediction -- Claude's own analytical opinion, not a
database lookup, not a claimed statistic, and not the same thing as either
(a) the model's simulated MVP in the synthetic league (always a fictional
player) or (b) dashboard/real_history.py's factual record of past, already-
decided real awards.

Why this exists as hand-entered judgment rather than computed data: the
rest of this project's "team rankings" run on a synthetic league (see
src/data_collection.py) because bulk real NBA box scores aren't reachable
from this environment (no network access -- confirmed directly, see
src/data_collection.py:fetch_real_season). There is no real per-player
performance data anywhere in this codebase to compute a real formula from.
What follows is a transparent, weighted rubric applied to Claude's general
basketball knowledge (training data through the stated cutoff) about real,
named, currently-notable players -- each score is a qualitative judgment
call, presented as one, not dressed up as a sourced number.

Real-world caveat: the actual MVP award is decided by a media panel (not
fans) after the season ends. There is no official *MVP* fan vote to cite --
real fan voting exists for the All-Star Game, a different award entirely --
so "fan popularity" below is an honest qualitative cultural-impact proxy,
not a real vote count.

This predicts the **2025-26** season specifically: 2024-25 already has a
real, confirmed outcome (Shai Gilgeous-Alexander won MVP and Finals MVP,
Oklahoma City beat Indiana in 7 games -- see dashboard/real_history.py,
where that now lives as settled fact, not a prediction). 2025-26 was still
in progress as of this project's January 2026 knowledge cutoff, so its
real outcome isn't yet knowable here. By the time anyone reads this, that
real season may already be decided in ways this cutoff doesn't reflect --
treat this as a snapshot opinion, not a live prediction, and check an
official source for the real outcome.
"""
from __future__ import annotations

KNOWLEDGE_CUTOFF = "January 2026"
PREDICTING_SEASON = "2025-26"

WEIGHTS = {"performance": 0.55, "team_success": 0.20, "media_press": 0.15, "fan_popularity": 0.10}

# Each score is 0-10, Claude's own qualitative judgment as of the knowledge
# cutoff above -- not derived from any dataset in this project.
CANDIDATES = [
    {
        "name": "Shai Gilgeous-Alexander", "team": "Oklahoma City Thunder", "position": "PG", "number": 2,
        "scores": {"performance": 9.5, "team_success": 9.0, "media_press": 8.0, "fan_popularity": 7.5},
        "blurb": "Won the real 2024-25 MVP and Finals MVP outright, anchoring the league's best "
                 "young contender with elite scoring efficiency -- a repeat bid for 2025-26 with "
                 "no obvious drop-off signal as of this knowledge cutoff.",
    },
    {
        "name": "Luka Doncic", "team": "Los Angeles Lakers", "position": "PG/SG", "number": 77,
        "scores": {"performance": 9.0, "team_success": 8.0, "media_press": 9.5, "fan_popularity": 9.0},
        "blurb": "The blockbuster trade to the Lakers put him in the league's biggest market "
                 "alongside LeBron James, and his usage/production profile remains among the "
                 "sport's best -- as much a media storyline as a stat line.",
    },
    {
        "name": "Nikola Jokic", "team": "Denver Nuggets", "position": "C", "number": 15,
        "scores": {"performance": 9.5, "team_success": 7.5, "media_press": 7.0, "fan_popularity": 7.5},
        "blurb": "Sustained, historically efficient all-around production (a multi-time real "
                 "MVP); the passing/scoring/rebounding combination has no real precedent, "
                 "though voter fatigue and a slightly lower media profile have cost him before.",
    },
    {
        "name": "Anthony Edwards", "team": "Minnesota Timberwolves", "position": "SG", "number": 1,
        "scores": {"performance": 8.5, "team_success": 7.5, "media_press": 8.0, "fan_popularity": 8.5},
        "blurb": "The league's clearest \"face of the next generation\" storyline -- highlight-"
                 "driven popularity, real two-way production growth, and a competitive team.",
    },
    {
        "name": "Giannis Antetokounmpo", "team": "Milwaukee Bucks", "position": "PF/C", "number": 34,
        "scores": {"performance": 9.0, "team_success": 6.5, "media_press": 7.5, "fan_popularity": 8.0},
        "blurb": "Still a top-handful two-way force by reputation; the case is capped mainly by "
                 "Milwaukee's less certain contender status relative to the other names here.",
    },
    {
        "name": "Victor Wembanyama", "team": "San Antonio Spurs", "position": "C", "number": 1,
        "scores": {"performance": 8.0, "team_success": 5.5, "media_press": 9.0, "fan_popularity": 8.5},
        "blurb": "The single biggest hype/media narrative in the league -- a generational "
                 "defensive/two-way talent -- but a young, still-improving Spurs team and real "
                 "injury-history questions temper the performance/team-success case.",
    },
    {
        "name": "Stephen Curry", "team": "Golden State Warriors", "position": "PG", "number": 30,
        "scores": {"performance": 7.5, "team_success": 7.0, "media_press": 8.0, "fan_popularity": 9.0},
        "blurb": "Still elite, and arguably the single most culturally influential player of his "
                 "generation, but age and a deeper league of younger stars make a real repeat "
                 "MVP case harder than the cultural-impact score alone suggests.",
    },
    {
        "name": "Jayson Tatum", "team": "Boston Celtics", "position": "SF", "number": 0,
        "scores": {"performance": 7.0, "team_success": 8.0, "media_press": 7.5, "fan_popularity": 7.0},
        "blurb": "A perennial contender's best player, but a serious real-world injury near the "
                 "end of the 2024-25 playoffs is a genuine, material availability question this "
                 "cutoff can't fully resolve -- scored more conservatively as a result.",
    },
]


def score_candidate(scores: dict[str, float]) -> float:
    return sum(scores[k] * w for k, w in WEIGHTS.items())


def build_prediction() -> dict:
    ranked = sorted(CANDIDATES, key=lambda c: score_candidate(c["scores"]), reverse=True)
    rows = []
    for c in ranked:
        total = score_candidate(c["scores"])
        rows.append({
            "name": c["name"], "team": c["team"], "position": c["position"], "number": c["number"],
            "blurb": c["blurb"], "scores": c["scores"],
            "total_score": round(total, 2), "total_pct": round(total / 10 * 100, 1),
        })
    return {
        "knowledge_cutoff": KNOWLEDGE_CUTOFF,
        "predicting_season": PREDICTING_SEASON,
        "weights": WEIGHTS,
        "candidates": rows,
        "predicted_mvp": rows[0]["name"],
    }
