"""A real-world NBA MVP prediction, computed the same way the rest of this
project predicts anything: real inputs, z-scored against the candidate
pool, combined into a weighted composite -- the same "data -> composite
score -> rank" shape `src/team_strength.py` uses for team ratings and
`dashboard/export_data.py`'s `_select_mvp` uses for the fictional model's
own Season MVP. It is NOT the same thing as either (a) the model's
simulated MVP in the synthetic league (always a fictional player) or
(b) dashboard/real_history.py's factual record of past, already-decided
real awards.

Why the underlying numbers are hand-entered rather than fetched: the rest
of this project's "team rankings" run on a synthetic league (see
src/data_collection.py) because bulk real NBA box scores aren't reachable
from this environment (no network access -- confirmed directly, see
src/data_collection.py:fetch_real_season). There is no live real-per-
player-data feed anywhere in this codebase. What follows instead is each
candidate's own real, publicly documented statistics -- their most
recently *completed* real NBA season (2024-25), plus real career-award
counts -- recalled from Claude's training data rather than pulled live.
Treat every number below as an approximate recollection of public
record, not an authoritative, freshly sourced statistic; verify exact
figures against Basketball-Reference or nba.com before citing formally.
Once real per-player data is reachable, swap CANDIDATES' hand-entered
numbers for a live query and this module's formula runs unchanged.

Real-world caveat: the actual MVP award is decided by a media panel (not
fans) after the season ends -- there's no official *MVP* fan vote to
model (real fan voting exists for the All-Star Game, a different award).
"recognition" below is a real, verifiable proxy (career MVP awards +
All-Star selections through 2024-25), not a cited poll or vote count.

This predicts the **2025-26** season specifically: 2024-25 already has a
real, confirmed outcome (Shai Gilgeous-Alexander won MVP and Finals MVP,
Oklahoma City beat Indiana in 7 games -- see dashboard/real_history.py,
where that now lives as settled fact, not a prediction). 2025-26 was still
in progress as of this project's January 2026 knowledge cutoff, so its
real outcome isn't yet knowable here, and per-game stats/team records this
early in a still-in-progress season would be moving targets even if this
environment could fetch them live -- using each candidate's last full,
complete real season is the closest thing to a stable, real, "final"
number available at this cutoff. By the time anyone reads this, the real
2025-26 season may already be decided in ways this cutoff doesn't
reflect -- treat this as a snapshot computed from old-but-real inputs, not
a live prediction, and check an official source for the real outcome.
"""
from __future__ import annotations

import numpy as np

KNOWLEDGE_CUTOFF = "January 2026"
PREDICTING_SEASON = "2025-26"
STATS_SEASON = "2024-25"  # the real season the per-game/team numbers below are drawn from

# Composite weights across the three *data-derived* components computed
# below -- mirrors the fictional Season MVP formula's stats-vs-context
# split (see dashboard/export_data.py's MVP_STATS_WEIGHT_TOTAL /
# MVP_MEDIA_WEIGHT_TOTAL): performance dominates, team success and real
# recognition history make up the rest, standing in for the media-
# press/fan-popularity split the previous version of this module scored
# by hand.
WEIGHTS = {"performance": 0.55, "team_success": 0.20, "recognition": 0.25}

# Sub-weights for the performance composite, applied to era-agnostic
# (there's only one real "era" here) z-scores of real per-game stats --
# same relative emphasis (scoring heaviest, then efficiency, then
# playmaking/rebounding/stocks) as MVP_STATS_BASE_WEIGHTS in
# dashboard/export_data.py.
PERFORMANCE_WEIGHTS = {"pts": 0.40, "ts_pct": 0.18, "ast": 0.16, "reb": 0.16, "stocks": 0.10}

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


def _zscore(values: list[float]) -> np.ndarray:
    arr = np.array(values, dtype=float)
    std = arr.std()
    return (arr - arr.mean()) / std if std > 1e-9 else np.zeros_like(arr)


def build_prediction() -> dict:
    stocks = [c["real_stats"]["stl"] + c["real_stats"]["blk"] for c in CANDIDATES]
    perf_z = (
        PERFORMANCE_WEIGHTS["pts"] * _zscore([c["real_stats"]["pts"] for c in CANDIDATES])
        + PERFORMANCE_WEIGHTS["ts_pct"] * _zscore([c["real_stats"]["ts_pct"] for c in CANDIDATES])
        + PERFORMANCE_WEIGHTS["ast"] * _zscore([c["real_stats"]["ast"] for c in CANDIDATES])
        + PERFORMANCE_WEIGHTS["reb"] * _zscore([c["real_stats"]["reb"] for c in CANDIDATES])
        + PERFORMANCE_WEIGHTS["stocks"] * _zscore(stocks)
    )
    team_z = _zscore([c["team_win_pct"] for c in CANDIDATES])
    recognition_raw = [c["prior_mvp_awards"] * 3 + c["all_star_selections"] * 0.5 for c in CANDIDATES]
    recognition_z = _zscore(recognition_raw)

    blended_z = WEIGHTS["performance"] * perf_z + WEIGHTS["team_success"] * team_z + WEIGHTS["recognition"] * recognition_z
    # Logistic squash (same log5/logistic-scale idea src/simulate.py uses
    # for win probabilities) rather than a hard min-max stretch, so the
    # spread reflects how far apart the composites actually are instead of
    # always forcing the field into a fixed 0-100 range.
    pct = 100.0 / (1.0 + np.exp(-1.2 * blended_z))

    order = np.argsort(-blended_z)
    rows = []
    for rank_idx in order:
        c = CANDIDATES[rank_idx]
        rows.append({
            "name": c["name"], "team": c["team"], "position": c["position"], "number": c["number"],
            "blurb": c["blurb"], "real_stats": c["real_stats"], "team_win_pct": c["team_win_pct"],
            "prior_mvp_awards": c["prior_mvp_awards"], "all_star_selections": c["all_star_selections"],
            "total_pct": round(float(pct[rank_idx]), 1),
        })
    return {
        "knowledge_cutoff": KNOWLEDGE_CUTOFF,
        "predicting_season": PREDICTING_SEASON,
        "stats_season": STATS_SEASON,
        "weights": WEIGHTS,
        "candidates": rows,
        "predicted_mvp": rows[0]["name"],
    }
