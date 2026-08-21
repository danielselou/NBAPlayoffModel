# NBA Playoff Prediction Model

An end-to-end pandas/scikit-learn pipeline that predicts NBA regular-season
and playoff outcomes from box-score stats, advanced efficiency metrics,
injury history, and playoff-specific performance trends.

## What it models

Rather than just win/loss records, this pipeline builds a team's strength
from every angle public NBA analytics communities (Basketball-Reference,
Thinking Basketball's "Era Ball", the "82-0" projection-modeling community)
typically use:

| Category | Signal | Module |
|---|---|---|
| Box score | PTS/REB/AST/STL/BLK/TOV, FG%/3P%/FT%, usage rate | `src/data_collection.py` |
| Advanced efficiency | True Shooting %, eFG%, Four Factors | `src/features.py` |
| Team on-court strength | Net Rating, Simple Rating System (SRS), Pythagorean win expectation | `src/features.py` |
| Era adjustment ("Era Ball") | Every rate stat z-scored against its **own season's** league context, so a 2003 3PA rate and a 2023 3PA rate are judged fairly instead of face-value | `src/features.py` |
| Injury proneness | Trailing games-missed rate + age-based risk curve, weighted by each player's on-court impact | `src/injury.py` |
| Playoff "riser/faller" tendency | Career playoff-vs-regular-season per-36 scoring delta (causal: only uses *prior* postseasons, never leaks the season being predicted) | `src/playoff_dropoff.py` |
| Off-court context | Roster continuity, coaching stability, altitude home-court edge, market size | `src/team_strength.py` |
| Playoff simulation | Monte Carlo full-bracket simulation using log5/Elo-style win probabilities with home-court advantage | `src/simulate.py` |
| Prediction | LogisticRegression + RandomForest + XGBoost ensemble | `src/model.py` |

## Data source

Real NBA stats can be pulled live via [`nba_api`](https://github.com/swar/nba_api)
(`src/data_collection.py:fetch_real_season`), but `stats.nba.com` frequently
blocks data-center IPs/proxies, so the pipeline ships with a **synthetic
league generator** (`generate_synthetic_league`) as a fully-functional
offline fallback and demo dataset. It isn't random noise: it builds a
persistent 20+ year player universe (so career injury history and playoff
deltas actually accumulate), simulates real free-agency-style roster
churn, and derives win totals from simulated offensive/defensive ratings
via the actual Pythagorean formula -- then plays out every playoff bracket
game-by-game with `src/simulate.py`, so the "labels" the model learns from
are driven by the same signals the feature pipeline computes, not arbitrary
noise. Point it at real box scores by populating `data/raw/` in the same
schema and swapping the loader in `main.py`.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

This generates ~25 seasons of league data, engineers every feature above,
trains the ensemble models (time-based train/test split so the model never
sees "future" seasons during training), and projects the most recent
season's playoffs both via the ML models and an independent Monte Carlo
bracket simulation -- printed metrics plus these files in `output/`:

- `feature_importance.csv` / `.png` — what actually predicts making the playoffs
- `team_predictions_<year>.csv` — per-team ML playoff/round/championship probabilities
- `monte_carlo_odds_<year>.csv` — bracket-simulation round-by-round odds
- `trained_models.joblib` — the fitted ensemble (reloadable via `joblib.load`)

## Tests

```bash
pytest tests/ -v
```

23 sanity tests check formula correctness (TS%/eFG%/Pythagorean against
known values), bounded outputs, that trailing/causal features never leak
future data, and — most importantly — that higher playoff seeds actually
advance further and win more often (a real calibration check: an earlier
version of this pipeline had home-court advantage overwhelming team skill
in the bracket simulator, and this test catches that class of bug).

## Project layout

```
config.py                 seasons, teams, RNG seed, simulation constants
src/data_collection.py    nba_api fetch (best-effort) + synthetic league generator
src/features.py           TS%/eFG%/Four Factors, SRS, Pythagorean, era z-scores
src/injury.py             player + team injury-proneness scoring
src/playoff_dropoff.py    playoff riser/faller scoring (causal/trailing)
src/team_strength.py      composite rating: on-court + off-court factors
src/model.py              ensemble ML training/evaluation
src/simulate.py           log5/Elo win probability + Monte Carlo bracket sim
main.py                   orchestrates the full pipeline
tests/test_pipeline.py    pytest sanity + calibration tests
```

## Honest limitations

This is a demonstration pipeline. The synthetic data is internally
consistent (skill -> rating -> wins -> seeding -> playoff results all
derive from the same underlying numbers, and several tests confirm the
correlations hold), but it is **not real NBA history** — treat reported
accuracy numbers as validation that the pipeline works end-to-end, not as
real-world predictive accuracy. For real predictions, feed it actual
`nba_api`/Basketball-Reference data in the same schema.
