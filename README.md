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
| Era regimes ("how the game changed") | Pace, 3PA rate, and big-man-vs-guard value all shift by era (Big Man Era &rarr; Hand-Check &rarr; Perimeter Freedom &rarr; 3PT Revolution &rarr; Modern) instead of one fixed rate for 55 seasons | `src/era.py` |
| Prediction | LogisticRegression + RandomForest + XGBoost ensemble | `src/model.py` |

Modeled seasons run **1979-80 through 10 years past the last historical
season** -- 1980 was chosen deliberately: it's the year the NBA introduced
the 3-point line, giving the era system (below) real historical texture to
work with instead of one flat window.

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

## Visual dashboard

```bash
python -m dashboard.build   # runs the pipeline, writes dashboard/index.html
```

Produces a self-contained HTML page with a season picker covering every
modeled year, 1982-83 through 10 seasons past the last historical season
(the walk-forward model needs a few seasons of history before its first
genuine out-of-sample prediction):

- **Historical seasons** show the generator's actual outcome (standings,
  bracket, champion) *and* a genuinely out-of-sample **walk-forward**
  prediction — a model retrained on only the strictly-earlier seasons, so
  the "Model Accuracy" panel and per-team comparison table are a real
  model-vs-actual check, not a model grading its own homework.
- **Future seasons** (type in any year in the supported range) show only
  the model's projection — standings, a bracket built from simulated
  favorites (dashed lines mark projected series), and a **confidence
  rating that decays with distance** from the last historical season,
  explicitly labeled as an estimate rather than a measured accuracy.

`dashboard/export_data.py` runs `src.model.walk_forward_predictions` (one
retrain per season, expanding window) to produce every year's prediction,
`src.model.year_accuracy_summary` to score historical years, and
`src.model.future_confidence` for the decay curve; brackets are
reconstructed from seeds + `rounds_won` for history (NBA brackets never
re-seed, so the winner at each node is fully determined) or from
re-simulated matchups for projections. `dashboard/team_meta.py` holds team
display names/colors; `dashboard/template.html` is the page itself. Full
export (55 seasons, walk-forward retraining included) takes 3-4 minutes.
A **per-era accuracy breakdown** (average walk-forward accuracy for each of
the four historical eras) sits in the Model Accuracy section, so you can
see where the fit holds up and where it doesn't, rather than one aggregate
number hiding era-to-era variation.

Each season also gets a **Season MVP** card, scored 85% stats / 15%
"outside the box score": the stats side reuses `era_adjusted_zscore` (the
same function used throughout the pipeline) on scoring, playmaking,
rebounding, stocks, and shooting efficiency, with the sub-weights
themselves shifting by era (rebounding/rim-protection matter more in the
Big Man Era, efficiency more in the 3PT Revolution — see `src/era.py`); the
media/market side blends team win% (the "your team has to win" voter
narrative), market size, and a modeled narrative-buzz term, deliberately
kept a minority weight. The card shows the *actual* stats-vs-media split
for that pick, not just the fixed formula weight. `dashboard/player_names.py`
gives each synthetic player ID a stable fictional name; `dashboard/portrait.py`
draws a small illustrated card (team-color gradient, silhouette, jersey
number) with Pillow and embeds it as a PNG data URI — there's no real photo
behind these players, so the card is deliberately abstract rather than
attempting a likeness.

A separate **"Real NBA History"** panel (`dashboard/real_history.py`) shows
the *actual* MVP and champion for each real season, 1982-83 through
2024-25 (Shai Gilgeous-Alexander won MVP and Finals MVP; Oklahoma City beat
Indiana in 7 games) — the most recent real season with a confirmed outcome
as of this project's knowledge cutoff. 2025-26 isn't included: it was
still in progress at that cutoff, and this environment has no live network
access to check it — see the "Real-World MVP Prediction" section instead
for a clearly-labeled opinion/prediction about that season, not a claimed
fact. Compiled from well-established public record, kept visually and
structurally distinct from the model's own (entirely fictional) simulated
season so the two are never confused. It ships with no real player photos
(licensing them requires rights this project doesn't have); drop a
licensed image at `dashboard/assets/real_mvps/<year>.jpg` and it's picked
up automatically — see the README in that folder.

The page also has scroll-triggered reveal animations on secondary text
(respecting `prefers-reduced-motion`) and an in-page **"How This Works"**
section mapping every library and stat/formula on the page to where it's
actually used — the same mapping as the table above, surfaced for anyone
looking at the dashboard itself rather than this file.

A third, standalone section — **"Real-World MVP Prediction"**
(`dashboard/real_mvp_prediction.py`) — is independent of the season picker
entirely: a real, current NBA player, ranked by a transparent rubric (55%
performance / 20% team success / 15% media press / 10% fan popularity),
where every 0-10 score is Claude's own qualitative judgment from general
basketball knowledge, not a database lookup or sourced statistic (there's
no real per-player data anywhere in this project to compute one from — see
Honest limitations). It's presented as a snapshot opinion with an explicit
knowledge-cutoff date, not a live prediction, and it corrects a common
misconception in-page: real MVP voting has no official fan-vote component
(that's the All-Star Game); "fan popularity" here is an honest
cultural-impact proxy, not a cited poll.

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
config.py                 seasons, teams, RNG seed, era anchors, simulation constants
src/era.py                era interpolation (pace, 3PA rate, big-man/guard value by year)
src/data_collection.py    nba_api fetch (best-effort) + synthetic league generator
src/features.py           TS%/eFG%/Four Factors, SRS, Pythagorean, era z-scores
src/injury.py             player + team injury-proneness scoring
src/playoff_dropoff.py    playoff riser/faller scoring (causal/trailing)
src/team_strength.py      composite rating: on-court + off-court factors
src/model.py              ensemble ML training/evaluation, walk-forward, era accuracy
src/simulate.py           log5/Elo win probability + Monte Carlo bracket sim
main.py                   orchestrates the full pipeline
dashboard/                visual dashboard: export, MVP/portrait, real-history reference
tests/test_pipeline.py    pytest sanity + calibration tests
```

## Honest limitations

This is a demonstration pipeline. The synthetic data is internally
consistent (skill -> rating -> wins -> seeding -> playoff results all
derive from the same underlying numbers, and several tests confirm the
correlations hold), but it is **not real NBA history** — treat reported
accuracy numbers as validation that the pipeline works end-to-end, not as
real-world predictive accuracy.

**Why not just train on real historical data?** `nba_api`/stats.nba.com is
network-blocked from the environment this was built in (confirmed directly
— see `fetch_real_season`'s docstring), and scraping Basketball-Reference
would violate its terms of service, so bulk real box scores genuinely
weren't reachable. The era system (`src/era.py`) is the honest middle
ground: real, well-documented trends (the pace trough of the 90s, the 3PT
rate's rise, big men's declining-then-partly-recovering value) shape the
*synthetic* generator's anchor points, without claiming those anchors are
verified historical statistics. The one place real facts do appear is the
dashboard's "Real NBA History" panel (actual MVPs/champions, 1982-83
onward) — kept strictly separate from, and never blended into, the
model's own simulated season. Point `src/data_collection.py` at real box
scores (same schema) from an environment with access, and everything
downstream — features, model, dashboard — works unchanged.
