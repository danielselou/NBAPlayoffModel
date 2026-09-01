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
| Injury proneness | Trailing games-missed rate + a `lifelines` Cox proportional-hazards fit (age + usage rate vs. games-survived-before-injury), weighted by each player's on-court impact | `src/injury.py` |
| Playoff "riser/faller" tendency | Career playoff-vs-regular-season per-36 scoring delta (causal: only uses *prior* postseasons, never leaks the season being predicted) | `src/playoff_dropoff.py` |
| Off-court context | Roster continuity, coaching stability, altitude home-court edge, market size | `src/team_strength.py` |
| Playoff simulation | Monte Carlo full-bracket simulation using log5/Elo-style win probabilities with home-court advantage | `src/simulate.py` |
| Era regimes ("how the game changed") | Pace, 3PA rate, and big-man-vs-guard value all shift by era (Big Man Era &rarr; Hand-Check &rarr; Perimeter Freedom &rarr; 3PT Revolution &rarr; Modern) instead of one fixed rate for 55 seasons | `src/era.py` |
| Prediction | LogisticRegression + RandomForest + XGBoost ensemble | `src/model.py` |

Modeled seasons run **1979-80 through 10 years past the last historical
season** -- 1980 was chosen deliberately: it's the year the NBA introduced
the 3-point line, giving the era system (below) real historical texture to
work with instead of one flat window.

`src/model.py`'s `FEATURE_COLS` feeds the classifiers the **era-adjusted
z-scores** of net rating, 3PA rate, TS%, and pace (`*_era_z`), not their
raw values: those four drift enormously across a 55-season range (league
pace alone runs ~105 in 1980 down to ~90 around 2000 back up to ~100 by
the 2020s), so a raw number doesn't mean the same thing in different eras.
Feeding raw values measurably hurt accuracy once the modeled range grew
past one narrow window; a controlled A/B on identical data showed the
era-normalized features winning on every metric (made-playoffs accuracy,
log-loss, and AUC for all three targets).

**Injuries factor into every stage, not just one.** `src/injury.py` scores
each player-season 0-1 by blending a trailing (prior-seasons-only, causal)
games-missed rate with a [`lifelines`](https://lifelines.readthedocs.io/)
Cox proportional-hazards fit -- the standard real sports-medicine survival-
analysis technique for "games survived before a significant injury,"
regressed on age and usage rate -- rather than a hand-picked age curve
(the fitted model recovers the expected real relationship on its own:
both age and usage rate come out as statistically significant, positive
hazard predictors). That per-player score rolls up into `team_injury_risk`
(minutes x usage weighted, so a starter's durability matters far more than
a bench piece's), which then feeds three separate places: (1) it's a
`FEATURE_COLS` input the made-playoffs/rounds-won/champion classifiers see
directly, (2) it's one of `team_strength.py`'s composite-rating components
(weighted -0.15, i.e. working against a team), and (3)
`dashboard/export_data.py` subtracts it (scaled) from the rating fed into
the actual bracket/series Monte Carlo simulation, so a banged-up team's
game-by-game win probability is lower in the *simulated* playoffs too, not
just in the standings. Separately, the synthetic generator's own game
simulation already bakes in an age-injury effect when it decides how many
games each player actually misses that season (`src/data_collection.py`) --
`src/injury.py`'s job is to *re-estimate* that risk from observable stats
alone (the same "figure it out from the data, don't just read the answer"
principle the rest of the pipeline follows), not to read the generator's
internal truth directly.

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

Each season's free-agent fill-in drafts in a **reshuffled team order**
(`_assign_rosters` in `src/data_collection.py`), not the fixed alphabetical
`TEAMS` list -- looping teams in the same order every year let early-
alphabet teams (Atlanta, Boston, Brooklyn, ...) claim the best available
free agent first *every single season for the full 55-year range*, baking
in a permanent skill hierarchy instead of realistic year-to-year
competitive churn (caught from oddly extreme, static-looking standings:
Atlanta and Brooklyn were structurally overrated in every season, several
teams' playoff odds were pinned near 0%/100% regardless of actual season
strength). Shuffling draft order per season dropped the 55-season win-pct
spread across teams from 0.58 to roughly 0.12-0.17 -- real-league-scale
parity -- and single-season records now land in a normal range (best/worst
team varies year to year) instead of the same teams posting 70-plus-win or
sub-20-win seasons on repeat.

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

Earlier versions of this dashboard also crowned a fictional "Season MVP"
for the synthetic league itself. It's been removed: once both the real
MVP history panel and the real, data-driven MVP prediction (below)
existed, a fictional stand-in MVP for a simulated season no longer earned
its place and mostly caused confusion about which MVP was real.

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
up automatically — see the README in that folder. Absent a real photo,
`dashboard/export_data.py` falls back to `dashboard/portrait.py`'s
illustrated-card generator (the same one the Real-World MVP Prediction
section below uses for its own candidates), using each real player's
actual jersey number/position/team color — still clearly a silhouette
card, not a photo, and captioned as such.

The page also has scroll-triggered reveal animations on secondary text
(respecting `prefers-reduced-motion`) and an in-page **"How This Works"**
section mapping every library and stat/formula on the page to where it's
actually used — the same mapping as the table above, surfaced for anyone
looking at the dashboard itself rather than this file.

A third, standalone section — **"Real-World MVP Prediction"**
(`dashboard/real_mvp_prediction.py`) — is independent of the season picker
entirely: a real, current NBA player, predicted the same way the rest of
this project predicts anything — real inputs, z-scored across the
candidate pool, combined into a weighted composite (55% real per-game
performance / 20% real team win% / 25% real recognition history, i.e.
career MVP awards + All-Star selections) — rather than hand-typed 0-10
judgment calls. The one thing this project genuinely can't do is fetch
that real per-player data live (there's no real per-player data source
reachable from this environment — see Honest limitations), so the numbers
themselves are Claude's own recollection of each candidate's well-
documented public stats from their most recently *completed* real season,
not a live feed: approximate, not freshly sourced, and clearly labeled as
such in the module and on the page — verify exact figures independently
before citing formally. It's presented as a snapshot computed from
old-but-real inputs with an explicit knowledge-cutoff date, not a live
prediction, and it corrects a common misconception in-page: real MVP
voting has no official fan-vote component (that's the All-Star Game).
Like the Real NBA History panel, each candidate gets the same
illustrated-card fallback (no real photos are licensed for this project)
using their real jersey number and team color, clearly captioned as an
illustration rather than a photo.

Right below it, an **"MVP Formula Backtest"** section runs that identical
formula against 10 real, already-decided seasons (2010-11 through
2024-25) instead of a still-open one: each season's actual top handful of
MVP-caliber candidates, scored on their real final per-game stats and
team record for that season, checked against who actually won (from
`dashboard/real_history.py`'s `REAL_MVP`). This answers a different
question than the live prediction does -- not "using last season's stats,
who wins next" but "given this season's real numbers, does the formula's
ranking rule land on the real result" -- the standard way to sanity-check
a hand-built scoring rubric against history it was never fit to.
Recognition inputs (career MVP awards/All-Star selections) only count
what had happened *before* that season, so the formula never sees a
season's own outcome while scoring it. Currently **4/10 (40%)**: it
correctly picks LeBron 2013, Westbrook 2017, Harden 2018, and Jokic 2024,
but misses on seasons where real voting rewarded team success or
narrative over the rawest box-score line (2011: LeBron's numbers and
recognition outscore Rose's, but Chicago's record and the "youngest MVP
ever" story carried real voters; 2019: Harden's 36 PPG outscores Giannis
on paper, but Giannis's two-way impact and Milwaukee's record won
comfortably) -- an honest look at where a stats-heavy formula and real
MVP voting actually diverge, not just where they agree.

## Public site

`docs/index.html` is a committed mirror of the same self-contained
dashboard (`dashboard/build.py` writes both paths from one build) --
unlike `dashboard/index.html`, which is gitignored as a build artifact,
`docs/` is meant to be served directly by
[GitHub Pages](https://docs.github.com/pages), free, with no server of
its own (it's one static HTML file with the data inlined).

**One-time setup** (repo admin, via github.com -- not automatable from
here): if the repo is private, make it public first (Pages needs a public
repo on the free tier: Settings -> General -> Danger Zone -> Change
visibility), then Settings -> Pages -> Source: "Deploy from a branch" ->
branch: this branch -> folder: `/docs` -> Save. The site goes live at
`https://<owner>.github.io/<repo>/` a minute or two later.

**Publishing an update** after that is just: make the change, `python -m
dashboard.build` (regenerates both `dashboard/index.html` and
`docs/index.html`), commit, push -- GitHub Pages redeploys automatically
on every push to the configured branch.

**Feedback loop:** the page's footer links to "Open an issue on GitHub"
(`/issues/new`) for anyone to report a bug or suggest a stat/formula
change; that only works once the repo is public, since a private repo's
issues aren't visible to non-collaborators either.

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
