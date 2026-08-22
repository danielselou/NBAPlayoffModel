"""Global configuration for the NBA playoff prediction pipeline."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
OUTPUT_DIR = ROOT_DIR / "output"

for _d in (DATA_RAW_DIR, DATA_PROCESSED_DIR, OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42

# Seasons to model. Starts at 1980 -- the season the 3-point line was
# introduced ("1979-80") -- so the era system below has real historical
# texture to work with (a pre-3pt Big Man Era through today's spacing game),
# not just one 25-year window.
START_SEASON_YEAR = 1980
END_SEASON_YEAR = 2024
CURRENT_SEASON_YEAR = END_SEASON_YEAR

# "Present" boundary: seasons up to here are treated as history (the
# generator's simulated outcome is revealed and can be compared against the
# model's retrospective, walk-forward prediction). Seasons beyond this are
# "the future" -- only the model's projection is shown, never the generator's
# own simulated outcome, even though the generator can mechanically produce
# one (it has to, to keep team strength evolving consistently year to year).
PRESENT_YEAR = END_SEASON_YEAR
FUTURE_HORIZON_YEARS = 10
MAX_MODELED_YEAR = END_SEASON_YEAR + FUTURE_HORIZON_YEARS

# --------------------------------------------------------------------------
# Era system: the synthetic league doesn't just ramp smoothly across 55
# seasons -- real NBA history didn't either. These are stylized, order-of-
# magnitude design choices (loosely tracking real, well-known trends: the
# 3-point rate explosion, the pace trough of the physical/hand-check 90s,
# the decline and partial recent revival of traditional big-man value) used
# to shape the *synthetic* generator -- not a claim of precise historical
# statistics. src/era.py interpolates between these anchor points.
# --------------------------------------------------------------------------
ERA_BANDS = [
    (1980, 1994, "Big Man Era"),
    (1995, 2004, "Hand-Check Era"),
    (2005, 2014, "Perimeter Freedom Era"),
    (2015, 2024, "Three-Point Revolution"),
    (2025, MAX_MODELED_YEAR, "Modern Positionless Era"),
]

# League-average pace (possessions/48 min): high in the run-and-gun 80s,
# a trough during the grinding, hand-check-legal 90s/early-2000s, rising
# again as rules opened up the perimeter and small-ball took hold.
PACE_ANCHORS = [
    (1980, 105.0), (1985, 102.0), (1990, 99.5), (1995, 95.0), (1999, 90.5),
    (2004, 90.5), (2008, 92.0), (2012, 94.0), (2016, 96.0), (2020, 100.0),
    (2024, 99.0), (2029, 100.5), (MAX_MODELED_YEAR, 101.0),
]

# League-average 3PA rate (share of FGA): near-zero when the line was new,
# a bump from the temporarily-shortened line (1994-97), reverting when the
# line moved back, then the modern acceleration -- plateauing in the 2030s
# in line with the real rate's recent deceleration.
THREE_RATE_ANCHORS = [
    (1980, 0.02), (1985, 0.04), (1990, 0.07), (1994, 0.11), (1997, 0.15),
    (1998, 0.12), (2000, 0.11), (2004, 0.12), (2008, 0.15), (2012, 0.19),
    (2015, 0.24), (2018, 0.32), (2020, 0.34), (2022, 0.37), (2024, 0.39),
    (2028, 0.44), (MAX_MODELED_YEAR, 0.47),
]

# Relative value multiplier for traditional big men (PF/C) vs. guards/wings
# in team strength and MVP scoring: dominant in the post-up 80s, declining
# through the pace-and-space 2010s, with a modest recent uptick (modern
# do-everything centers). Guard/wing value moves roughly inversely.
BIG_MAN_WEIGHT_ANCHORS = [
    (1980, 1.40), (1990, 1.35), (1995, 1.25), (2000, 1.15), (2004, 1.10),
    (2008, 1.02), (2012, 0.95), (2015, 0.88), (2018, 0.82), (2020, 0.80),
    (2022, 0.83), (2024, 0.85), (2028, 0.83), (MAX_MODELED_YEAR, 0.82),
]

# Walk-forward evaluation: a year needs at least this many strictly-earlier
# seasons of training data before it gets a genuine out-of-sample prediction.
MIN_TRAIN_SEASONS = 3

# Future-year confidence decay: baseline (measured historical walk-forward
# accuracy) shrinks by this fraction per year beyond PRESENT_YEAR, floored --
# compounding unknowns (rookies, injuries, trades, aging) make projections
# further out genuinely less reliable, not just cosmetically labeled so.
CONFIDENCE_DECAY_RATE = 0.06
CONFIDENCE_FLOOR = 0.30

TEAMS = [
    "ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GSW",
    "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NOP", "NYK",
    "OKC", "ORL", "PHI", "PHX", "POR", "SAC", "SAS", "TOR", "UTA", "WAS",
]

EASTERN_CONFERENCE = {
    "ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DET", "IND", "MIA", "MIL",
    "NYK", "ORL", "PHI", "TOR", "WAS",
}
WESTERN_CONFERENCE = set(TEAMS) - EASTERN_CONFERENCE

PLAYERS_PER_TEAM = 15
ROTATION_SIZE = 9

# rounds_won = number of playoff series actually won (0-4). Missed-playoffs
# teams and round-1 losers share value 0 (both won zero series); use the
# separate `made_playoffs` column to tell them apart.
ROUND_LABELS = {
    0: "Missed Playoffs / Lost Round 1",
    1: "Lost Conf Semis",
    2: "Lost Conf Finals",
    3: "Lost Finals",
    4: "Champion",
}
N_ROUND_CLASSES = len(ROUND_LABELS)

# Ratings fed into src.simulate are real NBA net-rating point differentials
# (points per 100 possessions), not abstract Elo units. NBA home-court worth
# is empirically ~2-3 points; a logistic scale of ~19 (base-10 form) makes a
# 10-point-better team win ~78% of a neutral-court single game, in line with
# real single-game NBA win probabilities implied by that size of net-rating gap.
HOME_COURT_ADVANTAGE_PTS = 3.0
WIN_PROB_SCALE = 19.0

N_MONTE_CARLO_SIMS = 10_000
