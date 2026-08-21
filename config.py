"""Global configuration for the NBA playoff prediction pipeline."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
OUTPUT_DIR = ROOT_DIR / "output"

for _d in (DATA_RAW_DIR, DATA_PROCESSED_DIR, OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42

# Seasons to model, e.g. "1999-00" through "2023-24"
START_SEASON_YEAR = 2000
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
