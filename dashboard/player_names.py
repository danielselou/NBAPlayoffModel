"""Deterministic synthetic player names.

The generator only tracks players by integer ID; for the MVP highlight to
read like a real card, each ID needs a stable display name. Names are drawn
from generic first/last name pools, seeded by player_id so the same player
always gets the same name across rebuilds -- and deliberately generic/common
rather than drawn from any real-player list, since these are fictional
players in a synthetic league, not real athletes.
"""
import random

FIRST_NAMES = [
    "Marcus", "Jalen", "Xavier", "Andre", "Malik", "Isaiah", "Trevon", "Elijah",
    "Cameron", "Devin", "Tyrell", "Jordan", "Darius", "Kaden", "Bryce", "Amir",
    "Jaylen", "Christian", "Dominic", "Miles", "Reggie", "Terrence", "Julian",
    "Aaron", "Corey", "DeShawn", "Emmanuel", "Nathaniel", "Quentin", "Sekou",
    "Tobias", "Wesley", "Zachary", "Antoine", "Brandon", "Cedric", "Donovan",
    "Fabian", "Grayson", "Harold",
]
LAST_NAMES = [
    "Whitfield", "Brantley", "Okafor", "Sinclair", "Mercer", "Holloway",
    "Ashford", "Delgado", "Reyes", "Winslow", "Castellano", "Marbury",
    "Dunlap", "Ferris", "Grayling", "Hobbes", "Ibarra", "Kellerman", "Lockhart",
    "Mabry", "Nakamura", "Osei", "Prescott", "Quintero", "Rutledge", "Sandoval",
    "Thackeray", "Underwood", "Vantassel", "Wexler", "Yardley", "Zellner",
    "Abernathy", "Boatwright", "Corrigan", "Duquesne", "Ekwueme", "Fontenot",
]


def player_name(player_id: int) -> str:
    rng = random.Random(player_id * 7919 + 13)
    return f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"


def jersey_number(player_id: int) -> int:
    rng = random.Random(player_id * 104729 + 7)
    return rng.randint(0, 55)
