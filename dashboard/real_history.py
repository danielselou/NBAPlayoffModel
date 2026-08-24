"""Real, well-known NBA history for the dashboard's reference panel.

This is NOT project output -- it's public record, included so a real
historical season can be compared side by side against the model's own
(entirely fictional) simulated season, without ever mixing the two: the
model's simulated MVP/champion is always a made-up player on a made-up
team-season, and this module is the only place real, named people appear,
strictly as a factual "here's what actually happened" reference.

Compiled from widely-corroborated public record (award winners and NBA
Finals results are about as well-established as sports trivia gets), keyed
by season-ending year to match this project's `year` convention (e.g. 2016
== the 2015-16 season, when Stephen Curry won a unanimous MVP). Covers
1979-80 through 2024-25 -- the most recent season with a real, confirmed
outcome as of this project's January 2026 knowledge cutoff; 2025-26 isn't
included; see the note by its (absent) dict entries below. Treat as a
convenience reference, not an authoritative source -- verify independently
before citing formally, and note gaps/uncertainty rather than guessing.

Image slot: this ships with no real photos (licensing real player photos
requires rights this project doesn't have). To add your own licensed
images, drop a file at dashboard/assets/real_mvps/<year>.<ext> (jpg/jpeg/png/
webp), e.g. dashboard/assets/real_mvps/2016.jpg for the 2015-16 MVP, and
`real_mvp_image_data_uri` below will pick it up automatically. Absent a
real photo, `dashboard/export_data.py` falls back to the same illustrated-
card generator (`dashboard/portrait.py`) used for the model's own fictional
MVPs, using each player's real jersey number/position/team color -- still
clearly not a photo, just a nicer placeholder than blank.
"""
from __future__ import annotations

import base64
from pathlib import Path

REAL_MVP: dict[int, dict[str, str]] = {
    1980: {"name": "Kareem Abdul-Jabbar", "team": "Los Angeles Lakers", "number": 33, "position": "C"},
    1981: {"name": "Julius Erving", "team": "Philadelphia 76ers", "number": 6, "position": "SF"},
    1982: {"name": "Moses Malone", "team": "Houston Rockets", "number": 24, "position": "C"},
    1983: {"name": "Moses Malone", "team": "Philadelphia 76ers", "number": 2, "position": "C"},
    1984: {"name": "Larry Bird", "team": "Boston Celtics", "number": 33, "position": "SF"},
    1985: {"name": "Larry Bird", "team": "Boston Celtics", "number": 33, "position": "SF"},
    1986: {"name": "Larry Bird", "team": "Boston Celtics", "number": 33, "position": "SF"},
    1987: {"name": "Magic Johnson", "team": "Los Angeles Lakers", "number": 32, "position": "PG"},
    1988: {"name": "Michael Jordan", "team": "Chicago Bulls", "number": 23, "position": "SG"},
    1989: {"name": "Magic Johnson", "team": "Los Angeles Lakers", "number": 32, "position": "PG"},
    1990: {"name": "Magic Johnson", "team": "Los Angeles Lakers", "number": 32, "position": "PG"},
    1991: {"name": "Michael Jordan", "team": "Chicago Bulls", "number": 23, "position": "SG"},
    1992: {"name": "Michael Jordan", "team": "Chicago Bulls", "number": 23, "position": "SG"},
    1993: {"name": "Charles Barkley", "team": "Phoenix Suns", "number": 34, "position": "PF"},
    1994: {"name": "Hakeem Olajuwon", "team": "Houston Rockets", "number": 34, "position": "C"},
    1995: {"name": "David Robinson", "team": "San Antonio Spurs", "number": 50, "position": "C"},
    1996: {"name": "Michael Jordan", "team": "Chicago Bulls", "number": 23, "position": "SG"},
    1997: {"name": "Karl Malone", "team": "Utah Jazz", "number": 32, "position": "PF"},
    1998: {"name": "Michael Jordan", "team": "Chicago Bulls", "number": 23, "position": "SG"},
    1999: {"name": "Karl Malone", "team": "Utah Jazz", "number": 32, "position": "PF"},
    2000: {"name": "Shaquille O'Neal", "team": "Los Angeles Lakers", "number": 34, "position": "C"},
    2001: {"name": "Allen Iverson", "team": "Philadelphia 76ers", "number": 3, "position": "PG"},
    2002: {"name": "Tim Duncan", "team": "San Antonio Spurs", "number": 21, "position": "PF"},
    2003: {"name": "Tim Duncan", "team": "San Antonio Spurs", "number": 21, "position": "PF"},
    2004: {"name": "Kevin Garnett", "team": "Minnesota Timberwolves", "number": 21, "position": "PF"},
    2005: {"name": "Steve Nash", "team": "Phoenix Suns", "number": 13, "position": "PG"},
    2006: {"name": "Steve Nash", "team": "Phoenix Suns", "number": 13, "position": "PG"},
    2007: {"name": "Dirk Nowitzki", "team": "Dallas Mavericks", "number": 41, "position": "PF"},
    2008: {"name": "Kobe Bryant", "team": "Los Angeles Lakers", "number": 24, "position": "SG"},
    2009: {"name": "LeBron James", "team": "Cleveland Cavaliers", "number": 23, "position": "SF"},
    2010: {"name": "LeBron James", "team": "Cleveland Cavaliers", "number": 23, "position": "SF"},
    2011: {"name": "Derrick Rose", "team": "Chicago Bulls", "number": 1, "position": "PG"},
    2012: {"name": "LeBron James", "team": "Miami Heat", "number": 6, "position": "SF"},
    2013: {"name": "LeBron James", "team": "Miami Heat", "number": 6, "position": "SF"},
    2014: {"name": "Kevin Durant", "team": "Oklahoma City Thunder", "number": 35, "position": "SF"},
    2015: {"name": "Stephen Curry", "team": "Golden State Warriors", "number": 30, "position": "PG"},
    2016: {"name": "Stephen Curry", "team": "Golden State Warriors", "number": 30, "position": "PG", "note": "Unanimous"},
    2017: {"name": "Russell Westbrook", "team": "Oklahoma City Thunder", "number": 0, "position": "PG"},
    2018: {"name": "James Harden", "team": "Houston Rockets", "number": 13, "position": "SG"},
    2019: {"name": "Giannis Antetokounmpo", "team": "Milwaukee Bucks", "number": 34, "position": "PF"},
    2020: {"name": "Giannis Antetokounmpo", "team": "Milwaukee Bucks", "number": 34, "position": "PF"},
    2021: {"name": "Nikola Jokic", "team": "Denver Nuggets", "number": 15, "position": "C"},
    2022: {"name": "Nikola Jokic", "team": "Denver Nuggets", "number": 15, "position": "C"},
    2023: {"name": "Joel Embiid", "team": "Philadelphia 76ers", "number": 21, "position": "C"},
    2024: {"name": "Nikola Jokic", "team": "Denver Nuggets", "number": 15, "position": "C"},
    2025: {"name": "Shai Gilgeous-Alexander", "team": "Oklahoma City Thunder", "number": 2, "position": "PG", "note": "Also Finals MVP"},
    # 2026 (2025-26 season) intentionally omitted: this project's knowledge
    # cutoff (January 2026) predates that season's conclusion, and this
    # environment has no live network access to check whether/how it's
    # been decided -- see dashboard/real_mvp_prediction.py for a labeled
    # opinion/prediction instead of a fabricated "fact" here.
}

REAL_CHAMPION: dict[int, str] = {
    1980: "Los Angeles Lakers", 1981: "Boston Celtics", 1982: "Los Angeles Lakers",
    1983: "Philadelphia 76ers", 1984: "Boston Celtics", 1985: "Los Angeles Lakers",
    1986: "Boston Celtics", 1987: "Los Angeles Lakers", 1988: "Los Angeles Lakers",
    1989: "Detroit Pistons", 1990: "Detroit Pistons", 1991: "Chicago Bulls",
    1992: "Chicago Bulls", 1993: "Chicago Bulls", 1994: "Houston Rockets",
    1995: "Houston Rockets", 1996: "Chicago Bulls", 1997: "Chicago Bulls",
    1998: "Chicago Bulls", 1999: "San Antonio Spurs", 2000: "Los Angeles Lakers",
    2001: "Los Angeles Lakers", 2002: "Los Angeles Lakers", 2003: "San Antonio Spurs",
    2004: "Detroit Pistons", 2005: "San Antonio Spurs", 2006: "Miami Heat",
    2007: "San Antonio Spurs", 2008: "Boston Celtics", 2009: "Los Angeles Lakers",
    2010: "Los Angeles Lakers", 2011: "Dallas Mavericks", 2012: "Miami Heat",
    2013: "Miami Heat", 2014: "San Antonio Spurs", 2015: "Golden State Warriors",
    2016: "Cleveland Cavaliers", 2017: "Golden State Warriors", 2018: "Golden State Warriors",
    2019: "Toronto Raptors", 2020: "Los Angeles Lakers", 2021: "Milwaukee Bucks",
    2022: "Golden State Warriors", 2023: "Denver Nuggets", 2024: "Boston Celtics",
    2025: "Oklahoma City Thunder",  # beat the Indiana Pacers in 7 games
}

ASSET_DIR = Path(__file__).parent / "assets" / "real_mvps"
_MIME_BY_EXT = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}


def real_mvp_image_data_uri(year: int) -> str | None:
    """Returns a data: URI for a user-supplied real MVP photo, if one has
    been placed at dashboard/assets/real_mvps/<year>.<ext>; otherwise None."""
    for ext, mime in _MIME_BY_EXT.items():
        path = ASSET_DIR / f"{year}.{ext}"
        if path.exists():
            return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")
    return None
