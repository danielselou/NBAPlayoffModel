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
== the 2015-16 season, when Stephen Curry won a unanimous MVP). Treat as a
convenience reference, not an authoritative source -- verify independently
before citing formally, and note gaps/uncertainty rather than guessing.

Image slot: this ships with no real photos (licensing real player photos
requires rights this project doesn't have). To add your own licensed
images, drop a file at dashboard/assets/real_mvps/<year>.<ext> (jpg/jpeg/png/
webp), e.g. dashboard/assets/real_mvps/2016.jpg for the 2015-16 MVP, and
`real_mvp_image_data_uri` below will pick it up automatically -- otherwise
the panel renders as plain text with no photo.
"""
from __future__ import annotations

import base64
from pathlib import Path

REAL_MVP: dict[int, dict[str, str]] = {
    1980: {"name": "Kareem Abdul-Jabbar", "team": "Los Angeles Lakers"},
    1981: {"name": "Julius Erving", "team": "Philadelphia 76ers"},
    1982: {"name": "Moses Malone", "team": "Houston Rockets"},
    1983: {"name": "Moses Malone", "team": "Philadelphia 76ers"},
    1984: {"name": "Larry Bird", "team": "Boston Celtics"},
    1985: {"name": "Larry Bird", "team": "Boston Celtics"},
    1986: {"name": "Larry Bird", "team": "Boston Celtics"},
    1987: {"name": "Magic Johnson", "team": "Los Angeles Lakers"},
    1988: {"name": "Michael Jordan", "team": "Chicago Bulls"},
    1989: {"name": "Magic Johnson", "team": "Los Angeles Lakers"},
    1990: {"name": "Magic Johnson", "team": "Los Angeles Lakers"},
    1991: {"name": "Michael Jordan", "team": "Chicago Bulls"},
    1992: {"name": "Michael Jordan", "team": "Chicago Bulls"},
    1993: {"name": "Charles Barkley", "team": "Phoenix Suns"},
    1994: {"name": "Hakeem Olajuwon", "team": "Houston Rockets"},
    1995: {"name": "David Robinson", "team": "San Antonio Spurs"},
    1996: {"name": "Michael Jordan", "team": "Chicago Bulls"},
    1997: {"name": "Karl Malone", "team": "Utah Jazz"},
    1998: {"name": "Michael Jordan", "team": "Chicago Bulls"},
    1999: {"name": "Karl Malone", "team": "Utah Jazz"},
    2000: {"name": "Shaquille O'Neal", "team": "Los Angeles Lakers"},
    2001: {"name": "Allen Iverson", "team": "Philadelphia 76ers"},
    2002: {"name": "Tim Duncan", "team": "San Antonio Spurs"},
    2003: {"name": "Tim Duncan", "team": "San Antonio Spurs"},
    2004: {"name": "Kevin Garnett", "team": "Minnesota Timberwolves"},
    2005: {"name": "Steve Nash", "team": "Phoenix Suns"},
    2006: {"name": "Steve Nash", "team": "Phoenix Suns"},
    2007: {"name": "Dirk Nowitzki", "team": "Dallas Mavericks"},
    2008: {"name": "Kobe Bryant", "team": "Los Angeles Lakers"},
    2009: {"name": "LeBron James", "team": "Cleveland Cavaliers"},
    2010: {"name": "LeBron James", "team": "Cleveland Cavaliers"},
    2011: {"name": "Derrick Rose", "team": "Chicago Bulls"},
    2012: {"name": "LeBron James", "team": "Miami Heat"},
    2013: {"name": "LeBron James", "team": "Miami Heat"},
    2014: {"name": "Kevin Durant", "team": "Oklahoma City Thunder"},
    2015: {"name": "Stephen Curry", "team": "Golden State Warriors"},
    2016: {"name": "Stephen Curry", "team": "Golden State Warriors", "note": "Unanimous"},
    2017: {"name": "Russell Westbrook", "team": "Oklahoma City Thunder"},
    2018: {"name": "James Harden", "team": "Houston Rockets"},
    2019: {"name": "Giannis Antetokounmpo", "team": "Milwaukee Bucks"},
    2020: {"name": "Giannis Antetokounmpo", "team": "Milwaukee Bucks"},
    2021: {"name": "Nikola Jokic", "team": "Denver Nuggets"},
    2022: {"name": "Nikola Jokic", "team": "Denver Nuggets"},
    2023: {"name": "Joel Embiid", "team": "Philadelphia 76ers"},
    2024: {"name": "Nikola Jokic", "team": "Denver Nuggets"},
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
