"""Era interpolation: turns config.py's anchor points into smooth,
year-by-year values, and the anchor points into a discrete display label.

Piecewise-linear interpolation is used deliberately over anything fancier:
it guarantees the curve passes exactly through each anchor (so the anchor
choices in config.py are directly legible in the output) and never
overshoots between them, unlike a spline.
"""
from __future__ import annotations

import bisect

from config import BIG_MAN_WEIGHT_ANCHORS, ERA_BANDS, PACE_ANCHORS, THREE_RATE_ANCHORS


def _interp(anchors: list[tuple[int, float]], year: int) -> float:
    years = [a[0] for a in anchors]
    if year <= years[0]:
        return anchors[0][1]
    if year >= years[-1]:
        return anchors[-1][1]
    i = bisect.bisect_right(years, year) - 1
    y0, v0 = anchors[i]
    y1, v1 = anchors[i + 1]
    t = (year - y0) / (y1 - y0)
    return v0 + (v1 - v0) * t


def era_pace(year: int) -> float:
    return _interp(PACE_ANCHORS, year)


def era_three_rate(year: int) -> float:
    return _interp(THREE_RATE_ANCHORS, year)


def era_big_man_weight(year: int) -> float:
    return _interp(BIG_MAN_WEIGHT_ANCHORS, year)


def era_guard_weight(year: int) -> float:
    """Roughly complementary to big-man weight: as traditional bigs lose
    relative value, perimeter players gain it, and vice versa."""
    return 2.0 - era_big_man_weight(year)


def era_name(year: int) -> str:
    for start, end, name in ERA_BANDS:
        if start <= year <= end:
            return name
    return ERA_BANDS[-1][2] if year > ERA_BANDS[-1][1] else ERA_BANDS[0][2]
