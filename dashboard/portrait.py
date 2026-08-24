"""Generates a small illustrated "player card" portrait as a PNG data URI.

There's no real photo to show -- these are fictional players from the
synthetic league generator -- so rather than fabricate something that could
pass as a real athlete photo, this draws an abstract, clearly-illustrated
silhouette card: a team-colored gradient, a generic bust silhouette, and the
player's jersey number, in the same spirit as a stat-card icon rather than a
portrait photograph.
"""
from __future__ import annotations

import base64
import io
import os
import random

import matplotlib
from PIL import Image, ImageDraw, ImageFont

_FONT_PATH = os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans-Bold.ttf")
_CARD_W, _CARD_H = 240, 300


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _mix(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def _rounded_mask(size: tuple[int, int], radius: int) -> Image.Image:
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, size[0] - 1, size[1] - 1], radius=radius, fill=255)
    return mask


def generate_portrait_data_uri(player_id: int, team_color: str, position: str, jersey_number: int) -> str:
    rng = random.Random(player_id * 2654435761 % (2**32))
    team_rgb = _hex_to_rgb(team_color)
    top_color = _mix(team_rgb, (255, 255, 255), 0.28)
    bottom_color = _mix(team_rgb, (0, 0, 0), 0.55)

    img = Image.new("RGB", (_CARD_W, _CARD_H))
    grad_draw = ImageDraw.Draw(img)
    for y in range(_CARD_H):
        grad_draw.line([(0, y), (_CARD_W, y)], fill=_mix(top_color, bottom_color, y / _CARD_H))

    draw = ImageDraw.Draw(img, "RGBA")

    # Soft radial glow behind the head for a bit of depth.
    head_cx, head_cy = _CARD_W // 2, 108
    for r in range(70, 0, -2):
        alpha = int(18 * (1 - r / 70))
        draw.ellipse([head_cx - r, head_cy - r, head_cx + r, head_cy + r], fill=(255, 255, 255, alpha))

    # Generic bust silhouette: head + shoulders, in a neutral charcoal so it
    # reads as an icon/avatar rather than an attempted likeness.
    silhouette = (24, 27, 34, 235)
    head_r = 38 + rng.randint(-3, 3)
    draw.ellipse([head_cx - head_r, head_cy - head_r, head_cx + head_r, head_cy + head_r], fill=silhouette)
    shoulder_w = 118 + rng.randint(-6, 6)
    draw.polygon([
        (head_cx - 30, head_cy + 30),
        (head_cx + 30, head_cy + 30),
        (head_cx + shoulder_w // 2, _CARD_H + 10),
        (head_cx - shoulder_w // 2, _CARD_H + 10),
    ], fill=silhouette)

    # Jersey number, translucent over the torso/shoulders.
    number_font = ImageFont.truetype(_FONT_PATH, 78)
    number_text = str(jersey_number)
    bbox = draw.textbbox((0, 0), number_text, font=number_font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text((head_cx - tw / 2 - bbox[0], 205 - th / 2 - bbox[1]), number_text,
              font=number_font, fill=(255, 255, 255, 210))

    # Position pill, bottom edge.
    pill_font = ImageFont.truetype(_FONT_PATH, 15)
    pill_bbox = draw.textbbox((0, 0), position, font=pill_font)
    pill_w = (pill_bbox[2] - pill_bbox[0]) + 26
    pill_x0 = head_cx - pill_w / 2
    draw.rounded_rectangle([pill_x0, _CARD_H - 34, pill_x0 + pill_w, _CARD_H - 12], radius=11, fill=(0, 0, 0, 130))
    draw.text((head_cx - (pill_bbox[2] - pill_bbox[0]) / 2 - pill_bbox[0], _CARD_H - 30 - pill_bbox[1]),
              position, font=pill_font, fill=(255, 255, 255, 235))

    mask = _rounded_mask((_CARD_W, _CARD_H), 18)
    rounded = Image.new("RGBA", (_CARD_W, _CARD_H))
    rounded.paste(img, (0, 0), mask)
    border_draw = ImageDraw.Draw(rounded, "RGBA")
    border_draw.rounded_rectangle([0.5, 0.5, _CARD_W - 1.5, _CARD_H - 1.5], radius=18,
                                   outline=(255, 255, 255, 60), width=1)

    buf = io.BytesIO()
    rounded.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def generate_real_player_card(name: str, team_full_name: str, position: str, number: int) -> str:
    """Same illustrated-card treatment, for a real player, when no real
    photo file has been supplied (see dashboard/real_history.py and
    dashboard/real_mvp_prediction.py). Still an abstract silhouette, not an
    attempted likeness -- the jersey number/team color are the only real,
    identifying details, same as an unofficial fan-made stat card.
    """
    from dashboard.team_meta import TEAM_META  # local import: avoids a module-load cycle

    color = "#999999"
    for meta in TEAM_META.values():
        if meta["name"] == team_full_name:
            color = meta["color"]
            break
    player_id = int.from_bytes(name.encode("utf-8"), "little", signed=False) % (2**31)
    return generate_portrait_data_uri(player_id, color, position, number)
