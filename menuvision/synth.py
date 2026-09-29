"""Synthetic photos of printed Italian menus with their ground truth.

No real business's menu is used for evaluation: every page is rendered here from a
seed, then photographed in software (perspective, lighting gradient, blur, noise).
"""
import random
from dataclasses import dataclass

import cv2 as cv
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .parse import Item, Menu, Section

FONTS = ["/usr/share/fonts/liberation/LiberationSerif-Regular.ttf",
         "/usr/share/fonts/liberation/LiberationSans-Regular.ttf",
         "/usr/share/fonts/noto/NotoSerif-Regular.ttf",
         "/usr/share/fonts/TTF/DejaVuSans.ttf"]
TITLES = ["Trattoria da Nino", "Osteria del Porto", "Ristorante Il Faro", "Pizzeria Etna", "Bar Centrale"]
DISHES = {
    "Antipasti": ["Caponata", "Bruschetta al pomodoro", "Arancine al ragu", "Tagliere di salumi", "Parmigiana"],
    "Primi": ["Pasta alla Norma", "Spaghetti alle vongole", "Maccheroni al ragu di maiale",
              "Pasta con le sarde", "Risotto ai funghi porcini", "Linguine al nero di seppia"],
    "Secondi": ["Involtini alla messinese", "Pesce spada alla griglia", "Salsiccia al finocchietto",
                "Frittura di calamari", "Braciole di maiale"],
    "Dolci": ["Cannolo siciliano", "Cassata", "Granita di mandorla", "Tiramisu della casa"],
}
CLOSING = ["Chiuso il lunedi", "Chiuso la domenica sera", "Chiuso il martedi"]


@dataclass
class Sample:
    photo: np.ndarray
    truth: Menu
    lines: list[str]  # the text lines as printed, top to bottom


def random_menu(rng: random.Random) -> tuple[Menu, list[str]]:
    menu = Menu(title=rng.choice(TITLES))
    lines = [menu.title]
    for name in rng.sample(list(DISHES), rng.randint(2, 3)):
        section = Section(name)
        lines.append(name)
        for dish in rng.sample(DISHES[name], rng.randint(2, 3)):
            price = f"{rng.randint(3, 22)},{rng.choice(['00', '50'])}"
            section.items.append(Item(dish, price))
            lines.append(f"{dish} {price} €")
        menu.sections.append(section)
    lunch = f"12:{rng.choice(['00', '30'])}-15:00"
    dinner = f"19:{rng.choice(['00', '30'])}-23:{rng.choice(['00', '30'])}"
    menu.hours = [lunch, dinner]
    lines.append(f"Aperto {lunch} e {dinner}")
    menu.notes = [rng.choice(CLOSING)]
    lines.append(menu.notes[0])
    return menu, lines


def render(menu: Menu, rng: random.Random) -> np.ndarray:
    W, H = 1000, 1400
    ink = tuple(rng.randint(15, 60) for _ in range(3))
    page = Image.new("RGB", (W, H), tuple(rng.randint(236, 252) for _ in range(3)))
    d = ImageDraw.Draw(page)
    font = rng.choice(FONTS)
    big, head, body = (ImageFont.truetype(font, s) for s in (54, 40, 34))
    y = 70
    d.text((W // 2, y), menu.title, font=big, fill=ink, anchor="mt")
    y += 110
    for section in menu.sections:
        d.text((80, y), section.name, font=head, fill=(130, 40, 20))
        y += 70
        for item in section.items:
            d.text((110, y), item.name, font=body, fill=ink)
            d.text((W - 110, y), f"{item.price} €", font=body, fill=ink, anchor="ra")
            y += 58
        y += 24
    d.text((80, y + 10), f"Aperto {menu.hours[0]} e {menu.hours[1]}", font=body, fill=ink)
    d.text((80, y + 70), menu.notes[0], font=body, fill=ink)
    return cv.cvtColor(np.array(page), cv.COLOR_RGB2BGR)


def photograph(page: np.ndarray, rng: random.Random, tilt: float = 0.08, blur: float = 1.0) -> np.ndarray:
    """Place the page on a table at an angle, with uneven light, blur and sensor noise."""
    H, W = 1500, 2000
    np_rng = np.random.default_rng(rng.randint(0, 2**31))
    table = cv.GaussianBlur(np_rng.integers(50, 120, (H, W, 3), dtype=np.uint8), (0, 0), 6)
    h, w = page.shape[:2]
    cx, cy, ph = W / 2 + rng.uniform(-150, 150), H / 2 + rng.uniform(-60, 60), H * 0.85
    pw = ph * w / h
    corners = np.float32([[cx - pw / 2, cy - ph / 2], [cx + pw / 2, cy - ph / 2],
                          [cx + pw / 2, cy + ph / 2], [cx - pw / 2, cy + ph / 2]])
    corners += np.float32([[rng.uniform(-1, 1) * tilt * pw, rng.uniform(-1, 1) * tilt * ph] for _ in range(4)])
    M = cv.getPerspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]), corners)
    warped = cv.warpPerspective(page, M, (W, H))
    mask = cv.warpPerspective(np.full((h, w), 255, np.uint8), M, (W, H))
    shot = np.where(mask[..., None] > 0, warped, table).astype(np.float32)
    light = np.linspace(rng.uniform(0.7, 0.9), rng.uniform(1.0, 1.1), W, dtype=np.float32)
    shot *= (light if rng.random() < 0.5 else light[::-1])[None, :, None]
    shot += np_rng.normal(0, 4, shot.shape)
    shot = np.clip(shot, 0, 255).astype(np.uint8)
    return cv.GaussianBlur(shot, (0, 0), blur) if blur > 0 else shot


def sample(seed: int, tilt: float = 0.08, blur: float = 1.0) -> Sample:
    rng = random.Random(seed)
    menu, lines = random_menu(rng)
    return Sample(photograph(render(menu, rng), rng, tilt, blur), menu, lines)
