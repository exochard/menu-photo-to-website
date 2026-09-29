"""Write the demo pair: a synthetic menu photo and a site config that is out of date.

The config lists the printed menu with two old prices and a dish that is no longer
printed, and its lunch hours differ from the photo, so one run shows every path:
proposals, a missing-item question and an hours question.

    .venv/bin/python demo/make_demo.py
"""
import random
from pathlib import Path

import cv2 as cv
import yaml

from menuvision.parse import Item, Menu, Section
from menuvision.synth import photograph, render

HERE = Path(__file__).resolve().parent
SEED = 7  # drives the camera only; evaluation seeds start at 1000

# Hand-written rather than sampled: the evaluation's random prices (3-22 euro in any
# section) read as implausible to anyone who knows a Sicilian menu.
PRINTED = [("Dolci", [("Cannolo siciliano", "3,50"), ("Granita di mandorla", "4,00")]),
           ("Secondi", [("Involtini alla messinese", "14,00"), ("Braciole di maiale", "12,50")])]
LUNCH, DINNER = "12:00-15:00", "19:30-23:00"


def printed_menu() -> tuple[Menu, list[str]]:
    menu = Menu(title="Ristorante Il Faro")
    lines = [menu.title]
    for name, dishes in PRINTED:
        section = Section(name)
        lines.append(name)
        for dish, price in dishes:
            section.items.append(Item(dish, price))
            lines.append(f"{dish} {price} €")
        menu.sections.append(section)
    menu.hours = [LUNCH, DINNER]
    lines.append(f"Aperto {LUNCH} e {DINNER}")
    menu.notes = ["Chiuso il lunedi"]
    lines.append(menu.notes[0])
    return menu, lines


def main() -> None:
    rng = random.Random(SEED)
    menu, _ = printed_menu()
    photo = photograph(render(menu, rng), rng, 0.06, 0.8)
    cv.imwrite(str(HERE / "menu.jpg"), photo, [cv.IMWRITE_JPEG_QUALITY, 88])
    groups = [{"name": sec.name, "items": [{"name": i.name, "price": f"{i.price} €"} for i in sec.items]}
              for sec in menu.sections]
    groups[0]["items"][0]["price"] = "3,00 €"
    groups[-1]["items"][-1]["price"] = "11,00 €"
    groups[-1]["items"].append({"name": "Piatto del giorno", "price": "13,00 €"})
    lunch, dinner = menu.hours
    cfg = {
        "site": {"name": menu.title, "tagline": "Cucina di casa", "lang": "it", "year": "2026",
                 "noindex": True, "skip_label": "Vai al contenuto",
                 "theme_toggle_label": "Cambia tema chiaro / scuro"},
        "nav": [{"label": "Menu", "page": "index"}],
        "pages": [{"slug": "index", "description": "Il menu di oggi.", "sections": [
            {"type": "menu", "title": "Il menu", "groups": groups},
            {"type": "hours", "title": "Orari", "items": [
                {"day": "Tutti i giorni", "time": f"12:00-14:30, {dinner}"}]},
        ]}],
    }
    (HERE / "site.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print("wrote demo/menu.jpg and demo/site.yaml; printed lunch", lunch)


if __name__ == "__main__":
    main()
