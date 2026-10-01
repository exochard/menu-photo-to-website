"""Turn OCR lines into a structured menu: sections, items with prices, opening hours."""
import re
from dataclasses import dataclass, field

# A price at the end of a line, with a euro or dollar sign before or after it. Under heavy
# blur the sign can still come back as a stray "6", "C", "E" or "e" after the price.
PRICE = re.compile(r"^(?P<name>.*?)(?:(?:\s*[.·…_]){3,})?\s*(?:[€$]\s*)?(?P<price>\d{1,3}[.,]\d{2})(?:\s*(?:[€$]|[6CEce]))?$")
TIME_RANGE = re.compile(r"(\d{1,2})[:.](\d{2})\s*[-–]?\s*(\d{1,2})[:.](\d{2})")
CLOSED = re.compile(r"\bchius[oa]\b", re.I)


@dataclass
class Item:
    name: str
    price: str  # "9,00", Italian format, as printed


@dataclass
class Section:
    name: str
    items: list[Item] = field(default_factory=list)


@dataclass
class Menu:
    title: str = ""
    sections: list[Section] = field(default_factory=list)
    hours: list[str] = field(default_factory=list)  # "12:30-15:00"
    notes: list[str] = field(default_factory=list)  # other lines, e.g. "Chiuso il lunedi"


def parse(lines: list[str]) -> Menu:
    menu = Menu()
    # The last line kept as a section heading or a note, with the list that holds it. A line
    # that is only a price right after it is the price of that dish, printed below its name.
    above: tuple[str, list] | None = None
    for raw in lines:
        text = re.sub(r"\s+", " ", raw).strip()
        if not text:
            continue
        unpriced, above = above, None
        m = PRICE.match(text)
        if m:
            name = m.group("name")
            if not name:
                if unpriced is None:
                    menu.notes.append(text)  # a price with no name: leave it for review
                    continue
                name = unpriced[0]
                unpriced[1].pop()
            if not menu.sections:
                menu.sections.append(Section(""))
            menu.sections[-1].items.append(Item(name, m.group("price").replace(".", ",")))
            continue
        ranges = TIME_RANGE.findall(text)
        if ranges:
            menu.hours += [f"{int(a):02d}:{b}-{int(c):02d}:{d}" for a, b, c, d in ranges]
            continue
        if CLOSED.search(text):
            menu.notes.append(text)
        elif not menu.title:
            menu.title = text
        elif len(text.split()) <= 3:
            menu.sections.append(Section(text))
            above = (text, menu.sections)
        else:
            menu.notes.append(text)
            above = (text, menu.notes)
    return menu
