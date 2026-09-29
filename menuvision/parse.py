"""Turn OCR lines into a structured menu: sections, items with prices, opening hours."""
import re
from dataclasses import dataclass, field

# A price at the end of a line, optionally followed by the euro sign. Under heavy blur the
# sign can still come back as a stray "6", "C", "E" or "e" after the price.
PRICE = re.compile(r"^(?P<name>.*?)\s*(?:€\s*)?(?P<price>\d{1,3}[.,]\d{2})(?:\s*(?:€|[6CEce]))?$")
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
    for raw in lines:
        text = re.sub(r"\s+", " ", raw).strip()
        if not text:
            continue
        m = PRICE.match(text)
        if m and not m.group("name"):
            menu.notes.append(text)  # a price with no name: leave it for review
            continue
        if m:
            if not menu.sections:
                menu.sections.append(Section(""))
            menu.sections[-1].items.append(Item(m.group("name"), m.group("price").replace(".", ",")))
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
        else:
            menu.notes.append(text)
    return menu
