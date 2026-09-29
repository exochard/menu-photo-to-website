"""The agent loop: a menu photo becomes owner-approved changes to the site config.

Each step is a tool call recorded in the trace, and what the camera saw decides the next
one. No page in the photo ends in a retake request. Low-confidence lines trigger a second
read of a sharpened page. A change read with low confidence becomes a question to the
owner, never a proposal, and nothing reaches the site before the owner approves it.

    python -m menuvision.agent photo.jpg --config site.yaml [--plan plan.json]
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import shutil
import subprocess
import sys
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2 as cv
import numpy as np
import yaml

from menuvision.ocr import Line, Reader
from menuvision.page import find_page
from menuvision.parse import PRICE, TIME_RANGE, parse

# Line confidence below this triggers a re-read, and a change read below it is a question.
# On tuning seeds 0-39 of the six conditions it flags 23% of misread item lines and 6% of
# correct ones, with no wrong proposal; 0.90 also gave none but flagged 9% of misreads.
LOW = 0.93
_HERE = Path(__file__).resolve().parents[1]
BUILD = next(p for p in (_HERE / "sitebuilder/build.py",
                         _HERE.parents[1] / "products/static-site/template/build.py") if p.exists())


@dataclass
class Change:
    kind: str  # "price" | "new_item"
    group: str
    name: str
    old: str | None
    new: str
    confidence: float


@dataclass
class Outcome:
    status: str  # "retake" | "proposals" | "no_changes"
    proposals: list[Change] = field(default_factory=list)
    questions: list[str] = field(default_factory=list)
    trace: list[dict[str, Any]] = field(default_factory=list)


def sharpen(image: np.ndarray, amount: float = 1.5) -> np.ndarray:
    """Unsharp mask; recovers strokes that defocus blur merged."""
    return cv.addWeighted(image, 1 + amount, cv.GaussianBlur(image, (0, 0), 2.0), -amount, 0)


def merge_reads(first: list[Line], second: list[Line], low: float = LOW) -> tuple[list[Line], int]:
    """Keep each confident line; for a weak one take the likelier reading of the same row."""
    merged, replaced = [], 0
    for line in first:
        best = line
        if line.confidence < low:
            same_row = [other for other in second if abs(other.y - line.y) < 12]
            best = max([line, *same_row], key=lambda candidate: candidate.confidence)
        replaced += best is not line
        merged.append(best)
    return merged, replaced


def key(name: str) -> str:
    """Compare names as the OCR sees them: no accents, no case, single spaces."""
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", ascii_name).strip().lower()


def price_of(text: str) -> str:
    match = re.search(r"\d{1,3}[.,]\d{2}", text or "")
    return match.group().replace(".", ",") if match else ""


def menu_section(cfg: dict[str, Any]) -> dict[str, Any] | None:
    for page in cfg.get("pages", []):
        for section in page.get("sections", []):
            if section.get("type") == "menu":
                return section
    return None


def diff(section: dict[str, Any], read: list[tuple[str, str, str, float]],
         low: float = LOW) -> tuple[list[Change], list[str]]:
    """Compare what was read, (group, name, price, confidence), with the site's menu."""
    site = {key(item["name"]): (group.get("name", ""), item) for group in section.get("groups", [])
            for item in group.get("items", [])}
    proposals, questions, seen = [], [], set()
    for group, name, price, confidence in read:
        match = difflib.get_close_matches(key(name), list(site), n=1, cutoff=0.8)
        if match:
            seen.add(match[0])
            site_group, item = site[match[0]]
            if price_of(item.get("price", "")) == price:
                continue
            change = Change("price", site_group, item["name"], item.get("price"), price, confidence)
        else:
            change = Change("new_item", group, name, None, price, confidence)
        if confidence < low:
            questions.append(f'I read "{name} {price}" but not clearly. Is that right?')
        else:
            proposals.append(change)
    for name_key, (_, item) in site.items():
        if name_key not in seen:
            # Absence in a photo is weak evidence (a crease, a missed line), so it is only asked.
            questions.append(f'"{item["name"]}" is on the site but I did not find it in the photo. '
                             "Is it still on the menu?")
    return proposals, questions


def hours_question(cfg: dict[str, Any], hours: list[str]) -> str | None:
    """A question when the printed opening times differ from the site's hours section.

    The photo gives times without days, so the owner decides which days change; the agent
    never edits hours itself.
    """
    site = {f"{int(a):02d}:{b}-{int(c):02d}:{d}"
            for page in cfg.get("pages", []) for section in page.get("sections", [])
            if section.get("type") == "hours" for row in section.get("items", [])
            for a, b, c, d in TIME_RANGE.findall(str(row.get("time", "")))}
    if not hours or not site or set(hours) == site:
        return None
    return (f"The photo says open {' and '.join(hours)}, the site says {' and '.join(sorted(site))}. "
            "Which days should change?")


def run(photo: np.ndarray, cfg: dict[str, Any], reader: Reader, low: float = LOW) -> Outcome:
    outcome = Outcome("no_changes")
    trace = outcome.trace
    page = find_page(photo)
    trace.append({"tool": "find_page", "found": page.quad is not None, "coverage": round(page.coverage, 3)})
    if page.quad is None:
        outcome.status = "retake"
        outcome.questions.append("I cannot find the edges of the page. Please take the photo again, "
                                 "with the whole sheet in the frame on a darker surface.")
        return outcome
    lines = reader.read(page.image)
    # Text running into the left or right edge of the flattened page means the outline cut
    # through the menu (a page partly outside the photo); names read there are fragments.
    width = page.image.shape[1]
    cut = sum(line.words[0].x0 <= 1 or line.words[-1].x1 >= width - 1 for line in lines)
    if cut >= 2:
        trace.append({"tool": "read", "lines": len(lines), "cut_at_edge": cut})
        outcome.status = "retake"
        outcome.questions.append("Part of the menu is outside the photo or hidden. Please take it again "
                                 "with the whole sheet in the frame.")
        return outcome
    weak = sum(line.confidence < low for line in lines)
    trace.append({"tool": "read", "lines": len(lines), "low_confidence": weak})
    if weak:
        lines, replaced = merge_reads(lines, reader.read(sharpen(page.image)), low)
        trace.append({"tool": "reread_region", "preprocess": "unsharp", "lines_replaced": replaced,
                      "still_low": sum(line.confidence < low for line in lines)})
    menu = parse([line.text for line in lines])
    confidence = {}
    for line in lines:
        match = PRICE.match(re.sub(r"\s+", " ", line.text).strip())
        if match and match.group("name"):
            confidence[(match.group("name"), match.group("price").replace(".", ","))] = line.confidence
    read = [(section.name, item.name, item.price, confidence.get((item.name, item.price), 0.0))
            for section in menu.sections for item in section.items]
    trace.append({"tool": "parse", "items": len(read), "hours": menu.hours})
    if not read:
        outcome.status = "retake"
        outcome.questions.append("I found the page but could not read any prices. "
                                 "Please take the photo again, closer and in focus.")
        return outcome
    section = menu_section(cfg)
    if section is None:
        outcome.questions.append("The site has no menu section yet. Should I add one?")
        section = {"groups": []}
    outcome.proposals, questions = diff(section, read, low)
    outcome.questions += questions
    if question := hours_question(cfg, menu.hours):
        outcome.questions.append(question)
    trace.append({"tool": "diff_against_site", "proposals": len(outcome.proposals),
                  "questions": len(outcome.questions)})
    outcome.status = "proposals" if outcome.proposals else "no_changes"
    return outcome


def apply_approved(config_path: Path, approved: list[Change]) -> dict[str, Any]:
    """Write the approved changes into the config, keeping a .bak copy of the old one."""
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    section = menu_section(cfg)
    if section is None:
        section = {"type": "menu", "title": "Menu", "groups": []}
        cfg["pages"][0].setdefault("sections", []).append(section)
    groups = section.setdefault("groups", [])
    for change in approved:
        group = next((g for g in groups if g.get("name", "") == change.group), None)
        if group is None:
            group = {"name": change.group, "items": []}
            groups.append(group)
        if change.kind == "price":
            for item in group.get("items", []):
                if item["name"] == change.name:
                    item["price"] = f"{change.new} €"
        else:
            group.setdefault("items", []).append({"name": change.name, "price": f"{change.new} €"})
    shutil.copy2(config_path, config_path.with_suffix(config_path.suffix + ".bak"))
    config_path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return cfg


def describe(change: Change) -> str:
    if change.kind == "price":
        return f'{change.name}: {change.old} -> {change.new} € (confidence {change.confidence:.2f})'
    return f'new item in "{change.group}": {change.name} {change.new} € (confidence {change.confidence:.2f})'


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("photo", type=Path)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--plan", type=Path, help="write the outcome as JSON and stop before any edit")
    args = ap.parse_args()
    photo = cv.imread(str(args.photo))
    if photo is None:
        sys.exit(f"cannot read {args.photo}")
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    outcome = run(photo, cfg, Reader())
    for step in outcome.trace:
        print("trace", json.dumps(step, ensure_ascii=False))
    if args.plan:
        args.plan.write_text(json.dumps(asdict(outcome), indent=1, ensure_ascii=False), encoding="utf-8")
        print("wrote", args.plan)
        return
    for question in outcome.questions:
        print("question:", question)
    approved = [c for c in outcome.proposals if input(f"apply {describe(c)}? [y/N] ").strip().lower() == "y"]
    if not approved:
        print("nothing approved; the site is unchanged")
        return
    apply_approved(args.config, approved)
    subprocess.run([sys.executable, str(BUILD), "--config", str(args.config),
                    "--out", str(args.config.parent / "dist")], check=True)


if __name__ == "__main__":
    main()
