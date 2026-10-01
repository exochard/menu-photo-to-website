import random
from pathlib import Path

import cv2 as cv
import numpy as np
import pytest
import yaml
from PIL import Image, ImageDraw, ImageFont

from menuvision.agent import Change, apply_approved, diff, hours_question, merge_reads, run
from menuvision.ocr import RECOGNIZER, Line, Reader, Word
from menuvision.synth import FONTS, photograph, sample


needs_models = pytest.mark.skipif(not RECOGNIZER.exists(), reason="models not downloaded (./fetch_models.sh)")


def line(y: float, text: str, confidence: float) -> Line:
    return Line(y, [Word(0, 10, text, confidence)])


def site(groups: list[dict]) -> dict:
    return {"site": {"name": "T"}, "pages": [{"slug": "index", "sections": [
        {"type": "menu", "title": "Menu", "groups": groups}]}]}


def test_reread_replaces_only_weak_lines_with_likelier_ones():
    first = [line(10, "Carbonara 12,00", 0.95), line(40, "Amatriclana 11,00", 0.5)]
    second = [line(11, "Carbonara 13,00", 0.99), line(42, "Amatriciana 11,00", 0.9)]
    merged, replaced = merge_reads(first, second)
    assert [l.text for l in merged] == ["Carbonara 12,00", "Amatriciana 11,00"]
    assert replaced == 1


def test_reread_never_takes_the_other_column_of_the_same_row():
    left = Line(10, [Word(60, 300, "Caffe 0,80", 0.5)])
    right = Line(11, [Word(600, 900, "Fanta 1,00", 0.99)])
    assert merge_reads([left], [right])[0] == [left]
    better = Line(11, [Word(60, 300, "Caffè 0,80", 0.97)])
    assert merge_reads([left], [right, better])[0] == [better]


def test_a_new_dish_read_like_a_piece_of_a_description_is_asked_not_proposed():
    section = site([{"name": "Primi", "items": [{"name": "Carbonara", "price": "12,00 €"}]}])["pages"][0]["sections"][0]
    read = [("", "Carbonara", "12,00", 0.99), ("", "Tagliatelle al ragù", "11,00", 0.99),
            ("", "FOUR", "5,99", 0.99), ("", "Onions, Green Peppers, Mozzarella", "14,99", 0.99),
            ("", "served with roasted potatoes", "14,99", 0.99), ("", "(v)", "3,00", 0.99)]
    proposals, questions = diff(section, read)
    assert [c.name for c in proposals] == ["Tagliatelle al ragù"]
    assert sum("Is that a new dish?" in q for q in questions) == 4


def test_diff_proposes_confident_changes_and_asks_about_the_rest():
    section = site([{"name": "Primi", "items": [
        {"name": "Carbonara", "price": "12,00 €"}, {"name": "Cacio e pepe", "price": "10,00 €"},
        {"name": "Caffè", "price": "1,20 €"}]}])["pages"][0]["sections"][0]
    read = [("Primi", "Carbonara", "13,00", 0.95),   # price changed
            ("Primi", "Caffe", "1,20", 0.95),        # same, accent lost by the OCR
            ("Primi", "Gricia", "11,00", 0.4)]       # new but unclear
    proposals, questions = diff(section, read)
    assert proposals == [Change("price", "Primi", "Carbonara", "12,00 €", "13,00", 0.95)]
    assert any("Gricia" in q for q in questions)
    assert any("Cacio e pepe" in q for q in questions)  # missing from the photo: asked, not removed
    assert len(questions) == 2


def test_a_new_dish_that_is_part_of_a_site_dish_is_asked_not_proposed():
    section = site([{"name": "Secondi", "items": [
        {"name": "Involtini alla messinese", "price": "14,00 €"}]}])["pages"][0]["sections"][0]
    proposals, questions = diff(section, [("Secondi", "Involtini alla", "14,00", 0.98)])
    assert proposals == []
    assert questions[0] == 'I read "Involtini alla 14,00". Is that "Involtini alla messinese"?'


def test_apply_writes_only_approved_changes_and_keeps_a_backup(tmp_path):
    config = tmp_path / "site.yaml"
    config.write_text(yaml.safe_dump(site([{"name": "Primi", "items": [
        {"name": "Carbonara", "price": "12,00 €"}]}])), encoding="utf-8")
    apply_approved(config, [Change("price", "Primi", "Carbonara", "12,00 €", "13,00", 0.9),
                            Change("new_item", "Dolci", "Tiramisu", None, "6,00", 0.9)])
    groups = yaml.safe_load(config.read_text())["pages"][0]["sections"][0]["groups"]
    assert groups[0]["items"] == [{"name": "Carbonara", "price": "13,00 €"}]
    assert groups[1] == {"name": "Dolci", "items": [{"name": "Tiramisu", "price": "6,00 €"}]}
    assert "12,00" in (tmp_path / "site.yaml.bak").read_text()


@needs_models
def test_blank_photo_asks_for_a_retake_and_edits_nothing():
    outcome = run(np.full((600, 400, 3), 128, np.uint8), site([]), Reader())
    assert outcome.status == "retake" and not outcome.proposals
    assert outcome.trace == [
        {"tool": "find_page", "outline": "single", "found": False, "coverage": 0.0,
         "next": "try the joined outline"},
        {"tool": "find_page", "outline": "joined", "found": False, "coverage": 0.0,
         "next": "ask for a retake"}]


@needs_models
def test_a_menu_cut_by_the_frame_asks_for_a_retake():
    # A tuning seed cropped through the price column: both outlines cut the text.
    s = sample(3, tilt=0.04, blur=0.6)
    groups = [{"name": sec.name, "items": [{"name": i.name, "price": f"{i.price} €"} for i in sec.items]}
              for sec in s.truth.sections]
    outcome = run(s.photo[:, :1200], site(groups), Reader())
    assert outcome.status == "retake" and not outcome.proposals
    assert "outside the photo" in outcome.questions[0]
    assert outcome.trace[2]["cut_at_edge"] >= 2 and outcome.trace[-1]["next"] == "ask for a retake"


@needs_models
def test_a_sheet_running_out_of_the_frame_is_read_from_its_joined_outline():
    # A tuning seed where the steep sheet leaves the frame but its text does not: the single
    # outline is a part of the sheet that cuts the text, the joined one is the whole sheet.
    s = sample(36, tilt=0.16, blur=2.2)
    groups = [{"name": sec.name, "items": [{"name": i.name, "price": f"{i.price} €"} for i in sec.items]}
              for sec in s.truth.sections]
    target = groups[0]["items"][0]
    target["price"] = "99,00 €"
    outcome = run(s.photo, site(groups), Reader())
    assert [(c.name, c.old) for c in outcome.proposals] == [(target["name"], "99,00 €")]
    assert outcome.trace[2].get("next") == "try the joined outline"
    assert outcome.trace[3]["outline"] == "joined" and outcome.trace[3]["found"]


@needs_models
def test_a_changed_price_on_a_real_photo_becomes_a_proposal():
    s = sample(7, tilt=0.04, blur=0.6)  # a tuning seed; evaluation seeds start at 1000
    groups = [{"name": sec.name, "items": [{"name": i.name, "price": f"{i.price} €"} for i in sec.items]}
              for sec in s.truth.sections]
    target = groups[0]["items"][0]
    target["price"] = "99,00 €"
    outcome = run(s.photo, site(groups), Reader())
    assert [(c.kind, c.name, c.old) for c in outcome.proposals] == [("price", target["name"], "99,00 €")]
    assert [step["tool"] for step in outcome.trace][:3] == ["find_page", "level", "read"]


def test_changed_hours_become_a_question_and_matching_hours_do_not():
    cfg = site([])
    cfg["pages"][0]["sections"].append({"type": "hours", "items": [
        {"day": "Mar-Dom", "time": "12:00 – 15:00, 19:30 – 23:00"}]})
    assert hours_question(cfg, ["12:00-15:00", "19:30-23:00"]) is None
    assert "12:30-15:00" in hours_question(cfg, ["12:30-15:00", "19:30-23:00"])
    assert hours_question(site([]), ["12:00-15:00"]) is None  # no hours section: nothing to compare


@needs_models
@pytest.mark.skipif(not Path(FONTS[0]).exists(), reason="font missing")
def test_a_price_printed_below_its_dish_is_asked_about_never_proposed():
    page = Image.new("RGB", (1000, 1400), (245, 242, 235))
    draw = ImageDraw.Draw(page)
    draw.text((500, 70), "Trattoria da Nino", font=ImageFont.truetype(FONTS[0], 54), fill=(30, 30, 30), anchor="mt")
    body = ImageFont.truetype(FONTS[0], 34)
    for i, (name, price) in enumerate([("Agnello panato alla frutta secca", "25,80"),
                                       ("Bocconcini d'anatra al miele", "22,50"), ("Tagliata di manzo al pepe", "24,00")]):
        draw.text((500, 220 + 190 * i), name, font=body, fill=(30, 30, 30), anchor="mt")
        draw.text((500, 280 + 190 * i), f"€ {price}", font=body, fill=(30, 30, 30), anchor="mt")
    photo = photograph(cv.cvtColor(np.array(page), cv.COLOR_RGB2BGR), random.Random(5), tilt=0.04, blur=0.6)
    cfg = site([{"name": "Secondi", "items": [{"name": "Agnello panato alla frutta secca", "price": "20,00 €"},
                                              {"name": "Bocconcini d'anatra al miele", "price": "22,50 €"}]}])
    outcome = run(photo, cfg, Reader())
    # The price was paired with the line above it by layout alone, so the owner confirms it.
    assert outcome.proposals == []
    assert 'I read "Agnello panato alla frutta secca 25,80" but not clearly. Is that right?' in outcome.questions
    assert outcome.trace[-2]["items"] == 3
