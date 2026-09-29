import numpy as np
import pytest
import yaml

from menuvision.agent import Change, apply_approved, diff, hours_question, merge_reads, run
from menuvision.ocr import RECOGNIZER, Line, Reader, Word
from menuvision.synth import sample


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
    assert outcome.trace[0] == {"tool": "find_page", "found": False, "coverage": 0.0}


@needs_models
def test_a_page_running_out_of_the_photo_asks_for_a_retake():
    # A tuning seed where the steep sheet leaves the frame and the outline cuts the text.
    s = sample(36, tilt=0.16, blur=2.2)
    groups = [{"name": sec.name, "items": [{"name": i.name, "price": f"{i.price} €"} for i in sec.items]}
              for sec in s.truth.sections]
    outcome = run(s.photo, site(groups), Reader())
    assert outcome.status == "retake" and not outcome.proposals
    assert outcome.trace[-1]["cut_at_edge"] >= 2


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
