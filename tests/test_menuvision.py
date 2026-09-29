import random

import cv2 as cv
import numpy as np
import pytest

from menuvision.ocr import CHARSET, RECOGNIZER, Reader, Word, ctc_decode, group_lines, text_slope
from menuvision.page import find_page
from menuvision.parse import parse
from menuvision.synth import photograph, random_menu, render, sample


def one_hot(indices: list[int]) -> np.ndarray:
    probs = np.full((len(indices), len(CHARSET) + 1), 0.001, np.float32)
    for t, i in enumerate(indices):
        probs[t, i] = 0.9
    return np.log(probs)  # the decoder takes log-probabilities


def test_ctc_keeps_real_hyphens_and_collapses_repeats():
    idx = {c: CHARSET.index(c) + 1 for c in "12:30-5"}
    seq = [idx["1"], idx["1"], 0, idx["2"], idx[":"], idx["3"], 0, idx["0"], idx["-"], idx["-"], 0, idx["5"]]
    text, conf = ctc_decode(one_hot(seq))
    assert text == "12:30-5"
    assert 0.8 < conf <= 1.0


def test_ctc_blank_separates_double_letters():
    l = CHARSET.index("l") + 1
    assert ctc_decode(one_hot([l, 0, l]))[0] == "ll"
    assert ctc_decode(one_hot([l, l]))[0] == "l"


def test_ctc_reads_italian_accents_and_the_euro_sign():
    text = "ragù lunedì 9,50 €"
    assert ctc_decode(one_hot([CHARSET.index(c) + 1 for c in text]))[0] == text


def test_rows_join_a_lower_price_but_not_the_next_dish():
    name, price, below = Word(100, 400, "Cassata", 0.99), Word(800, 900, "6,00 €", 0.99), \
        Word(100, 400, "Cannolo", 0.99)
    # The price box sits 15 px lower than its 40 px dish name; the next dish is 58 px down.
    lines = group_lines([(100, 140, name), (115, 155, price), (158, 198, below)])
    assert [line.text for line in lines] == ["Cassata 6,00 €", "Cannolo"]


def test_text_slope_is_the_median_of_long_boxes():
    def box(cx, cy, slope):
        dx = np.float32([150, 150 * slope])
        dy = np.float32([-20 * slope, 20])
        c = np.float32([cx, cy])
        return np.array([c - dx - dy, c + dx - dy, c + dx + dy, c - dx + dy], np.float32)
    boxes = [box(300, 100 + 60 * i, 0.05) for i in range(5)] + [box(300, 500, 0.0)]
    assert abs(text_slope(boxes) - 0.05) < 0.005


def test_parse_prices_hours_and_sections():
    menu = parse(["Trattoria da Nino", "Primi", "Pasta alla Norma 9,00 6", "Spaghetti alle vongole 13.50 C",
                  "Secondi", "Involtini alla messinese 12,00 €", "Aperto 12:30-15:00 e 19:30-23:00",
                  "Chiuso il lunedi"])
    assert menu.title == "Trattoria da Nino"
    assert [s.name for s in menu.sections] == ["Primi", "Secondi"]
    assert [(i.name, i.price) for i in menu.sections[0].items] == [
        ("Pasta alla Norma", "9,00"), ("Spaghetti alle vongole", "13,50")]
    assert menu.hours == ["12:30-15:00", "19:30-23:00"]
    assert menu.notes == ["Chiuso il lunedi"]


def test_parse_does_not_take_a_bare_price_as_an_item():
    assert parse(["Menu", "9,00 €"]).sections == []


def test_find_page_flattens_a_tilted_page():
    rng = random.Random(3)
    menu, _ = random_menu(rng)
    page = find_page(photograph(render(menu, rng), rng, tilt=0.1, blur=0.8))
    assert page.quad is not None
    h, w = page.image.shape[:2]
    # The page is 1000x1400; random corner jitter is not a true camera, so the flattened
    # aspect only approximates 1.4.
    assert 1.2 < h / w < 1.7 and page.coverage > 0.3


def test_find_page_reports_a_missing_page():
    blank = np.full((600, 800, 3), 90, np.uint8)
    page = find_page(blank)
    assert page.quad is None and page.coverage == 0.0


@pytest.mark.skipif(not RECOGNIZER.exists(), reason="models not downloaded (./fetch_models.sh)")
def test_end_to_end_reads_names_with_accents_prices_and_hours():
    s = sample(7, tilt=0.06, blur=0.8)
    menu = parse([line.text for line in Reader().read(find_page(s.photo).image)])
    assert menu.hours == s.truth.hours
    want = [(i.name, i.price) for sec in s.truth.sections for i in sec.items]
    got = [(i.name, i.price) for sec in menu.sections for i in sec.items]
    assert sum(item in got for item in want) >= len(want) - 1
