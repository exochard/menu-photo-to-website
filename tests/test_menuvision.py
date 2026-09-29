import random

import cv2 as cv
import numpy as np
import pytest

from menuvision.ocr import CHARSET, RECOGNIZER, Reader, ctc_decode, split_words
from menuvision.page import find_page
from menuvision.parse import parse
from menuvision.synth import photograph, random_menu, render, sample


def one_hot(indices: list[int]) -> np.ndarray:
    probs = np.full((len(indices), len(CHARSET) + 1), 0.001, np.float32)
    for t, i in enumerate(indices):
        probs[t, i] = 0.9
    return np.log(probs)  # the CRNN emits log-probabilities


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


def test_split_words_on_rendered_line():
    line = np.full((50, 400), 255, np.uint8)
    cv.putText(line, "Pasta   alla   Norma", (5, 38), cv.FONT_HERSHEY_SIMPLEX, 1.1, 0, 2)
    assert len(split_words(line)) == 3


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
def test_end_to_end_reads_prices_and_hours():
    s = sample(7, tilt=0.06, blur=0.8)
    menu = parse([line.text for line in Reader().read(find_page(s.photo).image)])
    assert menu.hours == s.truth.hours
    want = [i.price for sec in s.truth.sections for i in sec.items]
    got = [i.price for sec in menu.sections for i in sec.items]
    assert sum(p in got for p in want) >= len(want) - 1
