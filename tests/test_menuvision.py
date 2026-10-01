import random
from pathlib import Path

import cv2 as cv
import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from menuvision.ocr import CHARSET, RECOGNIZER, Reader, Word, ctc_decode, group_lines, text_slope
from menuvision.page import find_page
from menuvision.parse import parse
from menuvision.synth import FONTS, photograph, random_menu, render, sample


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


def texts(lines):
    return [line.text for line in lines]


def test_columns_far_apart_are_two_lines_and_a_lone_price_stays_with_its_dish():
    # Row of 40 px words: a left column, a right column 300 px away, and each one's price.
    row = [(100, 140, Word(60, 300, "Cassata", 0.99)), (100, 140, Word(600, 700, "6,00 €", 0.99)),
           (100, 140, Word(1000, 1300, "Granita di mandorla", 0.99)),
           (100, 140, Word(1600, 1700, "4,50 €", 0.99))]
    assert texts(group_lines(row)) == ["Cassata 6,00 €", "Granita di mandorla 4,50 €"]


def test_columns_as_close_as_a_dish_and_its_price_split_after_the_price():
    # A table: name, price, name, price, every gap 28 px between 28 px words.
    row = [(244, 272, Word(70, 213, "Caffè Espresso", 0.9)), (244, 272, Word(241, 326, "€ 0,80", 0.9)),
           (244, 272, Word(355, 520, "Fanta latt.cl.33", 0.9)), (244, 272, Word(548, 638, "€ 1,00", 0.9))]
    assert texts(group_lines(row)) == ["Caffè Espresso € 0,80", "Fanta latt.cl.33 € 1,00"]


def test_a_box_that_spans_two_columns_is_cut_after_the_price():
    row = [(244, 272, Word(70, 213, "Caffè Espresso", 0.9)),
           (244, 272, Word(241, 638, "€ 0,80 Fanta latt.cl.33 € 1,00", 0.9))]
    lines = group_lines(row)
    assert texts(lines) == ["Caffè Espresso € 0,80", "Fanta latt.cl.33 € 1,00"]
    assert lines[1].x0 > lines[0].x1 - 100 and lines[1].x1 == 638  # the pieces keep their share of the width


def test_a_dotted_leader_and_a_price_at_the_right_margin_stay_on_one_line():
    leader = [(100, 140, Word(60, 520, "CHEESE ........", 0.9)), (100, 140, Word(900, 980, "2.25", 0.9))]
    assert texts(group_lines(leader)) == ["CHEESE ........ 2.25"]
    margin = [(100, 140, Word(110, 600, "Pasta con le sarde", 0.9)), (100, 140, Word(1500, 1700, "12,50 €", 0.9))]
    assert texts(group_lines(margin)) == ["Pasta con le sarde 12,50 €"]


def test_a_price_first_row_stays_whole_so_its_price_is_not_taken_by_the_row_above():
    near = [(100, 140, Word(60, 200, "€ 12,00", 0.9)), (100, 140, Word(260, 600, "Pasta al pomodoro", 0.9))]
    far = [(100, 140, Word(60, 200, "€ 12,00", 0.9)), (100, 140, Word(900, 1200, "Pasta al pomodoro", 0.9))]
    boxed = [(100, 140, Word(60, 600, "€ 12,00 Pasta al pomodoro", 0.9))]
    for row in (near, far, boxed):
        assert texts(group_lines(row)) == ["€ 12,00 Pasta al pomodoro"]


def test_text_without_a_price_in_the_next_column_is_cut_at_the_gap_only():
    row = [(760, 803, Word(249, 516, "Meatball Breadstick", 0.9)), (760, 803, Word(616, 702, "$3.99", 0.9)),
           (762, 797, Word(799, 1032, "Pineapple & Mozzarella", 0.9))]
    assert texts(group_lines(row)) == ["Meatball Breadstick $3.99", "Pineapple & Mozzarella"]
    # In one box there is no gap to see: a capital starts a dish, a lower-case word trails the price.
    box = [(760, 803, Word(249, 1032, "Meatball Breadstick $3.99 Pineapple & Mozzarella", 0.9))]
    assert texts(group_lines(box)) == ["Meatball Breadstick $3.99", "Pineapple & Mozzarella"]


def test_a_stray_sign_a_second_price_and_a_time_range_do_not_open_a_column():
    for text in ["Pasta alla Norma 9,00 6", "Spaghetti 9,00 (v)", "Margherita 8,00 9,00 10,00",
                 "Pizza 8,00 pomodoro e mozzarella", "Aperto 12.30-15.00 e 19.30-23.00"]:
        assert texts(group_lines([(100, 140, Word(60, 800, text, 0.9))])) == [text]


def test_parse_joins_a_price_printed_below_its_dish():
    menu = parse(["Trattoria da Nino", "Primi", "Agnello panato alla frutta secca", "25,80 €",
                  "Calamari Saltati", "€ 19,00", "Pasta alla Norma 9,00"])
    assert [(i.name, i.price) for s in menu.sections for i in s.items] == [
        ("Agnello panato alla frutta secca", "25,80"), ("Calamari Saltati", "19,00"), ("Pasta alla Norma", "9,00")]
    assert [s.name for s in menu.sections] == ["Primi"] and menu.notes == []


def test_parse_takes_a_price_below_a_dish_once_and_never_below_the_title_or_hours():
    menu = parse(["Menu", "9,00", "Pasta al forno con besciamella", "8,00", "7,00", "Aperto 12:30-15:00", "6,00"])
    assert [(i.name, i.price) for s in menu.sections for i in s.items] == [
        ("Pasta al forno con besciamella", "8,00")]
    assert menu.notes == ["9,00", "7,00", "6,00"]


def test_parse_drops_dotted_leaders_and_a_dollar_sign_from_the_name():
    menu = parse(["Menu", "CHEESE ........ 2.25", "Spaghetti . . . . 3,00", "Meatball Breadstick $3.99"])
    assert [(i.name, i.price) for s in menu.sections for i in s.items] == [
        ("CHEESE", "2,25"), ("Spaghetti", "3,00"), ("Meatball Breadstick", "3,99")]


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
    reader = Reader()
    menu = parse([line.text for line in reader.read(reader.level(find_page(s.photo).image))])
    assert menu.hours == s.truth.hours
    want = [(i.name, i.price) for sec in s.truth.sections for i in sec.items]
    got = [(i.name, i.price) for sec in menu.sections for i in sec.items]
    assert sum(item in got for item in want) >= len(want) - 1


@pytest.mark.skipif(not RECOGNIZER.exists() or not Path(FONTS[0]).exists(),
                    reason="models not downloaded (./fetch_models.sh) or font missing")
def test_a_two_column_price_list_reads_each_dish_with_its_own_price():
    rows = [("Caffè Espresso", "0,80", "Fanta latt.cl.33", "1,00"), ("Cappuccino", "1,20", "Birra Peroni cl.33", "1,30"),
            ("Latte Macchiato", "1,20", "Birra Heineken cl.66", "2,50"), ("Cioccolata Calda", "1,30", "Spremuta d'arancia", "2,00")]
    font = ImageFont.truetype(FONTS[0], 30)
    page = Image.new("RGB", (1000, 80 + 60 * len(rows)), (250, 250, 250))
    draw = ImageDraw.Draw(page)
    for i, (left, left_price, right, right_price) in enumerate(rows):
        y = 40 + 60 * i
        # The price follows its dish 30 px away, as the next column does.
        for x, name, price in ((60, left, left_price), (520, right, right_price)):
            draw.text((x, y), name, font=font, fill=(20, 20, 20))
            draw.text((x + draw.textlength(name, font=font) + 30, y), f"€ {price}", font=font, fill=(20, 20, 20))
    menu = parse([line.text for line in Reader().read(cv.cvtColor(np.array(page), cv.COLOR_RGB2BGR))])
    got = {(item.name, item.price) for section in menu.sections for item in section.items}
    assert got == {(name, price) for left, lp, right, rp in rows for name, price in ((left, lp), (right, rp))}
