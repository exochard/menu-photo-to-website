"""Read the text of a flattened page with OpenCV 5 only.

PP-OCRv3 DB (cv.dnn.TextDetectionModel_DB) finds text phrases, and the PP-OCRv5 Latin
recogniser (cv.dnn) reads each phrase whole. Its alphabet covers Italian accents and "€",
so "ragù", "lunedì" and "12,50 €" come back as printed.
"""
import re
from dataclasses import dataclass, field
from pathlib import Path

import cv2 as cv
import numpy as np

from menuvision.parse import PRICE

MODELS = Path(__file__).resolve().parent.parent / "models"
DETECTOR = MODELS / "text_detection_en_ppocrv3_2023may.onnx"
RECOGNIZER = MODELS / "latin_PP-OCRv5_mobile_rec.onnx"
# The recogniser's alphabet, copied from its PaddleOCR config (Apache-2.0). Index 0 of its
# output is the CTC blank, 1..836 are these characters and the last class is a space.
CHARSET = (Path(__file__).with_name("latin_dict.txt").read_text(encoding="utf-8").split("\n")[:-1]
           + [" "])
REC_HEIGHT, REC_MAX_WIDTH = 48, 3200


@dataclass
class Word:
    x0: int
    x1: int
    text: str
    confidence: float


@dataclass
class Line:
    y: float
    words: list[Word] = field(default_factory=list)
    y0: int = 0  # the row's vertical extent on the page it was read from
    y1: int = 0

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)

    @property
    def confidence(self) -> float:
        return min((w.confidence for w in self.words), default=0.0)

    @property
    def x0(self) -> int:
        return self.words[0].x0

    @property
    def x1(self) -> int:
        return self.words[-1].x1


def ctc_decode(logprobs: np.ndarray) -> tuple[str, float]:
    """Greedy CTC decode of a (T, 1+len(CHARSET)) log-probability matrix; blank is index 0.

    The confidence is the geometric mean of the per-step best probabilities, in 0..1.
    Keeping the blank as an index, not as a character, keeps real hyphens
    ("12:30-15:00").
    """
    best = logprobs.argmax(1)
    text, prev = [], 0
    for idx in best:
        if idx != 0 and idx != prev:
            text.append(CHARSET[idx - 1])
        prev = idx
    return "".join(text).strip(), float(np.exp(logprobs.max(1).mean()))


class Reader:
    def __init__(self, detector: Path = DETECTOR, recognizer: Path = RECOGNIZER, size=(1152, 1600)):
        self.size = size
        self.det = cv.dnn.TextDetectionModel_DB(str(detector))
        self.det.setBinaryThreshold(0.3).setPolygonThreshold(0.5).setMaxCandidates(400).setUnclipRatio(2.0)
        self.det.setInputParams(1.0 / 255.0, size, (122.68, 116.67, 104.01), True)
        self.rec = cv.dnn.readNet(str(recognizer))

    def recognize(self, page: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> tuple[str, float]:
        crop = page[y0:y1, x0:x1]
        width = int(np.clip(np.ceil(REC_HEIGHT * crop.shape[1] / crop.shape[0]), 16, REC_MAX_WIDTH))
        # PaddleOCR's preprocessing: height 48, aspect kept, pixels scaled to -1..1.
        self.rec.setInput(cv.dnn.blobFromImage(crop, 1 / 127.5, (width, REC_HEIGHT), (127.5,) * 3))
        probs = self.rec.forward()[0]  # (T, classes), softmax
        return ctc_decode(np.log(np.maximum(probs, 1e-12)))

    def detect(self, page: np.ndarray) -> list[np.ndarray]:
        """Text boxes as 4x2 corner arrays in page pixels."""
        H, W = page.shape[:2]
        boxes, _ = self.det.detect(cv.resize(page, self.size))
        scale = np.float32([W / self.size[0], H / self.size[1]])
        return [np.asarray(box, np.float32) * scale for box in boxes]

    def level(self, page: np.ndarray, max_skew: float = 0.01) -> np.ndarray:
        """The page rotated so its text lines run level; unchanged when already level.

        A page flattened from a steep or blurred photo can keep a slope; left uncorrected, a
        price at the right margin lines up with the dish one row above.
        """
        slope = text_slope(self.detect(page))
        if abs(slope) <= max_skew:
            return page
        H, W = page.shape[:2]
        turn = cv.getRotationMatrix2D((W / 2, H / 2), float(np.degrees(np.arctan(slope))), 1.0)
        return cv.warpAffine(page, turn, (W, H), borderMode=cv.BORDER_REPLICATE)

    def read(self, page: np.ndarray, pad: int = 3) -> list[Line]:
        """Rows of text on a page that `level` has already levelled."""
        boxes = self.detect(page)
        H, W = page.shape[:2]
        gray = cv.cvtColor(page, cv.COLOR_BGR2GRAY) if page.ndim == 3 else page
        spans = []
        for pts in boxes:
            x0, y0 = np.maximum(np.floor(pts.min(0)).astype(int) - pad, 0)
            x1, y1 = np.minimum(np.ceil(pts.max(0)).astype(int) + pad, [W, H])
            if x1 - x0 >= 4 and y1 - y0 >= 4:
                spans.append(extend_along_ink(gray, int(x0), int(y0), int(x1), int(y1)))
        words: list[tuple[int, int, Word]] = []
        for x0, y0, x1, y1 in merge_overlapping(spans):
            text, conf = self.recognize(page, x0, y0, x1, y1)
            if text:
                words.append((y0, y1, Word(x0, x1, text, conf)))
        return group_lines(words)


def extend_along_ink(gray: np.ndarray, x0: int, y0: int, x1: int, y1: int,
                     gap: float = 1.0) -> tuple[int, int, int, int]:
    """Grow a box sideways while ink continues within `gap` line heights.

    Under blur the detector's box can stop mid-phrase ("Involtini alla" of "Involtini alla
    messinese"), and the recogniser reads the fragment with full confidence. Word gaps are
    about a third of a line height; the space between a dish and its price is many.
    """
    band = gray[y0:y1].astype(np.int16)
    # Ink is what is darker than the paper around it: a closing estimates the paper, so a
    # shadow or a glare spot shifts both and cancels out.
    h = y1 - y0
    paper = cv.morphologyEx(gray[y0:y1], cv.MORPH_CLOSE, np.ones((h, h), np.uint8)).astype(np.int16)
    depth = (paper - band).max(0)
    ink = depth > 0.35 * max(1, int(depth[x0:x1].max()))
    # The flattened page's outer strip can hold the table's edge; it is never text.
    margin = int(0.02 * len(ink))
    ink[:margin] = ink[len(ink) - margin:] = False
    limit = max(4, int(gap * h))
    right = x1
    x = x1
    while x < len(ink) and x - right <= limit:
        right = x + 1 if ink[x] else right
        x += 1
    left = x0
    x = x0 - 1
    while x >= 0 and left - x <= limit:
        left = x if ink[x] else left
        x -= 1
    return left, y0, right, y1


def merge_overlapping(spans: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    """Union boxes that overlap horizontally and share half of the shorter height."""
    merged: list[list[int]] = []
    for x0, y0, x1, y1 in sorted(spans):
        for m in merged:
            share = min(y1, m[3]) - max(y0, m[1])
            if x0 <= m[2] and x1 >= m[0] and share >= 0.5 * min(y1 - y0, m[3] - m[1]):
                m[:] = [min(x0, m[0]), min(y0, m[1]), max(x1, m[2]), max(y1, m[3])]
                break
        else:
            merged.append([x0, y0, x1, y1])
    return [tuple(m) for m in merged]


def text_slope(boxes: list[np.ndarray]) -> float:
    """Median dy/dx of the long side of text boxes at least 2.5 times wider than tall."""
    slopes = []
    for pts in boxes:
        (cx, cy), (w, h), angle = cv.minAreaRect(pts)
        corners = cv.boxPoints(((cx, cy), (w, h), angle))
        edge = max((corners[1] - corners[0], corners[2] - corners[1]), key=lambda v: np.hypot(*v))
        if max(w, h) >= 2.5 * min(w, h) and abs(edge[0]) > 1e-6:
            slopes.append(edge[1] / edge[0])
    return float(np.median(slopes)) if slopes else 0.0


# A price, with its currency sign and a stray sign the blur can leave after it. Not a time
# ("12.30-15.00") and not part of a longer number.
PRICE_AT = re.compile(r"(?<![\d.,:\-–])(?:[€$]\s*)?\d{1,3}[.,]\d{2}(?:\s*[€$])?(?:\s+[6CEce](?=\s|$))?"
                      r"(?![\d]|[.,]\d|\s*[-–:]\s*\d)")
# The start of a cell: a word of three letters after a bullet or a bracket at most, so a
# trailing mark such as "(v)" or a second price stays with its item.
OPENS_CELL = re.compile(r"\s*[^\w\s]{0,2}([^\W\d_])[^\W\d_]{2,}")


def opens_cell(rest: str, priced: bool) -> bool:
    """Whether the text after a price starts another column's cell and not a note on the dish.

    With a price of its own it does; without one only a capital marks a dish name
    ("Pineapple & Mozzarella"), where "con rucola" or "(vegan)" trails the price.
    """
    match = OPENS_CELL.match(rest)
    return match is not None and (priced or match.group(1).isupper())


def is_lone_price(text: str) -> bool:
    match = PRICE.match(text.strip())
    return text.strip() in ("€", "$") or bool(match and not match.group("name"))


def ends_with_price(text: str) -> bool:
    last = None
    for last in PRICE_AT.finditer(text):
        pass
    return last is not None and last.end() == len(text.rstrip())


def split_at_prices(y0: int, y1: int, word: Word) -> list[tuple[int, int, Word]]:
    """Cut a recognised box where a price is followed by the start of another cell.

    A box can span two columns ("€ 0,80 Fanta lt.33 € 1,00") when the gap between them is
    as small as the one between a dish and its price. The left part of the box keeps its
    share of the width, by characters.
    """
    text, cuts = word.text, []
    for m in PRICE_AT.finditer(text):
        rest = text[m.end():]
        if opens_cell(rest, PRICE_AT.search(rest) is not None):
            cuts.append(m.end())
    if not cuts:
        return [(y0, y1, word)]
    pieces, start = [], 0
    for end in [*cuts, len(text)]:
        part = text[start:end].strip()
        pieces.append((part, word.x0 + round((word.x1 - word.x0) * start / len(text)),
                       word.x0 + round((word.x1 - word.x0) * end / len(text))))
        start = end
    return [(y0, y1, Word(a, b, part, word.confidence)) for part, a, b in pieces if part]


def split_row(members: list[tuple[int, int, Word]], gutter: float) -> list[Line]:
    """One row of words as the cells it holds, one Line each, left to right.

    A new cell starts where the row jumps a `gutter` of text heights, and after a price
    when the next column's cell follows it (a dish and its price, then the next dish). A lone
    price stays with the cell to its left however far away: a price at the right margin
    belongs to its dish. A row that starts with a price is not cut after it: its price
    belongs to the text that follows.
    """
    parts = [piece for m in members for piece in split_at_prices(*m)]
    parts.sort(key=lambda t: t[2].x0)
    cells: list[list[tuple[int, int, Word]]] = []
    for i, part in enumerate(parts):
        y0, y1, word = part
        if cells and not is_lone_price(word.text) and not all(is_lone_price(t[2].text) for t in cells[-1]):
            py0, py1, prev = cells[-1][-1]
            after_price = (ends_with_price(prev.text)
                           and opens_cell(word.text, any(PRICE_AT.search(p[2].text) for p in parts[i:])))
            wide = word.x0 - prev.x1 >= gutter * min(y1 - y0, py1 - py0)
            if after_price or wide:
                cells.append([part])
                continue
        if cells:
            cells[-1].append(part)
        else:
            cells.append([part])
    lines = []
    for cell in cells:
        top = min(cell, key=lambda t: t[0] + t[1])
        lines.append(Line((top[0] + top[1]) / 2, [t[2] for t in cell],
                          min(t[0] for t in cell), max(t[1] for t in cell)))
    return lines


def group_lines(words: list[tuple[int, int, Word]], overlap: float = 0.5,
                gutter: float = 2.0) -> list[Line]:
    """Rows of words, joined when their boxes share at least `overlap` of the shorter height,
    then cut into the cells of a multi-column row (`split_row`).

    Overlap, not a fixed pixel tolerance: on a page flattened from a tilted photo the price
    at the right margin can sit a third of a line lower than its dish name.
    """
    rows: list[tuple[int, int, list[tuple[int, int, Word]]]] = []
    for y0, y1, word in sorted(words, key=lambda t: t[0] + t[1]):
        if rows:
            r0, r1, members = rows[-1]
            if min(y1, r1) - max(y0, r0) >= overlap * min(y1 - y0, r1 - r0):
                members.append((y0, y1, word))
                continue
        rows.append((y0, y1, [(y0, y1, word)]))
    return [line for _, _, members in rows for line in split_row(members, gutter)]
