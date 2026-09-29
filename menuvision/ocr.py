"""Read the text of a flattened page with OpenCV 5 only.

PP-OCRv3 DB (cv.dnn.TextDetectionModel_DB) finds text phrases; each phrase is split
into words at ink gaps; a CRNN (cv.dnn) reads each word. The CRNN reads 100x32 crops,
so reading whole phrases squashes them; word crops keep the characters legible.
"""
import string
from dataclasses import dataclass, field
from pathlib import Path

import cv2 as cv
import numpy as np

MODELS = Path(__file__).resolve().parent.parent / "models"
DETECTOR = MODELS / "text_detection_en_ppocrv3_2023may.onnx"
RECOGNIZER = MODELS / "text_recognition_CRNN_CH_2023feb_fp16.onnx"
# Index 0 of the CRNN output is the CTC blank; the rest map onto this charset.
CHARSET = string.digits + string.ascii_lowercase + string.ascii_uppercase + string.punctuation.replace("\\", "")
CRNN_SIZE = (100, 32)


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

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)

    @property
    def confidence(self) -> float:
        return min((w.confidence for w in self.words), default=0.0)


def ctc_decode(logprobs: np.ndarray) -> tuple[str, float]:
    """Greedy CTC decode of a (T, 1+len(CHARSET)) log-probability matrix; blank is index 0.

    The confidence is the geometric mean of the per-step best probabilities, in 0..1.

    The OpenCV Zoo wrapper marks blanks with "-", which is also a charset character,
    so it drops every real hyphen ("12:30-15:00" becomes "12:3015:00"). Keeping the
    blank as an index avoids that.
    """
    best = logprobs.argmax(1)
    text, prev = [], 0
    for idx in best:
        if idx != 0 and idx != prev:
            text.append(CHARSET[idx - 1])
        prev = idx
    return "".join(text), float(np.exp(logprobs.max(1).mean()))


def split_words(crop: np.ndarray, gap_ratio: float = 0.28) -> list[tuple[int, int]]:
    """Column spans of ink separated by gaps wider than gap_ratio x crop height."""
    gray = cv.cvtColor(crop, cv.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    ink = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1].any(0)
    gap_min = max(3, int(gap_ratio * crop.shape[0]))
    spans, start, end, run = [], None, 0, 0
    for x, on in enumerate(list(ink) + [False] * (gap_min + 1)):
        if on:
            start = x if start is None else start
            end, run = x, 0
        elif start is not None:
            run += 1
            if run >= gap_min:
                spans.append((start, end + 1))
                start = None
    return [(a, b) for a, b in spans if b - a > 2]


class Reader:
    def __init__(self, detector: Path = DETECTOR, recognizer: Path = RECOGNIZER, size=(1152, 1600),
                 min_aspect: float = 1.5):
        self.size = size
        self.min_aspect = min_aspect
        self.det = cv.dnn.TextDetectionModel_DB(str(detector))
        self.det.setBinaryThreshold(0.3).setPolygonThreshold(0.5).setMaxCandidates(400).setUnclipRatio(2.0)
        self.det.setInputParams(1.0 / 255.0, size, (122.68, 116.67, 104.01), True)
        self.rec = cv.dnn.readNet(str(recognizer))

    def recognize(self, page: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> tuple[str, float]:
        crop = page[y0:y1, x0:x1]
        # Stretching a short word to 100x32 makes the CRNN repeat letters ("al" -> "all"),
        # so crops narrower than min_aspect are padded with the paper colour.
        missing = int(crop.shape[0] * self.min_aspect) - crop.shape[1]
        if missing > 0:
            paper = np.median(np.concatenate([crop[:, 0], crop[:, -1]]), axis=0)
            crop = cv.copyMakeBorder(crop, 0, 0, missing // 2, missing - missing // 2,
                                     cv.BORDER_CONSTANT, value=paper.tolist())
        self.rec.setInput(cv.dnn.blobFromImage(crop, size=CRNN_SIZE, mean=127.5, scalefactor=1 / 127.5))
        out = self.rec.forward()  # (T, 1, classes), log-softmax already applied
        return ctc_decode(out[:, 0, :])

    def read(self, page: np.ndarray, pad: int = 4) -> list[Line]:
        H, W = page.shape[:2]
        boxes, _ = self.det.detect(cv.resize(page, self.size))
        scale = np.float32([W / self.size[0], H / self.size[1]])
        words: list[tuple[float, Word]] = []
        for box in boxes:
            pts = np.asarray(box, np.float32) * scale
            x0, y0 = np.maximum(np.floor(pts.min(0)).astype(int), 0)
            x1, y1 = np.minimum(np.ceil(pts.max(0)).astype(int), [W, H])
            if x1 - x0 < 4 or y1 - y0 < 4:
                continue
            for a, b in split_words(page[y0:y1, x0:x1]):
                wx0, wx1 = max(0, x0 + a - pad), min(W, x0 + b + pad)
                text, conf = self.recognize(page, wx0, y0, wx1, y1)
                if text:
                    words.append(((y0 + y1) / 2, Word(wx0, wx1, text, conf)))
        return group_lines(words)


def group_lines(words: list[tuple[float, Word]], tolerance: float = 18) -> list[Line]:
    lines: list[Line] = []
    for y, word in sorted(words, key=lambda t: t[0]):
        if lines and abs(lines[-1].y - y) < tolerance:
            lines[-1].words.append(word)
        else:
            lines.append(Line(y, [word]))
    for line in lines:
        line.words.sort(key=lambda w: w.x0)
    return lines
