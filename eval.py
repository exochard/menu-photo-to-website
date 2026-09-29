"""Measure the vision pipeline on synthetic menu photos.

    python eval.py [--n 30] [--out out/eval.json]

Reports, per condition (tilt, blur): page found, character error rate over all text,
items whose name and price both come back exactly, and opening hours recovered.
"""
import argparse
import json
import re
import time
from pathlib import Path

from menuvision.ocr import Reader
from menuvision.page import find_page
from menuvision.parse import parse
from menuvision.synth import sample

# (tilt, blur, phone): phone adds a hand's shadow, a glare spot and JPEG compression.
CONDITIONS = [(0.04, 0.6, False), (0.08, 1.0, False), (0.12, 1.6, False),
              (0.16, 2.2, False), (0.10, 1.2, True), (0.16, 1.8, True)]
# Parameters were tuned on seeds 0-99; the evaluation never uses those.
EVAL_SEED = 1000


def cer(got: str, want: str) -> float:
    prev = list(range(len(want) + 1))
    for i, a in enumerate(got, 1):
        cur = [i]
        for j, b in enumerate(want, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a != b)))
        prev = cur
    return prev[-1] / max(1, len(want))


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("€", "")).strip().lower()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--out", default="out/eval.json")
    args = ap.parse_args()
    reader = Reader()
    report = {}
    for tilt, blur, phone in CONDITIONS:
        found, cers, items_ok, items_all, hours_ok, secs = 0, [], 0, 0, 0, []
        for seed in range(EVAL_SEED, EVAL_SEED + args.n):
            s = sample(seed, tilt, blur, phone)
            t = time.perf_counter()
            page = find_page(s.photo)
            lines = [line.text for line in reader.read(reader.level(page.image))]
            secs.append(time.perf_counter() - t)
            found += page.quad is not None
            cers.append(cer(norm(" ".join(lines)), norm(" ".join(s.lines))))
            menu = parse(lines)
            got = {(i.name.lower(), i.price) for sec in menu.sections for i in sec.items}
            want = [(i.name.lower(), i.price) for sec in s.truth.sections for i in sec.items]
            items_ok += sum(w in got for w in want)
            items_all += len(want)
            hours_ok += menu.hours == s.truth.hours
        key = f"tilt={tilt},blur={blur}" + (",phone" if phone else "")
        report[key] = {
            "photos": args.n,
            "page_found": f"{found}/{args.n}",
            "cer_mean": round(sum(cers) / len(cers), 3),
            "items_exact": f"{items_ok}/{items_all}",
            "hours_exact": f"{hours_ok}/{args.n}",
            "seconds_per_photo": round(sum(secs) / len(secs), 2),
        }
        print(key, report[key])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
