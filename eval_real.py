"""Measure the frozen pipeline on real menu photos found online.

    python eval_real.py [--dir docs/real-photos] [--out docs/results/eval-real.json]

Each photo in `labels.json` carries up to six hand-labelled (name, price) pairs, taken from
the first priced items in reading order before the pipeline was run on it. The agent's own
read step runs unchanged: the single outline, then the joined outline, then a retake request.

Per photo it reports what the agent did (read the page or asked for a retake), how many
labelled items came back with the right price, and how many came back with a wrong price.
A wrong price is the failure that matters: it would become a proposal on the live site,
where a retake request only costs the owner a second photo.

The full agent then runs against a site whose menu holds exactly the labelled dishes at
their labelled prices. A price change it proposes on a labelled dish is always wrong. A
"new dish" proposal can be an unlabelled dish read correctly, or a misread line, so those
are listed for a person to classify. Questions are counted apart, because the owner
answers them instead of approving a change.
"""
import argparse
import json
import re
import time
import unicodedata
from pathlib import Path

import cv2 as cv

from menuvision.agent import LOW, read_page, run
from menuvision.ocr import Reader


def norm(name: str) -> str:
    text = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"^\s*\d+\s+(?=[a-z])", "", text)
    return re.sub(r"\s+", " ", text).strip()


def same_dish(label: str, read: str) -> bool:
    """A read name counts as the labelled dish when it contains the label, or most words agree.

    Containment runs one way only: "Spaghetti" read off the card must not count as the
    labelled "Spaghetti with Salad", which is a different dish at a different price.
    """
    a, b = norm(label), norm(read)
    if not a or not b:
        return False
    if a in b:
        return True
    wa, wb = set(a.split()), set(b.split())
    return len(wa & wb) / len(wa | wb) >= 0.5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="docs/real-photos")
    ap.add_argument("--out", default="docs/results/eval-real.json")
    args = ap.parse_args()
    folder = Path(args.dir)
    labels = json.loads((folder / "labels.json").read_text(encoding="utf-8"))
    reader = Reader()
    rows = []
    totals = {"labelled": 0, "right_price": 0, "wrong_price": 0, "retakes": 0,
              "price_changes": 0, "new_dish_proposals": 0, "questions": 0}
    for entry in labels:
        photo = cv.imread(str(folder / entry["photo"]))
        trace, quad, failures = [], None, []
        t = time.perf_counter()
        for joined in (False, True):
            read, _, failure, quad = read_page(photo, reader, joined, trace, LOW, quad)
            if not failure:
                break
            failures.append(failure)
        secs = time.perf_counter() - t
        status = "read" if not failure else "retake"
        right, wrong = [], []
        for name, price in entry["items"]:
            hits = [(n, p) for _, n, p, _ in read if same_dish(name, n)]
            if any(p == price for _, p in hits):
                right.append(name)
            elif hits:
                wrong.append({"label": [name, price], "read": hits[0]})
        site = {"pages": [{"sections": [{"type": "menu", "groups": [{"name": "Menu", "items": [
            {"name": name, "price": f"{price} €"} for name, price in entry["items"]]}]}]}]}
        outcome = run(photo, site, reader)
        proposals = [[c.kind, c.name, c.old, c.new] for c in outcome.proposals]
        totals["price_changes"] += sum(c.kind == "price" for c in outcome.proposals)
        totals["new_dish_proposals"] += sum(c.kind == "new_item" for c in outcome.proposals)
        totals["questions"] += len(outcome.questions)
        totals["labelled"] += len(entry["items"])
        totals["right_price"] += len(right)
        totals["wrong_price"] += len(wrong)
        totals["retakes"] += status == "retake"
        rows.append({"photo": entry["photo"], "status": status, "failures": failures,
                     "items_read": len(read), "labelled": len(entry["items"]),
                     "right_price": len(right), "wrong_price": wrong,
                     "agent": outcome.status, "proposals": proposals,
                     "questions": outcome.questions,
                     "read": [[n, p] for _, n, p, _ in read], "seconds": round(secs, 1)})
        print(f"{entry['photo']}: {status} {failures or ''} read={len(read)} "
              f"right={len(right)}/{len(entry['items'])} wrong={len(wrong)} "
              f"agent={outcome.status} proposals={len(proposals)} questions={len(outcome.questions)} "
              f"({secs:.1f}s)")
    report = {"photos": len(rows), **totals, "rows": rows}
    print({k: v for k, v in report.items() if k != "rows"})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
