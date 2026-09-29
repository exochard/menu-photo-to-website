"""Measure the agent loop end to end on synthetic photos with a known site config.

Each photo's site config is the printed menu with two prices changed and one extra item,
so a perfect agent proposes exactly the two old-to-new price changes, asks about the extra
item, and proposes nothing else. `--no-reread` switches the sharpened second read off.

    .venv/bin/python eval_agent.py --n 30 [--no-reread] [--first-seed 0]
"""
import argparse
import json
import random
from pathlib import Path

from menuvision import agent
from menuvision.ocr import Reader
from menuvision.synth import sample

# (tilt, blur, phone): phone adds a hand's shadow, a glare spot and JPEG compression.
CONDITIONS = [(0.04, 0.6, False), (0.08, 1.0, False), (0.12, 1.6, False),
              (0.16, 2.2, False), (0.10, 1.2, True), (0.16, 1.8, True)]
EVAL_SEED = 1000  # parameters were tuned on seeds 0-99


def scenario(s, rng: random.Random) -> tuple[dict, set[tuple[str, str]]]:
    groups = [{"name": sec.name, "items": [{"name": i.name, "price": f"{i.price} €"} for i in sec.items]}
              for sec in s.truth.sections]
    items = [item for group in groups for item in group["items"]]
    expected = set()
    for item in rng.sample(items, 2):
        expected.add((item["name"], item["price"][:-2]))
        item["price"] = f"{rng.randint(30, 60)},00 €"
    groups[-1]["items"].append({"name": "Piatto del giorno", "price": "14,00 €"})
    cfg = {"site": {"name": "T"}, "pages": [{"slug": "index", "sections": [
        {"type": "menu", "title": "Menu", "groups": groups}]}]}
    return cfg, expected


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--no-reread", action="store_true")
    ap.add_argument("--out", default="out/eval-agent.json")
    ap.add_argument("--first-seed", type=int, default=EVAL_SEED, help="0 for the tuning seeds")
    args = ap.parse_args()
    if args.no_reread:
        agent.merge_reads = lambda first, _second, low=agent.LOW: (first, 0)
    reader = Reader()
    report = {}
    for tilt, blur, phone in CONDITIONS:
        found = right = wrong = asked_extra = questions = retakes = rereads = 0
        for seed in range(args.first_seed, args.first_seed + args.n):
            s = sample(seed, tilt, blur, phone)
            cfg, expected = scenario(s, random.Random(seed))
            outcome = agent.run(s.photo, cfg, reader)
            retakes += outcome.status == "retake"
            rereads += any(step["tool"] == "reread_region" for step in outcome.trace)
            got = {(c.name, c.new) for c in outcome.proposals if c.kind == "price"}
            right += len(got & expected)
            found += len(expected)
            wrong += len(outcome.proposals) - len(got & expected)
            asked_extra += any("Piatto del giorno" in q for q in outcome.questions)
            questions += len(outcome.questions)
        key = f"tilt={tilt},blur={blur}" + (",phone" if phone else "")
        report[key] = {
            "photos": args.n,
            "price_changes_proposed": f"{right}/{found}",
            "wrong_proposals": wrong,
            "asked_about_extra_item": f"{asked_extra}/{args.n}",
            "questions_per_photo": round(questions / args.n, 2),
            "retake_requests": retakes,
            "photos_reread": rereads,
        }
        print(key, report[key])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
