# Menu photo to website

A restaurant owner photographs the printed menu. OpenCV 5 finds the page, flattens it and
reads it; an agent compares what it read with the restaurant's website config, proposes
each change, and rebuilds the site only after the owner approves it. Entry for the OpenCV
AI Competition 2026, Agentic Vision.

- Live demo (AWS Lambda): https://rdnkmfzkqy6xfzvguayc3sneva0igvic.lambda-url.eu-central-1.on.aws/
- Technical report, with the evaluation and its failures: [`docs/REPORT.md`](docs/REPORT.md)
- Architecture: [`docs/architecture.png`](docs/architecture.png)

## How it decides

The agent's tools are `find_page`, `read`, `reread_region`, `parse`, `diff_against_site`
and `apply_approved`. What the camera measured picks the next step:

- no page in the photo: a retake request, and nothing is edited;
- lines read under 0.95 confidence: the page is sharpened and read again;
- a change still under 0.95, a dish missing from the photo, or different opening hours:
  a question for the owner, not an edit;
- everything else: a proposal the owner approves or rejects one by one.

## Results (synthetic photos, 30 per condition, seeds never used for tuning)

| Tilt, blur | Items read exactly | Changed prices proposed | Wrong proposals |
|---|---:|---:|---:|
| 0.04, 0.6 | 83% | 54/60 | 0 |
| 0.08, 1.0 | 75% | 54/60 | 0 |
| 0.12, 1.6 | 39% | 36/60 | 0 |

## Run it

```
python -m venv .venv && .venv/bin/pip install -r requirements.txt
./fetch_models.sh                    # OpenCV Zoo text models (Apache-2.0)
.venv/bin/python -m pytest -q
.venv/bin/python webapp.py           # http://localhost:8080
.venv/bin/python eval.py --n 30      # reading
.venv/bin/python eval_agent.py --n 30 [--no-reread]
./deploy.sh                          # AWS Lambda + Function URL, after `aws login`
./deploy.sh --down                   # remove it
```

`sitebuilder/` is the config-driven static-site builder the agent updates; each client is a
`site.yaml`, and every string from it is escaped.

Built with an AI coding assistant (Claude Code); see "How it was built" in the report.
