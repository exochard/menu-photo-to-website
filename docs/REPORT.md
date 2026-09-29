# Menu photo to website: technical report

OpenCV AI Competition 2026, Agentic Vision. Team exochard (Giuseppe Castelluccio), report of
2026-09-29. Every number below comes from a script in this repository and says which one.

- Live demo: https://rdnkmfzkqy6xfzvguayc3sneva0igvic.lambda-url.eu-central-1.on.aws/ (a
  synthetic menu and site are preloaded; "Upload your own" reads any photo)
- Code: https://github.com/exochard/menu-photo-to-website
- Video: https://vimeo.com/1231246365

## Problem and users

Small restaurants and bars in Italy change prices and opening hours on paper first: the
printed menu, the price list by the till, the sign on the door. Their website, when they
have one, lags weeks behind, and a customer who finds a pasta at 9 euro online and pays 11
at the table does not come back happy. The owner rarely has the time or the login to fix
it. They do have a phone.

The user is the owner. They photograph the printed menu; the system reads it, compares it
with the website, and shows each difference as a change they can approve or reject. The
site is rebuilt only from approved changes.

## What the system does

1. **find_page.** Canny edges, dilation, convex hulls, and the largest four-corner hull
   covering at least 15% of the photo; a perspective warp flattens it. Hulls rather than
   raw contours, because glare or a thumb breaks the page edge into an open contour with
   almost no area while its hull is still the page. No page found ends the run with a
   retake request.
2. **level and read.** OpenCV 5 `dnn` runs both networks. The PP-OCRv3 DB text detector
   (`cv.dnn.TextDetectionModel_DB`) finds phrases, and PaddlePaddle's PP-OCRv5 Latin
   recogniser (`cv.dnn.readNet`, Apache-2.0) reads each phrase whole at its own aspect
   ratio. Its alphabet of 836 characters covers Italian accents and "€", so "ragù",
   "lunedì" and "12,50 €" come back as printed. A greedy CTC decoder keeps the blank as an
   index, so real hyphens survive ("12:30-15:00"); a phrase's confidence is the geometric
   mean of its per-step best probabilities, and a line's is its weakest phrase. Two steps
   keep a price on its own dish's row. The page is levelled by the median slope of its long
   text boxes when that slope exceeds 1%, because a page flattened from a steep photo can
   keep a tilt that puts the right-hand price level with the dish above. Phrases are then
   joined into a row when their boxes share at least half of the shorter box's height.
   Under blur the detector's box can stop mid-phrase ("Involtini alla" of "Involtini alla
   messinese"), and the recogniser reads the fragment with full confidence, so each box is
   first grown sideways while ink continues within one line height. Ink is measured
   against the paper around it (a morphological closing), so shadows and glare cancel out.
3. **Cut text check.** When two or more rows run into the left or right edge of the
   flattened page, the page outline cut through the menu (the sheet runs out of the photo,
   or a hand covers it). Names read there are fragments, so the run ends with a retake
   request instead of proposing them as new dishes.
4. **reread_region.** When any line is under 0.93 confidence, the page is read again after
   an unsharp mask, and each weak line keeps the likelier of its two readings.
5. **parse.** Lines become sections, items with prices, opening hours and closing notes.
6. **diff_against_site.** Names are compared without accents or case with a 0.8
   similarity cut-off, so a site that writes "Caffe" still matches a printed "Caffè"; then
   prices and hours are compared with the site's `menu` and `hours` sections. A "new" dish
   whose name is part of a site dish becomes a question ("Is that Involtini alla
   messinese?"), because it is more likely a clipped read than a new dish.
7. **Plan.** A change read at 0.93 or more becomes a proposal. Anything weaker becomes a
   question. A dish on the site but not in the photo is always a question, because absence
   in a photo is weak evidence (a crease, a missed line). Changed hours are always a
   question too: the photo gives times without days.
8. **apply_approved and build.** Approved changes are written to the site's config (the
   old one kept as `.bak`) and the static site is rebuilt by a config-driven builder that
   escapes every client string.

What the camera measured decides the next step four times: page or retake, whole menu or
retake, one read or two, proposal or question.

## Architecture on AWS

![Architecture](architecture.png)

- **Compute:** one AWS Lambda function from a container image
  (`public.ecr.aws/lambda/python:3.13`, 2048 MB, 60 s timeout), image stored in Amazon ECR.
- **Endpoint:** a Lambda Function URL serves the review page and two JSON routes,
  `POST /plan` and `POST /apply`.
- **State:** none on the server. The site config travels with each request and uploaded
  photos are never written to disk or storage.
- **Observability:** every plan logs one JSON line to CloudWatch Logs with the tool
  sequence, counts, status and timing.
- **Permissions:** the function's role has `AWSLambdaBasicExecutionRole` only (writing its
  own logs). It can read nothing else in the account.
- **Delivery:** `deploy.sh` creates or updates the repository, image, role, function and
  URL; `deploy.sh --down` removes them.

Measured on the live URL, 2026-09-29, with the PP-OCRv5 reader: a plan takes 6.0 s end to
end on a cold start and 2.8 s warm (two warm runs; levelling and growing the text boxes cost
about a second); approval and rebuild take under 0.2 s. The CRNN version took 7.6 s cold and
4.2 s warm.

## Evaluation

There is no public dataset of Italian menu photos with ground truth, so the evaluation uses
synthetic ones: `menuvision/synth.py` renders a random menu (sections, dishes with their
accents such as "ragù" and "Babà", prices with "€", hours, a closing day such as "lunedì"),
then photographs it on a table at a random angle, with uneven light, blur and sensor noise.
The two "phone" conditions add what a quick phone shot brings: a hand's soft shadow across
part of the page, a glare spot from a ceiling light, and JPEG compression at quality 55 to
75. Six conditions, 30 photos each. Every parameter was tuned on seeds 0 to 99; all numbers
below use seeds from 1000. The result files are in `docs/results/`.

The first version of this entry read words with the OpenCV Zoo CRNN, whose charset is
ASCII. The baseline columns are that version, run on the same photos.

**Reading** (`eval.py`, `docs/results/reading.json`):

| Tilt, blur | Page found | Character error rate | Items exact (name and price) | Baseline (CRNN) | Hours exact |
|---|---:|---:|---:|---:|---:|
| 0.04, 0.6 | 30/30 | 0.000 | 199/199 (100%) | 128/199 (64%) | 30/30 |
| 0.08, 1.0 | 30/30 | 0.000 | 199/199 (100%) | 117/199 (59%) | 30/30 |
| 0.12, 1.6 | 29/30 | 0.002 | 195/199 (98%) | 58/199 (29%) | 30/30 |
| 0.16, 2.2 | 26/30 | 0.011 | 191/199 (96%) | 6/199 (3%) | 29/30 |
| 0.10, 1.2, phone | 30/30 | 0.000 | 198/199 (99%) | 89/199 (45%) | 30/30 |
| 0.16, 1.8, phone | 26/30 | 0.010 | 194/199 (97%) | 18/199 (9%) | 29/30 |

Finding and reading a page takes 0.22 s per photo on a laptop CPU, against 0.73 s for
the word-by-word CRNN (30 photos, nothing else running).

**The agent, end to end** (`eval_agent.py`, `docs/results/agent*.json`). Each photo's site
config is its printed menu with two prices changed and one extra dish, so a perfect agent
proposes exactly 60 price changes per condition, asks about the extra dish, and proposes
nothing else.

| Tilt, blur | Price changes proposed | Without the re-read | Baseline (CRNN) | Wrong proposals | Baseline wrong | Asked about the extra dish | Questions per photo | Retakes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.04, 0.6 | 60/60 | 59/60 | 56/60 | 0 | 0 | 30/30 | 1.00 | 0 |
| 0.08, 1.0 | 60/60 | 60/60 | 53/60 | 0 | 0 | 30/30 | 1.00 | 0 |
| 0.12, 1.6 | 58/60 | 56/60 | 34/60 | 0 | 0 | 29/30 | 1.00 | 1 |
| 0.16, 2.2 | 49/60 | 46/60 | 13/60 | 0 | 4 | 26/30 | 1.10 | 4 |
| 0.10, 1.2, phone | 60/60 | 57/60 | 37/60 | 0 | 1 | 30/30 | 1.00 | 0 |
| 0.16, 1.8, phone | 50/60 | 46/60 | 16/60 | 0 | 6 | 26/30 | 1.07 | 4 |

Across 180 photos the agent proposed no wrong change. The CRNN version proposed 11 on the
harder photos, where a confident misread became a proposal. A perfect run asks one question
per photo (the extra dish); the agent asks 1.00 to 1.10.

**Failures, as measured.**

- Every retake (1, 4 and 4) is a photo where no page outline was found: at a steep angle
  the sheet runs out of the frame. The agent asks for a new photo rather than guessing.
- Every changed price the agent did not propose on a photo it read became a question to the
  owner (0 to 3 per condition); none was dropped silently.
- The re-read now adds little (0 to 4 proposals per condition), because the first read is
  rarely unsure.
- Past what a person can read, a confident misread still passes. The demo photo blurred at
  sigma 4 gives "Cannolo siciliano 1,50" at 0.933, just over the threshold, and at 4.5 the
  page outline is lost and the agent asks for a retake. The owner's approval of each change
  is the guard for the first case.
- The confidence threshold is imperfect. On tuning seeds 0 to 39 it flags 23% of misread
  item lines and 6% of correctly read ones. A misread the recogniser is sure of can pass as
  a proposal; the owner's approval step is the guard for those.

## Limitations

- The evaluation is synthetic. Real menus have handwriting, chalkboards, two columns,
  decorative fonts and photos on laminated card; none of those is measured here.
- A photo that runs past the page edge ends in a retake; the agent does not read a partial
  menu.
- Hours are compared as time ranges only. The owner decides which days change.
- One page per photo; one menu section per site.
- The demo's Function URL is public and has no login, so anyone can make the function read
  a photo. It stores nothing and can reach nothing but its own logs; the account's Lambda
  concurrency limit (10, checked 2026-09-29) caps the load. A real
  deployment would put each owner's page behind a sign-in.

## Responsible use and human control

- Nothing reaches the website without the owner's approval of that specific change.
- Low confidence, a missing page, a missing dish and changed hours all become questions
  rather than edits.
- Uploaded photos are processed in memory and not stored. The public demo holds no
  personal data; its menu is synthetic.
- Client config is untrusted input: the site builder escapes every string it renders and
  whitelists what reaches CSS, and a test checks that a dish named `<script>` is rendered
  as text.
- The function's AWS role can only write its own logs.

## How it was built

I set the goal, the constraints (owner approval for every edit, no stored uploads, zero
spend) and the acceptance checks. The code, the evaluation scripts and this report were
drafted with an AI coding assistant (Claude Code), which also ran the measurements; I
reviewed the result before submitting it. Every number comes from the script named next to
it, and the failures are reported with the successes.

## Reproduce

```
python -m venv .venv && .venv/bin/pip install -r requirements.txt
./fetch_models.sh
.venv/bin/python -m pytest -q                    # 17 tests
.venv/bin/python eval.py --n 30                  # reading table
.venv/bin/python eval_agent.py --n 30            # agent table
.venv/bin/python eval_agent.py --n 30 --no-reread
.venv/bin/python webapp.py                       # http://localhost:8080
./deploy.sh                                      # AWS, after `aws login`
```
