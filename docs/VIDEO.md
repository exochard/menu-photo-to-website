# Demo video

The rules ask for "a public or unlisted judge-accessible video of no more than five minutes
that shows the team, the application working, its architecture, and its principal results"
(https://opencv26.devpost.com/, read 2026-09-29). The video runs 3:21 at 1920x1080, 30 fps:
narration by a synthetic voice (Qwen3-TTS-12Hz-1.7B-CustomVoice, speaker "aiden", run
locally), the same words as large captions, and a quiet music bed generated for the video.
The team is shown by name and photo on the title and closing cards; the closing card says the
voice is synthetic.

## Story

| Chapter | What the viewer sees | What it proves |
|---|---|---|
| The problem | The demo restaurant's site on a phone says 3,00 €; its printed menu says 3,50 € | why anyone needs this |
| The idea | Title, team, then five steps: photo, OpenCV reads, agent compares, owner approves, site rebuilt | the whole loop in ten seconds |
| 1 · Seeing the menu | The real edge map, the page outline drawn corner by corner, the flattened page, then every DB text box and the PP-OCRv5 reading with its confidence | substantive OpenCV 5 |
| 2 · Deciding | Photo vs. website table: two proposals, two questions, with the agent's own wording | the agent's autonomy and its limits |
| 3 · The owner approves | The live app on its Lambda URL: untick one proposal, apply, rebuilt page, the phone updates | the application working, human control |
| 4 · When the photo is bad | Blurred photo: two weak lines, sharpened and re-read, the braciole price still unclear, so a question. A steep held-out test photo whose sheet leaves the frame: the outline in two pieces, the joined outline, the trace, both changes found (retakes on the two steepest sets 8 to 0). A blurrier photo: both outlines fail, a retake request and no change | failure handling |
| 5 · On AWS | Animated architecture: Function URL, Lambda container with the four tools, ECR, CloudWatch with the real log line of the blurred-photo run, IAM and storage badges | architecture, responsible operation |
| 6 · Results | Sample photo per condition, bars for 60, 60, 60, 57, 60 and 57 of 60 beside the first version's, and 0 wrong proposals (the first version: 11) | principal results |
| Close | Live URL, repository, team, AI-assistance note | where to check it |

The caption lines are in `video/script.json`.

## How it is built

Every image comes from the real system: `video/assets.py` runs the same `menuvision`
code as the live endpoint to produce the edge map, page outline, flattened page, word
boxes and reads, and the blurred copies; `video/capture.mjs` drives the live URL for the
app screenshots and the site before and after approval. `video/stage.html` animates them;
every frame is a pure function of time, and `video/render.mjs` captures it frame by frame
and mixes the narration.

```
.venv/bin/python -m video.assets                         # from the entry folder
cd video
PLAYWRIGHT=<path>/node_modules/playwright node capture.mjs     # about 8 live requests
.venv/bin/python portrait.py <photo>                     # round portrait for the cards
<torch venv>/bin/python narrate.py --engine qwen --speaker aiden   # voice + timeline
.venv/bin/python music.py                                # out/music.wav, synthesised here
PLAYWRIGHT=<path>/node_modules/playwright node render.mjs --person "<name>" --team "<team>" \
  --repo "<repo URL>" --voice-note "Narration: synthetic voice (Qwen3-TTS)." \
  --audio out/narration.wav --music out/music.wav
```

Output: `video/out/demo.mp4` and `video/out/captions.srt`.

## Narration

`script.json` is written as spoken narration, one argument from problem to proof; each line
is its caption too, or `[caption, spoken]` where the voice reads a price aloud. Captions show
one sentence at a time. Every sentence was transcribed
back with Qwen3-ASR-1.7B to catch misread words. `narrate.py --text` gives a caption-only cut;
`voice.sh` records a human voice instead; each re-times the whole video.
