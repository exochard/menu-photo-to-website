# Held-out real menu photos

Ten more real menu photos with open licences (CC BY, CC BY-SA), from Openverse, found with
searches in other languages ("speisekarte", "carta restaurante precios", "cafe price list"
and similar). They were labelled by hand, up to six (name, price) pairs each, before any
code ran on them, and kept away from the work that tuned the pipeline on
`docs/real-photos/`. They were run twice: once on the code before the real-photo changes
(`docs/results/eval-real-heldout-before.json`) and once on the code after them
(`docs/results/eval-real-heldout.json`). Nothing was changed between or after those runs.

Where several readings of a handwritten price column were possible, the price was
assigned by its vertical position next to the dish; items whose name could not be read
with certainty were skipped. `ho022.jpg` shows prices with the names hidden by a fold and
has no labels, to see whether the agent asks for a retake.

```
.venv/bin/python eval_real.py --dir docs/real-photos-heldout --out docs/results/eval-real-heldout.json
```

## Credits

| Photo | Author | Licence | Source |
|---|---|---|---|
| `ho008.jpg` | Cyrond | BY-SA 2.0 | https://www.flickr.com/photos/8829996@N07/542089922 |
| `ho013.jpg` | ThomasKohler | BY 2.0 | https://www.flickr.com/photos/28077296@N02/3665587930 |
| `ho016.jpg` | fjludo | BY 2.0 | https://www.flickr.com/photos/28832189@N04/3572208559 |
| `ho017.jpg` | domjisch | BY-SA 2.0 | https://www.flickr.com/photos/46604778@N00/13984232606 |
| `ho022.jpg` | domjisch | BY-SA 2.0 | https://www.flickr.com/photos/46604778@N00/14027358033 |
| `ho034.jpg` | Wolf Gang | BY-SA 2.0 | https://www.flickr.com/photos/81669195@N00/4786608403 |
| `ho045.jpg` | Thomas Locke Hobbs | BY-SA 2.0 | https://www.flickr.com/photos/22185138@N00/363199442 |
| `ho088.jpg` | markus119 | BY 2.0 | https://www.flickr.com/photos/35850894@N08/50562764606 |
| `ho206.jpg` | Bernt Rostad | BY 2.0 | https://www.flickr.com/photos/67975030@N00/6809457906 |
| `ho211.jpg` | Bernt Rostad | BY 2.0 | https://www.flickr.com/photos/67975030@N00/6357845565 |

CC BY-SA photos stay under CC BY-SA here.
