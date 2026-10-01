# Real menu photos

Eleven photos of real menus, found on Openverse and Wikimedia Commons under licences that
allow reuse (CC0, public domain, CC BY, CC BY-SA). They are the only real photos in the
evaluation; the rest is synthetic. Photos wider than 2000 px were scaled down to 2000 px
before the run, and the scaled copies here are the ones measured. No other change was made.

## How they were chosen and labelled

The search covered about 400 candidates ("menu", "pizzeria menu", "listino prezzi",
"menu trattoria" and similar). A photo was kept when it shows a printed or handwritten menu
with prices written with two decimals, the only price format the parser reads. Menus that
give one price per size in separate columns were left out, because the site holds one
price per dish.

For each photo, up to six (name, price) pairs were written down by hand from the first
priced items in reading order, before the pipeline was run on it (`labels.json`). Where a
handwritten name could not be read with certainty, that item was skipped. One photo
(`ov076.jpg`) has no complete name and price pair in view and is kept with no labels, to
see whether the agent asks for a retake.

## Credits

| Photo | Author | Licence | Source |
|---|---|---|---|
| `ov076.jpg` | sidewalk flying | BY 2.0 | https://www.flickr.com/photos/76994867@N00/4038872674 |
| `ov116.jpg` | Scotscowgirls | BY 2.0 | https://www.flickr.com/photos/16491477@N07/6822426049 |
| `ov133.jpg` | avlxyz | BY-SA 2.0 | https://www.flickr.com/photos/10559879@N00/3427783161 |
| `ov141.jpg` | grongar | BY 2.0 | https://www.flickr.com/photos/70757891@N00/6447341935 |
| `ov145.jpg` | qcom | BY-SA 2.0 | https://www.flickr.com/photos/11253414@N00/9064258137 |
| `wc099.jpg` | Missvain | CC0 | https://commons.wikimedia.org/wiki/File:Brozinni_Pizzeria_-_October_2023_-_Sarah_Stierch_04.jpg |
| `wc113.jpg` | not named (scan via DPLA) | Public domain | https://commons.wikimedia.org/wiki/File:Fireside_Pizza_Menu_-_DPLA_-_d11d53752afd42fba6255d1b3b13bb3d_(page_2).jpg |
| `wc137.jpg` | Joanbanjo | CC BY-SA 4.0 | https://commons.wikimedia.org/wiki/File:Men%C3%BA_d%27una_pizzeria_del_carrer_de_Baix,_Val%C3%A8ncia.jpg |
| `wc145.jpg` | StevenBjerke97 | CC BY-SA 4.0 | https://commons.wikimedia.org/wiki/File:Quebec_pizzeria_menu.jpg |
| `wc148.jpg` | Missvain | CC0 | https://commons.wikimedia.org/wiki/File:Della_Santina%27s_Trattoria_-_February_2025_-_Sarah_Stierch_01.jpg |
| `wc162.jpg` | Nemo bis | CC BY-SA 3.0 | https://commons.wikimedia.org/wiki/File:2016-07-05_osteria_la_Chitarra,_Napoli.jpg |

CC BY-SA photos stay under CC BY-SA here.
