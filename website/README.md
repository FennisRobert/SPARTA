# SPARTA website

A single-page site: what SPARTA is, how it works, features, performance,
requirements, where to get it, and questions.

## Files

| Path | What it is |
|---|---|
| `index.html` | The whole page: HTML, styles and a little JavaScript, in one file |
| `assets/logo.svg` | The helmet logo, copied from `documentation/logo_header.svg` |

It needs no build step. The only outside resource is the Roboto font from
Google Fonts; without it the page falls back to the system font.

## Preview

Open `index.html` in a browser, or serve the folder:

```sh
python3 -m http.server -d website 8000
```

and visit <http://localhost:8000>.

## Before it goes live

- **The GitHub links point at a private repository.** The page sends visitors
  to <https://github.com/FennisRobert/SPARTA> and its `pip install` line
  installs from there. Both work only once the repository is public.
- **The licence is the GNU AGPL, version 3 or later**, with a commercial
  licence for closed-source software; inquiries go to
  robertfennis@emerge-numerics.com. Keep the page in step with `LICENSE` and the
  main README.
- **The performance numbers** come from three EMerge models in the
  `sparta_compare` folder next to this repository, measured on 16 November
  2026 with EMerge 3.0.7 on an Apple M4 Pro (14-core CPU, 20-core GPU, 24 GB),
  default settings: `python run_bench.py 3 > results3.jsonl`, then
  `python to_website.py results3.jsonl <this folder>/index.html`, which
  rewrites only the part between the `EMERGE-BENCH` markers. Re-run it before
  quoting new numbers.

## Hosting

Any static host works, since it's one HTML file and one image: GitHub Pages
pointed at this folder, Netlify, or a plain web server.
