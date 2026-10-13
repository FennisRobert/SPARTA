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
- **The performance numbers** were measured with `tools/website_bench.py` on
  13 October 2026: Apple M4 Pro (14-core CPU, 20-core GPU, 24 GB), macOS 26.6.2,
  the four `test_matrices/*1.sparta` systems, medians of three runs, with
  `RAYON_NUM_THREADS=8 VECLIB_MAXIMUM_THREADS=8`, without iterative refinement. Problem #1
  is `em_m1`, #2 `em_m2`, #3 `em_m3_asym`, #4 `em_bandcond`. Re-run it before
  quoting new numbers.

## Hosting

Any static host works, since it's one HTML file and one image: GitHub Pages
pointed at this folder, Netlify, or a plain web server.
