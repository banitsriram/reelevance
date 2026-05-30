# 🎬 Reelevance

> *Recommendations with genuine **reel**evance.*

[![CI](https://github.com/banitsriram/reelevance/actions/workflows/ci.yml/badge.svg)](https://github.com/banitsriram/reelevance/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)
![scikit-learn](https://img.shields.io/badge/built%20with-scikit--learn-orange.svg)
![Code style](https://img.shields.io/badge/code%20style-PEP%208-blueviolet.svg)

A content-based movie recommender built on the [TMDB 5000 Movies](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) dataset. Tell it a movie you like and it suggests similar films by comparing their genres, keywords, and plot overviews. It also clusters the catalogue, ranks films with a Bayesian weighted score, and ships an interactive command-line interface.

Built with **pandas**, **scikit-learn**, and **matplotlib**.

---

## Demo

Ask for movies similar to *The Dark Knight*:

```
══════════════════════════════════════════════════════════════════════
  Movies similar to 'The Dark Knight'
══════════════════════════════════════════════════════════════════════
  #    Title                                      Year    Rating   Score
  ─────────────────────────────────────────────────────────────────
  1    The Dark Knight Rises                      2012    7.6      7.51
  2    Batman Returns                             1992    6.6      6.47
  3    Batman Begins                              2005    7.5      7.40
  4    Batman                                     1989    7.0      6.80
  5    Batman Forever                             1995    5.2      5.45
  6    Batman & Robin                             1997    4.2      4.75
  7    Batman v Superman: Dawn of Justice         2016    5.7      5.73
  8    Batman: The Dark Knight Returns, Part 2    2013    7.9      —
  9    Defendor                                   2009    6.5      —
  10   Sherlock Holmes: A Game of Shadows         2011    7.0      6.88
══════════════════════════════════════════════════════════════════════
```

| Movie clusters (PCA + KMeans) | Genre distribution |
|:---:|:---:|
| ![Movie clusters](assets/clusters.png) | ![Genre distribution](assets/genres.png) |

---

## What it does

- **Recommend by title** — finds the 10 most similar movies using TF-IDF + cosine similarity over a combined "soup" of genres, keywords, and overview text. Includes a fuzzy fallback for partial/misspelled titles.
- **Browse by genre** — lists top-rated movies in a genre, ranked by a Bayesian weighted score (so a movie with 9.0 from 12 votes doesn't outrank an 8.5 from 12,000 votes).
- **Visualisations** — five charts: genre distribution, rating distribution, releases per year, top-rated films, and a 2D PCA scatter of KMeans clusters.
- **Interactive CLI** — menu-driven; no arguments to remember.

---

## How it works

| Stage | Technique |
|-------|-----------|
| Text features | `TfidfVectorizer` (5,000 features) over genres + top-10 keywords + overview |
| Similarity | Cosine similarity between TF-IDF vectors |
| Ranking | Bayesian weighted rating (`m` = 70th percentile of vote count) |
| Clustering | TF-IDF (500 feats) → PCA (50 dims) → KMeans (k=8) |

---

## Results

The TMDB dataset is movie **metadata only** — there are no user ratings or interactions, so a traditional user-based `precision@k` isn't possible (there are no users to hold out). Reporting one would be misleading.

Instead, `evaluate.py` measures **content relevance**: for a random sample of 300 query movies, do the recommendations actually share genres with the query? The content-based recommender is compared against a random baseline and a popularity baseline (always recommend the globally top-ranked films).

| Method | Genre hit-rate@10 | Genre Jaccard@10 |
|--------|:-----------------:|:----------------:|
| **Content-based (this system)** | **0.774** | **0.358** |
| Popularity baseline | 0.537 | 0.190 |
| Random baseline | 0.499 | 0.169 |

The recommender clearly beats both baselines: ~77% of its suggestions share a genre with the query movie, versus ~50% for random picks. (Reproduce with `python evaluate.py`.)

---

## Setup

> Tested on Python 3.9–3.14 with pandas 2.x and 3.x.

```bash
# 1. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the dataset
#    Get tmdb_5000_movies.csv from:
#    https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata
#    and place it in the project root (next to main.py).
```

## Usage

```bash
python main.py        # launch the interactive recommender
python evaluate.py    # reproduce the evaluation table above
pytest                # run the test suite
```

---

## Tests

```bash
pytest                          # full suite
pytest tests/test_pipeline.py   # hermetic tests only — no dataset required
```

The suite has two layers:

- **`test_pipeline.py`** — hermetic tests that run on a tiny synthetic fixture (`tests/sample_movies.csv`). They exercise the full load → cluster → recommend pipeline and pass on a fresh clone with **no dataset download**. This is what CI runs.
- **`test_recommender.py`** — integration tests that assert real-world behaviour (a *Batman* query returns Batman films) against the actual TMDB dataset. They auto-skip when the dataset isn't present.

---

## Project structure

```
.
├── main.py                       # data loading, recommender, clustering, plots, CLI
├── evaluate.py                   # content-relevance evaluation vs. baselines
├── tests/
│   ├── test_pipeline.py          # hermetic tests — run on any clone, no dataset needed
│   ├── test_recommender.py       # integration tests against the real TMDB dataset
│   └── sample_movies.csv         # tiny synthetic fixture for the hermetic tests
├── assets/                       # demo images for this README
├── .github/workflows/ci.yml      # CI: tests on Python 3.9–3.12
├── requirements.txt
├── LICENSE
└── tmdb_5000_movies.csv          # dataset (download separately — see Setup)
```

---

## Limitations & next steps

- **Content-based only.** Recommendations are based purely on movie attributes, not on what other users enjoyed. It favours similarity over novelty — ask for movies like *The Dark Knight* and you'll get the other Batman films, not a surprising-but-fitting pick.
- **No personalisation.** There's no user model; everyone asking for the same title gets the same results. Adding collaborative filtering (e.g. with the MovieLens ratings dataset) would enable personalised, novelty-aware recommendations.
- **Recomputed each run.** The similarity matrix is rebuilt on startup (~seconds for 4,800 movies). For a larger catalogue, this would be precomputed and cached.

---

## License

Released under the [MIT License](LICENSE).
