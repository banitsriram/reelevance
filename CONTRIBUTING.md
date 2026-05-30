# Contributing

Thanks for taking a look! This is a small, focused project, so contributing is straightforward.

## Getting set up

```bash
git clone <your-fork-url>
cd movierecomendation

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The interactive app needs the TMDB 5000 dataset (`tmdb_5000_movies.csv`) in the
project root — see the [README](README.md#setup) for the download link. The
hermetic test suite does **not** need it.

## Running the tests

```bash
pytest                          # full suite
pytest tests/test_pipeline.py   # hermetic tests only — no dataset required
```

- `tests/test_pipeline.py` runs on a tiny synthetic fixture and must pass on any
  clone. **Add new logic tests here** so they run in CI.
- `tests/test_recommender.py` checks real-world relevance against the full TMDB
  dataset and auto-skips when it's absent.

CI (GitHub Actions) runs the suite on Python 3.9–3.12 for every push and pull
request. Please make sure `pytest` is green before opening a PR.

## Style

- Follow [PEP 8](https://peps.python.org/pep-0008/); match the surrounding code.
- Keep changes focused — one logical change per commit.
- This repo uses [Conventional Commits](https://www.conventionalcommits.org/)
  (`feat:`, `fix:`, `test:`, `docs:`, `ci:`).

## Ideas worth picking up

See the **Limitations & next steps** section of the README — collaborative
filtering, caching the similarity matrix, and embedding-based similarity are all
good first contributions.
