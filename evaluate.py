"""
============================================================
  EVALUATION — Content-Based Recommender
============================================================
Honest evaluation for a content-based system.

NOTE ON METHODOLOGY:
The TMDB 5000 dataset contains movie *metadata* only — there are no
user ratings or interactions. So a classic user-based precision@k is
not possible (there are no users to hold out). Reporting one would be
meaningless.

Instead we measure CONTENT RELEVANCE: given a query movie, do the
recommendations actually share genres with it? This directly tests
whether the TF-IDF + cosine similarity captures real relatedness.

We report two metrics over a random sample of query movies, and compare
the content-based recommender against two baselines:
  - random      : k movies picked at random
  - popularity  : top-k movies by weighted score (ignores the query)

Metrics:
  - Genre hit-rate@k : fraction of recommendations sharing >=1 genre
                       with the query movie.
  - Mean genre Jaccard@k : average Jaccard overlap of genre sets.

A good content-based recommender should clearly beat both baselines on
these metrics. (It will not beat them on novelty — that trade-off is
discussed in the README.)
"""

import random
import numpy as np

import main


def _genre_sets(df):
    return df["genres_list"].apply(set).tolist()


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _topk_content(idx, sim, k):
    """Top-k most similar movie indices (excluding the movie itself)."""
    scores = list(enumerate(sim[idx]))
    scores.sort(key=lambda x: x[1], reverse=True)
    return [i for i, _ in scores[1 : k + 1]]


def evaluate(df, sim, k=10, sample_size=300, seed=42):
    rng = random.Random(seed)
    genres = _genre_sets(df)

    # Only evaluate query movies that actually have genres.
    candidates = [i for i in range(len(df)) if genres[i]]
    sample = rng.sample(candidates, min(sample_size, len(candidates)))

    # Popularity baseline: same top-k by score for every query.
    pop_order = df["score"].to_numpy().argsort()[::-1]

    results = {m: {"hit": [], "jac": []} for m in ("content", "popularity", "random")}

    for q in sample:
        q_genres = genres[q]

        recs = {
            "content": _topk_content(q, sim, k),
            "popularity": [i for i in pop_order if i != q][:k],
            "random": rng.sample([i for i in candidates if i != q], k),
        }

        for method, idxs in recs.items():
            hits = [1 for r in idxs if q_genres & genres[r]]
            jacs = [_jaccard(q_genres, genres[r]) for r in idxs]
            results[method]["hit"].append(len(hits) / k)
            results[method]["jac"].append(np.mean(jacs) if jacs else 0.0)

    return {
        m: {
            "hit_rate@k": float(np.mean(v["hit"])),
            "genre_jaccard@k": float(np.mean(v["jac"])),
        }
        for m, v in results.items()
    }


def main_eval():
    print("\nLoading data and building similarity matrix ...")
    df = main.load_data(main.DATASET_PATH)
    df, _ = main.cluster_movies(df)
    sim, _ = main.build_recommender(df)

    k = 10
    scores = evaluate(df, sim, k=k)

    print("\n" + "=" * 60)
    print(f"  CONTENT-RELEVANCE EVALUATION  (k={k}, 300 query movies)")
    print("=" * 60)
    print(f"  {'Method':<14}{'Genre hit-rate@k':<20}{'Genre Jaccard@k'}")
    print("  " + "-" * 52)
    for method in ("content", "popularity", "random"):
        s = scores[method]
        print(f"  {method:<14}{s['hit_rate@k']:<20.3f}{s['genre_jaccard@k']:.3f}")
    print("=" * 60)
    print("\n  Interpretation: the content-based recommender should score")
    print("  much higher than the random and popularity baselines, showing")
    print("  its recommendations are genuinely related to the query movie.\n")


if __name__ == "__main__":
    main_eval()
