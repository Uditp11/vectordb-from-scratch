# Testing

The tests in `tests/` check that the search algorithms return correct results. They run on the CPU only and need no Ollama and no network.

## Run them

```bash
pip install -r requirements-dev.txt
python -m pytest
```

On my laptop the full run takes under a minute. Almost all of that is building the HNSW indexes for the recall tests, because insertion is pure Python.

## What each test proves

### `tests/test_algorithms.py`

| Test | What it proves |
|---|---|
| `test_distances_on_hand_checked_values` | Cosine, Euclidean and Manhattan distance give the values I worked out by hand (for example, the Euclidean distance from (0, 0) to (3, 4) is 5 and the Manhattan distance is 7) |
| `test_cosine_of_zero_vector_is_one` | A zero vector has no direction, so `cosine_distance` returns 1.0 instead of dividing by zero |
| `test_brute_force_matches_numpy_sort` | `BruteForce` returns exactly the same 10 ids, in the same order, as sorting all distances with NumPy. This makes brute force a trusted ground truth for the other tests |
| `test_brute_force_on_empty_input` | Searching an empty list returns an empty list |
| `test_kdtree_equals_brute_force` | KD-Tree search returns exactly what brute force returns, for 3 seeds, 4 dimensions (2, 3, 8, 16) and both Euclidean and Manhattan distance. KD-Tree is meant to be exact, so anything less than equal is a bug |
| `test_cosine_kdtree_equals_brute_force` | A cosine KD-Tree (`KDTree(dim, normalize=True)`) returns exactly what brute force returns with cosine distance, for 3 seeds and the same 4 dimensions. See "KD-Tree and cosine distance" below |
| `test_cosine_kdtree_with_zero_vectors` | Zero vectors cannot be scaled to length 1. The cosine tree keeps them, still matches brute force, still returns every point when k covers them all, and a zero query gets distance 1.0 to everything |
| `test_kdtree_with_k_zero_returns_nothing` | Asking for 0 neighbours returns an empty list. Before the fix this raised an IndexError |
| `test_kdtree_insert_and_delete_rebuild` | After an insert or a delete the tree is rebuilt and search sees the change |
| `test_hnsw_recall_at_10` | HNSW finds at least a measured share of the true 10 nearest neighbours, for 3 seeds, with Euclidean and cosine distance |
| `test_hnsw_is_deterministic_with_seed` | With the same seed, two builds give the same node levels and the same layer 0 graph. Without this, a recall test could pass on one run and fail on the next |
| `test_hnsw_empty_index_and_delete` | An empty index returns nothing, and a deleted node no longer appears in results |

### `tests/test_pca.py`

| Test | What it proves |
|---|---|
| `test_pca_matches_svd_up_to_sign` | `pca_2d` (covariance matrix plus `numpy.linalg.eigh`) gives the same 2D projection as an SVD of the centred data. An eigenvector is only defined up to its sign, so each axis is compared after flipping it to match |
| `test_pca_small_inputs` | No vectors give an empty list, and one vector gives the point (0, 0) |

### `tests/test_api.py`

These call the FastAPI routes through `TestClient`, which runs the app in the same process without starting a server. `app.py` imports without Ollama running: it builds the demo database at import time and only talks to Ollama inside `/doc/insert`, `/doc/ask` and `/status`. Those three routes are not tested here because they need Ollama. `verify_rag_pipeline.py` covers the first two with mocked Ollama responses.

| Test | What it proves |
|---|---|
| `test_stats` | `/stats` reports 16 dimensions and the 20 demo vectors |
| `test_items_include_pca_coords` | `/items` returns one PCA point per item |
| `test_search_finds_the_query_vector_itself` | Searching with a stored vector returns that vector first, for all three algorithms |
| `test_kdtree_route_matches_brute_force` | Through the API, KD-Tree returns the same distances as brute force for 20 random queries with each metric. Distances are compared rather than ids because some hand-made demo vectors are at exactly the same distance from a query, and tied items can come back in either order |
| `test_search_with_k_zero` | `k=0` returns an empty result for all three algorithms instead of an error |
| `test_search_rejects_bad_input` | A vector that is not numbers, or an unknown algorithm, gives HTTP 400 |
| `test_benchmark_runs_all_three` | `/benchmark` returns results for brute, kdtree and hnsw |
| `test_hnsw_info` | `/hnsw-info` counts all 20 demo nodes |
| `test_insert_then_delete` | Insert adds an item, delete removes it, a second delete gives 404, and a body with missing fields gives 400 |

## How the HNSW thresholds were chosen

HNSW is approximate, so it cannot be tested for an exact match. It is tested by recall@10 against brute force instead. The threshold was measured first and then set below the worst value seen, so the test does not pass only because the bar was set low.

Setup, the same as in the test: 500 random vectors from a standard normal distribution, 16 dimensions, 50 random queries, `M=16`, `ef_construction=200` (the values `app.py` uses), search with `ef=50`, seeds 0 to 9. Measured on 2026-10-19:

| Seed | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| Euclidean | 1.000 | 1.000 | 0.998 | 1.000 | 0.998 | 1.000 | 0.998 | 0.998 | 0.998 | 0.996 |
| Cosine | 0.996 | 0.980 | 0.980 | 0.984 | 0.988 | 0.990 | 0.992 | 0.992 | 0.992 | 0.994 |

The worst seed was 0.996 for Euclidean and 0.980 for cosine. I set the floors 0.05 below those, rounded down: **0.95** for Euclidean and **0.93** for cosine. The margin leaves room for small differences between machines and library versions, while a real bug (for example, broken edges) would drop recall far below it.

Cosine recall is a little lower because the graph is always built with squared Euclidean distance (`HNSW._sq_dist`), while the search uses whatever distance function it is given. For cosine that means the graph links neighbours by one measure and the search ranks them by another.

To repeat the measurement:

```bash
python -c "import sys; sys.path[:0] = ['.', 'tests']; from test_algorithms import hnsw_recall; from database import euclidean_distance as e, cosine_distance as c; print('euclidean', [hnsw_recall(s, e) for s in range(10)]); print('cosine', [hnsw_recall(s, c) for s in range(10)])"
```

## KD-Tree and cosine distance

KD-Tree skips the far side of a split when the distance from the query to the splitting plane is larger than the current k-th best distance. That shortcut is safe only if the true distance to any point on the far side is at least some bound computed from the gap to the plane. For Euclidean and Manhattan distance the gap itself is that bound, because both distances are at least as large as the difference in any one coordinate. Cosine distance measures the angle between vectors and ignores their length. On raw vectors it can be small even when one coordinate differs a lot, so the gap is not a bound.

Before the fix, the KD-Tree used the gap as the bound for cosine too. On random 8D data (5 seeds, 20 queries each, k=10), its cosine results differed from brute force on 97 of 100 queries, while Euclidean and Manhattan matched on all 100.

The fix uses one identity. For two vectors of length 1, cosine distance equals half the squared Euclidean distance:

```
|a - b|^2 = |a|^2 + |b|^2 - 2 a.b = 2 - 2 cos(a, b)   so   1 - cos(a, b) = |a - b|^2 / 2
```

So a cosine tree (`KDTree(dim, normalize=True)`) scales every stored vector and the query to length 1 before building and searching. The Euclidean distance between two unit vectors is at least the gap to the plane, so their cosine distance is at least half the squared gap, and that is the bound the cosine tree prunes with. Scaling does not change cosine distance, so the results are the same as on the raw vectors. Zero vectors cannot be scaled and stay zero. The bound still holds for them, because a zero vector is at cosine distance 1.0 from everything and the bound is at most 0.5 for any zero vector on the far side.

`VectorDB` keeps two KD-Trees: the original one for Euclidean and Manhattan, and a cosine tree that it uses when the metric is cosine. Both are rebuilt on every insert and delete. Before the fix, `k=0` also crashed the KD-Tree search; it now returns an empty list.

## Terms

- **Nearest neighbours:** the stored vectors with the smallest distance to the query.
- **Ground truth:** the answer known to be correct. Here it is brute force, because it checks every vector, and `test_brute_force_matches_numpy_sort` confirms it against NumPy.
- **Exact vs approximate search:** exact search (brute force, KD-Tree) always returns the true nearest neighbours. Approximate search (HNSW) can miss some in exchange for checking far fewer vectors.
- **recall@k:** of the true k nearest neighbours, the share the algorithm returned. If brute force says the 10 nearest are A to J and HNSW returns 9 of them, recall@10 is 0.9. Averaged over all queries.
- **Seed:** the starting value for a random number generator. The same seed gives the same "random" numbers every time, so a test that uses random data gives the same result on every run. The tests seed both NumPy (for the data) and Python's `random` module (HNSW uses it to pick each node's top layer).
- **Threshold (floor):** the lowest recall the test accepts.
- **Latency:** how long one query takes, from call to result.
- **Percentile:** the value below which a given share of measurements fall. The 50th percentile (the median) is the middle value: half the queries were faster and half slower. The median is less affected by a few unusually slow runs than the average is.
- **Median latency:** the 50th percentile of the per-query times. `scripts/benchmark.py` reports it for each algorithm (see [BENCHMARK.md](BENCHMARK.md)).
- **Build time:** how long it takes to create an index before the first query. KD-Tree sorts the points once. HNSW inserts them one by one, running a beam search for each.
- **ef:** the beam width of an HNSW search, meaning how many candidates it keeps while walking the graph. A larger ef checks more nodes, so recall goes up and latency goes up.
- **Synthetic data:** vectors generated at random by the script, not taken from a real dataset. They are easy to reproduce with a seed, but they have no clusters, so results on real embeddings can differ.

## Interview questions

1. Why is brute force a fair ground truth for testing HNSW, and what would you do if brute force itself had a bug?
2. Why does the KD-Tree pruning rule work for Euclidean distance but not for cosine distance, and how does scaling the vectors to length 1 fix it?
3. Why test HNSW with a recall threshold instead of exact equality, and how would you pick that threshold?
4. What could make a seeded HNSW test still give different results on another machine?
5. Why compare PCA results "up to sign", and what would go wrong if you did not?
