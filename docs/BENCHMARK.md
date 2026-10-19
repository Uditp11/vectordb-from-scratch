# Benchmark

Measured with `scripts/benchmark.py`. The vectors are synthetic: random draws from a standard normal distribution, not real embeddings. Results on real data, other settings or another machine will differ.

- Date: 2026-10-19
- Machine: Windows-11-10.0.26200-SP0, Intel64 Family 6 Model 183 Stepping 1, GenuineIntel, 28 logical CPUs
- Python: 3.13.5, NumPy 2.5.4
- Command: `python scripts/benchmark.py --sizes 1000 10000 --dims 16 128 --queries 100`
- Setup: 100 queries per configuration, seed 0, k=10, Euclidean distance, HNSW with M=16, ef_construction=200 (as in `app.py`) and search ef=50

Recall@10 is the share of the true 10 nearest neighbours (from brute force) that each algorithm returned, averaged over the queries. Brute force is the ground truth, so its recall is 1.000 by definition. Median latency is the middle value of the per-query search times. Build time is the time to build the index once; brute force has no index.

| N | d | Algorithm | Recall@10 | Median latency (ms) | Build time (s) |
|---|---|---|---|---|---|
| 1,000 | 16 | BruteForce | 1.000 | 35.17 | 0.00 |
| 1,000 | 16 | KDTree | 1.000 | 40.88 | 0.00 |
| 1,000 | 16 | HNSW | 0.995 | 26.91 | 18.03 |
| 1,000 | 128 | BruteForce | 1.000 | 53.12 | 0.00 |
| 1,000 | 128 | KDTree | 1.000 | 54.81 | 0.02 |
| 1,000 | 128 | HNSW | 0.970 | 38.77 | 160.91 |
| 10,000 | 16 | BruteForce | 1.000 | 289.31 | 0.00 |
| 10,000 | 16 | KDTree | 1.000 | 299.35 | 0.14 |
| 10,000 | 16 | HNSW | 0.975 | 33.48 | 548.82 |
| 10,000 | 128 | BruteForce | 1.000 | 558.03 | 0.00 |
| 10,000 | 128 | KDTree | 1.000 | 582.73 | 0.16 |
| 10,000 | 128 | HNSW | 0.769 | 79.24 | 3325.66 |

## What this run shows

These notes were written by hand after the run above. Rerunning the script rewrites this file without them.

- **HNSW answers queries fastest in every configuration.** At N=10,000 and d=16 its median was 33.48 ms against 289.31 ms for brute force, about 9 times faster.
- **HNSW pays for that at build time.** Insertion is pure Python. Building the index took 548.82 s for 10,000 vectors at d=16 and 3,325.66 s (about 55 minutes) at d=128.
- **HNSW recall falls at high dimension.** At N=10,000 and d=128 recall@10 was 0.769 with ef=50. Random normal data has no clusters, which makes it a hard case for graph search. A larger ef should raise recall at the cost of latency; that was not measured in this run.
- **KD-Tree was exact but not faster than brute force here.** Its recall was 1.000 everywhere, but its median latency was 3% to 16% slower than brute force in every configuration. With 10,000 points the balanced tree is about 14 levels deep, so even at d=16 it splits on each dimension at most once along any path and can rarely skip a subtree. This matches the curse of dimensionality described in the README.
- **The machine was not idle.** Other programs were running during the measurement, so the absolute times are noisy. All three algorithms ran on the same data in the same run, so the comparison between them is fairer than the absolute values.

d=768 (the size of the `nomic-embed-text` embeddings) was not run: the HNSW build at d=128 already took about 55 minutes for 10,000 vectors, and 768 dimensions would take several hours.
