"""Benchmark BruteForce, KDTree and HNSW on synthetic seeded vectors.

For each (N, d) it builds all three indexes, runs the same queries on each,
and reports recall@10 against brute force and the median query latency.
Results go to docs/BENCHMARK.md. The data is random (standard normal), not
real embeddings.

Usage:
    python scripts/benchmark.py --sizes 1000 10000 --dims 16 128
"""

import argparse
import datetime
import os
import platform
import random
import statistics
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from algorithms import BruteForce, HNSW, KDTree  # noqa: E402
from database import euclidean_distance  # noqa: E402

K = 10


def timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, (time.perf_counter() - t0) * 1000


def run(n, dim, queries, seed):
    random.seed(seed)  # HNSW draws node levels from Python's global random
    rng = np.random.default_rng(seed)
    points = list(enumerate(rng.standard_normal((n, dim)).tolist()))
    qs = rng.standard_normal((queries, dim)).tolist()

    brute = BruteForce()
    kdtree = KDTree(dim)
    hnsw = HNSW(dim, M=16, ef_construction=200)  # same settings as app.py

    _, kd_build = timed(lambda: kdtree.build(points))
    _, hnsw_build = timed(lambda: [hnsw.insert(i, v) for i, v in points])

    search = {
        "BruteForce": lambda q: brute.search(q, points, K, euclidean_distance),
        "KDTree": lambda q: kdtree.search(q, K, euclidean_distance),
        "HNSW": lambda q: hnsw.search(q, K, euclidean_distance, ef=50),
    }
    build_ms = {"BruteForce": 0.0, "KDTree": kd_build, "HNSW": hnsw_build}

    truth = [{i for i, _ in search["BruteForce"](q)} for q in qs]
    rows = []
    for name, fn in search.items():
        latencies, hits = [], 0
        for q, true_ids in zip(qs, truth):
            res, ms = timed(lambda: fn(q))
            latencies.append(ms)
            hits += len(true_ids & {i for i, _ in res})
        rows.append((n, dim, name, hits / (queries * K),
                     statistics.median(latencies), build_ms[name] / 1000))
        print(f"N={n} d={dim} {name}: recall@10={rows[-1][3]:.3f} "
              f"median={rows[-1][4]:.2f} ms build={rows[-1][5]:.2f} s", flush=True)
    return rows


def write_report(path, rows, args, started):
    cmd = "python scripts/benchmark.py " + " ".join(sys.argv[1:])
    lines = [
        "# Benchmark",
        "",
        "Measured with `scripts/benchmark.py`. The vectors are synthetic: random "
        "draws from a standard normal distribution, not real embeddings. Results "
        "on real data, other settings or another machine will differ.",
        "",
        f"- Date: {started:%Y-%m-%d}",
        f"- Machine: {platform.platform()}, {platform.processor() or platform.machine()}, "
        f"{os.cpu_count()} logical CPUs",
        f"- Python: {platform.python_version()}, NumPy {np.__version__}",
        f"- Command: `{cmd}`",
        f"- Setup: {args.queries} queries per configuration, seed {args.seed}, "
        "k=10, Euclidean distance, HNSW with M=16, ef_construction=200 (as in "
        "`app.py`) and search ef=50",
        "",
        "Recall@10 is the share of the true 10 nearest neighbours (from brute force) "
        "that each algorithm returned, averaged over the queries. Brute force is "
        "the ground truth, so its recall is 1.000 by definition. Median latency is "
        "the middle value of the per-query search times. Build time is the time to "
        "build the index once; brute force has no index.",
        "",
        "| N | d | Algorithm | Recall@10 | Median latency (ms) | Build time (s) |",
        "|---|---|---|---|---|---|",
    ]
    for n, d, name, recall, median, build in rows:
        lines.append(f"| {n:,} | {d} | {name} | {recall:.3f} | {median:.2f} | {build:.2f} |")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--sizes", type=int, nargs="+", default=[1000, 10000])
    p.add_argument("--dims", type=int, nargs="+", default=[16, 128])
    p.add_argument("--queries", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=os.path.join(ROOT, "docs", "BENCHMARK.md"))
    args = p.parse_args()

    started = datetime.datetime.now()
    rows = []
    for n in args.sizes:
        for d in args.dims:
            rows += run(n, d, args.queries, args.seed)
    write_report(args.out, rows, args, started)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
