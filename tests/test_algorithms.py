"""Correctness tests for the search algorithms and distance functions.

CPU only. No Ollama, no network. See docs/TESTING.md for what each test proves.
"""

import functools
import math
import random

import numpy as np
import pytest

from algorithms import BruteForce, HNSW, KDTree
from database import cosine_distance, euclidean_distance, manhattan_distance

K = 10


def random_points(seed, n, dim):
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n, dim))
    queries = rng.standard_normal((20, dim))
    return list(enumerate(data.tolist())), queries.tolist()


def brute_ids(query, points, k, fn):
    return [i for i, _ in BruteForce().search(query, points, k, fn)]


# Distance functions, checked against values worked out by hand.

def test_distances_on_hand_checked_values():
    a, b = [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]
    assert cosine_distance(a, a) == pytest.approx(0.0)
    assert cosine_distance(a, b) == pytest.approx(1.0)
    assert cosine_distance(a, [-1.0, 0.0, 0.0]) == pytest.approx(2.0)
    assert euclidean_distance([0, 0], [3, 4]) == pytest.approx(5.0)
    assert euclidean_distance(a, b) == pytest.approx(math.sqrt(2))
    assert manhattan_distance([0, 0], [3, 4]) == pytest.approx(7.0)
    assert manhattan_distance([1, -2, 3], [-1, 2, 0]) == pytest.approx(9.0)


def test_cosine_of_zero_vector_is_one():
    assert cosine_distance([0.0, 0.0], [1.0, 2.0]) == 1.0


# Brute force, checked against an independent NumPy sort.

@pytest.mark.parametrize("seed", [0, 1, 2])
def test_brute_force_matches_numpy_sort(seed):
    points, queries = random_points(seed, 300, 8)
    data = np.array([v for _, v in points])
    for q in queries:
        expected = np.argsort(np.linalg.norm(data - np.array(q), axis=1))[:K]
        assert brute_ids(q, points, K, euclidean_distance) == expected.tolist()


def test_brute_force_on_empty_input():
    assert BruteForce().search([0.0], [], K, euclidean_distance) == []


# KD-Tree exact search must equal brute force.

@pytest.mark.parametrize("fn", [euclidean_distance, manhattan_distance])
@pytest.mark.parametrize("dim", [2, 3, 8, 16])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_kdtree_equals_brute_force(seed, dim, fn):
    points, queries = random_points(seed, 300, dim)
    tree = KDTree(dim)
    tree.build(points)
    for q in queries:
        assert [i for i, _ in tree.search(q, K, fn)] == brute_ids(q, points, K, fn)


@pytest.mark.parametrize("dim", [2, 3, 8, 16])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_cosine_kdtree_equals_brute_force(seed, dim):
    points, queries = random_points(seed, 300, dim)
    tree = KDTree(dim, normalize=True)
    tree.build(points)
    for q in queries:
        got = [i for i, _ in tree.search(q, K, cosine_distance)]
        assert got == brute_ids(q, points, K, cosine_distance)


def test_cosine_kdtree_with_zero_vectors():
    points, queries = random_points(0, 200, 4)
    points = [(str(i), v) for i, v in points]
    points += [("zero_a", [0.0] * 4), ("zero_b", [0.0] * 4)]
    tree = KDTree(4, normalize=True)
    tree.build(points)
    for q in queries:
        got = [i for i, _ in tree.search(q, 210, cosine_distance)]
        assert got[:K] == brute_ids(q, points, K, cosine_distance)
        assert set(got) == {i for i, _ in points}
    # A zero query is at cosine distance 1.0 from everything.
    dists = [d for _, d in tree.search([0.0] * 4, K, cosine_distance)]
    assert dists == [1.0] * K


def test_kdtree_with_k_zero_returns_nothing():
    points, _ = random_points(0, 50, 3)
    tree = KDTree(3)
    tree.build(points)
    assert tree.search([0.0, 0.0, 0.0], 0, euclidean_distance) == []


def test_kdtree_insert_and_delete_rebuild():
    tree = KDTree(2)
    tree.build([("a", [0.0, 0.0]), ("b", [5.0, 5.0])])
    tree.insert("c", [1.0, 1.0])
    assert tree.search([0.9, 0.9], 1, euclidean_distance)[0][0] == "c"
    tree.delete("c")
    assert tree.search([0.9, 0.9], 1, euclidean_distance)[0][0] == "a"


# HNSW is approximate, so it is checked by recall@10 against brute force.
# The floors come from measurements, not guesses (see docs/TESTING.md).
# Measured on N=500, d=16, 50 queries, seeds 0 to 9, on 2026-10-19:
#   euclidean worst seed 0.996, cosine worst seed 0.980.
HNSW_RECALL_FLOOR = {"euclidean": 0.95, "cosine": 0.93}


@functools.lru_cache(maxsize=None)
def seeded_hnsw(seed, n=500, dim=16, queries=50):
    """Build once per seed; the Euclidean and cosine tests share the index."""
    random.seed(seed)  # HNSW draws node levels from Python's global random
    rng = np.random.default_rng(seed)
    points = list(enumerate(rng.standard_normal((n, dim)).tolist()))
    index = HNSW(dim, M=16, ef_construction=200)
    for i, v in points:
        index.insert(i, v)
    return points, rng.standard_normal((queries, dim)).tolist(), index


def hnsw_recall(seed, fn, n=500):
    points, queries, index = seeded_hnsw(seed, n)
    hits = 0
    for q in queries:
        truth = set(brute_ids(q, points, K, fn))
        got = {i for i, _ in index.search(q, K, fn, ef=50)}
        hits += len(truth & got)
    return hits / (len(queries) * K)


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("name, fn", [("euclidean", euclidean_distance),
                                      ("cosine", cosine_distance)])
def test_hnsw_recall_at_10(name, fn, seed):
    assert hnsw_recall(seed, fn) >= HNSW_RECALL_FLOOR[name]


def test_hnsw_is_deterministic_with_seed():
    first = seeded_hnsw.__wrapped__(5, n=200)[2]
    second = seeded_hnsw.__wrapped__(5, n=200)[2]
    assert first.node_level == second.node_level
    assert dict(first.graphs[0]) == dict(second.graphs[0])


def test_hnsw_empty_index_and_delete():
    index = HNSW(2, M=4, ef_construction=10)
    assert index.search([0.0, 0.0], 3, euclidean_distance) == []
    random.seed(0)
    for i, v in enumerate([[0, 0], [1, 1], [5, 5]]):
        index.insert(i, v)
    index.delete(0)
    ids = [i for i, _ in index.search([0.0, 0.0], 3, euclidean_distance)]
    assert ids == [1, 2]
