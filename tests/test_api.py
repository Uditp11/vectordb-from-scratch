"""FastAPI routes that do not need Ollama, called through TestClient.

app.py builds the demo VectorDB at import time and only contacts Ollama inside
the /doc/* and /status routes, so those routes are left out here.
"""

import pytest
from fastapi.testclient import TestClient

from app import app
from database import DEMO_VECTORS

client = TestClient(app)
QUERY = ",".join(str(x) for x in DEMO_VECTORS[0]["vector"])


def test_stats():
    body = client.get("/stats").json()
    assert body["dimensions"] == 16
    assert body["total_items"] == len(DEMO_VECTORS)


def test_items_include_pca_coords():
    body = client.get("/items").json()
    assert len(body["items"]) == len(body["pca_coords"]) == len(DEMO_VECTORS)


@pytest.mark.parametrize("algo", ["brute", "kdtree", "hnsw"])
def test_search_finds_the_query_vector_itself(algo):
    params = {"v": QUERY, "k": 3, "metric": "euclidean", "algo": algo}
    r = client.get("/search", params=params)
    assert r.status_code == 200
    assert r.json()["results"][0]["id"] == DEMO_VECTORS[0]["id"]


def test_search_rejects_bad_input():
    assert client.get("/search", params={"v": "a,b"}).status_code == 400
    assert client.get("/search", params={"v": QUERY, "algo": "nope"}).status_code == 400


def test_benchmark_runs_all_three():
    body = client.get("/benchmark", params={"v": QUERY, "k": 3}).json()
    assert set(body) == {"brute", "kdtree", "hnsw"}


def test_hnsw_info():
    assert client.get("/hnsw-info").json()["total_nodes"] == len(DEMO_VECTORS)


def test_insert_then_delete():
    item = {"id": "t1", "label": "Test", "category": "CS", "vector": [0.5] * 16}
    assert client.post("/insert", json=item).status_code == 200
    assert client.get("/stats").json()["total_items"] == len(DEMO_VECTORS) + 1
    assert client.delete("/delete/t1").status_code == 200
    assert client.delete("/delete/t1").status_code == 404
    assert client.post("/insert", json={"id": "x"}).status_code == 400
