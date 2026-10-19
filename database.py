"""
database.py — VectorDB and DocumentDB: in-memory vector databases
backed by the from-scratch algorithms in algorithms.py.

Distance functions (cosine, euclidean, manhattan) are also defined here.
"""

import math
import time
import numpy as np

from algorithms import BruteForce, KDTree, HNSW
from pca import pca_2d


# =====================================================================
# Distance Functions
# =====================================================================

def cosine_distance(a, b):
    """
    Cosine distance: 1 − cosine_similarity.

    Returns 0 for identical directions, 2 for opposite directions.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 1.0
    return 1.0 - dot / (norm_a * norm_b)


def euclidean_distance(a, b):
    """L2 (Euclidean) distance."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(a - b))


def manhattan_distance(a, b):
    """L1 (Manhattan / city-block) distance."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.sum(np.abs(a - b)))


def get_distance_fn(name):
    """
    Look up a distance function by string name.

    Supported names: 'cosine', 'euclidean', 'manhattan'.
    """
    fns = {
        'cosine': cosine_distance,
        'euclidean': euclidean_distance,
        'manhattan': manhattan_distance,
    }
    fn = fns.get(name.lower())
    if fn is None:
        raise ValueError(f"Unknown distance metric: {name!r}. "
                         f"Choose from {list(fns.keys())}")
    return fn


# =====================================================================
# Demo Vectors — 20 items × 16 dimensions, 4 categories
# =====================================================================
# Dimensions loosely represent:
#   0-3  : CS-related features
#   4-7  : Math-related features
#   8-11 : Food-related features
#   12-15: Sports-related features
#
# Items in their own category have high values in the matching range,
# with some cross-category signal and noise for realism.

DEMO_VECTORS = [
    # ── CS (dims 0-3 high) ──────────────────────────────────────────
    {
        'id': 'cs_1', 'label': 'Binary Search Tree', 'category': 'CS',
        'vector': [0.92, 0.85, 0.78, 0.88,  # CS core
                   0.55, 0.60, 0.40, 0.35,   # Math overlap
                   0.10, 0.08, 0.12, 0.05,   # Food (low)
                   0.15, 0.12, 0.10, 0.08],  # Sports (low)
    },
    {
        'id': 'cs_2', 'label': 'Neural Network', 'category': 'CS',
        'vector': [0.90, 0.88, 0.82, 0.75,
                   0.70, 0.72, 0.55, 0.50,   # Stronger math overlap
                   0.08, 0.10, 0.06, 0.12,
                   0.10, 0.08, 0.14, 0.06],
    },
    {
        'id': 'cs_3', 'label': 'Hash Table', 'category': 'CS',
        'vector': [0.88, 0.90, 0.85, 0.80,
                   0.40, 0.45, 0.38, 0.30,
                   0.12, 0.05, 0.10, 0.08,
                   0.08, 0.10, 0.06, 0.12],
    },
    {
        'id': 'cs_4', 'label': 'Recursion', 'category': 'CS',
        'vector': [0.85, 0.82, 0.90, 0.78,
                   0.60, 0.55, 0.50, 0.45,
                   0.06, 0.10, 0.08, 0.04,
                   0.12, 0.06, 0.10, 0.08],
    },
    {
        'id': 'cs_5', 'label': 'TCP/IP Protocol', 'category': 'CS',
        'vector': [0.80, 0.78, 0.72, 0.92,
                   0.30, 0.35, 0.28, 0.40,
                   0.10, 0.12, 0.06, 0.08,
                   0.14, 0.10, 0.08, 0.12],
    },

    # ── Math (dims 4-7 high) ────────────────────────────────────────
    {
        'id': 'math_1', 'label': 'Linear Algebra', 'category': 'Math',
        'vector': [0.55, 0.60, 0.50, 0.45,   # CS overlap
                   0.92, 0.88, 0.85, 0.80,    # Math core
                   0.08, 0.10, 0.06, 0.12,
                   0.10, 0.08, 0.12, 0.06],
    },
    {
        'id': 'math_2', 'label': 'Calculus', 'category': 'Math',
        'vector': [0.40, 0.45, 0.38, 0.42,
                   0.90, 0.85, 0.88, 0.82,
                   0.10, 0.06, 0.12, 0.08,
                   0.08, 0.12, 0.06, 0.10],
    },
    {
        'id': 'math_3', 'label': 'Probability', 'category': 'Math',
        'vector': [0.50, 0.55, 0.48, 0.52,
                   0.85, 0.90, 0.80, 0.88,
                   0.12, 0.08, 0.10, 0.06,
                   0.06, 0.10, 0.08, 0.12],
    },
    {
        'id': 'math_4', 'label': 'Graph Theory', 'category': 'Math',
        'vector': [0.62, 0.58, 0.55, 0.50,   # Stronger CS overlap
                   0.88, 0.82, 0.90, 0.78,
                   0.06, 0.12, 0.08, 0.10,
                   0.12, 0.06, 0.10, 0.08],
    },
    {
        'id': 'math_5', 'label': 'Fourier Transform', 'category': 'Math',
        'vector': [0.48, 0.52, 0.45, 0.55,
                   0.82, 0.88, 0.78, 0.92,
                   0.08, 0.10, 0.12, 0.06,
                   0.10, 0.08, 0.06, 0.12],
    },

    # ── Food (dims 8-11 high) ───────────────────────────────────────
    {
        'id': 'food_1', 'label': 'Sushi', 'category': 'Food',
        'vector': [0.08, 0.12, 0.06, 0.10,
                   0.10, 0.08, 0.12, 0.06,
                   0.90, 0.85, 0.88, 0.82,    # Food core
                   0.06, 0.10, 0.08, 0.12],
    },
    {
        'id': 'food_2', 'label': 'Pizza Margherita', 'category': 'Food',
        'vector': [0.10, 0.06, 0.12, 0.08,
                   0.06, 0.12, 0.08, 0.10,
                   0.88, 0.92, 0.80, 0.85,
                   0.12, 0.08, 0.06, 0.10],
    },
    {
        'id': 'food_3', 'label': 'Chocolate Cake', 'category': 'Food',
        'vector': [0.06, 0.10, 0.08, 0.12,
                   0.12, 0.06, 0.10, 0.08,
                   0.85, 0.88, 0.92, 0.80,
                   0.08, 0.12, 0.10, 0.06],
    },
    {
        'id': 'food_4', 'label': 'Caesar Salad', 'category': 'Food',
        'vector': [0.12, 0.08, 0.10, 0.06,
                   0.08, 0.10, 0.06, 0.12,
                   0.82, 0.80, 0.85, 0.92,
                   0.10, 0.06, 0.12, 0.08],
    },
    {
        'id': 'food_5', 'label': 'Espresso', 'category': 'Food',
        'vector': [0.10, 0.08, 0.06, 0.12,
                   0.06, 0.10, 0.08, 0.12,
                   0.80, 0.82, 0.78, 0.88,
                   0.12, 0.10, 0.08, 0.06],
    },

    # ── Sports (dims 12-15 high) ────────────────────────────────────
    {
        'id': 'sport_1', 'label': 'Basketball', 'category': 'Sports',
        'vector': [0.12, 0.10, 0.08, 0.06,
                   0.08, 0.06, 0.10, 0.12,
                   0.06, 0.12, 0.10, 0.08,
                   0.92, 0.88, 0.85, 0.80],   # Sports core
    },
    {
        'id': 'sport_2', 'label': 'Swimming', 'category': 'Sports',
        'vector': [0.06, 0.12, 0.10, 0.08,
                   0.10, 0.08, 0.06, 0.12,
                   0.12, 0.06, 0.08, 0.10,
                   0.88, 0.85, 0.90, 0.82],
    },
    {
        'id': 'sport_3', 'label': 'Chess', 'category': 'Sports',
        'vector': [0.35, 0.30, 0.28, 0.32,    # Mild CS/Math overlap
                   0.40, 0.38, 0.35, 0.42,
                   0.08, 0.10, 0.06, 0.12,
                   0.82, 0.85, 0.80, 0.88],
    },
    {
        'id': 'sport_4', 'label': 'Marathon Running', 'category': 'Sports',
        'vector': [0.08, 0.06, 0.12, 0.10,
                   0.06, 0.10, 0.12, 0.08,
                   0.15, 0.18, 0.12, 0.20,    # Slight food overlap
                   0.85, 0.90, 0.82, 0.88],
    },
    {
        'id': 'sport_5', 'label': 'Rock Climbing', 'category': 'Sports',
        'vector': [0.10, 0.08, 0.06, 0.12,
                   0.12, 0.06, 0.08, 0.10,
                   0.10, 0.12, 0.08, 0.06,
                   0.90, 0.82, 0.88, 0.85],
    },
]


# =====================================================================
# VectorDB — Main demo vector database
# =====================================================================

class VectorDB:
    """
    In-memory vector database backed by three simultaneous indices:
    BruteForce, KDTree, and HNSW.

    Pre-loaded with 20 demo vectors across 4 categories.
    """

    DIM = 16

    def __init__(self):
        # Storage: id -> {id, label, category, vector}
        self.items = {}

        # Indices
        self.brute = BruteForce()
        self.kdtree = KDTree(self.DIM)                         # Euclidean, Manhattan
        self.kdtree_cosine = KDTree(self.DIM, normalize=True)  # cosine
        self.hnsw = HNSW(self.DIM, M=16, ef_construction=200)

        # Load demo data
        for item in DEMO_VECTORS:
            self.items[item['id']] = dict(item)
            self.hnsw.insert(item['id'], item['vector'])

        # Build the KD-Trees once with all demo points
        self._rebuild_kdtree()

    # ── CRUD ─────────────────────────────────────────────────────────

    def insert(self, id, label, category, vector):
        """Insert a new vector (updates all three indices)."""
        self.items[id] = {
            'id': id,
            'label': label,
            'category': category,
            'vector': list(vector),
        }
        self.hnsw.insert(id, vector)
        self._rebuild_kdtree()

    def delete(self, id):
        """Delete a vector by id (updates all three indices)."""
        if id not in self.items:
            return False
        del self.items[id]
        self.hnsw.delete(id)
        self._rebuild_kdtree()
        return True

    # ── Search ───────────────────────────────────────────────────────

    def search(self, query, k=5, metric='cosine', algo='hnsw'):
        """
        Search for the k nearest neighbors of *query*.

        Parameters
        ----------
        query : list[float]
        k : int
        metric : str   — 'cosine', 'euclidean', or 'manhattan'
        algo : str     — 'brute', 'kdtree', or 'hnsw'

        Returns
        -------
        list[dict]
            Each dict: {id, label, category, distance}.
        """
        dist_fn = get_distance_fn(metric)

        if algo == 'brute':
            vectors = [(item['id'], item['vector'])
                       for item in self.items.values()]
            raw = self.brute.search(query, vectors, k, dist_fn)
        elif algo == 'kdtree':
            tree = self.kdtree_cosine if dist_fn is cosine_distance else self.kdtree
            raw = tree.search(query, k, dist_fn)
        elif algo == 'hnsw':
            raw = self.hnsw.search(query, k, dist_fn, ef=max(50, k * 2))
        else:
            raise ValueError(f"Unknown algorithm: {algo!r}")

        results = []
        for vec_id, dist in raw:
            if vec_id in self.items:
                item = self.items[vec_id]
                results.append({
                    'id': item['id'],
                    'label': item['label'],
                    'category': item['category'],
                    'distance': round(float(dist), 6),
                })
        return results

    # ── Benchmark ────────────────────────────────────────────────────

    def benchmark(self, query, k=5, metric='cosine'):
        """
        Run the same search on all three algorithms and return timing.

        Returns
        -------
        dict
            {brute: {time_ms, results}, kdtree: {...}, hnsw: {...}}
        """
        output = {}
        for algo in ['brute', 'kdtree', 'hnsw']:
            t0 = time.perf_counter()
            results = self.search(query, k, metric, algo)
            elapsed = (time.perf_counter() - t0) * 1000  # ms
            output[algo] = {
                'time_ms': round(elapsed, 4),
                'results': results,
            }
        return output

    # ── Getters ──────────────────────────────────────────────────────

    def get_items(self):
        """Return all stored items (without full vectors for brevity)."""
        return [
            {
                'id': item['id'],
                'label': item['label'],
                'category': item['category'],
                'vector': item['vector'],
            }
            for item in self.items.values()
        ]

    def get_stats(self):
        """Return summary statistics."""
        categories = {}
        for item in self.items.values():
            cat = item['category']
            categories[cat] = categories.get(cat, 0) + 1
        return {
            'total_items': len(self.items),
            'dimensions': self.DIM,
            'categories': categories,
            'algorithms': ['brute', 'kdtree', 'hnsw'],
        }

    def get_hnsw_info(self):
        """Return HNSW graph diagnostics."""
        return self.hnsw.get_info()

    def get_pca_coords(self):
        """Project all stored vectors to 2D for visualization."""
        ids = []
        vectors = []
        for item in self.items.values():
            ids.append(item['id'])
            vectors.append(item['vector'])
        if not vectors:
            return []
        coords = pca_2d(vectors)
        return [
            {'id': ids[i], 'x': coords[i][0], 'y': coords[i][1]}
            for i in range(len(ids))
        ]

    # ── Internal ─────────────────────────────────────────────────────

    def _rebuild_kdtree(self):
        """Rebuild both KD-Trees from the current items."""
        points = [(item['id'], item['vector'])
                  for item in self.items.values()]
        self.kdtree.build(points)
        self.kdtree_cosine.build(points)


# =====================================================================
# DocumentDB — For RAG with 768D Ollama embeddings
# =====================================================================

class DocumentDB:
    """
    Document store backed by an HNSW index for 768-dimensional
    embeddings (from Ollama's nomic-embed-text or similar model).

    Each document is split into chunks, embedded, and stored for
    semantic retrieval.
    """

    DIM = 768

    def __init__(self):
        # Storage: id -> {id, title, text, chunk_index, embedding}
        self.docs = {}
        self.hnsw = HNSW(self.DIM, M=16, ef_construction=200)
        self._next_id = 0

    # ── Insert ───────────────────────────────────────────────────────

    def insert_chunks(self, title, chunks, embeddings):
        """
        Insert pre-chunked and pre-embedded document pieces.

        Parameters
        ----------
        title : str
            Document title.
        chunks : list[str]
            Text chunks.
        embeddings : list[list[float]]
            Corresponding embedding for each chunk.
        """
        inserted_ids = []
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            doc_id = f"doc_{self._next_id}"
            self._next_id += 1
            self.docs[doc_id] = {
                'id': doc_id,
                'title': title,
                'text': chunk,
                'chunk_index': i,
                'embedding': emb,
            }
            self.hnsw.insert(doc_id, emb)
            inserted_ids.append(doc_id)
        return inserted_ids

    # ── Delete ───────────────────────────────────────────────────────

    def delete(self, id):
        """Delete a single document chunk by id."""
        if id not in self.docs:
            return False
        self.hnsw.delete(id)
        del self.docs[id]
        return True

    # ── Search ───────────────────────────────────────────────────────

    def search(self, query_embedding, k=5):
        """
        Find the k most relevant document chunks for the given embedding.

        Uses cosine distance for semantic similarity.
        """
        results = self.hnsw.search(
            query_embedding, k, cosine_distance, ef=max(50, k * 2)
        )
        output = []
        for doc_id, dist in results:
            if doc_id in self.docs:
                doc = self.docs[doc_id]
                output.append({
                    'id': doc['id'],
                    'title': doc['title'],
                    'text': doc['text'],
                    'chunk_index': doc['chunk_index'],
                    'distance': round(float(dist), 6),
                })
        return output

    # ── Getters ──────────────────────────────────────────────────────

    def get_docs(self):
        """Return all stored document chunks (without embeddings)."""
        return [
            {
                'id': doc['id'],
                'title': doc['title'],
                'text': doc['text'],
                'chunk_index': doc['chunk_index'],
            }
            for doc in self.docs.values()
        ]

    def get_count(self):
        """Return the number of stored chunks."""
        return len(self.docs)


# =====================================================================
# Chunking Helper
# =====================================================================

def chunk_text(text, chunk_size=250, overlap=50):
    """
    Split *text* into overlapping chunks by word count.

    Parameters
    ----------
    text : str
        The full document text.
    chunk_size : int
        Maximum number of words per chunk.
    overlap : int
        Number of overlapping words between consecutive chunks.

    Returns
    -------
    list[str]
        The text chunks.
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)
        # Advance by (chunk_size - overlap) words
        start += chunk_size - overlap

    return chunks
