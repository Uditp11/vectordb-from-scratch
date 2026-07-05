"""
algorithms.py — From-scratch implementations of vector search algorithms.

Implements three search structures:
  1. BruteForce  — Linear scan (exact nearest neighbor)
  2. KDTree      — KD-Tree with hyperrectangle pruning
  3. HNSW        — Hierarchical Navigable Small World graph

No external vector-DB libraries are used; only Python stdlib + math.
"""

import math
import random
import heapq
from collections import defaultdict


# ──────────────────────────────────────────────────────────────────────
# 1. BruteForce
# ──────────────────────────────────────────────────────────────────────

class BruteForce:
    """
    Exact k-NN via exhaustive linear scan.

    This is the simplest possible approach: compute the distance from the
    query to every stored vector, then return the k closest.  O(n·d) per
    query — impractical for large datasets but useful as a correctness
    baseline.
    """

    def __init__(self):
        pass

    def search(self, query, vectors, k, distance_fn):
        """
        Parameters
        ----------
        query : list[float]
            The query vector.
        vectors : list[tuple[any, list[float]]]
            Each element is (id, vector).
        k : int
            Number of nearest neighbors to return.
        distance_fn : callable(a, b) -> float
            Distance function (lower = more similar).

        Returns
        -------
        list[tuple[any, float]]
            Up to k results as (id, distance), sorted ascending by distance.
        """
        if not vectors:
            return []

        # Compute distance to every vector
        distances = []
        for vec_id, vec in vectors:
            dist = distance_fn(query, vec)
            distances.append((vec_id, dist))

        # Sort by distance and return top-k
        distances.sort(key=lambda x: x[1])
        return distances[:k]


# ──────────────────────────────────────────────────────────────────────
# 2. KDTree
# ──────────────────────────────────────────────────────────────────────

class _KDNode:
    """Internal node for the KD-Tree."""
    __slots__ = ('id', 'point', 'left', 'right', 'split_dim')

    def __init__(self, id, point, split_dim, left=None, right=None):
        self.id = id
        self.point = point
        self.split_dim = split_dim
        self.left = left
        self.right = right


class KDTree:
    """
    KD-Tree for k-nearest-neighbor search in arbitrary dimensions.

    The tree cycles through dimensions for the splitting axis and uses
    the median element as the pivot at each level.  Search uses
    hyperrectangle pruning to skip subtrees whose closest possible point
    is farther than the current k-th nearest neighbor.

    Insertions and deletions trigger a full rebuild — this keeps the tree
    balanced and the implementation simple.
    """

    def __init__(self, dim):
        """
        Parameters
        ----------
        dim : int
            Dimensionality of the vectors.
        """
        self.dim = dim
        self.root = None
        self.points = []  # list of (id, vector) for rebuilds

    # ── Build / Rebuild ──────────────────────────────────────────────

    def build(self, points):
        """
        Build (or rebuild) the KD-Tree from a list of (id, vector) pairs.
        """
        self.points = list(points)
        self.root = self._build(list(range(len(self.points))), depth=0)

    def _build(self, indices, depth):
        """Recursively partition *indices* into a balanced KD-Tree."""
        if not indices:
            return None

        # Cycle through dimensions
        split_dim = depth % self.dim

        # Sort indices by the splitting dimension and pick the median
        indices.sort(key=lambda i: self.points[i][1][split_dim])
        median = len(indices) // 2

        idx = indices[median]
        node = _KDNode(
            id=self.points[idx][0],
            point=self.points[idx][1],
            split_dim=split_dim,
            left=self._build(indices[:median], depth + 1),
            right=self._build(indices[median + 1:], depth + 1),
        )
        return node

    # ── Insert / Delete (rebuild-based) ──────────────────────────────

    def insert(self, id, vector):
        """Insert a new point and rebuild the tree."""
        self.points.append((id, vector))
        self.build(self.points)

    def delete(self, id):
        """Remove the point with the given id and rebuild."""
        self.points = [(pid, vec) for pid, vec in self.points if pid != id]
        self.build(self.points)

    # ── k-NN Search ──────────────────────────────────────────────────

    def search(self, query, k, distance_fn):
        """
        Find the k nearest neighbors to *query*.

        Uses a max-heap of size k so that the farthest current neighbor
        can be compared against the splitting plane to decide whether the
        opposite subtree needs visiting (hyperrectangle pruning).

        Returns
        -------
        list[tuple[any, float]]
            Up to k results as (id, distance), sorted ascending.
        """
        if self.root is None:
            return []

        # Max-heap (negate distances so heapq gives us the *farthest*)
        # Each element: (-distance, id)
        best = []

        def _search(node):
            if node is None:
                return

            dist = distance_fn(query, node.point)

            # If we have fewer than k neighbors, always push;
            # otherwise push only if this point is closer than the farthest.
            if len(best) < k:
                heapq.heappush(best, (-dist, node.id))
            elif dist < -best[0][0]:
                heapq.heapreplace(best, (-dist, node.id))

            # Decide which subtree to visit first (the one on the same
            # side as the query along the splitting dimension).
            diff = query[node.split_dim] - node.point[node.split_dim]
            if diff <= 0:
                first, second = node.left, node.right
            else:
                first, second = node.right, node.left

            _search(first)

            # Prune: only visit the other subtree if the splitting plane
            # is closer than the farthest current neighbor.
            if len(best) < k or abs(diff) < -best[0][0]:
                _search(second)

        _search(self.root)

        # Convert max-heap to sorted ascending list
        results = [(_id, -neg_dist) for neg_dist, _id in best]
        results.sort(key=lambda x: x[1])
        return results


# ──────────────────────────────────────────────────────────────────────
# 3. HNSW  (Hierarchical Navigable Small World)
# ──────────────────────────────────────────────────────────────────────

class HNSW:
    """
    Hierarchical Navigable Small World graph for approximate k-NN.

    This is a faithful implementation of the algorithm described in
    Malkov & Yashunin (2018).  Key ideas:

    * Multiple layers of proximity graphs; higher layers are sparser and
      act as "express lanes".
    * Insert assigns a random level to each new node (exponentially
      distributed).  The node is connected to its nearest neighbors in
      every layer from its level down to layer 0.
    * Search starts at the entry point on the highest layer and greedily
      descends, then performs a beam search (ef-width) on layer 0.

    Parameters
    ----------
    dim : int
        Vector dimensionality.
    M : int
        Max edges per node per layer (M_max).  Layer 0 allows 2·M.
    ef_construction : int
        Beam width used during insertion.
    max_level_mult : float
        Multiplier for the random level formula: floor(-ln(uniform) * mL).
    """

    def __init__(self, dim, M=16, ef_construction=200,
                 max_level_mult=1.0 / math.log(2.0)):
        self.dim = dim
        self.M = M
        self.M0 = 2 * M          # max edges on layer 0
        self.ef_construction = ef_construction
        self.ml = max_level_mult

        # Storage ----------------------------------------------------------
        # vectors[id]  -> list[float]
        self.vectors = {}
        # graphs[layer][id] -> set of neighbor ids
        self.graphs = defaultdict(lambda: defaultdict(set))
        # node_level[id] -> highest layer this node lives on
        self.node_level = {}

        self.entry_point = None
        self.max_layer = -1

    # ── Helpers ──────────────────────────────────────────────────────

    def _random_level(self):
        """
        Draw a random layer for a new node.

        The distribution is geometric: most nodes only appear on layer 0,
        and the probability of appearing on layer L decreases exponentially.
        """
        r = random.random()
        # Avoid log(0)
        if r == 0:
            r = 1e-10
        return int(-math.log(r) * self.ml)

    def _search_layer(self, query, entry_point, ef, layer, distance_fn):
        """
        Beam search within a single layer of the HNSW graph.

        Parameters
        ----------
        query : list[float]
            Query vector.
        entry_point : any
            Starting node id.
        ef : int
            Beam width (number of candidates to track).
        layer : int
            Which layer to search.
        distance_fn : callable

        Returns
        -------
        list[tuple[float, any]]
            Found neighbors as (distance, id), length up to ef, sorted ascending.
        """
        ep_dist = distance_fn(query, self.vectors[entry_point])
        # candidates: min-heap of (distance, id) — closest first
        candidates = [(ep_dist, entry_point)]
        # visited set
        visited = {entry_point}
        # result: max-heap (negate distance) — farthest first
        results = [(-ep_dist, entry_point)]

        while candidates:
            c_dist, c_id = heapq.heappop(candidates)

            # If the closest candidate is farther than the farthest result,
            # we can stop — no unvisited node can improve the result set.
            farthest_dist = -results[0][0]
            if c_dist > farthest_dist:
                break

            # Expand neighbors of c_id on this layer
            for neighbor_id in self.graphs[layer].get(c_id, set()):
                if neighbor_id in visited:
                    continue
                visited.add(neighbor_id)

                n_dist = distance_fn(query, self.vectors[neighbor_id])
                farthest_dist = -results[0][0]

                if n_dist < farthest_dist or len(results) < ef:
                    heapq.heappush(candidates, (n_dist, neighbor_id))
                    heapq.heappush(results, (-n_dist, neighbor_id))
                    if len(results) > ef:
                        heapq.heappop(results)  # drop the farthest

        # Convert results to ascending list
        out = [(- neg_d, nid) for neg_d, nid in results]
        out.sort(key=lambda x: x[0])
        return out

    @staticmethod
    def _select_neighbors(candidates, M):
        """
        Simple neighbor selection: keep the M closest candidates.

        Parameters
        ----------
        candidates : list[tuple[float, any]]
            (distance, id) pairs.
        M : int
            Max neighbors to keep.

        Returns
        -------
        list[tuple[float, any]]
            The M closest, sorted ascending.
        """
        candidates.sort(key=lambda x: x[0])
        return candidates[:M]

    # ── Insert ───────────────────────────────────────────────────────

    def insert(self, id, vector):
        """
        Insert a new node into the HNSW index.

        Steps:
        1. Assign a random level L to the new node.
        2. Starting from the entry point on the top layer, greedily
           descend until reaching layer L+1 (single nearest neighbor
           at each layer).
        3. From layer L down to 0, perform a beam search (width =
           ef_construction), select up to M neighbors, and create
           bidirectional edges.
        4. If L > max_layer, update the entry point.
        """
        self.vectors[id] = list(vector)
        level = self._random_level()
        self.node_level[id] = level

        # First node ever inserted
        if self.entry_point is None:
            self.entry_point = id
            self.max_layer = level
            # Ensure the node exists in each of its layers
            for lyr in range(level + 1):
                self.graphs[lyr][id]  # creates empty set via defaultdict
            return

        # ------ Phase 1: greedy descent from top to level+1 -----------
        ep = self.entry_point
        for lyr in range(self.max_layer, level, -1):
            # If the entry point doesn't exist on this layer, skip
            if ep not in self.graphs[lyr]:
                continue
            # Greedy walk: always move to the closest neighbor
            changed = True
            while changed:
                changed = False
                for neighbor_id in self.graphs[lyr].get(ep, set()):
                    if (self._sq_dist(vector, self.vectors[neighbor_id]) <
                            self._sq_dist(vector, self.vectors[ep])):
                        ep = neighbor_id
                        changed = True

        # ------ Phase 2: insert into layers level..0 ------------------
        for lyr in range(min(level, self.max_layer), -1, -1):
            # Beam search to find ef_construction nearest neighbors
            results = self._search_layer(
                vector, ep, self.ef_construction, lyr,
                distance_fn=self._sq_dist_query(vector),
            )

            # Select up to M (or M0 on layer 0) neighbors
            M_layer = self.M0 if lyr == 0 else self.M
            neighbors = self._select_neighbors(results, M_layer)

            # Create bidirectional edges
            self.graphs[lyr][id]  # ensure node set exists
            for dist, neighbor_id in neighbors:
                self.graphs[lyr][id].add(neighbor_id)
                self.graphs[lyr][neighbor_id].add(id)

                # Trim neighbor's edges if they exceed M_layer
                if len(self.graphs[lyr][neighbor_id]) > M_layer:
                    # Keep only the M_layer closest
                    scored = [
                        (self._sq_dist(self.vectors[neighbor_id],
                                       self.vectors[nid]), nid)
                        for nid in self.graphs[lyr][neighbor_id]
                    ]
                    scored.sort(key=lambda x: x[0])
                    self.graphs[lyr][neighbor_id] = set(
                        nid for _, nid in scored[:M_layer]
                    )

            # Use the closest found neighbor as entry point for the next layer
            if neighbors:
                ep = neighbors[0][1]

        # Update entry point if new node has a higher level
        if level > self.max_layer:
            self.entry_point = id
            self.max_layer = level

    # ── Delete ───────────────────────────────────────────────────────

    def delete(self, id):
        """
        Remove a node from the HNSW graph.

        Removes the node from every layer it appears in and cleans up
        all edges pointing to it.  If the deleted node was the entry
        point, a new entry point is chosen from the highest populated
        layer.
        """
        if id not in self.vectors:
            return

        level = self.node_level.get(id, 0)

        # Remove edges in every layer
        for lyr in range(level + 1):
            # Remove edges from neighbors → id
            for neighbor_id in list(self.graphs[lyr].get(id, set())):
                self.graphs[lyr][neighbor_id].discard(id)
            # Remove the node's adjacency list
            if id in self.graphs[lyr]:
                del self.graphs[lyr][id]

        del self.vectors[id]
        del self.node_level[id]

        # Handle entry-point removal
        if self.entry_point == id:
            if not self.vectors:
                self.entry_point = None
                self.max_layer = -1
            else:
                # Find the highest layer that still has nodes
                while self.max_layer >= 0 and not self.graphs[self.max_layer]:
                    self.max_layer -= 1
                if self.max_layer >= 0:
                    self.entry_point = next(iter(self.graphs[self.max_layer]))
                else:
                    # Fallback: pick any remaining node
                    self.entry_point = next(iter(self.vectors))
                    self.max_layer = 0

    # ── Search ───────────────────────────────────────────────────────

    def search(self, query, k, distance_fn, ef=50):
        """
        Find the k approximate nearest neighbors of *query*.

        Parameters
        ----------
        query : list[float]
            Query vector.
        k : int
            Number of neighbors to return.
        distance_fn : callable
            Distance function (lower = closer).
        ef : int
            Beam width for the bottom-layer search (higher = more accurate
            but slower).

        Returns
        -------
        list[tuple[any, float]]
            Up to k results as (id, distance), sorted ascending.
        """
        if self.entry_point is None:
            return []

        # Greedy descent from top layer to layer 1
        ep = self.entry_point
        for lyr in range(self.max_layer, 0, -1):
            results = self._search_layer(query, ep, 1, lyr, distance_fn)
            if results:
                ep = results[0][1]

        # Full beam search on layer 0
        results = self._search_layer(query, ep, max(ef, k), 0, distance_fn)

        # Return top-k
        return [(nid, dist) for dist, nid in results[:k]]

    # ── Info ─────────────────────────────────────────────────────────

    def get_info(self):
        """
        Return diagnostic information about the HNSW index.

        Returns
        -------
        dict
            Keys: layers (list of {layer, node_count, edge_count}),
                  total_nodes, total_edges, entry_point, max_layer.
        """
        layers = []
        total_edges = 0
        for lyr in range(self.max_layer + 1):
            node_count = len(self.graphs[lyr])
            edge_count = sum(len(adj) for adj in self.graphs[lyr].values())
            total_edges += edge_count
            layers.append({
                'layer': lyr,
                'node_count': node_count,
                'edge_count': edge_count,
            })

        return {
            'layers': layers,
            'total_nodes': len(self.vectors),
            'total_edges': total_edges,
            'entry_point': self.entry_point,
            'max_layer': self.max_layer,
        }

    # ── Private distance helpers ─────────────────────────────────────

    @staticmethod
    def _sq_dist(a, b):
        """Squared Euclidean distance (avoids sqrt for speed)."""
        return sum((ai - bi) ** 2 for ai, bi in zip(a, b))

    @staticmethod
    def _sq_dist_query(query):
        """Return a closure that computes squared distance to *query*."""
        def fn(a, b):
            return sum((ai - bi) ** 2 for ai, bi in zip(a, b))
        return fn
