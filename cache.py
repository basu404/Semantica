"""
cache.py
--------
A semantic cache built entirely from scratch — no Redis, Memcached, or any
caching middleware. Just Python dicts, numpy, and deliberate design.

CORE IDEA:
A traditional string-keyed cache only hits when queries are *identical*.
We want hits when queries are *semantically equivalent*, e.g.:
    "what is machine learning?" ≈ "explain ML to me"

DESIGN — Data Structure:
We organise the cache as a dict of lists, keyed by dominant cluster:

    {
        cluster_id (int): [
            CacheEntry(query, embedding, result, cluster_weights),
            ...
        ]
    }

WHY cluster-partitioned instead of a flat list?
- A flat list requires O(n) similarity scan for every lookup (n = total entries).
- With cluster partitioning, a new query only searches entries in its dominant
  cluster(s). If the cache has 1000 entries evenly across 15 clusters, that's
  ~67 comparisons instead of 1000 — a 15× speedup that grows with cache size.
- This is where Part 2 (clustering) does *real work* for Part 3 (caching).
- We search the top-2 clusters by query weight to handle boundary cases where
  a query straddles two clusters.

DESIGN — Similarity Metric:
Cosine similarity between L2-normalised embeddings reduces to a dot product,
which is O(dim) and extremely fast. We don't use Euclidean distance because:
    * Cosine is invariant to vector magnitude — "machine learning" and
      "MACHINE LEARNING" embed to vectors of different norms but same direction
    * All embeddings are already L2-normalised, so dot product IS cosine sim

DESIGN — The Similarity Threshold (the key tunable):
The threshold τ determines what counts as a "cache hit".

    τ = 0.95  → Very strict. Only catches near-identical rephrasing.
               False-positive rate ≈ 0%. Miss rate is high.
               Use case: when wrong cached results are very costly.

    τ = 0.85  → Balanced. Catches paraphrases and synonym substitution.
               "What is deep learning?" matches "Explain deep learning".
               Recommended default for a search cache.

    τ = 0.75  → Loose. Catches topically similar but distinct questions.
               "GPU performance" might hit "graphics card benchmarks".
               Risk: returning a result that doesn't match the new query.

    τ = 0.65  → Too loose. Different questions in the same domain collide.
               High false-positive rate. Degrades result quality noticeably.

We default to τ = 0.85. The system exposes this as a constructor argument
so it can be tuned without code changes (the API could accept it as a param).

The interesting insight: lower τ → higher hit rate → lower quality.
This is a precision/recall trade-off on the cache itself.
"""

import numpy as np
import time
from dataclasses import dataclass, field
from typing import Optional
from clustering import FuzzyClustering


DEFAULT_THRESHOLD = 0.85   # Cosine similarity threshold for a cache hit
TOP_K_CLUSTERS = 2         # How many top clusters to search per query


@dataclass
class CacheEntry:
    """One stored query-result pair in the cache."""
    query: str
    embedding: np.ndarray          # shape (dim,), L2-normalised
    result: str                    # The stored result for this query
    cluster_weights: np.ndarray    # shape (n_clusters,), sums to 1
    dominant_cluster: int
    timestamp: float = field(default_factory=time.time)


class SemanticCache:
    """
    Cluster-partitioned semantic cache using cosine similarity for lookup.

    Args:
        clusterer: fitted FuzzyClustering instance (needed to assign new queries)
        threshold: cosine similarity threshold for cache hits (default 0.85)
    """

    def __init__(self, clusterer: FuzzyClustering, threshold: float = DEFAULT_THRESHOLD):
        self.clusterer = clusterer
        self.threshold = threshold

        # Main storage: cluster_id → list of CacheEntry
        # Using a plain dict of lists — no external library
        self._store: dict[int, list[CacheEntry]] = {}

        # Stats tracking
        self._hit_count = 0
        self._miss_count = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def lookup(
        self,
        query: str,
        query_embedding: np.ndarray,
    ) -> Optional[tuple[CacheEntry, float]]:
        """
        Search the cache for a semantically similar prior query.

        Algorithm:
          1. Assign the query to clusters via NMF projection
          2. Collect candidate entries from the top-K dominant clusters
          3. Compute cosine similarity (dot product, since embeddings are normalised)
          4. Return the best match if similarity >= threshold, else None

        Args:
            query: raw query string (used for logging only)
            query_embedding: L2-normalised float32 array, shape (dim,)

        Returns:
            (CacheEntry, similarity_score) on hit, or None on miss
        """
        cluster_weights = self.clusterer.assign_query(query_embedding)

        # Pick the top-K clusters to search
        top_clusters = np.argsort(cluster_weights)[::-1][:TOP_K_CLUSTERS]

        best_entry: Optional[CacheEntry] = None
        best_sim: float = -1.0

        for cluster_id in top_clusters:
            cluster_id = int(cluster_id)
            if cluster_id not in self._store:
                continue

            for entry in self._store[cluster_id]:
                # Dot product of two L2-normalised vectors = cosine similarity
                sim = float(np.dot(query_embedding, entry.embedding))
                if sim > best_sim:
                    best_sim = sim
                    best_entry = entry

        if best_entry is not None and best_sim >= self.threshold:
            self._hit_count += 1
            return best_entry, best_sim

        # Cache miss
        self._miss_count += 1
        return None

    def store(
        self,
        query: str,
        query_embedding: np.ndarray,
        result: str,
    ) -> CacheEntry:
        """
        Store a new query-result pair in the cache.

        The entry is placed in the bucket of its dominant cluster.
        We also store the full cluster_weights for potential future use
        (e.g. multi-cluster indexing, cache analytics).

        Returns the created CacheEntry.
        """
        cluster_weights = self.clusterer.assign_query(query_embedding)
        dominant_cluster = int(np.argmax(cluster_weights))

        entry = CacheEntry(
            query=query,
            embedding=query_embedding.copy(),  # copy to avoid mutation
            result=result,
            cluster_weights=cluster_weights,
            dominant_cluster=dominant_cluster,
        )

        if dominant_cluster not in self._store:
            self._store[dominant_cluster] = []
        self._store[dominant_cluster].append(entry)

        return entry

    def flush(self) -> None:
        """Clear all cache entries and reset stats."""
        self._store.clear()
        self._hit_count = 0
        self._miss_count = 0

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def total_entries(self) -> int:
        return sum(len(entries) for entries in self._store.values())

    @property
    def hit_count(self) -> int:
        return self._hit_count

    @property
    def miss_count(self) -> int:
        return self._miss_count

    @property
    def hit_rate(self) -> float:
        total = self._hit_count + self._miss_count
        return round(self._hit_count / total, 4) if total > 0 else 0.0

    def stats(self) -> dict:
        return {
            "total_entries": self.total_entries,
            "hit_count": self._hit_count,
            "miss_count": self._miss_count,
            "hit_rate": self.hit_rate,
        }

    def cluster_distribution(self) -> dict[int, int]:
        """How many cache entries live in each cluster bucket. Useful for debugging."""
        return {cid: len(entries) for cid, entries in sorted(self._store.items())}
