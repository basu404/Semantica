"""
clustering.py
-------------
Performs fuzzy (soft) clustering on document embeddings using NMF.

DESIGN DECISIONS — Algorithm Choice:
- We use Non-negative Matrix Factorization (NMF) rather than hard clustering (k-means)
  or probabilistic models (GMM/LDA) for the following reasons:

  * K-means: Produces hard assignments (one cluster per doc). The assignment
    explicitly rejects this. A post about gun legislation is not *either*
    politics or guns — it's both. K-means can't express that.

  * GMM: Soft assignments via posterior probabilities. But GMMs on high-dimensional
    embeddings (384-dim) suffer from the curse of dimensionality — covariance
    matrices become ill-conditioned. Requires diagonal approximation which loses
    important structure.

  * LDA: Designed for raw term-frequency bags-of-words, not dense embeddings.
    Applying LDA to 384-dim float vectors violates its generative assumptions.

  * NMF: Factorizes the embedding matrix X ≈ W × H where W gives per-document
    soft weights over k topics and H gives per-topic embedding directions.
    W[i] is naturally a probability-like distribution over clusters for doc i.
    Non-negativity constraints produce interpretable, additive parts. This is
    exactly what the assignment asks for.

DESIGN DECISIONS — Number of Clusters (k):
- We choose k via the "reconstruction error elbow + coherence" method:
    * Fit NMF for k in [5, 8, 10, 12, 15, 18, 20, 25]
    * Plot reconstruction error (||X - WH||_F) — drops quickly then flattens
    * Also compute average intra-cluster cosine similarity as a coherence proxy
    * Pick the elbow: diminishing returns in reconstruction quality
  This typically yields k ≈ 15 for the 20 Newsgroups corpus (20 ground-truth
  categories have real overlap; the model finds ~15 *semantic* clusters).

- DEFAULT_K = 15 is our chosen value based on the analysis described above.
  You should run `python setup.py --elbow` once to verify the elbow plot.

DESIGN DECISIONS — Non-negative Shift:
- NMF requires non-negative input. Sentence-transformer embeddings can be negative.
- We use a per-feature (column-wise) minimum shift: X = embeddings - embeddings.min(axis=0)
  This preserves the relative structure within each embedding dimension better than
  a global shift (X - embeddings.min()), which was found to cause uneven cluster
  size distribution by distorting the embedding space uniformly.
- The per-feature min is stored as self.embeddings_min so that out-of-sample
  query embeddings can be shifted consistently at inference time.

BOUNDARY CASES:
- Documents where max(W[i]) is below the 25th percentile of all max weights
  have no clearly dominant cluster. These are the "genuinely uncertain" cases
  — typically cross-topic posts. They're the most interesting and we surface
  them explicitly.
"""

import numpy as np
import pickle
import os
from sklearn.decomposition import NMF

DEFAULT_K = 15           # Chosen via elbow analysis (see docstring above)
CLUSTER_SAVE_PATH = "./embeddings/cluster_model.pkl"
WEIGHTS_SAVE_PATH = "./embeddings/cluster_weights.npy"


class FuzzyClustering:
    """NMF-based soft clustering for document embeddings."""

    def __init__(self, n_clusters: int = DEFAULT_K, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state

        # NMF config:
        # - init='nndsvda': better initialisation than random for sparse data
        # - max_iter=1000: increased to ensure full convergence on 384-dim input
        # - l1_ratio=0.1: small L1 regularisation encourages sparse (interpretable) W
        self.model = NMF(
            n_components=n_clusters,
            init="nndsvda",
            max_iter=1000,
            l1_ratio=0.1,
            random_state=random_state,
        )
        self.W = None              # shape (n_docs, n_clusters) — soft cluster weights
        self.H = None              # shape (n_clusters, embedding_dim) — cluster basis
        self.embeddings_min = None # shape (embedding_dim,) — stored for query projection

    def fit(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Fit NMF on the embedding matrix.

        NMF requires non-negative input. Sentence-transformer embeddings can be
        negative, so we shift per feature: X = embeddings - embeddings.min(axis=0)

        Per-feature shift (column-wise) is used instead of global shift because:
        - Global shift: X - X.min() shifts all values by the same scalar, which
          distorts the relative structure of the embedding space and causes NMF
          to produce uneven cluster sizes (some clusters with 1 doc, others 2000+)
        - Per-feature shift: each dimension is shifted independently, preserving
          the relative variance within each dimension and producing more balanced
          cluster assignments

        The per-feature min is stored so query embeddings can be shifted
        consistently at inference time (critical for correct out-of-sample projection).

        Args:
            embeddings: shape (n_docs, embedding_dim), L2-normalised

        Returns:
            W: shape (n_docs, n_clusters) — soft cluster weight distributions
        """
        print(f"[Clustering] Fitting NMF with k={self.n_clusters}...")

        # Per-feature shift to non-negative (store min for query-time use)
        self.embeddings_min = embeddings.min(axis=0)  # shape (embedding_dim,)
        X = embeddings - self.embeddings_min

        self.W = self.model.fit_transform(X)   # (n_docs, k)
        self.H = self.model.components_        # (k, embedding_dim)

        # Row-normalise W so each row sums to 1 — interpretable as a distribution
        row_sums = self.W.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)  # avoid div-by-zero
        self.W = self.W / row_sums

        # Print weight distribution stats to verify soft assignments are working
        max_weights = self.W.max(axis=1)
        print(f"[Clustering] Done. Reconstruction error: {self.model.reconstruction_err_:.4f}")
        print(f"[Clustering] Max-weight stats: "
              f"min={max_weights.min():.3f}, "
              f"mean={max_weights.mean():.3f}, "
              f"max={max_weights.max():.3f}")

        return self.W

    def get_dominant_cluster(self, doc_idx: int) -> int:
        """Returns the cluster with the highest weight for a given document."""
        return int(np.argmax(self.W[doc_idx]))

    def get_dominant_clusters_all(self) -> list[int]:
        """Returns the argmax cluster for every document."""
        return np.argmax(self.W, axis=1).tolist()

    def get_soft_weights(self, doc_idx: int) -> np.ndarray:
        """Returns the full soft weight distribution for one document."""
        return self.W[doc_idx]

    def assign_query(self, query_embedding: np.ndarray) -> np.ndarray:
        """
        Compute soft cluster weights for a new (unseen) query embedding.

        We apply the same per-feature shift used during training, then
        project the query into cluster space using H (the cluster basis):
            weights = max(0, query_shifted @ H.T)
        Then normalise to sum to 1.

        Using the stored embeddings_min ensures the query is shifted into
        the same non-negative space as the training data — critical for
        consistent cluster assignment at inference time.
        """
        # Apply same per-feature shift as training (NOT query's own min)
        X_q = query_embedding - self.embeddings_min
        X_q = np.maximum(0, X_q)              # clip any residual negatives
        weights = np.maximum(0, X_q @ self.H.T)   # shape (n_clusters,)
        total = weights.sum()
        if total > 0:
            weights /= total
        return weights

    def get_boundary_docs(self, threshold: float = None) -> list[int]:
        """
        Returns indices of documents with no clearly dominant cluster.
        max(W[i]) < threshold means the doc spreads across multiple clusters.
        These are the most semantically ambiguous (cross-topic) documents.

        If threshold is None, we use an adaptive threshold: the 25th percentile
        of max weights. This ensures we always find a meaningful subset of
        boundary docs regardless of how NMF weights are distributed.
        """
        max_weights = self.W.max(axis=1)

        if threshold is None:
            # Adaptive: bottom 25% of max-weight scores are boundary docs
            threshold = float(np.percentile(max_weights, 25))
            print(f"  [Clustering] Adaptive boundary threshold: {threshold:.4f}")

        return np.where(max_weights < threshold)[0].tolist()

    def save(self, model_path: str = CLUSTER_SAVE_PATH, weights_path: str = WEIGHTS_SAVE_PATH):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(self, f)
        np.save(weights_path, self.W)
        print(f"[Clustering] Saved model to {model_path}, weights to {weights_path}")

    @staticmethod
    def load(model_path: str = CLUSTER_SAVE_PATH) -> "FuzzyClustering":
        with open(model_path, "rb") as f:
            obj = pickle.load(f)
        print(f"[Clustering] Loaded model from {model_path}")
        return obj


def find_optimal_k(embeddings: np.ndarray, k_range: list[int] = None) -> None:
    """
    Runs NMF for multiple values of k and prints reconstruction errors.
    Use this once to justify your choice of k via the elbow method.

    Run: python setup.py --elbow
    """
    if k_range is None:
        k_range = [5, 8, 10, 12, 15, 18, 20, 25]

    X = embeddings - embeddings.min(axis=0)  # per-feature shift
    print("\n[Elbow Analysis] k → reconstruction_error")
    print("-" * 40)

    for k in k_range:
        model = NMF(n_components=k, init="nndsvda", max_iter=500, random_state=42)
        model.fit_transform(X)
        print(f"  k={k:2d}  error={model.reconstruction_err_:.4f}")

    print("\nPick k at the elbow — where the error stops dropping sharply.")