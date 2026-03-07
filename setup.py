"""
setup.py
--------
One-time setup script. Run this BEFORE starting the FastAPI server.

What it does:
  1. Loads and cleans the 20 Newsgroups corpus
  2. Encodes all documents with sentence-transformers
  3. Fits NMF fuzzy clustering on the embeddings
  4. Persists everything to disk (ChromaDB + pickle)

Run:
    python setup.py

This takes ~5–15 minutes depending on your hardware (GPU speeds it up).
You only need to run it once — the server loads from disk on startup.
"""

import numpy as np
import os
import sys

from preprocessor import build_corpus
from embedder import Embedder
from vector_store import VectorStore
from clustering import FuzzyClustering, find_optimal_k, DEFAULT_K

EMBEDDINGS_CACHE = "./embeddings/corpus_embeddings.npy"
TEXTS_CACHE = "./embeddings/corpus_texts.pkl"
LABELS_CACHE = "./embeddings/corpus_labels.pkl"
LABEL_NAMES_CACHE = "./embeddings/label_names.pkl"

import pickle


def save_pkl(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def run_setup(run_elbow: bool = False):
    os.makedirs("./embeddings", exist_ok=True)

    # ------------------------------------------------------------------ #
    # Step 1: Load & clean corpus
    # ------------------------------------------------------------------ #
    print("\n=== Step 1: Loading corpus ===")
    if os.path.exists(TEXTS_CACHE):
        print("  [Cache] Found existing corpus cache, loading...")
        texts = load_pkl(TEXTS_CACHE)
        labels = load_pkl(LABELS_CACHE)
        label_names = load_pkl(LABEL_NAMES_CACHE)
    else:
        texts, labels, label_names = build_corpus(subset="all")
        save_pkl(texts, TEXTS_CACHE)
        save_pkl(labels, LABELS_CACHE)
        save_pkl(label_names, LABEL_NAMES_CACHE)

    print(f"  Corpus: {len(texts)} documents, {len(label_names)} categories")

    # ------------------------------------------------------------------ #
    # Step 2: Embed corpus
    # ------------------------------------------------------------------ #
    print("\n=== Step 2: Embedding corpus ===")
    if os.path.exists(EMBEDDINGS_CACHE):
        print("  [Cache] Found existing embeddings, loading...")
        embeddings = np.load(EMBEDDINGS_CACHE)
    else:
        embedder = Embedder()
        embeddings = embedder.encode_corpus(texts)
        np.save(EMBEDDINGS_CACHE, embeddings)
        print(f"  Saved embeddings: {embeddings.shape}")

    print(f"  Embeddings shape: {embeddings.shape}")

    # ------------------------------------------------------------------ #
    # Step 3: Optional elbow analysis (run once to justify k choice)
    # ------------------------------------------------------------------ #
    if run_elbow:
        print("\n=== Step 3a: Elbow Analysis (k selection) ===")
        find_optimal_k(embeddings)
        print("\nReview the output above and update DEFAULT_K in clustering.py if needed.")

    # ------------------------------------------------------------------ #
    # Step 4: Fuzzy clustering
    # ------------------------------------------------------------------ #
    print(f"\n=== Step 3: Fuzzy Clustering (NMF, k={DEFAULT_K}) ===")
    from clustering import CLUSTER_SAVE_PATH

    if os.path.exists(CLUSTER_SAVE_PATH):
        print("  [Cache] Found existing cluster model, loading...")
        clusterer = FuzzyClustering.load()
        W = clusterer.W
    else:
        clusterer = FuzzyClustering(n_clusters=DEFAULT_K)
        W = clusterer.fit(embeddings)
        clusterer.save()

    dominant_clusters = clusterer.get_dominant_clusters_all()
    cluster_weights_list = [W[i] for i in range(len(texts))]

    # Print cluster summary
    print("\n  Cluster size distribution:")
    from collections import Counter
    dist = Counter(dominant_clusters)
    for cid in sorted(dist):
        print(f"    Cluster {cid:2d}: {dist[cid]:4d} docs")

    boundary_docs = clusterer.get_boundary_docs(threshold=0.4)
    print(f"\n  Boundary docs (no clear dominant cluster): {len(boundary_docs)}")

    # ------------------------------------------------------------------ #
    # Step 5: Store in ChromaDB
    # ------------------------------------------------------------------ #
    print("\n=== Step 4: Storing in ChromaDB ===")
    store = VectorStore()

    if store.is_populated():
        print(f"  [Cache] ChromaDB already has {store.count()} docs — skipping insert.")
    else:
        store.add_documents(
            texts=texts,
            embeddings=embeddings,
            labels=labels,
            label_names=label_names,
            dominant_clusters=dominant_clusters,
            cluster_weights=cluster_weights_list,
        )

    # ------------------------------------------------------------------ #
    # Step 6: Print sample cluster analysis (for the assignment write-up)
    # ------------------------------------------------------------------ #
    print("\n=== Step 5: Cluster Analysis Sample ===")
    _print_cluster_analysis(texts, labels, label_names, W, dominant_clusters, boundary_docs)

    print("\n✅ Setup complete! Run the server with:")
    print("   uvicorn main:app --host 0.0.0.0 --port 8000 --reload\n")


def _print_cluster_analysis(texts, labels, label_names, W, dominant_clusters, boundary_docs):
    """
    Prints evidence that clusters are semantically meaningful.
    This output is your justification for the assignment.
    """
    from collections import Counter, defaultdict

    # For each cluster, show the top ground-truth categories it contains
    cluster_to_labels = defaultdict(list)
    for i, cid in enumerate(dominant_clusters):
        cluster_to_labels[cid].append(label_names[labels[i]])

    print("\n  Top ground-truth categories per cluster (semantic coherence check):")
    for cid in sorted(cluster_to_labels.keys()):
        cat_dist = Counter(cluster_to_labels[cid]).most_common(3)
        top_cats = ", ".join(f"{c}({n})" for c, n in cat_dist)
        print(f"    Cluster {cid:2d} ({len(cluster_to_labels[cid])} docs): {top_cats}")

    # Boundary doc examples
    if boundary_docs:
        print(f"\n  Sample boundary documents (spread across multiple clusters):")
        for idx in boundary_docs[:3]:
            weights = W[idx]
            top2 = np.argsort(weights)[::-1][:2]
            print(f"    Doc {idx}: cluster {top2[0]}({weights[top2[0]]:.2f}) + "
                  f"cluster {top2[1]}({weights[top2[1]]:.2f})")
            print(f"    Category: {label_names[labels[idx]]}")
            print(f"    Preview: {texts[idx][:120].strip()}...")
            print()


if __name__ == "__main__":
    # Pass --elbow flag to run k-selection analysis first
    run_elbow = "--elbow" in sys.argv
    run_setup(run_elbow=run_elbow)
