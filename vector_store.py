"""
vector_store.py
---------------
Wraps ChromaDB for persistent storage and retrieval of document embeddings.

DESIGN DECISIONS — Vector Store Choice:
- ChromaDB was chosen over alternatives for the following reasons:
    * FAISS: Extremely fast ANN search, but no metadata filtering and no built-in
      persistence without extra serialization code. It's a pure index, not a DB.
    * Pinecone/Weaviate: Cloud-hosted. Adds network latency and requires an account.
      Overkill for a local prototype on 20k documents.
    * ChromaDB: Local, file-backed persistence, supports metadata filtering
      (we store cluster assignments as metadata), cosine similarity built-in,
      and a clean Python API. Perfect fit for this use case.
- We store the following metadata per document:
    * label: ground-truth category (int) — useful for evaluation
    * label_name: human-readable category string
    * dominant_cluster: the cluster with highest NMF weight (int)
    * cluster_weights: serialized comma-separated floats (Chroma doesn't support arrays)
- Chroma handles its own embedding persistence, so we pass pre-computed numpy
  embeddings directly (embedding_function=None) to avoid re-encoding on reload.
"""

import numpy as np
import chromadb
from chromadb.config import Settings


COLLECTION_NAME = "newsgroups"
CHROMA_PATH = "./embeddings/chroma_db"


class VectorStore:
    """Interface to ChromaDB for the newsgroups corpus."""

    def __init__(self, persist_path: str = CHROMA_PATH):
        self.client = chromadb.PersistentClient(
            path=persist_path,
            settings=Settings(anonymized_telemetry=False),
        )
        # embedding_function=None because we supply our own pre-computed vectors
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},  # use cosine distance for ANN index
        )

    def add_documents(
        self,
        texts: list[str],
        embeddings: np.ndarray,
        labels: list[int],
        label_names: list[str],
        dominant_clusters: list[int],
        cluster_weights: list[np.ndarray],
        batch_size: int = 512,
    ) -> None:
        """
        Upsert documents into ChromaDB in batches.
        Chroma has a default batch limit, so we chunk large corpora.
        """
        n = len(texts)
        print(f"[VectorStore] Inserting {n} documents into ChromaDB...")

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_ids = [f"doc_{i}" for i in range(start, end)]

            # Chroma metadata values must be str, int, float, or bool — no lists/arrays
            batch_meta = [
                {
                    "label": labels[i],
                    "label_name": label_names[labels[i]],
                    "dominant_cluster": dominant_clusters[i],
                    # Serialize float array as comma-separated string
                    "cluster_weights": ",".join(f"{w:.4f}" for w in cluster_weights[i]),
                }
                for i in range(start, end)
            ]

            self.collection.upsert(
                ids=batch_ids,
                documents=texts[start:end],
                embeddings=embeddings[start:end].tolist(),
                metadatas=batch_meta,
            )

        print(f"[VectorStore] Done. Collection size: {self.collection.count()}")

    def query(
        self,
        query_embedding: np.ndarray,
        n_results: int = 5,
        where: dict | None = None,
    ) -> list[dict]:
        """
        Find the top-n most similar documents to a query vector.

        Args:
            query_embedding: shape (dim,) — L2-normalised float32
            n_results: how many results to return
            where: optional ChromaDB metadata filter dict

        Returns:
            List of dicts with keys: id, text, metadata, distance
        """
        kwargs = {
            "query_embeddings": [query_embedding.tolist()],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = self.collection.query(**kwargs)

        # Unpack ChromaDB's nested list format
        docs = []
        for i in range(len(results["ids"][0])):
            docs.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],  # cosine distance (lower = better)
            })
        return docs

    def is_populated(self) -> bool:
        return self.collection.count() > 0

    def count(self) -> int:
        return self.collection.count()
