"""
embedder.py
-----------
Wraps sentence-transformers for encoding documents and queries.

DESIGN DECISIONS — Embedding Model Choice:
- Model: all-MiniLM-L6-v2
    * 384-dimensional output — compact enough to store 20k vectors in RAM and Chroma
    * Strong semantic performance on sentence similarity benchmarks (MTEB)
    * ~6× faster than all-mpnet-base-v2 with only ~5% quality loss on STS tasks
    * The quality trade-off is acceptable here: we're doing semantic retrieval,
      not fine-grained NLI. Speed matters for a live API.
- Alternative considered: all-mpnet-base-v2 (768-dim)
    * Better accuracy, but 2× memory, slower inference, and the extra quality
      doesn't justify it for a newsgroup search system.
- We encode with batch_size=64 and show_progress_bar=True for the one-time
  corpus setup. For query-time (single strings), we call encode directly.
- normalize_embeddings=True: L2-normalises output vectors so that dot product
  equals cosine similarity. This is critical — our cache and Chroma both rely
  on cosine similarity, and normalization makes the math consistent.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


class Embedder:
    """Singleton-style wrapper around a sentence-transformer model."""

    def __init__(self, model_name: str = MODEL_NAME):
        print(f"[Embedder] Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dim = EMBEDDING_DIM

    def encode_corpus(
        self,
        texts: list[str],
        batch_size: int = 64,
    ) -> np.ndarray:
        """
        Encode a large list of texts in batches.
        Returns: float32 array of shape (n_docs, EMBEDDING_DIM), L2-normalised.
        """
        print(f"[Embedder] Encoding {len(texts)} documents in batches of {batch_size}...")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,  # cosine sim = dot product after this
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        """
        Encode a single query string.
        Returns: float32 array of shape (EMBEDDING_DIM,), L2-normalised.
        """
        vec = self.model.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vec.astype(np.float32)
