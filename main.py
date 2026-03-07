"""
main.py
-------
FastAPI service exposing semantic search with a semantic cache layer.

STATE MANAGEMENT:
- All state (embedder, vector store, clusterer, cache) lives in module-level
  singletons, loaded once at startup via FastAPI's lifespan context manager.
- This avoids re-loading models on every request (which would be catastrophic
  for latency — loading sentence-transformers takes ~3 seconds).
- The SemanticCache object is in-memory and lives for the duration of the process.
  A DELETE /cache resets it entirely.

QUERY FLOW (POST /query):
  1. Embed the incoming query string
  2. Assign it to clusters (for cache lookup routing)
  3. Check semantic cache → if hit, return cached result immediately
  4. On miss: query ChromaDB for the top matching document
  5. Store query + result in cache
  6. Return response

The "result" on a cache miss is the text of the closest matching document
in the vector store. In a production system this might be a generated answer,
but here the retrieval result itself is the output (this is a retrieval system).
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
import numpy as np
import os
import pickle

from models import QueryRequest, QueryResponse, CacheStats, CacheFlushResponse
from embedder import Embedder
from vector_store import VectorStore
from clustering import FuzzyClustering
from cache import SemanticCache


# ---------------------------------------------------------------------------
# Application state — populated at startup
# ---------------------------------------------------------------------------
class AppState:
    embedder: Embedder = None
    vector_store: VectorStore = None
    clusterer: FuzzyClustering = None
    cache: SemanticCache = None


state = AppState()


# ---------------------------------------------------------------------------
# Lifespan: load all models once at startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs on startup (before the app starts serving) and teardown.
    We load heavy models here so they're ready before the first request.
    """
    print("[Startup] Loading models and index...")

    # 1. Embedding model
    state.embedder = Embedder()

    # 2. Vector store (connects to persisted ChromaDB)
    state.vector_store = VectorStore()
    if not state.vector_store.is_populated():
        raise RuntimeError(
            "ChromaDB is empty. Run `python setup.py` first to ingest the corpus."
        )
    print(f"[Startup] Vector store loaded: {state.vector_store.count()} documents")

    # 3. Cluster model
    from clustering import CLUSTER_SAVE_PATH
    if not os.path.exists(CLUSTER_SAVE_PATH):
        raise RuntimeError(
            "Cluster model not found. Run `python setup.py` first."
        )
    state.clusterer = FuzzyClustering.load()
    print(f"[Startup] Cluster model loaded: k={state.clusterer.n_clusters}")

    # 4. Semantic cache — starts empty, grows as queries come in
    # threshold=0.65 chosen based on empirical testing:
    # - catches genuine paraphrases (diabetes/blood-sugar: 0.726)
    # - rejects related-but-distinct questions (Windows/bluescreen: 0.449)
    # - rejects cross-topic boundary cases (guns-in-church/Bible: 0.502)
    state.cache = SemanticCache(clusterer=state.clusterer, threshold=0.65)
    print("[Startup] Semantic cache initialised (empty)")

    print("[Startup] Ready to serve requests.\n")
    yield  # App runs here

    # Teardown (nothing to clean up — models are GC'd)
    print("[Shutdown] Shutting down.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Newsgroups Semantic Search",
    description="Semantic search over 20 Newsgroups with fuzzy clustering and semantic cache.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# POST /query
# ---------------------------------------------------------------------------
@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    Accepts a natural language query, checks the semantic cache, and returns
    the most relevant document from the corpus.

    Cache hit: returns stored result + similarity score.
    Cache miss: queries ChromaDB, stores result, returns it.
    """
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    query_text = request.query.strip()

    # Step 1: Embed the query
    query_embedding = state.embedder.encode_query(query_text)

    # Step 2: Compute query's own cluster assignment once, reuse for both
    # cache lookup and response. dominant_cluster in the response always
    # reflects the CURRENT query's cluster — not the cached entry's cluster.
    # This fixes a bug where cache hits were returning dominant_cluster=0
    # due to NMF projection variance on stored entries.
    query_cluster_weights = state.clusterer.assign_query(query_embedding)
    query_dominant_cluster = int(np.argmax(query_cluster_weights))

    # Step 3: Check semantic cache
    cache_result = state.cache.lookup(query_text, query_embedding)

    if cache_result is not None:
        # ---- CACHE HIT ----
        entry, similarity = cache_result
        return QueryResponse(
            query=query_text,
            cache_hit=True,
            matched_query=entry.query,
            similarity_score=round(similarity, 4),
            result=entry.result,
            dominant_cluster=query_dominant_cluster,  # current query's cluster
        )

    # ---- CACHE MISS ----
    # Step 4: Query ChromaDB for the closest matching document
    results = state.vector_store.query(query_embedding, n_results=1)
    if not results:
        raise HTTPException(status_code=500, detail="No results found in vector store.")

    top_doc = results[0]
    result_text = top_doc["text"][:1000]  # Truncate very long documents for response

    # Step 5: Store in cache for future similar queries
    state.cache.store(
        query=query_text,
        query_embedding=query_embedding,
        result=result_text,
    )

    return QueryResponse(
        query=query_text,
        cache_hit=False,
        matched_query=None,
        similarity_score=None,
        result=result_text,
        dominant_cluster=query_dominant_cluster,  # consistent — always query's cluster
    )


# ---------------------------------------------------------------------------
# GET /cache/stats
# ---------------------------------------------------------------------------
@app.get("/cache/stats", response_model=CacheStats)
async def cache_stats():
    """Returns current cache statistics."""
    s = state.cache.stats()
    return CacheStats(
        total_entries=s["total_entries"],
        hit_count=s["hit_count"],
        miss_count=s["miss_count"],
        hit_rate=s["hit_rate"],
    )


# ---------------------------------------------------------------------------
# DELETE /cache
# ---------------------------------------------------------------------------
@app.delete("/cache", response_model=CacheFlushResponse)
async def flush_cache():
    """Clears all cache entries and resets statistics."""
    state.cache.flush()
    return CacheFlushResponse(message="Cache flushed. All entries and stats reset.")


# ---------------------------------------------------------------------------
# GET / — health check
# ---------------------------------------------------------------------------
@app.get("/")
async def root():
    return {
        "status": "ok",
        "docs_indexed": state.vector_store.count(),
        "cache_entries": state.cache.total_entries,
        "cache_threshold": state.cache.threshold,
    }


# ---------------------------------------------------------------------------
# POST /debug — inspect similarity scores and cluster assignments
# Useful for tuning the threshold and verifying cache behaviour
# ---------------------------------------------------------------------------
@app.post("/debug")
async def debug_similarity(request: QueryRequest):
    """
    Debug endpoint — shows the actual cosine similarity between the incoming
    query and every entry currently in the cache, plus cluster assignments.

    Use this to understand why a query did or did not hit the cache,
    and to tune the similarity threshold.
    """
    query_text = request.query.strip()
    q_emb = state.embedder.encode_query(query_text)
    q_weights = state.clusterer.assign_query(q_emb)

    # Top 3 cluster assignments for this query
    top_clusters = sorted(enumerate(q_weights), key=lambda x: x[1], reverse=True)[:3]

    # Compute cosine similarity against ALL cache entries (brute force for debug)
    comparisons = []
    for cid, entries in state.cache._store.items():
        for entry in entries:
            sim = float(np.dot(q_emb, entry.embedding))
            comparisons.append({
                "cached_query": entry.query,
                "similarity": round(sim, 4),
                "would_hit_at_threshold": sim >= state.cache.threshold,
                "cached_in_cluster": entry.dominant_cluster,
                "query_top_cluster": int(top_clusters[0][0]),
                "same_cluster_bucket": entry.dominant_cluster == int(top_clusters[0][0]),
            })

    # Sort by similarity descending
    comparisons.sort(key=lambda x: x["similarity"], reverse=True)

    return {
        "query": query_text,
        "current_threshold": state.cache.threshold,
        "query_top_3_clusters": [
            {"cluster": int(c), "weight": round(float(w), 4)}
            for c, w in top_clusters
        ],
        "cache_size": state.cache.total_entries,
        "comparisons": comparisons,
        "diagnosis": (
            "No cache entries yet — send a query first via POST /query"
            if not comparisons else
            f"Best similarity: {comparisons[0]['similarity']:.4f} "
            f"(threshold is {state.cache.threshold}) — "
            f"{'WOULD HIT' if comparisons[0]['similarity'] >= state.cache.threshold else 'MISS — similarity below threshold'}"
        )
    }