# Newsgroups Semantic Search

A semantic search system over the 20 Newsgroups dataset featuring fuzzy NMF clustering, a from-scratch semantic cache, and a FastAPI service.

---

## Architecture Overview

```
Query → Embedder → Cache Lookup → ChromaDB (on miss) → Response
                       ↑
              Cluster-Partitioned
                SemanticCache
                       ↑
                NMF Fuzzy Clustering
                (soft assignments)
```

**Stack:**
- `sentence-transformers` — `all-MiniLM-L6-v2` for 384-dim embeddings
- `ChromaDB` — local vector store with cosine similarity
- `scikit-learn` NMF — fuzzy (soft) clustering, k=15
- `FastAPI` + `uvicorn` — REST API
- Pure Python `dict` — semantic cache (no Redis, no external libraries)

---

## Project Structure

```
newsgroups_search/
├── main.py            # FastAPI app — endpoints and query flow
├── cache.py           # SemanticCache — cluster-partitioned from-scratch cache
├── clustering.py      # FuzzyClustering — NMF soft assignments
├── embedder.py        # Embedder — sentence-transformer wrapper
├── vector_store.py    # VectorStore — ChromaDB interface
├── preprocessor.py    # Corpus loading and cleaning
├── models.py          # Pydantic schemas
├── setup.py           # One-time corpus ingestion script
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── ANALYSIS.md        # Full design justifications and empirical results
└── README.md
```

---

## Setup

### Prerequisites
- Python 3.11
- The 20 Newsgroups dataset extracted to `./20_newsgroups/`

### 1. Create virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt

# Windows — if torch install fails, use:
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cpu
```

### 3. Run corpus ingestion (one-time, ~20 minutes)

```bash
python setup.py
```

This will:
- Load and clean 15,200 documents from the corpus
- Generate 384-dim embeddings using `all-MiniLM-L6-v2`
- Fit NMF fuzzy clustering with k=15
- Store everything in ChromaDB at `./embeddings/chroma_db/`

### 4. Start the server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

API docs available at: `http://localhost:8000/docs`

---

## Docker

```bash
# Run setup.py first to generate embeddings, then:
docker-compose up --build
```

The `embeddings/` directory is mounted as a volume — ChromaDB and the cluster model persist across container restarts.

---

## API Endpoints

### `POST /query`

Submit a natural language query. Returns the most relevant document, with cache hit/miss status.

**Request:**
```json
{
  "query": "What are the symptoms of diabetes?"
}
```

**Response (cache miss):**
```json
{
  "query": "What are the symptoms of diabetes?",
  "cache_hit": false,
  "matched_query": null,
  "similarity_score": null,
  "result": "...(most relevant document text)...",
  "dominant_cluster": 13
}
```

**Response (cache hit):**
```json
{
  "query": "How do I know if I have high blood sugar?",
  "cache_hit": true,
  "matched_query": "What are the symptoms of diabetes?",
  "similarity_score": 0.7264,
  "result": "...(cached result)...",
  "dominant_cluster": 13
}
```

---

### `GET /cache/stats`

Returns current cache statistics.

```json
{
  "total_entries": 5,
  "hit_count": 3,
  "miss_count": 2,
  "hit_rate": 0.6
}
```

---

### `DELETE /cache`

Flushes all cache entries and resets statistics.

```json
{
  "message": "Cache flushed. All entries and stats reset."
}
```

---

### `GET /`

Health check.

```json
{
  "status": "ok",
  "docs_indexed": 15200,
  "cache_entries": 0,
  "cache_threshold": 0.65
}
```

---

### `POST /debug`

Inspect exact similarity scores between a query and all cached entries. Useful for threshold tuning.

```json
{
  "query": "How do I know if I have high blood sugar?",
  "current_threshold": 0.65,
  "query_top_3_clusters": [...],
  "cache_size": 1,
  "comparisons": [
    {
      "cached_query": "What are the symptoms of diabetes?",
      "similarity": 0.7264,
      "would_hit_at_threshold": true,
      "cached_in_cluster": 13,
      "query_top_cluster": 13,
      "same_cluster_bucket": true
    }
  ],
  "diagnosis": "Best similarity: 0.7264 (threshold is 0.65) — WOULD HIT"
}
```

---

## How the Semantic Cache Works

The cache stores query embeddings in a **cluster-partitioned dict** (`cluster_id → list[CacheEntry]`).

On each new query:
1. Embed the query → 384-dim vector
2. Assign it to top-2 NMF clusters
3. Search only entries in those cluster buckets (O(n/k) vs O(n))
4. If best cosine similarity ≥ τ=0.65 → **cache hit**, return stored result
5. Otherwise → **cache miss**, query ChromaDB, store result for future hits

The cluster partitioning is where the fuzzy clustering does real work — it's not just a label, it routes cache lookups efficiently.

### Similarity Threshold τ=0.65

Chosen empirically by testing real query pairs:

| Query Pair | Similarity | Hit at τ=0.65 | Correct? |
|---|---|---|---|
| diabetes symptoms / high blood sugar | 0.7264 | ✅ | ✅ should hit |
| treat headache / medicine for head pain | 0.6946 | ✅ | ✅ should hit |
| guns in church / Bible on violence | 0.5017 | ❌ | ✅ should miss |
| Windows crashing / blue screen of death | 0.4490 | ❌ | ✅ should miss |

τ=0.65 correctly separates genuine paraphrases (same question, different words) from topically related but distinct questions.

---

## Elbow Analysis (k selection)

```
k= 5   error=112.74
k= 8   error=109.63
k=10   error=107.76
k=12   error=106.14
k=15   error=103.87  ← chosen
k=18   error=101.81
k=20   error=100.53
k=25   error=97.63
```

Run it yourself:
```bash
python setup.py --elbow
```

---

## Design Decisions

See [ANALYSIS.md](./ANALYSIS.md) for full justifications covering:
- All preprocessing decisions
- Embedding model and vector store selection
- NMF algorithm choice vs GMM/LDA/K-means
- Cluster coherence analysis with all 14 clusters
- Threshold exploration with empirical data
- Cache architecture decisions

---

## Requirements

```
fastapi==0.111.0
uvicorn==0.30.1
sentence-transformers==3.0.1
chromadb==0.5.3
scikit-learn==1.5.0
numpy==1.26.4
pydantic==2.7.4
httpx==0.27.0
tqdm==4.66.4
torch==2.2.0
```