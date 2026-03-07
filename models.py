from pydantic import BaseModel
from typing import Optional


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    query: str
    cache_hit: bool
    matched_query: Optional[str]       # The cached query that was matched (None on miss)
    similarity_score: Optional[float]  # Cosine similarity to matched query (None on miss)
    result: str                        # The semantic search result (top matching document)
    dominant_cluster: int              # The cluster this query most strongly belongs to


class CacheStats(BaseModel):
    total_entries: int
    hit_count: int
    miss_count: int
    hit_rate: float


class CacheFlushResponse(BaseModel):
    message: str
