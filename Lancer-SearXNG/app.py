"""Lancer-SearXNG: Experimental search API using only SearXNG.

This version uses SearXNG meta-search instead of paid APIs,
returning 50+ results where embedding reranking actually makes sense.
"""

import os
import json
from typing import Optional
from functools import lru_cache

import httpx
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel


# === CONFIG ===

# SearXNG instances to try
SEARXNG_INSTANCES = [
    os.getenv("SEARXNG_URL", "https://searx.be"),
    "https://search.sapti.me",
    "https://searx.tiekoetter.com",
    "https://search.bus-hit.me",
]

# Embedding model
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "Madras1/minilm-gooaq-mnr-v5")


# === MODELS ===

class SearchRequest(BaseModel):
    query: str
    max_results: int = 10
    use_reranking: bool = True


class SearchResult(BaseModel):
    title: str
    url: str
    content: str
    score: float
    source: str


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    total_raw: int
    reranked: bool


# === EMBEDDING ===

@lru_cache(maxsize=1)
def get_embedder():
    """Load embedding model (cached)."""
    from sentence_transformers import SentenceTransformer
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    return SentenceTransformer(EMBEDDING_MODEL)


def rerank_with_embeddings(
    query: str,
    results: list[dict],
    top_k: int = 10,
) -> list[dict]:
    """Rerank results using semantic similarity."""
    if not results:
        return []
    
    model = get_embedder()
    
    # Create documents from results
    docs = [
        f"{r.get('title', '')}. {r.get('content', '')[:500]}"
        for r in results
    ]
    
    # Encode query and docs
    query_emb = model.encode(query, normalize_embeddings=True)
    doc_embs = model.encode(docs, normalize_embeddings=True)
    
    # Compute similarities
    similarities = np.dot(doc_embs, query_emb)
    
    # Add scores to results
    for i, result in enumerate(results):
        result["embedding_score"] = float(similarities[i])
        # Combine with original position score
        orig_score = result.get("score", 0.5)
        result["score"] = (result["embedding_score"] * 0.7) + (orig_score * 0.3)
    
    # Sort by combined score
    results.sort(key=lambda x: x["score"], reverse=True)
    
    return results[:top_k]


# === SEARXNG CLIENT ===

async def search_searxng(
    query: str,
    max_results: int = 50,
    time_range: Optional[str] = None,
) -> list[dict]:
    """Search using SearXNG meta-search."""
    
    params = {
        "q": query,
        "format": "json",
        "language": "all",
    }
    if time_range:
        params["time_range"] = time_range
    
    # Try each instance
    for instance in SEARXNG_INSTANCES:
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(
                    f"{instance.rstrip('/')}/search",
                    params=params,
                )
                response.raise_for_status()
                data = response.json()
            
            results = []
            for i, item in enumerate(data.get("results", [])[:max_results]):
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "content": item.get("content", ""),
                    "source": f"searxng:{item.get('engine', 'unknown')}",
                    "score": max(0.3, 1.0 - (i * 0.02)),  # Position-based score
                })
            
            if results:
                print(f"SearXNG ({instance}): {len(results)} results")
                return results
                
        except Exception as e:
            print(f"SearXNG {instance} failed: {e}")
            continue
    
    return []


# === APP ===

app = FastAPI(
    title="Lancer-SearXNG",
    description="Experimental search API using SearXNG + embedding reranking",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "name": "Lancer-SearXNG",
        "version": "0.1.0",
        "docs": "/docs",
        "embedding_model": EMBEDDING_MODEL,
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/api/v1/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Search using SearXNG and optionally rerank with embeddings.
    
    With 50+ results from SearXNG, embedding reranking actually helps!
    """
    # Get raw results from SearXNG
    raw_results = await search_searxng(
        query=request.query,
        max_results=50,  # Get many results
    )
    
    if not raw_results:
        raise HTTPException(status_code=404, detail="No results found")
    
    total_raw = len(raw_results)
    
    # Optionally rerank with embeddings
    if request.use_reranking and len(raw_results) > request.max_results:
        results = rerank_with_embeddings(
            query=request.query,
            results=raw_results,
            top_k=request.max_results,
        )
        reranked = True
    else:
        results = raw_results[:request.max_results]
        reranked = False
    
    return SearchResponse(
        query=request.query,
        results=[
            SearchResult(
                title=r["title"],
                url=r["url"],
                content=r["content"][:500],
                score=r["score"],
                source=r["source"],
            )
            for r in results
        ],
        total_raw=total_raw,
        reranked=reranked,
    )


@app.get("/api/v1/engines")
async def list_engines():
    """List available SearXNG engines."""
    for instance in SEARXNG_INSTANCES:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{instance}/config")
                if response.status_code == 200:
                    data = response.json()
                    return {
                        "instance": instance,
                        "engines": [
                            e["name"] for e in data.get("engines", [])
                            if not e.get("disabled", False)
                        ],
                    }
        except:
            continue
    
    return {"instance": None, "engines": []}
