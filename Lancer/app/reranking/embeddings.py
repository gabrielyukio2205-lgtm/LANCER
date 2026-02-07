"""Embedding-based reranking using sentence-transformers.

Provides bi-encoder and cross-encoder reranking for better relevance scoring.
"""

from functools import lru_cache
from typing import Optional

import numpy as np

from app.config import get_settings


@lru_cache(maxsize=1)
def get_bi_encoder():
    """Load and cache the bi-encoder model."""
    from sentence_transformers import SentenceTransformer
    settings = get_settings()
    return SentenceTransformer(settings.bi_encoder_model)


@lru_cache(maxsize=1)
def get_cross_encoder():
    """Load and cache the cross-encoder model."""
    from sentence_transformers import CrossEncoder
    settings = get_settings()
    return CrossEncoder(settings.cross_encoder_model)


def compute_bi_encoder_scores(
    query: str,
    documents: list[str],
) -> list[float]:
    """
    Compute semantic similarity scores using bi-encoder.
    
    Fast but less accurate than cross-encoder.
    Good for initial filtering of large result sets.
    
    Args:
        query: Search query
        documents: List of document texts
        
    Returns:
        List of similarity scores (0-1)
    """
    if not documents:
        return []
    
    model = get_bi_encoder()
    
    # Encode query and documents
    query_embedding = model.encode(query, normalize_embeddings=True)
    doc_embeddings = model.encode(documents, normalize_embeddings=True)
    
    # Compute cosine similarities (embeddings are normalized, so dot product = cosine)
    similarities = np.dot(doc_embeddings, query_embedding)
    
    # Convert to list and ensure values are in [0, 1]
    scores = [(float(s) + 1) / 2 for s in similarities]  # Map from [-1, 1] to [0, 1]
    
    return scores


def compute_cross_encoder_scores(
    query: str,
    documents: list[str],
) -> list[float]:
    """
    Compute relevance scores using cross-encoder.
    
    More accurate than bi-encoder but slower.
    Use after initial filtering for precise ranking.
    
    Args:
        query: Search query
        documents: List of document texts
        
    Returns:
        List of relevance scores (0-1)
    """
    if not documents:
        return []
    
    model = get_cross_encoder()
    
    # Create query-document pairs
    pairs = [[query, doc] for doc in documents]
    
    # Get scores
    scores = model.predict(pairs)
    
    # Normalize to [0, 1] using sigmoid if needed
    min_score = float(np.min(scores))
    max_score = float(np.max(scores))
    
    if max_score > min_score:
        normalized = [(float(s) - min_score) / (max_score - min_score) for s in scores]
    else:
        normalized = [0.5] * len(scores)
    
    return normalized
