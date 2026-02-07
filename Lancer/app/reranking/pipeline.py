"""Multi-stage reranking pipeline.

Implements a 3-stage reranking approach:
1. Bi-Encoder: Fast semantic similarity (for large result sets)
2. Cross-Encoder: Accurate relevance scoring
3. Temporal + Authority: Freshness and domain trust weighting
"""

import logging
from typing import Optional

from app.temporal.freshness_scorer import calculate_freshness_score, adjust_score_by_freshness
from app.reranking.authority_scorer import calculate_authority_score

logger = logging.getLogger(__name__)

# Flag to enable/disable embedding-based reranking
ENABLE_EMBEDDING_RERANKING = True


async def rerank_results(
    query: str,
    results: list[dict],
    temporal_urgency: float = 0.5,
    max_results: int = 10,
    use_embeddings: bool = True,
) -> list[dict]:
    """
    Apply multi-stage reranking to search results.
    
    Pipeline:
    1. Bi-encoder: Quick semantic filtering (if results > 20)
    2. Cross-encoder: Precise relevance scoring (top candidates)
    3. Temporal + Authority: Freshness and trust weighting
    
    Args:
        query: Original search query
        results: Raw search results
        temporal_urgency: How important freshness is (0-1)
        max_results: Maximum results to return
        use_embeddings: Whether to use embedding models
        
    Returns:
        Reranked results with updated scores
    """
    if not results:
        return []
    
    scored_results = results.copy()
    
    # Stage 1 & 2: Embedding-based reranking
    if use_embeddings and ENABLE_EMBEDDING_RERANKING:
        try:
            scored_results = await _apply_embedding_reranking(query, scored_results)
            logger.info(f"Applied embedding reranking to {len(scored_results)} results")
        except Exception as e:
            logger.warning(f"Embedding reranking failed, using fallback: {e}")
            # Fall through to basic scoring
    
    # Stage 3: Apply temporal + authority scoring
    for result in scored_results:
        # Calculate freshness score
        freshness = calculate_freshness_score(result.get("published_date"))
        result["freshness_score"] = freshness
        
        # Calculate authority score
        authority = calculate_authority_score(result.get("url", ""))
        result["authority_score"] = authority
        
        # Get base score (from search source or embedding)
        base_score = result.get("score", 0.5)
        
        # Adjust for freshness based on temporal urgency
        adjusted_score = adjust_score_by_freshness(
            base_score=base_score,
            freshness_score=freshness,
            temporal_urgency=temporal_urgency,
        )
        
        # Also factor in authority (10% weight)
        final_score = (adjusted_score * 0.9) + (authority * 0.1)
        result["score"] = final_score
    
    # Sort by final score (descending)
    scored_results.sort(key=lambda x: x["score"], reverse=True)
    
    return scored_results[:max_results]


async def _apply_embedding_reranking(
    query: str,
    results: list[dict],
) -> list[dict]:
    """Apply bi-encoder and cross-encoder reranking."""
    from app.reranking.embeddings import compute_bi_encoder_scores, compute_cross_encoder_scores
    
    # Extract document contents for embedding
    documents = [
        f"{r.get('title', '')}. {r.get('content', '')[:500]}"
        for r in results
    ]
    
    # Stage 1: Bi-encoder for initial scoring (fast)
    if len(results) > 15:
        bi_scores = compute_bi_encoder_scores(query, documents)
        for i, result in enumerate(results):
            result["bi_encoder_score"] = bi_scores[i]
        
        # Sort by bi-encoder and keep top 15 for cross-encoder
        results.sort(key=lambda x: x.get("bi_encoder_score", 0), reverse=True)
        results = results[:15]
        documents = documents[:15]
    
    # Stage 2: Cross-encoder for precise scoring (slower but accurate)
    cross_scores = compute_cross_encoder_scores(query, documents)
    
    for i, result in enumerate(results):
        # Blend cross-encoder score with original source score
        original_score = result.get("score", 0.5)
        cross_score = cross_scores[i]
        
        # Cross-encoder gets 70% weight, original 30%
        result["score"] = (cross_score * 0.7) + (original_score * 0.3)
        result["cross_encoder_score"] = cross_score
    
    return results

