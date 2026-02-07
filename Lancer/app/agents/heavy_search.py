"""Heavy Search Agent.

Middle-ground between Quick Search and Deep Research.
Scrapes full content from top results for richer answers.
"""

import json
import time
from typing import AsyncIterator

from app.agents.llm_client import generate_completion_stream
from app.sources.aggregator import aggregate_search
from app.sources.scraper import scrape_multiple_urls
from app.reranking.pipeline import rerank_results
from app.temporal.intent_detector import detect_temporal_intent


async def run_heavy_search(
    query: str,
    max_results: int = 15,
    max_scrape: int = 8,
    freshness: str = "any",
) -> AsyncIterator[str]:
    """
    Run heavy search with content scraping.
    
    Steps:
    1. Aggregate search from multiple sources
    2. Rerank results
    3. Scrape full content from top N results
    4. Stream synthesized answer
    
    Yields:
        SSE event strings
    """
    start_time = time.perf_counter()
    
    try:
        # Step 1: Status
        yield _sse_event("status", {"phase": "searching", "message": "Searching multiple sources..."})
        
        # Step 2: Aggregate search
        temporal_intent, temporal_urgency = detect_temporal_intent(query)
        
        raw_results = await aggregate_search(
            query=query,
            max_results=max_results + 5,
            freshness=freshness,
            include_wikipedia=True,
        )
        
        if not raw_results:
            yield _sse_event("error", {"message": "No results found"})
            return
        
        yield _sse_event("search_complete", {
            "results_count": len(raw_results),
            "sources": list(set(r.get("source", "unknown") for r in raw_results)),
        })
        
        # Step 3: Rerank (use embeddings when we have many results from SearXNG)
        yield _sse_event("status", {"phase": "ranking", "message": "Ranking results..."})
        
        # Enable embeddings when we have many results (SearXNG provides volume)
        use_embeddings = len(raw_results) > 20
        
        ranked_results = await rerank_results(
            query=query,
            results=raw_results,
            temporal_urgency=temporal_urgency,
            max_results=max_results,
            use_embeddings=use_embeddings,
        )
        
        # Step 4: Scrape top results
        yield _sse_event("status", {"phase": "scraping", "message": f"Reading top {max_scrape} sources..."})
        
        urls_to_scrape = [r.get("url") for r in ranked_results[:max_scrape] if r.get("url")]
        scraped_content = await scrape_multiple_urls(
            urls=urls_to_scrape,
            max_chars_per_url=4000,
            max_concurrent=3,
        )
        
        # Merge scraped content into results
        for result in ranked_results:
            url = result.get("url", "")
            if url in scraped_content and scraped_content[url]:
                result["full_content"] = scraped_content[url]
                result["scraped"] = True
            else:
                result["full_content"] = result.get("content", "")
                result["scraped"] = False
        
        scraped_count = sum(1 for r in ranked_results if r.get("scraped"))
        yield _sse_event("scrape_complete", {
            "scraped_count": scraped_count,
            "total": len(urls_to_scrape),
        })
        
        # Step 5: Send results
        yield _sse_event("results", {
            "results": [
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "score": r.get("score", 0),
                    "source": r.get("source", ""),
                    "scraped": r.get("scraped", False),
                }
                for r in ranked_results
            ],
            "temporal_intent": temporal_intent,
            "temporal_urgency": temporal_urgency,
        })
        
        # Step 6: Synthesize answer
        yield _sse_event("status", {"phase": "synthesizing", "message": "Generating answer..."})
        yield _sse_event("answer_start", {})
        
        async for chunk in _synthesize_heavy_answer(query, ranked_results, temporal_intent):
            yield _sse_event("answer_chunk", {"content": chunk})
        
        # Done
        total_time = time.perf_counter() - start_time
        yield _sse_event("done", {
            "total_sources": len(ranked_results),
            "scraped_sources": scraped_count,
            "total_time_seconds": round(total_time, 2),
        })
        
    except Exception as e:
        yield _sse_event("error", {"message": str(e)})


async def _synthesize_heavy_answer(
    query: str,
    results: list[dict],
    temporal_intent: str,
) -> AsyncIterator[str]:
    """Synthesize answer from scraped content."""
    
    # Build context with full content
    context_parts = []
    for i, r in enumerate(results[:8], 1):
        content = r.get("full_content", r.get("content", ""))[:3000]
        scraped_tag = "[FULL]" if r.get("scraped") else "[SNIPPET]"
        
        context_parts.append(
            f"[{i}] {r.get('title', 'Untitled')} {scraped_tag}\n"
            f"URL: {r.get('url', '')}\n"
            f"Content:\n{content}\n"
        )
    
    context = "\n---\n".join(context_parts)
    
    prompt = f"""You are a research assistant providing comprehensive answers.

QUERY: {query}
TEMPORAL INTENT: {temporal_intent}

SOURCES (some with full content [FULL], some with snippets [SNIPPET]):
{context}

INSTRUCTIONS:
1. Provide a comprehensive, well-structured answer
2. Use information from [FULL] sources more extensively
3. Cite sources using [1], [2], etc.
4. Write in the same language as the query
5. Be thorough but clear

Answer:"""

    messages = [
        {"role": "system", "content": "You are a helpful research assistant."},
        {"role": "user", "content": prompt},
    ]
    
    async for chunk in generate_completion_stream(messages, temperature=0.3):
        yield chunk
    
    # Add citations
    yield "\n\n---\n**Sources:**\n"
    for i, r in enumerate(results[:8], 1):
        scraped = "📄" if r.get("scraped") else "📋"
        yield f"{scraped} [{i}] [{r.get('title', 'Untitled')}]({r.get('url', '')})\n"


def _sse_event(event_type: str, data: dict) -> str:
    """Format an SSE event."""
    payload = {"type": event_type, **data}
    return f"data: {json.dumps(payload)}\n\n"
