"""Search API routes."""

import json
import time
from datetime import datetime

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.api.schemas import (
    SearchRequest,
    SearchResponse,
    SearchResult,
    TemporalContext,
    Citation,
    ErrorResponse,
    DeepResearchRequest,
    BrowseRequest,
)
from app.config import get_settings
from app.temporal.intent_detector import detect_temporal_intent
from app.temporal.freshness_scorer import calculate_freshness_score
from app.sources.tavily import search_tavily
from app.sources.duckduckgo import search_duckduckgo
from app.reranking.pipeline import rerank_results
from app.agents.synthesizer import synthesize_answer, synthesize_answer_stream
from app.middleware.rate_limiter import limiter

router = APIRouter()


@router.post(
    "/search",
    response_model=SearchResponse,
    responses={500: {"model": ErrorResponse}},
    summary="Search with AI synthesis",
    description="Perform a search with temporal intelligence and return an AI-synthesized answer.",
)
@limiter.limit("30/minute")
async def search(request: Request, body: SearchRequest) -> SearchResponse:
    """
    Perform an intelligent search with:
    - Temporal intent detection
    - Multi-source search
    - Multi-stage reranking
    - AI-powered answer synthesis
    """
    start_time = time.perf_counter()
    settings = get_settings()
    
    try:
        # Step 1: Analyze temporal intent
        temporal_intent, temporal_urgency = detect_temporal_intent(body.query)
        
        temporal_context = TemporalContext(
            query_temporal_intent=temporal_intent,
            temporal_urgency=temporal_urgency,
            current_date=datetime.now().strftime("%Y-%m-%d"),
        )
        
        # Step 2: Search multiple sources
        raw_results = []
        
        # Try Tavily first (best quality)
        if settings.tavily_api_key:
            tavily_results = await search_tavily(
                query=body.query,
                max_results=settings.max_search_results,
                freshness=body.freshness,
                include_domains=body.include_domains,
                exclude_domains=body.exclude_domains,
            )
            raw_results.extend(tavily_results)
        
        # Fallback to DuckDuckGo if needed
        if not raw_results:
            ddg_results = await search_duckduckgo(
                query=body.query,
                max_results=settings.max_search_results,
            )
            raw_results.extend(ddg_results)
        
        if not raw_results:
            return SearchResponse(
                query=body.query,
                answer="No results found for your query.",
                results=[],
                citations=[],
                temporal_context=temporal_context,
                processing_time_ms=(time.perf_counter() - start_time) * 1000,
            )
        
        # Step 3: Apply multi-stage reranking
        ranked_results = await rerank_results(
            query=body.query,
            results=raw_results,
            temporal_urgency=temporal_urgency,
            max_results=body.max_results,
        )
        
        # Step 4: Convert to SearchResult models
        search_results = []
        for i, result in enumerate(ranked_results):
            freshness = calculate_freshness_score(result.get("published_date"))
            search_results.append(
                SearchResult(
                    title=result.get("title", ""),
                    url=result.get("url", ""),
                    content=result.get("content", ""),
                    score=result.get("score", 0.5),
                    published_date=result.get("published_date"),
                    freshness_score=freshness,
                    authority_score=result.get("authority_score", 0.5),
                )
            )
        
        # Step 5: Synthesize answer (if requested)
        answer = None
        citations = []
        
        if body.include_answer and search_results:
            answer, citations = await synthesize_answer(
                query=body.query,
                results=search_results,
                temporal_context=temporal_context,
            )
        
        processing_time = (time.perf_counter() - start_time) * 1000
        
        return SearchResponse(
            query=body.query,
            answer=answer,
            results=search_results,
            citations=citations,
            temporal_context=temporal_context,
            processing_time_ms=processing_time,
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@router.post(
    "/search/raw",
    response_model=SearchResponse,
    summary="Search without synthesis",
    description="Perform a search and return raw results without AI synthesis (faster).",
)
@limiter.limit("30/minute")
async def search_raw(request: Request, body: SearchRequest) -> SearchResponse:
    """Fast search without answer synthesis."""
    body.include_answer = False
    return await search(request, body)


@router.post(
    "/search/stream",
    summary="Search with streaming synthesis",
    description="Perform a search and stream the AI-synthesized answer in real-time using SSE.",
)
@limiter.limit("30/minute")
async def search_stream(request: Request, body: SearchRequest):
    """
    Streaming search with Server-Sent Events.
    
    Returns results first, then streams the answer as it's generated.
    """
    settings = get_settings()
    
    async def event_generator():
        try:
            # Step 1: Analyze temporal intent
            temporal_intent, temporal_urgency = detect_temporal_intent(body.query)
            
            temporal_context = TemporalContext(
                query_temporal_intent=temporal_intent,
                temporal_urgency=temporal_urgency,
                current_date=datetime.now().strftime("%Y-%m-%d"),
            )
            
            # Step 2: Search sources
            raw_results = []
            
            if settings.tavily_api_key:
                tavily_results = await search_tavily(
                    query=body.query,
                    max_results=settings.max_search_results,
                    freshness=body.freshness,
                    include_domains=body.include_domains,
                    exclude_domains=body.exclude_domains,
                )
                raw_results.extend(tavily_results)
            
            if not raw_results:
                ddg_results = await search_duckduckgo(
                    query=body.query,
                    max_results=settings.max_search_results,
                )
                raw_results.extend(ddg_results)
            
            if not raw_results:
                yield f"data: {json.dumps({'type': 'error', 'content': 'No results found'})}\n\n"
                return
            
            # Step 3: Rerank
            ranked_results = await rerank_results(
                query=body.query,
                results=raw_results,
                temporal_urgency=temporal_urgency,
                max_results=body.max_results,
            )
            
            # Step 4: Convert to SearchResult models
            search_results = []
            for result in ranked_results:
                freshness = calculate_freshness_score(result.get("published_date"))
                search_results.append(
                    SearchResult(
                        title=result.get("title", ""),
                        url=result.get("url", ""),
                        content=result.get("content", ""),
                        score=result.get("score", 0.5),
                        published_date=result.get("published_date"),
                        freshness_score=freshness,
                        authority_score=result.get("authority_score", 0.5),
                    )
                )
            
            # Send results first
            results_data = {
                "type": "results",
                "results": [r.model_dump(mode="json") for r in search_results],
                "temporal_context": temporal_context.model_dump(),
            }
            yield f"data: {json.dumps(results_data)}\n\n"
            
            # Step 5: Stream answer
            yield f"data: {json.dumps({'type': 'answer_start'})}\n\n"
            
            async for chunk in synthesize_answer_stream(
                query=body.query,
                results=search_results,
                temporal_context=temporal_context,
            ):
                yield f"data: {json.dumps({'type': 'answer_chunk', 'content': chunk})}\n\n"
            
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# === Deep Research Endpoints ===

@router.post(
    "/research/deep",
    summary="Deep research with multi-dimensional analysis",
    description="Decompose a query into dimensions, search each in parallel, and generate a comprehensive report.",
)
@limiter.limit("5/minute")
async def deep_research(request: Request, body: DeepResearchRequest):
    """
    Run deep research with streaming progress updates.
    
    Returns SSE events:
    - plan_ready: Research plan with dimensions
    - dimension_start/complete: Progress per dimension
    - report_chunk: Streaming report content
    - done: Final summary
    """
    from app.agents.deep_research import run_deep_research
    
    return StreamingResponse(
        run_deep_research(
            query=body.query,
            max_dimensions=body.max_dimensions,
            max_sources_per_dim=body.max_sources_per_dim,
            max_total_searches=body.max_total_searches,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post(
    "/search/heavy",
    summary="Heavy search with content scraping",
    description="Search with full content extraction from top sources for richer answers.",
)
@limiter.limit("10/minute")
async def heavy_search(request: Request, body: SearchRequest):
    """
    Heavy search with content scraping.
    
    Scrapes full content from top results instead of just snippets,
    providing richer context for answer generation.
    """
    from app.agents.heavy_search import run_heavy_search
    
    return StreamingResponse(
        run_heavy_search(
            query=body.query,
            max_results=body.max_results,
            max_scrape=5,
            freshness=body.freshness,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get(
    "/images",
    summary="Search for images",
    description="Search for images related to a query using Brave Image Search.",
)
@limiter.limit("60/minute")
async def image_search(request: Request, query: str, max_results: int = 6):
    """
    Search for images related to a query.
    
    Returns a list of image results with thumbnails and source URLs.
    """
    from app.sources.images import search_images
    
    if not query:
        raise HTTPException(status_code=400, detail="Query is required")
    
    images = await search_images(query=query, max_results=max_results)
    
    return {"query": query, "images": images}


# === SearXNG Search (pure - no LLM) ===

@router.post(
    "/search/searxng",
    summary="Search using SearXNG + embedding reranking",
    description="Uses SearXNG meta-search with embedding reranking. No LLM synthesis.",
)
@limiter.limit("20/minute")
async def searxng_search(request: Request, body: SearchRequest):
    """
    Search using SearXNG with embedding reranking only.
    
    This endpoint uses your SearXNG instance for 50+ results
    and reranks with embeddings. No LLM synthesis.
    """
    import json
    from app.sources.searxng import search_searxng
    from app.reranking.embeddings import compute_bi_encoder_scores
    
    async def event_generator():
        try:
            # Step 1: Search SearXNG
            yield f"data: {json.dumps({'type': 'status', 'message': 'Searching SearXNG...'})}\n\n"
            
            time_range = {"day": "day", "week": "week", "month": "month"}.get(body.freshness)
            raw_results = await search_searxng(
                query=body.query,
                max_results=50,
                time_range=time_range,
            )
            
            if not raw_results:
                yield f"data: {json.dumps({'type': 'error', 'message': 'No results from SearXNG'})}\n\n"
                return
            
            yield f"data: {json.dumps({'type': 'searxng_complete', 'count': len(raw_results)})}\n\n"
            
            # Step 2: Rerank with embeddings
            yield f"data: {json.dumps({'type': 'status', 'message': 'Reranking with embeddings...'})}\n\n"
            
            docs = [f"{r.get('title', '')}. {r.get('content', '')[:500]}" for r in raw_results]
            scores = compute_bi_encoder_scores(body.query, docs)
            
            for i, result in enumerate(raw_results):
                result["embedding_score"] = scores[i]
                orig_score = result.get("score", 0.5)
                result["score"] = (scores[i] * 0.7) + (orig_score * 0.3)
            
            raw_results.sort(key=lambda x: x["score"], reverse=True)
            final_results = raw_results[:body.max_results]
            
            # Step 3: Return results (no LLM)
            yield f"data: {json.dumps({'type': 'results', 'results': [{'title': r.get('title'), 'url': r.get('url'), 'content': r.get('content', '')[:300], 'score': round(r.get('score', 0), 3), 'source': r.get('source')} for r in final_results]})}\n\n"
            
            yield f"data: {json.dumps({'type': 'done', 'total_raw': len(raw_results), 'returned': len(final_results)})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


# === Code Search (GitHub, StackOverflow) ===

@router.post(
    "/search/code",
    summary="Search code repositories and programming Q&A",
    description="Uses SearXNG with GitHub, StackOverflow, and code-focused engines.",
)
@limiter.limit("20/minute")
async def code_search(request: Request, body: SearchRequest):
    """
    Search for code, programming solutions, and documentation.
    Uses GitHub, StackOverflow, GitLab, and other code-focused engines.
    """
    import json
    from app.sources.searxng import search_searxng
    from app.reranking.embeddings import compute_bi_encoder_scores
    
    async def event_generator():
        try:
            yield f"data: {json.dumps({'type': 'status', 'message': 'Searching code repositories...'})}\n\n"
            
            # Use code-specific engines
            raw_results = await search_searxng(
                query=body.query,
                max_results=50,
                categories=["it"],  # IT category includes code engines
                engines=["github", "stackoverflow", "gitlab", "npm", "pypi", "crates.io", "packagist"],
            )
            
            if not raw_results:
                yield f"data: {json.dumps({'type': 'error', 'message': 'No code results found'})}\n\n"
                return
            
            yield f"data: {json.dumps({'type': 'search_complete', 'count': len(raw_results)})}\n\n"
            
            # Rerank with embeddings
            yield f"data: {json.dumps({'type': 'status', 'message': 'Ranking by relevance...'})}\n\n"
            
            docs = [f"{r.get('title', '')}. {r.get('content', '')[:500]}" for r in raw_results]
            scores = compute_bi_encoder_scores(body.query, docs)
            
            for i, result in enumerate(raw_results):
                result["embedding_score"] = scores[i]
                orig_score = result.get("score", 0.5)
                result["score"] = (scores[i] * 0.7) + (orig_score * 0.3)
            
            raw_results.sort(key=lambda x: x["score"], reverse=True)
            final_results = raw_results[:body.max_results]
            
            yield f"data: {json.dumps({'type': 'results', 'results': [{'title': r.get('title'), 'url': r.get('url'), 'content': r.get('content', '')[:300], 'score': round(r.get('score', 0), 3), 'source': r.get('source')} for r in final_results]})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'total_raw': len(raw_results), 'returned': len(final_results)})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


# === Academic Search (arXiv, Google Scholar) ===

@router.post(
    "/search/academic",
    summary="Search academic papers and research",
    description="Uses SearXNG with arXiv, Google Scholar, Semantic Scholar, and academic engines.",
)
@limiter.limit("20/minute")
async def academic_search(request: Request, body: SearchRequest):
    """
    Search for academic papers, research, and scientific content.
    Uses arXiv, Google Scholar, Semantic Scholar, PubMed, and other academic engines.
    """
    import json
    from app.sources.searxng import search_searxng
    from app.reranking.embeddings import compute_bi_encoder_scores
    
    async def event_generator():
        try:
            yield f"data: {json.dumps({'type': 'status', 'message': 'Searching academic sources...'})}\n\n"
            
            # Use academic engines
            raw_results = await search_searxng(
                query=body.query,
                max_results=50,
                categories=["science"],
                engines=["arxiv", "google scholar", "semantic scholar", "pubmed", "base", "crossref"],
            )
            
            if not raw_results:
                yield f"data: {json.dumps({'type': 'error', 'message': 'No academic results found'})}\n\n"
                return
            
            yield f"data: {json.dumps({'type': 'search_complete', 'count': len(raw_results)})}\n\n"
            
            # Rerank with embeddings
            yield f"data: {json.dumps({'type': 'status', 'message': 'Ranking by relevance...'})}\n\n"
            
            docs = [f"{r.get('title', '')}. {r.get('content', '')[:500]}" for r in raw_results]
            scores = compute_bi_encoder_scores(body.query, docs)
            
            for i, result in enumerate(raw_results):
                result["embedding_score"] = scores[i]
                orig_score = result.get("score", 0.5)
                result["score"] = (scores[i] * 0.7) + (orig_score * 0.3)
            
            raw_results.sort(key=lambda x: x["score"], reverse=True)
            final_results = raw_results[:body.max_results]
            
            yield f"data: {json.dumps({'type': 'results', 'results': [{'title': r.get('title'), 'url': r.get('url'), 'content': r.get('content', '')[:300], 'score': round(r.get('score', 0), 3), 'source': r.get('source')} for r in final_results]})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'total_raw': len(raw_results), 'returned': len(final_results)})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


# === Browser Agent ===

@router.post(
    "/agent/browse",
    summary="Browser agent - navigate and extract from websites",
    description="Uses E2B sandbox. stream_visual=true for Chrome with live video, false for Camoufox stealth.",
)
@limiter.limit("10/minute")
async def browser_agent(request: Request, body: BrowseRequest):
    """
    Browser agent with two modes:
    - stream_visual=true: Chrome with live video stream (5 min timeout)
    - stream_visual=false: Camoufox stealth headless (faster, anti-bot)
    """
    
    async def event_generator():
        try:
            if body.stream_visual:
                from app.agents.browser_agent import run_browser_agent
                async for event in run_browser_agent(body.task, body.url):
                    yield f"data: {json.dumps(event)}\n\n"
            else:
                from app.agents.browser_agent_v2 import run_browser_agent_v2
                async for event in run_browser_agent_v2(body.task, body.url):
                    yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )

