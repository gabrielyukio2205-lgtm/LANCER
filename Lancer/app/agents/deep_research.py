"""Deep Research Orchestrator.

Coordinates the full deep research pipeline:
1. Planning (query decomposition)
2. Parallel searching (multiple dimensions)
3. Report synthesis
"""

import asyncio
import json
import time
from typing import AsyncIterator, Optional

from app.agents.planner import create_research_plan, ResearchPlan, ResearchDimension
from app.agents.llm_client import generate_completion_stream
from app.reranking.pipeline import rerank_results
from app.config import get_settings


class DimensionResult:
    """Results from researching a single dimension."""
    
    def __init__(self, dimension: ResearchDimension):
        self.dimension = dimension
        self.results: list[dict] = []
        self.error: Optional[str] = None


async def run_deep_research(
    query: str,
    max_dimensions: int = 6,
    max_sources_per_dim: int = 5,
    max_total_searches: int = 20,
) -> AsyncIterator[str]:
    """
    Run a deep research pipeline with streaming progress.
    
    Yields SSE-formatted events as the research progresses.
    
    Args:
        query: The research query
        max_dimensions: Maximum dimensions to research
        max_sources_per_dim: Max results per dimension
        max_total_searches: Total Tavily API calls allowed
        
    Yields:
        SSE event strings in format: data: {json}\n\n
    """
    start_time = time.perf_counter()
    settings = get_settings()
    
    try:
        # === PHASE 1: PLANNING ===
        yield _sse_event("status", {"phase": "planning", "message": "Analyzing query..."})
        
        plan = await create_research_plan(query, max_dimensions)
        
        yield _sse_event("plan_ready", {
            "refined_query": plan.refined_query,
            "dimensions": [
                {"name": d.name, "description": d.description, "priority": d.priority}
                for d in plan.dimensions
            ],
            "estimated_sources": plan.estimated_sources,
        })
        
        # === PHASE 2: PARALLEL SEARCHING ===
        yield _sse_event("status", {"phase": "searching", "message": "Researching dimensions..."})
        
        # Distribute search budget across dimensions
        num_dimensions = len(plan.dimensions)
        searches_per_dim = max(1, max_total_searches // num_dimensions)
        
        dimension_results: list[DimensionResult] = []
        
        # Search dimensions in parallel batches
        for i, dimension in enumerate(plan.dimensions):
            yield _sse_event("dimension_start", {
                "index": i + 1,
                "total": num_dimensions,
                "name": dimension.name,
                "query": dimension.search_query,
            })
            
            # Search this dimension
            result = await _search_dimension(
                dimension=dimension,
                max_results=max_sources_per_dim,
                max_searches=searches_per_dim,
            )
            dimension_results.append(result)
            
            yield _sse_event("dimension_complete", {
                "index": i + 1,
                "name": dimension.name,
                "results_count": len(result.results),
                "error": result.error,
            })
            
            # Small delay to avoid rate limits
            await asyncio.sleep(0.1)
        
        # === PHASE 3: SYNTHESIS ===
        yield _sse_event("status", {"phase": "synthesizing", "message": "Generating report..."})
        yield _sse_event("synthesis_start", {})
        
        # Stream the report generation
        async for chunk in _synthesize_report_stream(query, plan, dimension_results):
            yield _sse_event("report_chunk", {"content": chunk})
        
        # === COMPLETE ===
        total_time = time.perf_counter() - start_time
        total_sources = sum(len(r.results) for r in dimension_results)
        
        yield _sse_event("done", {
            "total_sources": total_sources,
            "total_dimensions": num_dimensions,
            "total_time_seconds": round(total_time, 2),
        })
        
    except Exception as e:
        yield _sse_event("error", {"message": str(e)})


async def _search_dimension(
    dimension: ResearchDimension,
    max_results: int = 5,
    max_searches: int = 2,
) -> DimensionResult:
    """Search a single dimension using the aggregator."""
    from app.sources.aggregator import aggregate_search
    
    result = DimensionResult(dimension)
    
    try:
        # Use aggregator to search all sources
        all_results = await aggregate_search(
            query=dimension.search_query,
            max_results=max_results + 3,  # Get extra for reranking
            include_wikipedia=True,
        )
        
        # Light reranking (use embeddings when we have many results from SearXNG)
        if all_results:
            use_embeddings = len(all_results) > 15
            ranked = await rerank_results(
                query=dimension.search_query,
                results=all_results,
                temporal_urgency=0.5,
                max_results=max_results,
                use_embeddings=use_embeddings,
            )
            result.results = ranked
        
    except Exception as e:
        result.error = str(e)
    
    return result


async def _synthesize_report_stream(
    original_query: str,
    plan: ResearchPlan,
    dimension_results: list[DimensionResult],
) -> AsyncIterator[str]:
    """Stream the synthesis of the final report."""
    
    # Build context from all dimension results
    context_parts = []
    all_sources = []
    source_index = 1
    
    for dr in dimension_results:
        if dr.results:
            context_parts.append(f"\n## {dr.dimension.name}\n")
            for r in dr.results:
                context_parts.append(
                    f"[{source_index}] {r.get('title', 'Untitled')}\n"
                    f"   URL: {r.get('url', '')}\n"
                    f"   Content: {r.get('content', '')[:400]}...\n"
                )
                all_sources.append({
                    "index": source_index,
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                })
                source_index += 1
    
    context = "\n".join(context_parts)
    
    # Build synthesis prompt
    prompt = f"""You are a research analyst. Create a comprehensive research report based on the gathered information.

ORIGINAL QUERY: {original_query}
REFINED QUERY: {plan.refined_query}

RESEARCH DIMENSIONS:
{', '.join(d.name for d in plan.dimensions)}

GATHERED INFORMATION:
{context}

INSTRUCTIONS:
1. Write a comprehensive research report in Markdown format
2. Start with an Executive Summary (2-3 paragraphs)
3. Create a section for each research dimension
4. Use citations [1], [2], etc. to reference sources
5. Include a Conclusion section
6. Be thorough but concise
7. Write in the same language as the query
8. Use headers (##) to organize sections

Generate the report:"""

    messages = [
        {"role": "system", "content": "You are a research analyst creating detailed reports."},
        {"role": "user", "content": prompt},
    ]
    
    try:
        async for chunk in generate_completion_stream(messages, temperature=0.4):
            yield chunk
        
        # Append sources at the end
        yield "\n\n---\n\n## Sources\n\n"
        for src in all_sources:
            yield f"[{src['index']}] [{src['title']}]({src['url']})\n"
            
    except Exception as e:
        yield f"\n\n**Error generating report:** {e}"


def _sse_event(event_type: str, data: dict) -> str:
    """Format an SSE event."""
    payload = {"type": event_type, **data}
    return f"data: {json.dumps(payload)}\n\n"
