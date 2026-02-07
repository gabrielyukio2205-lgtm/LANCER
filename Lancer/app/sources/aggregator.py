"""Multi-source search aggregator.

Combines results from multiple search sources in parallel.
"""

import asyncio
from typing import Optional
from urllib.parse import urlparse

from app.config import get_settings
from app.sources.tavily import search_tavily
from app.sources.brave import search_brave
from app.sources.duckduckgo import search_duckduckgo
from app.sources.wikipedia import search_wikipedia
from app.sources.searxng import search_searxng


async def aggregate_search(
    query: str,
    max_results: int = 15,
    freshness: str = "any",
    include_wikipedia: bool = True,
    include_domains: Optional[list[str]] = None,
    exclude_domains: Optional[list[str]] = None,
) -> list[dict]:
    """
    Aggregate search results from multiple sources in parallel.
    
    Args:
        query: Search query
        max_results: Maximum total results to return
        freshness: Freshness filter (day, week, month, year, any)
        include_wikipedia: Whether to include Wikipedia results
        include_domains: Only include these domains (Tavily only)
        exclude_domains: Exclude these domains (Tavily only)
        
    Returns:
        Deduplicated, merged list of search results
    """
    settings = get_settings()
    
    # Build list of search tasks
    tasks = []
    source_names = []
    
    # SearXNG (if configured - free, high volume)
    if hasattr(settings, 'searxng_url') and settings.searxng_url:
        time_range = {"day": "day", "week": "week", "month": "month"}.get(freshness)
        tasks.append(search_searxng(
            query=query,
            max_results=15,
            time_range=time_range,
        ))
        source_names.append("searxng")
    
    # Tavily (primary source - if API key available)
    if settings.tavily_api_key:
        tasks.append(search_tavily(
            query=query,
            max_results=12,  # Primary source
            freshness=freshness,
            include_domains=include_domains,
            exclude_domains=exclude_domains,
        ))
        source_names.append("tavily")
    
    # Brave (secondary - limited quota, use sparingly)
    if settings.brave_api_key:
        tasks.append(search_brave(
            query=query,
            max_results=5,  # Reduced to save quota
            freshness=freshness,
        ))
        source_names.append("brave")
    
    # DuckDuckGo (always available, free)
    tasks.append(search_duckduckgo(
        query=query,
        max_results=12,  # Free, can use more
    ))
    source_names.append("duckduckgo")
    
    # Wikipedia (for context/background)
    if include_wikipedia:
        tasks.append(search_wikipedia(
            query=query,
            max_results=5,
        ))
        source_names.append("wikipedia")
    
    # Run all searches in parallel
    results_lists = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Merge results
    all_results = []
    for i, results in enumerate(results_lists):
        if isinstance(results, Exception):
            print(f"Source {source_names[i]} failed: {results}")
            continue
        if results:
            all_results.extend(results)
    
    # Deduplicate by URL
    seen_urls = set()
    unique_results = []
    
    for result in all_results:
        url = result.get("url", "")
        normalized_url = _normalize_url(url)
        
        if normalized_url not in seen_urls:
            seen_urls.add(normalized_url)
            unique_results.append(result)
    
    # Sort by score (descending)
    unique_results.sort(key=lambda x: x.get("score", 0), reverse=True)
    
    return unique_results[:max_results]


def _normalize_url(url: str) -> str:
    """Normalize URL for deduplication."""
    try:
        parsed = urlparse(url)
        # Remove www., trailing slashes, and query params for comparison
        host = parsed.netloc.replace("www.", "")
        path = parsed.path.rstrip("/")
        return f"{host}{path}".lower()
    except:
        return url.lower()


async def get_available_sources() -> list[str]:
    """Get list of available search sources based on configuration."""
    settings = get_settings()
    sources = ["duckduckgo", "wikipedia"]  # Always available
    
    if hasattr(settings, 'searxng_url') and settings.searxng_url:
        sources.append("searxng")
    if settings.tavily_api_key:
        sources.append("tavily")
    if settings.brave_api_key:
        sources.append("brave")
    
    return sources
