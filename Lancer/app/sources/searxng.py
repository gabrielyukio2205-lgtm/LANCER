"""SearXNG meta-search source.

Uses a self-hosted SearXNG instance for comprehensive search results
from multiple engines (Google, Bing, DDG, etc.) without API costs.
"""

from typing import Optional
from datetime import datetime

import httpx

from app.config import get_settings


# Default SearXNG instance (your HF Space)
DEFAULT_SEARXNG_URL = "https://madras1-searxng-space.hf.space"

# No fallbacks - use only your instance
FALLBACK_INSTANCES = []


async def search_searxng(
    query: str,
    max_results: int = 50,
    categories: Optional[list[str]] = None,
    engines: Optional[list[str]] = None,
    language: str = "all",
    time_range: Optional[str] = None,
    searxng_url: Optional[str] = None,
) -> list[dict]:
    """
    Search using SearXNG meta-search engine.
    
    Returns many more results than API-based sources, making
    embedding-based reranking valuable.
    
    Args:
        query: Search query
        max_results: Maximum results to return (can be 50-100+)
        categories: Search categories (general, news, science, etc.)
        engines: Specific engines to use (google, bing, etc.)
        language: Language code (en, pt, all)
        time_range: Time filter (day, week, month, year)
        searxng_url: Custom SearXNG instance URL
        
    Returns:
        List of search results with title, url, content, source
    """
    settings = get_settings()
    
    # Build instance list
    instances = []
    if searxng_url:
        instances.append(searxng_url)
    if hasattr(settings, 'searxng_url') and settings.searxng_url:
        instances.append(settings.searxng_url)
    instances.append(DEFAULT_SEARXNG_URL)
    instances.extend(FALLBACK_INSTANCES)
    
    # Build params
    params = {
        "q": query,
        "format": "json",
        "language": language,
    }
    
    if categories:
        params["categories"] = ",".join(categories)
    if engines:
        params["engines"] = ",".join(engines)
    if time_range:
        params["time_range"] = time_range
    
    # Try each instance
    for instance in instances:
        try:
            results = await _fetch_searxng(instance, params, max_results)
            if results:
                return results
        except Exception as e:
            print(f"SearXNG instance {instance} failed: {e}")
            continue
    
    return []


async def _fetch_searxng(
    instance_url: str,
    params: dict,
    max_results: int,
) -> list[dict]:
    """Fetch results from a SearXNG instance."""
    
    # Use browser-like headers to avoid blocks
    headers = {
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
    }
    
    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.get(
            f"{instance_url.rstrip('/')}/search",
            params=params,
            headers=headers,
        )
        response.raise_for_status()
        data = response.json()
    
    results = []
    for item in data.get("results", [])[:max_results]:
        result = {
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "content": item.get("content", ""),
            "source": f"searxng:{item.get('engine', 'unknown')}",
            "score": _calculate_score(item),
        }
        
        # Extract date if available
        published_date = item.get("publishedDate")
        if published_date:
            result["published_date"] = published_date
        
        results.append(result)
    
    return results


def _calculate_score(item: dict) -> float:
    """Calculate initial score based on position and engine."""
    # Base score from position (if available)
    position = item.get("position", 10)
    position_score = max(0.3, 1.0 - (position * 0.05))
    
    # Bonus for certain engines
    engine = item.get("engine", "").lower()
    engine_bonus = {
        "google": 0.1,
        "bing": 0.05,
        "duckduckgo": 0.05,
        "wikipedia": 0.1,
        "arxiv": 0.15,
        "google scholar": 0.15,
    }.get(engine, 0)
    
    return min(1.0, position_score + engine_bonus)


async def get_searxng_engines(searxng_url: Optional[str] = None) -> list[str]:
    """Get list of available engines from SearXNG instance."""
    url = searxng_url or DEFAULT_SEARXNG_URL
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{url}/config")
            response.raise_for_status()
            data = response.json()
        
        return [
            engine["name"] 
            for engine in data.get("engines", [])
            if not engine.get("disabled", False)
        ]
    except Exception:
        return []
