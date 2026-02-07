"""Tavily search source integration.

Tavily provides high-quality, AI-optimized search results.
"""

from datetime import datetime
from typing import Literal, Optional

import httpx

from app.config import get_settings


async def search_tavily(
    query: str,
    max_results: int = 10,
    freshness: Literal["day", "week", "month", "year", "any"] = "any",
    include_domains: Optional[list[str]] = None,
    exclude_domains: Optional[list[str]] = None,
    search_depth: Literal["basic", "advanced"] = "advanced",
) -> list[dict]:
    """
    Search using Tavily API.
    
    Args:
        query: Search query
        max_results: Maximum results to return
        freshness: Filter by recency
        include_domains: Only include these domains
        exclude_domains: Exclude these domains
        search_depth: "basic" (fast) or "advanced" (thorough)
        
    Returns:
        List of result dicts with title, url, content, published_date, score
    """
    settings = get_settings()
    
    if not settings.tavily_api_key:
        return []
    
    # Map freshness to Tavily's days parameter
    days_map = {
        "day": 1,
        "week": 7,
        "month": 30,
        "year": 365,
        "any": None,
    }
    
    payload = {
        "api_key": settings.tavily_api_key,
        "query": query,
        "search_depth": search_depth,
        "max_results": max_results,
        "include_answer": False,
        "include_raw_content": False,
    }
    
    # Add optional filters
    if days_map.get(freshness):
        payload["days"] = days_map[freshness]
    
    if include_domains:
        payload["include_domains"] = include_domains
    
    if exclude_domains:
        payload["exclude_domains"] = exclude_domains
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.tavily.com/search",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
        
        results = []
        for item in data.get("results", []):
            # Parse published date if available
            pub_date = None
            if "published_date" in item and item["published_date"]:
                try:
                    pub_date = datetime.fromisoformat(
                        item["published_date"].replace("Z", "+00:00")
                    )
                except (ValueError, TypeError):
                    pass
            
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": item.get("content", ""),
                "published_date": pub_date,
                "score": item.get("score", 0.5),
                "source": "tavily",
            })
        
        return results
        
    except httpx.HTTPError as e:
        print(f"Tavily search error: {e}")
        return []
    except Exception as e:
        print(f"Tavily unexpected error: {e}")
        return []
