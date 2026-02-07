"""Image Search source.

Uses Tavily API with include_images=True for image search.
Falls back to Brave Image Search if Tavily unavailable.
"""

from typing import Optional

import httpx

from app.config import get_settings


async def search_images(
    query: str,
    max_results: int = 6,
) -> list[dict]:
    """
    Search for images using available APIs.
    
    Priority:
    1. Tavily (include_images=True) - uses existing API key
    2. Brave Image Search - fallback
    
    Args:
        query: Search query
        max_results: Maximum images to return
        
    Returns:
        List of image results with url, thumbnail, title
    """
    settings = get_settings()
    
    # Try Tavily first (same API key as main search)
    if settings.tavily_api_key:
        images = await _search_tavily_images(query, max_results)
        if images:
            return images
    
    # Fallback to Brave
    if settings.brave_api_key:
        return await _search_brave_images(query, max_results)
    
    return []


async def _search_tavily_images(query: str, max_results: int) -> list[dict]:
    """Search images using Tavily API."""
    settings = get_settings()
    
    payload = {
        "api_key": settings.tavily_api_key,
        "query": query,
        "search_depth": "basic",
        "max_results": 5,  # We just need images, not full results
        "include_images": True,
        "include_image_descriptions": True,
        "include_answer": False,
    }
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                "https://api.tavily.com/search",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
        
        results = []
        images = data.get("images", [])
        
        for img in images[:max_results]:
            if isinstance(img, str):
                # Simple URL format
                results.append({
                    "url": img,
                    "thumbnail": img,
                    "title": "",
                })
            elif isinstance(img, dict):
                # Dict format with description
                results.append({
                    "url": img.get("url", ""),
                    "thumbnail": img.get("url", ""),
                    "title": img.get("description", ""),
                })
        
        return results
        
    except Exception as e:
        print(f"Tavily image search error: {e}")
        return []


async def _search_brave_images(query: str, max_results: int) -> list[dict]:
    """Search images using Brave Image Search API."""
    settings = get_settings()
    
    params = {
        "q": query,
        "count": min(max_results, 20),
        "safesearch": "moderate",
    }
    
    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": settings.brave_api_key,
    }
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                "https://api.search.brave.com/res/v1/images/search",
                params=params,
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()
        
        results = []
        images = data.get("results", [])
        
        for img in images[:max_results]:
            results.append({
                "url": img.get("properties", {}).get("url", ""),
                "thumbnail": img.get("thumbnail", {}).get("src", ""),
                "title": img.get("title", ""),
            })
        
        return results
        
    except Exception as e:
        print(f"Brave image search error: {e}")
        return []
