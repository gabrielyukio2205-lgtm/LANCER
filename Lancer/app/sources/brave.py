"""Brave Search API source.

Official Brave Search API with 2000 free queries/month.
https://api.search.brave.com/
"""

from datetime import datetime
from typing import Optional

import httpx

from app.config import get_settings


async def search_brave(
    query: str,
    max_results: int = 10,
    freshness: str = "any",
    country: str = "BR",
) -> list[dict]:
    """
    Search using Brave Search API.
    
    Args:
        query: Search query
        max_results: Maximum results (1-20)
        freshness: 'pd' (day), 'pw' (week), 'pm' (month), 'py' (year), or None
        country: Country code for results
        
    Returns:
        List of search results with title, url, content, published_date, score
    """
    settings = get_settings()
    
    if not settings.brave_api_key:
        return []
    
    # Map freshness to Brave format
    freshness_map = {
        "day": "pd",
        "week": "pw", 
        "month": "pm",
        "year": "py",
        "any": None,
    }
    brave_freshness = freshness_map.get(freshness)
    
    params = {
        "q": query,
        "count": min(max_results, 20),
        "country": country,
        "search_lang": "pt",
        "text_decorations": False,
    }
    
    if brave_freshness:
        params["freshness"] = brave_freshness
    
    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": settings.brave_api_key,
    }
    
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(
                "https://api.search.brave.com/res/v1/web/search",
                params=params,
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()
        
        results = []
        web_results = data.get("web", {}).get("results", [])
        
        for i, item in enumerate(web_results):
            # Try to parse age/date
            published_date = None
            age = item.get("age")
            if age:
                published_date = _parse_brave_age(age)
            
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": item.get("description", ""),
                "published_date": published_date,
                "score": 0.8 - (i * 0.05),  # Decay score by position
                "source": "brave",
            })
        
        return results
        
    except httpx.HTTPStatusError as e:
        print(f"Brave API error: {e.response.status_code}")
        return []
    except Exception as e:
        print(f"Brave search error: {e}")
        return []


def _parse_brave_age(age: str) -> Optional[datetime]:
    """Parse Brave's age string like '2 days ago' to datetime."""
    import re
    
    now = datetime.now()
    
    patterns = [
        (r"(\d+)\s*hour", lambda m: now.replace(hour=now.hour - int(m.group(1)))),
        (r"(\d+)\s*day", lambda m: now.replace(day=now.day - int(m.group(1)))),
        (r"(\d+)\s*week", lambda m: now.replace(day=now.day - int(m.group(1)) * 7)),
        (r"(\d+)\s*month", lambda m: now.replace(month=now.month - int(m.group(1)))),
    ]
    
    for pattern, func in patterns:
        match = re.search(pattern, age, re.IGNORECASE)
        if match:
            try:
                return func(match)
            except ValueError:
                pass
    
    return None
