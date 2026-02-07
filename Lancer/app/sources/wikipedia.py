"""Wikipedia Search source.

Uses Wikipedia's free API for background/context information.
No API key required, unlimited usage.
"""

from datetime import datetime
from typing import Optional

import httpx


async def search_wikipedia(
    query: str,
    max_results: int = 5,
    language: str = "pt",
) -> list[dict]:
    """
    Search Wikipedia for relevant articles.
    
    Args:
        query: Search query
        max_results: Maximum results (1-10)
        language: Wikipedia language code (pt, en, es, etc)
        
    Returns:
        List of search results with title, url, content, score
    """
    base_url = f"https://{language}.wikipedia.org/w/api.php"
    
    # First, search for pages
    search_params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": min(max_results, 10),
        "format": "json",
        "utf8": 1,
    }
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Search for articles
            response = await client.get(base_url, params=search_params)
            response.raise_for_status()
            search_data = response.json()
            
            results = []
            search_results = search_data.get("query", {}).get("search", [])
            
            for i, item in enumerate(search_results):
                title = item.get("title", "")
                page_id = item.get("pageid")
                snippet = item.get("snippet", "")
                
                # Clean HTML from snippet
                snippet = _clean_html(snippet)
                
                # Get extract for better content
                extract = await _get_page_extract(client, base_url, page_id)
                
                results.append({
                    "title": f"Wikipedia: {title}",
                    "url": f"https://{language}.wikipedia.org/wiki/{title.replace(' ', '_')}",
                    "content": extract or snippet,
                    "published_date": None,  # Wikipedia doesn't provide this easily
                    "score": 0.7 - (i * 0.05),  # Lower base score (reference material)
                    "source": "wikipedia",
                })
            
            return results
            
    except Exception as e:
        print(f"Wikipedia search error: {e}")
        return []


async def _get_page_extract(
    client: httpx.AsyncClient,
    base_url: str,
    page_id: int,
) -> Optional[str]:
    """Get a short extract from a Wikipedia page."""
    params = {
        "action": "query",
        "pageids": page_id,
        "prop": "extracts",
        "exintro": True,
        "explaintext": True,
        "exsentences": 5,
        "format": "json",
    }
    
    try:
        response = await client.get(base_url, params=params)
        data = response.json()
        pages = data.get("query", {}).get("pages", {})
        page = pages.get(str(page_id), {})
        return page.get("extract", "")
    except:
        return None


def _clean_html(text: str) -> str:
    """Remove HTML tags from text."""
    import re
    clean = re.sub(r'<[^>]+>', '', text)
    return clean.strip()
