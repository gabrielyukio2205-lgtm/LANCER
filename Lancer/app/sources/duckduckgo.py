"""DuckDuckGo search source (free fallback).

Uses the duckduckgo_search library for free web search.
"""

from datetime import datetime, timedelta
from typing import Optional

import httpx


async def search_duckduckgo(
    query: str,
    max_results: int = 10,
    region: str = "wt-wt",  # Worldwide
) -> list[dict]:
    """
    Search using DuckDuckGo (free, no API key required).
    
    This is a fallback when other sources are unavailable.
    Uses the HTML endpoint for basic search.
    
    Args:
        query: Search query
        max_results: Maximum results to return
        region: Region code
        
    Returns:
        List of result dicts with title, url, content
    """
    try:
        # Use DuckDuckGo HTML API (lightweight, no JS needed)
        params = {
            "q": query,
            "kl": region,
            "kp": "-1",  # Safe search off
        }
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        
        async with httpx.AsyncClient(timeout=15.0) as client:
            # Use DuckDuckGo Lite (simpler to parse)
            response = await client.get(
                "https://lite.duckduckgo.com/lite/",
                params=params,
                headers=headers,
                follow_redirects=True,
            )
            response.raise_for_status()
            html = response.text
        
        # Simple HTML parsing for results
        results = parse_ddg_lite_results(html, max_results)
        return results
        
    except Exception as e:
        print(f"DuckDuckGo search error: {e}")
        return []


def parse_ddg_lite_results(html: str, max_results: int) -> list[dict]:
    """
    Parse DuckDuckGo Lite HTML results.
    
    This is a simple parser for the lite version of DDG.
    """
    import re
    
    results = []
    
    # Find all result links (class="result-link")
    # Pattern: <a rel="nofollow" href="URL" class='result-link'>TITLE</a>
    link_pattern = r'<a[^>]*class=["\']result-link["\'][^>]*href=["\']([^"\']+)["\'][^>]*>([^<]+)</a>'
    
    # Find snippets (class="result-snippet")
    snippet_pattern = r'<td[^>]*class=["\']result-snippet["\'][^>]*>([^<]+)</td>'
    
    links = re.findall(link_pattern, html, re.IGNORECASE)
    snippets = re.findall(snippet_pattern, html, re.IGNORECASE)
    
    for i, (url, title) in enumerate(links[:max_results]):
        content = snippets[i] if i < len(snippets) else ""
        
        # Clean up HTML entities
        title = title.strip()
        content = content.strip()
        
        # Skip DuckDuckGo internal links
        if "duckduckgo.com" in url:
            continue
        
        results.append({
            "title": title,
            "url": url,
            "content": content,
            "published_date": None,  # DDG Lite doesn't provide dates
            "score": 0.5,  # Neutral score, will be reranked
            "source": "duckduckgo",
        })
    
    return results[:max_results]
