"""Content Scraper.

Extracts clean text content from URLs for deeper analysis.
"""

import asyncio
from typing import Optional

import httpx


async def scrape_url_content(
    url: str,
    max_chars: int = 5000,
    timeout: float = 10.0,
) -> Optional[str]:
    """
    Scrape and extract clean text content from a URL.
    
    Args:
        url: URL to scrape
        max_chars: Maximum characters to return
        timeout: Request timeout in seconds
        
    Returns:
        Extracted text content or None if failed
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }
        
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            html = response.text
        
        # Try trafilatura first (best quality)
        try:
            import trafilatura
            text = trafilatura.extract(
                html,
                include_comments=False,
                include_tables=True,
                no_fallback=False,
            )
            if text:
                return text[:max_chars]
        except ImportError:
            pass
        
        # Fallback: simple HTML extraction
        text = _simple_extract(html)
        return text[:max_chars] if text else None
        
    except Exception as e:
        print(f"Scrape error for {url}: {e}")
        return None


def _simple_extract(html: str) -> str:
    """Simple HTML text extraction without external libs."""
    import re
    
    # Remove script and style tags
    html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<head[^>]*>.*?</head>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<nav[^>]*>.*?</nav>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<footer[^>]*>.*?</footer>', '', html, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove all HTML tags
    text = re.sub(r'<[^>]+>', ' ', html)
    
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text


async def scrape_multiple_urls(
    urls: list[str],
    max_chars_per_url: int = 3000,
    max_concurrent: int = 5,
) -> dict[str, Optional[str]]:
    """
    Scrape multiple URLs concurrently.
    
    Args:
        urls: List of URLs to scrape
        max_chars_per_url: Max chars per URL
        max_concurrent: Max concurrent requests
        
    Returns:
        Dict mapping URL to extracted content (or None if failed)
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def scrape_with_semaphore(url: str) -> tuple[str, Optional[str]]:
        async with semaphore:
            content = await scrape_url_content(url, max_chars_per_url)
            return url, content
    
    tasks = [scrape_with_semaphore(url) for url in urls]
    results = await asyncio.gather(*tasks)
    
    return dict(results)
