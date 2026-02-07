"""Browser Agent v3 - Batch extraction, minimal LLM calls.

Flow:
1. Search DuckDuckGo → Get top links
2. Batch extract content from 3-5 pages (NO LLM calls)
3. Send ALL content to LLM in ONE call
4. LLM either responds OR requests specific follow-up

Target: 2-4 LLM calls max instead of 40+
"""

import os
import re
import shlex
import logging
import time
from typing import AsyncGenerator, Optional, List, Dict

from app.config import get_settings
from app.agents.llm_client import generate_completion

logger = logging.getLogger(__name__)

# Config
MAX_PAGES_TO_EXTRACT = 4
TIMEOUT_SECONDS = 300
CONTENT_PER_PAGE = 2000


async def run_browser_agent_v3(
    task: str,
    url: Optional[str] = None,
) -> AsyncGenerator[dict, None]:
    """Run browser agent with batch extraction - minimal LLM calls."""
    settings = get_settings()
    
    if not settings.e2b_api_key:
        yield {"type": "error", "message": "E2B_API_KEY not configured"}
        return
    
    start_time = time.time()
    yield {"type": "status", "message": "🚀 Initializing agent..."}
    
    desktop = None
    
    try:
        from e2b_desktop import Sandbox
        
        os.environ["E2B_API_KEY"] = settings.e2b_api_key
        
        yield {"type": "status", "message": "🖥️ Creating virtual desktop..."}
        desktop = Sandbox.create(timeout=600)
        
        # Start streaming
        stream_url = None
        try:
            desktop.stream.start(require_auth=True)
            auth_key = desktop.stream.get_auth_key()
            stream_url = desktop.stream.get_url(auth_key=auth_key)
            yield {"type": "stream", "url": stream_url}
            desktop.wait(2000)
        except Exception as e:
            logger.warning(f"Could not start stream: {e}")
        
        # Launch Chrome
        yield {"type": "status", "message": "🌐 Launching browser..."}
        chrome_flags = "--no-sandbox --disable-gpu --start-maximized --no-first-run --disable-default-apps --disable-popup-blocking --disable-translate --no-default-browser-check"
        desktop.commands.run(f"google-chrome {chrome_flags} 'about:blank' &", background=True)
        desktop.wait(3000)
        desktop.press("enter")
        desktop.wait(1000)
        
        # Phase 1: Search
        yield {"type": "status", "message": f"🔍 Searching: {task[:50]}..."}
        search_query = task.replace(' ', '+')
        search_url = f"https://html.duckduckgo.com/html/?q={search_query}"
        
        desktop.commands.run(f"google-chrome {shlex.quote(search_url)} &", background=True)
        desktop.wait(3000)
        
        # Extract search results page
        search_content = await _extract_page_content(desktop, search_url)
        
        # Parse links from search results
        links = _extract_links_from_search(search_content, task)
        logger.info(f"Found {len(links)} relevant links")
        
        if not links:
            # Fallback: just use search content
            links = [search_url]
        
        # Phase 2: Batch extract from top pages
        extracted_pages: List[Dict] = []
        
        for i, link in enumerate(links[:MAX_PAGES_TO_EXTRACT]):
            remaining = int(TIMEOUT_SECONDS - (time.time() - start_time))
            if remaining < 30:
                break
                
            yield {"type": "status", "message": f"📊 Extracting page {i+1}/{min(len(links), MAX_PAGES_TO_EXTRACT)}... ({remaining}s remaining)"}
            
            try:
                desktop.commands.run(f"google-chrome {shlex.quote(link)} &", background=True)
                desktop.wait(2500)
                
                content = await _extract_page_content(desktop, link)
                if content and len(content) > 100:
                    extracted_pages.append({
                        "url": link,
                        "content": content[:CONTENT_PER_PAGE]
                    })
                    logger.info(f"Extracted {len(content)} chars from {link[:50]}")
            except Exception as e:
                logger.warning(f"Failed to extract {link}: {e}")
        
        # Phase 3: ONE LLM call with all content
        yield {"type": "status", "message": "🤔 Analyzing all sources..."}
        
        # Build context
        pages_context = "\n\n---\n\n".join([
            f"SOURCE {i+1}: {p['url']}\n{p['content']}"
            for i, p in enumerate(extracted_pages)
        ])
        
        prompt = f"""Você é um assistente de pesquisa. Analise as fontes abaixo e responda à pergunta.

PERGUNTA: {task}

FONTES COLETADAS:
{pages_context if pages_context else "(Nenhum conteúdo extraído)"}

INSTRUÇÕES:
1. Responda baseado APENAS nas fontes acima
2. Use **negrito** para valores importantes (preços, números, nomes)
3. Cite as fontes quando possível (ex: "Segundo o site X...")
4. Se as fontes não respondem a pergunta, diga isso honestamente
5. Seja direto e organizado

Responda em português:"""

        response = await generate_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500
        )
        
        final_result = response.strip() if response else "Não foi possível gerar resposta."
        
        # Yield final result
        yield {"type": "stream_end", "message": "Stream ended"}
        
        yield {
            "type": "result",
            "content": final_result,
            "links": [p["url"] for p in extracted_pages],
            "success": True
        }
        
        elapsed = int(time.time() - start_time)
        yield {"type": "complete", "message": f"Completed in {elapsed}s with {len(extracted_pages)} sources"}
        
        logger.info(f"Agent complete. Sources: {len(extracted_pages)}, Time: {elapsed}s, LLM calls: 1")
        
    except ImportError as e:
        yield {"type": "error", "message": "e2b-desktop not installed"}
    except Exception as e:
        logger.exception("Browser agent error")
        yield {"type": "error", "message": f"Error: {str(e)}"}
    finally:
        if desktop:
            try:
                desktop.stream.stop()
            except Exception:
                pass
            try:
                desktop.kill()
            except Exception:
                pass


async def _extract_page_content(desktop, url: str) -> str:
    """Extract text content from a page using curl."""
    try:
        result = desktop.commands.run(
            f"curl -sL --max-time 8 --connect-timeout 5 "
            f"-A 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36' "
            f"{shlex.quote(url)} 2>/dev/null | "
            "sed -e 's/<script[^>]*>.*<\\/script>//g' -e 's/<style[^>]*>.*<\\/style>//g' | "
            "sed 's/<[^>]*>//g' | "
            "tr -s ' \\n' ' ' | "
            "head -c 8000",
            timeout=12
        )
        return result.stdout.strip() if hasattr(result, 'stdout') else ""
    except Exception as e:
        logger.warning(f"Extract failed for {url}: {e}")
        return ""


def _extract_links_from_search(content: str, task: str) -> List[str]:
    """Extract relevant links from DuckDuckGo search results."""
    # DuckDuckGo HTML links pattern
    links = []
    
    # Find URLs in the content
    url_pattern = r'https?://[^\s<>"\']+[a-zA-Z0-9/]'
    found_urls = re.findall(url_pattern, content)
    
    # Filter out search engine URLs and duplicates
    seen = set()
    for url in found_urls:
        # Clean URL
        url = url.rstrip('.,;:)')
        
        # Skip search engines, trackers, etc
        skip_domains = ['duckduckgo.com', 'google.com', 'bing.com', 'facebook.com', 'twitter.com', 'instagram.com']
        if any(d in url.lower() for d in skip_domains):
            continue
        
        # Skip if already seen
        domain = url.split('/')[2] if len(url.split('/')) > 2 else url
        if domain in seen:
            continue
        seen.add(domain)
        
        links.append(url)
        
        if len(links) >= 8:
            break
    
    return links
