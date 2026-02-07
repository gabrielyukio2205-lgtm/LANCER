"""Simplified agent nodes - ONE LLM call per cycle.

DAG:
  START → THINK_ACT ←→ EXECUTE → RESPOND
              ↑______________|

ThinkAndAct: Analyzes content + decides action in ONE call
Execute: Runs the action (search, navigate, scroll) - NO LLM
Respond: Final synthesis
"""

import json
import logging
import shlex
import time
from abc import ABC, abstractmethod
from typing import Tuple, Optional, List

from app.agents.llm_client import generate_completion

logger = logging.getLogger(__name__)


class SimpleState:
    """Minimal state for the agent."""
    
    def __init__(self, task: str, url: Optional[str], desktop, timeout: float = 300):
        self.task = task
        self.url = url
        self.desktop = desktop
        self.timeout = timeout
        self.start_time = time.time()
        
        # Memory - content cache (URL -> content)
        self.content_cache: dict = {}  # {url: content}
        self.visited_urls: List[str] = []
        self.action_history: List[str] = []
        
        # Accumulated knowledge
        self.findings: List[str] = []  # Key findings extracted
        
        # Result
        self.final_result = ""
        self.done = False
    
    def elapsed(self) -> float:
        return time.time() - self.start_time
    
    def remaining(self) -> float:
        return max(0, self.timeout - self.elapsed())
    
    def should_continue(self) -> bool:
        return not self.done and self.remaining() > 20
    
    def add_page(self, url: str, content: str):
        """Add page to cache - no duplicate fetching."""
        if url not in self.content_cache:
            self.content_cache[url] = content[:4000]
        if url not in self.visited_urls:
            self.visited_urls.append(url)
    
    def get_cached_content(self, url: str) -> Optional[str]:
        """Get content from cache if available."""
        return self.content_cache.get(url)
    
    def add_finding(self, finding: str):
        """Add a key finding to memory."""
        if finding and finding not in self.findings:
            self.findings.append(finding)
    
    def get_all_content(self) -> str:
        """Get all cached content for final synthesis."""
        parts = []
        for url in self.visited_urls[-5:]:
            content = self.content_cache.get(url, "")
            if content:
                parts.append(f"[{url[:60]}]\n{content[:1500]}")
        return "\n\n---\n\n".join(parts)
    
    def get_recent_content(self) -> str:
        """Get last 2 pages content for context."""
        recent_urls = self.visited_urls[-2:] if self.visited_urls else []
        parts = []
        for url in recent_urls:
            content = self.content_cache.get(url, "")
            if content:
                parts.append(f"[{url[:60]}]\n{content[:2000]}")
        return "\n\n---\n\n".join(parts)


async def think_and_act(state: SimpleState) -> Tuple[str, dict]:
    """
    ONE LLM call that analyzes current state and decides next action.
    Returns: (action_type, action_params)
    
    Actions:
    - search: {"query": "..."}
    - navigate: {"url": "..."}
    - scroll: {}
    - complete: {"result": "..."}
    """
    
    content = state.get_recent_content() or "(No content yet)"
    history = ", ".join(state.action_history[-5:]) if state.action_history else "(starting)"
    
    # Memory: show visited URLs so LLM doesn't repeat
    visited = "\n".join([f"  - {u[:70]}" for u in state.visited_urls[-10:]]) if state.visited_urls else "(none)"
    
    prompt = f"""You are a web research agent. Analyze the current state and decide your next action.

TASK: {state.task}

ALREADY VISITED (DO NOT visit again):
{visited}

CURRENT PAGE CONTENT:
{content}

HISTORY: {history}
TIME REMAINING: {int(state.remaining())}s

Decide ONE action. Return JSON:

If you need to search: {{"action": "search", "query": "search terms"}}
If you found a NEW relevant link to visit: {{"action": "navigate", "url": "https://..."}}
If you need to scroll for more content: {{"action": "scroll"}}
If you have enough info to answer: {{"action": "complete", "result": "Your answer with **bold** for important values. Cite sources."}}

RULES:
- DO NOT navigate to URLs already in "ALREADY VISITED" list
- Only use URLs you see in the content above
- If you see the answer, return complete immediately
- Use **bold** for prices, numbers, names
- Be efficient - don't repeat searches

Return ONLY valid JSON:"""

    try:
        response = await generate_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800
        )
        
        # Parse JSON
        response = response.strip()
        if response.startswith("```"):
            response = response.split("```")[1]
            if response.startswith("json"):
                response = response[4:]
        
        decision = json.loads(response)
        action = decision.get("action", "search")
        
        # Safety check: prevent navigating to already visited URL
        if action == "navigate":
            url = decision.get("url", "").rstrip("/")
            
            # Check if URL already visited (normalize by removing trailing slash)
            visited_normalized = [u.rstrip("/") for u in state.visited_urls]
            if url in visited_normalized or url in state.visited_urls:
                logger.warning(f"LLM tried to revisit {url}, trying different approach")
                
                # If we have good content, finish
                good_content = [c for c in state.content_cache.values() 
                               if c and c not in ["[BLOCKED]", "[LOGIN_REQUIRED]"]]
                if good_content:
                    return "complete", {"result": f"Informação coletada: {state.get_recent_content()[:800]}"}
                
                # Otherwise, search with different terms
                return "search", {"query": f"{state.task} site:wikipedia.org OR site:gov.br"}
        
        logger.info(f"ThinkAndAct decision: {action}")
        return action, decision
        
    except Exception as e:
        logger.error(f"ThinkAndAct failed: {e}")
        # Fallback: if we have content, try to respond
        if state.content_cache:
            return "complete", {"result": f"Based on collected data: {state.get_recent_content()[:500]}"}
        return "search", {"query": state.task}


async def execute_action(state: SimpleState, action: str, params: dict) -> bool:
    """
    Execute action WITHOUT LLM call.
    Uses cache to avoid repeated requests.
    Returns True if should continue, False if done.
    """
    desktop = state.desktop
    
    if action == "complete":
        state.final_result = params.get("result", "")
        state.done = True
        return False
    
    elif action == "search":
        query = params.get("query", state.task)
        search_url = f"https://html.duckduckgo.com/html/?q={query.replace(' ', '+')}"
        
        # Check cache first
        cached = state.get_cached_content(search_url)
        if cached:
            logger.info(f"Using cached content for search: {query[:30]}")
            state.action_history.append(f"search(cached):{query[:30]}")
            return True
        
        desktop.commands.run(f"google-chrome {shlex.quote(search_url)} &", background=True)
        desktop.wait(3000)
        
        content = await _extract_content(desktop, search_url)
        state.add_page(search_url, content)
        state.action_history.append(f"search:{query[:30]}")
        
        return True
    
    elif action == "navigate":
        url = params.get("url", "")
        if not url.startswith("http"):
            return True  # Invalid URL, continue
        
        # Check cache first - don't re-fetch
        cached = state.get_cached_content(url)
        if cached:
            logger.info(f"Using cached content for: {url[:50]}")
            state.action_history.append(f"nav(cached):{url[:30]}")
            return True
        
        desktop.commands.run(f"google-chrome {shlex.quote(url)} &", background=True)
        desktop.wait(3000)
        
        content = await _extract_content(desktop, url)
        
        # Check for Cloudflare/bot detection - just skip if blocked
        from app.agents.flaresolverr import is_cloudflare_blocked, is_login_wall
        
        if is_cloudflare_blocked(content):
            logger.warning(f"Cloudflare block detected at {url[:50]}, skipping...")
            # Mark as visited so LLM doesn't try again
            if url not in state.visited_urls:
                state.visited_urls.append(url)
            state.content_cache[url] = "[BLOCKED]"  # Mark as blocked in cache
            state.action_history.append(f"nav(blocked):{url[:30]}")
            return True
        
        if is_login_wall(content):
            logger.warning(f"Login wall detected at {url[:50]}, skipping...")
            # Mark as visited so LLM doesn't try again
            if url not in state.visited_urls:
                state.visited_urls.append(url)
            state.content_cache[url] = "[LOGIN_REQUIRED]"  # Mark in cache
            state.action_history.append(f"nav(login_wall):{url[:30]}")
            return True
        
        state.add_page(url, content)
        state.action_history.append(f"nav:{url[:30]}")
        
        return True
    
    elif action == "scroll":
        desktop.scroll(-3)
        desktop.wait(1500)
        
        # Update cache for current page with new content
        if state.visited_urls:
            current_url = state.visited_urls[-1]
            content = await _extract_content(desktop, current_url)
            state.content_cache[current_url] = content[:4000]  # Update cache
        
        state.action_history.append("scroll")
        return True
    
    return True


async def _extract_content(desktop, url: str) -> str:
    """Extract page content via curl."""
    try:
        result = desktop.commands.run(
            f"curl -sL --max-time 8 --connect-timeout 5 "
            f"-A 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36' "
            f"'{url}' 2>/dev/null | "
            "sed -e 's/<script[^>]*>.*<\\/script>//g' -e 's/<style[^>]*>.*<\\/style>//g' | "
            "sed 's/<[^>]*>//g' | "
            "tr -s ' \\n' ' ' | "
            "head -c 6000",
            timeout=12
        )
        return result.stdout.strip() if hasattr(result, 'stdout') else ""
    except Exception as e:
        logger.warning(f"Extract failed: {e}")
        return ""


async def generate_final_response(state: SimpleState) -> str:
    """Generate response if agent timed out without completing."""
    if state.final_result:
        return state.final_result
    
    content = state.get_recent_content()
    
    prompt = f"""Based on the research done, answer the question.

TASK: {state.task}

COLLECTED DATA:
{content if content else "(No data collected)"}

SOURCES VISITED: {', '.join(state.visited_urls[:5]) if state.visited_urls else 'None'}

Provide a helpful answer based on what was found. Use **bold** for important values. If you couldn't find the answer, say so honestly.

Answer in Portuguese:"""

    try:
        response = await generate_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000
        )
        return response.strip()
    except Exception as e:
        return f"Não foi possível completar a pesquisa. Erro: {e}"
