"""Browser Agent v2 - Camoufox stealth with full agentic navigation.

Camoufox = Firefox stealth que passa anti-bot.
Roda DENTRO do E2B sandbox.
Full agentic loop with LLM-driven navigation.
Time limit: 5 minutes (300 seconds)
"""

import os
import json
import logging
import time
from typing import AsyncGenerator, Optional

from app.config import get_settings
from app.agents.llm_client import generate_completion
from app.agents.graph.state import AgentState

logger = logging.getLogger(__name__)

MAX_TIME_SECONDS = 300  # 5 minutes
MAX_PAGES = 5
STEALTH_SANDBOX_TIMEOUT_SECONDS = 600


async def run_browser_agent_v2(
    task: str,
    url: Optional[str] = None,
) -> AsyncGenerator[dict, None]:
    """Run stealth browser agent with Camoufox - full agentic navigation."""
    settings = get_settings()
    
    if not settings.e2b_api_key:
        yield {"type": "error", "message": "E2B_API_KEY not configured"}
        return
    
    # Initialize agent state
    state = AgentState(
        task=task,
        url=url,
        timeout_seconds=MAX_TIME_SECONDS,
        start_time=time.time()
    )
    
    yield {"type": "status", "message": "🚀 Initializing stealth agent..."}
    
    desktop = None
    
    try:
        from e2b_desktop import Sandbox
        
        os.environ["E2B_API_KEY"] = settings.e2b_api_key
        
        yield {"type": "status", "message": "🖥️ Creating sandbox..."}
        desktop = Sandbox.create(timeout=STEALTH_SANDBOX_TIMEOUT_SECONDS)
        state.desktop = desktop
        
        # Install Camoufox
        yield {"type": "status", "message": "📦 Installing stealth browser..."}
        
        try:
            desktop.commands.run("pip install --user camoufox playwright -q", timeout=120)
            yield {"type": "status", "message": "🔽 Downloading Firefox stealth (~30s)..."}
            desktop.commands.run("camoufox fetch", timeout=180)
            desktop.commands.run("sudo apt-get update -qq && sudo apt-get install -y -qq libgtk-3-0 libasound2 libdbus-glib-1-2 2>/dev/null || true", timeout=60)
            yield {"type": "status", "message": "✅ Browser ready!"}
        except Exception as e:
            logger.error(f"Camoufox install failed: {e}")
            yield {"type": "error", "message": f"Install failed: {e}"}
            return
        
        # Build initial URL
        if url:
            start_url = url
        else:
            search_query = task.replace(' ', '+')
            start_url = f"https://html.duckduckgo.com/html/?q={search_query}"
            state.add_query(task)
        
        state.visited_urls.append(start_url)
        state.add_action({"type": "start", "url": start_url})
        
        # Agentic loop
        while state.should_continue() and state.step_count < MAX_PAGES:
            state.step_count += 1
            elapsed = int(state.get_elapsed_time())
            remaining = int(state.get_remaining_time())
            
            current_url = state.visited_urls[-1]
            
            yield {"type": "status", "message": f"🔍 Step {state.step_count}: Fetching {current_url[:40]}... ({elapsed}s)"}
            
            # Fetch page with Camoufox
            script = _build_fetch_script(current_url)
            desktop.commands.run(f"cat > /tmp/fetch.py << 'EOF'\n{script}\nEOF", timeout=10)
            
            result = desktop.commands.run("python3 /tmp/fetch.py", timeout=60)
            output = result.stdout.strip() if hasattr(result, 'stdout') else ""
            
            # Parse result
            page_content = ""
            page_links = []
            is_blocked = False
            
            try:
                data = json.loads(output)
                page_content = data.get("content", "")
                page_links = data.get("links", [])
                is_blocked = data.get("blocked", False)
                
                if data.get("error"):
                    state.add_error(data["error"])
            except json.JSONDecodeError:
                page_content = output[:3000]
            
            if is_blocked:
                yield {"type": "status", "message": f"🚫 Blocked at {current_url[:30]}..., trying next..."}
                state.add_error(f"Blocked: {current_url}")
            else:
                state.extracted_data.append({
                    "url": current_url,
                    "content_length": len(page_content),
                    "links_found": len(page_links),
                    "preview": page_content[:300]
                })
            
            # Ask LLM what to do next
            memory_context = state.get_context_for_llm()
            links_str = "\n".join([f"- {l}" for l in page_links[:10]])
            known_str = "\n".join([f"- {f}" for f in state.known_facts[-6:]]) if state.known_facts else "(none yet)"
            missing_str = "\n".join([f"- {m}" for m in state.missing_points[-6:]]) if state.missing_points else "(none)"
            recent_queries_str = "\n".join([f"- {q}" for q in state.last_queries[-6:]]) if state.last_queries else "(none)"
            
            prompt = f"""You are a stealth browser agent. Analyze and decide next action.

TASK: {task}
CURRENT URL: {current_url}
TIME: {elapsed}s / {MAX_TIME_SECONDS}s
STEP: {state.step_count} / {MAX_PAGES}

MEMORY:
{memory_context}

RESEARCH PROGRESS:
KNOWN FACTS:
{known_str}

MISSING INFO:
{missing_str}

RECENT QUERIES:
{recent_queries_str}

VISITED:
{chr(10).join(['- ' + u for u in state.visited_urls])}

PAGE CONTENT (blocked={is_blocked}):
{page_content[:2000] if page_content else "(empty)"}

LINKS FOUND:
{links_str if page_links else "(none)"}

Reply with JSON:
{{
  "action": "NAVIGATE|SEARCH|DONE",
  "value": "url or query",
  "reason": "why",
  "known_facts": ["short fact", "..."],
  "missing_points": ["what is still missing", "..."]
}}

- NAVIGATE: Go to one of the links above (MUST be new, not already visited)
- SEARCH: New search query
- DONE: Task complete, include "answer" field
- Keep known_facts/missing_points concise and session-specific

If you have enough info, respond DONE with answer."""

            response = await generate_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600
            )
            
            # Parse decision
            try:
                json_match = response[response.find('{'):response.rfind('}')+1]
                decision = json.loads(json_match)
            except:
                decision = {"action": "DONE", "answer": response}
            
            action = decision.get("action", "DONE")
            value = decision.get("value", "")
            reason = decision.get("reason", "")
            known_facts = decision.get("known_facts", [])
            missing_points = decision.get("missing_points", [])

            if action == "SEARCH":
                state.add_query(value)

            if isinstance(known_facts, list) or isinstance(missing_points, list):
                state.update_research_progress(
                    known_facts=known_facts if isinstance(known_facts, list) else None,
                    missing_points=missing_points if isinstance(missing_points, list) else None,
                )
            
            state.add_action({"type": action.lower(), "value": value, "reason": reason})
            
            yield {"type": "status", "message": f"🤔 {action}: {reason[:40]}"}
            
            yield {
                "type": "progress",
                "known_facts": state.known_facts[-8:],
                "missing_points": state.missing_points[-8:],
                "last_queries": state.last_queries[-8:],
            }

            if action == "DONE":
                state.success = True
                final_answer = decision.get("answer", "")
                
                if not final_answer:
                    # Generate from memory
                    all_content = "\n\n".join([
                        f"Source: {d['url']}\n{d.get('preview', '')}"
                        for d in state.extracted_data[-5:]
                    ])
                    known_summary = "\n".join([f"- {f}" for f in state.known_facts[-8:]]) or "(none)"
                    missing_summary = "\n".join([f"- {m}" for m in state.missing_points[-8:]]) or "(none)"
                    final_prompt = (
                        f"Answer this: {task}\n\n"
                        f"Known facts:\n{known_summary}\n\n"
                        f"Missing points:\n{missing_summary}\n\n"
                        f"Content:\n{all_content}"
                    )
                    final_answer = await generate_completion(
                        messages=[{"role": "user", "content": final_prompt}],
                        max_tokens=1200
                    )
                
                state.final_result = final_answer
                
                yield {"type": "stream_end", "message": "Done"}
                yield {
                    "type": "result",
                    "content": final_answer,
                    "links": state.visited_urls,
                    "steps": state.step_count,
                    "success": True
                }
                yield {"type": "complete", "message": f"Done in {int(state.get_elapsed_time())}s (stealth)"}
                return
            
            elif action == "NAVIGATE":
                if value and value.startswith("http"):
                    if value not in state.visited_urls:
                        state.visited_urls.append(value)
                    else:
                        state.add_error(f"Tried revisit: {value}")
            
            elif action == "SEARCH":
                new_url = f"https://html.duckduckgo.com/html/?q={value.replace(' ', '+')}"
                if new_url not in state.visited_urls:
                    state.visited_urls.append(new_url)
        
        # Timeout or max pages - generate from memory
        yield {"type": "status", "message": "⏰ Generating final answer from memory..."}
        
        all_content = "\n\n".join([
            f"Source: {d['url']}\n{d.get('preview', '')}"
            for d in state.extracted_data[-5:]
        ])
        known_summary = "\n".join([f"- {f}" for f in state.known_facts[-8:]]) or "(none)"
        missing_summary = "\n".join([f"- {m}" for m in state.missing_points[-8:]]) or "(none)"
        final_prompt = (
            f"Answer this: {task}\n\n"
            f"Known facts:\n{known_summary}\n\n"
            f"Missing points:\n{missing_summary}\n\n"
            f"Content:\n{all_content}"
        )
        final_answer = await generate_completion(
            messages=[{"role": "user", "content": final_prompt}],
            max_tokens=1200
        )
        
        state.final_result = final_answer
        
        yield {"type": "stream_end", "message": "Done"}
        yield {
            "type": "result",
            "content": final_answer,
            "links": state.visited_urls,
            "steps": state.step_count,
            "success": True
        }
        yield {"type": "complete", "message": f"Done in {int(state.get_elapsed_time())}s (stealth, {state.step_count} pages)"}
        
    except ImportError:
        yield {"type": "error", "message": "e2b-desktop not installed"}
    except Exception as e:
        logger.exception("Stealth agent error")
        yield {"type": "error", "message": str(e)}
    finally:
        if desktop:
            try:
                desktop.kill()
            except:
                pass


def _build_fetch_script(url: str) -> str:
    """Build Python script to fetch a URL with Camoufox."""
    return f'''
import json
import sys

try:
    from camoufox.sync_api import Camoufox
except:
    print(json.dumps({{"error": "Camoufox not found"}}))
    sys.exit(1)

def is_blocked(text):
    t = text.lower()
    blocks = ["checking your browser", "cloudflare", "access denied", "just a moment", "enable javascript"]
    return len(text) < 800 and any(b in t for b in blocks)

try:
    with Camoufox(headless=True) as browser:
        page = browser.new_page()
        page.goto("{url}", timeout=30000)
        page.wait_for_timeout(2000)
        
        # Extract text
        content = page.evaluate("""() => {{
            document.querySelectorAll('script,style,noscript').forEach(e => e.remove());
            return document.body.innerText || '';
        }}""")[:5000]
        
        # Extract links
        links = page.evaluate("""() => {{
            return Array.from(document.querySelectorAll('a[href^="http"]'))
                .map(a => a.href)
                .filter(h => !h.includes('duckduckgo') && !h.includes('google'))
                .slice(0, 10);
        }}""")
        
        blocked = is_blocked(content)
        
        print(json.dumps({{
            "content": content,
            "links": links,
            "blocked": blocked
        }}))

except Exception as e:
    print(json.dumps({{"error": str(e)}}))
'''
