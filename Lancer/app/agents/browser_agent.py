"""Browser Agent - Chrome with live stream and agent memory.

Uses E2B Desktop sandbox with Chrome browser.
Time limit: 5 minutes (300 seconds)
Shows live video stream.
Includes full memory/history tracking via AgentState.
"""

import os
import json
import shlex
import logging
import base64
import time
from typing import AsyncGenerator, Optional

from app.config import get_settings
from app.agents.llm_client import generate_completion
from app.agents.graph.state import AgentState, NodeType
from app.agents.flaresolverr import is_cloudflare_blocked

logger = logging.getLogger(__name__)

MAX_TIME_SECONDS = 300  # 5 minutes


async def run_browser_agent(
    task: str,
    url: Optional[str] = None,
) -> AsyncGenerator[dict, None]:
    """Run browser agent with Chrome and live stream."""
    settings = get_settings()
    
    if not settings.e2b_api_key:
        yield {"type": "error", "message": "E2B_API_KEY not configured"}
        return
    
    # Initialize agent state with memory
    state = AgentState(
        task=task,
        url=url,
        timeout_seconds=MAX_TIME_SECONDS,
        start_time=time.time()
    )
    
    yield {"type": "status", "message": "🚀 Initializing agent..."}
    
    desktop = None
    
    try:
        from e2b_desktop import Sandbox
        
        os.environ["E2B_API_KEY"] = settings.e2b_api_key
        
        yield {"type": "status", "message": "🖥️ Creating virtual desktop..."}
        desktop = Sandbox.create(timeout=600)
        state.desktop = desktop
        
        # Start streaming
        stream_url = None
        try:
            desktop.stream.start(require_auth=True)
            auth_key = desktop.stream.get_auth_key()
            stream_url = desktop.stream.get_url(auth_key=auth_key)
            yield {"type": "stream", "url": stream_url}
            logger.info(f"Stream started: {stream_url}")
            desktop.wait(2000)
        except Exception as e:
            logger.warning(f"Could not start stream: {e}")
        
        # Launch Chrome
        yield {"type": "status", "message": "🌐 Launching browser..."}
        
        if url:
            start_url = url
        else:
            search_query = task.replace(' ', '+')
            start_url = f"https://html.duckduckgo.com/html/?q={search_query}"
            state.add_query(task)
        
        chrome_flags = "--no-sandbox --disable-gpu --start-maximized --no-first-run --disable-default-apps --disable-popup-blocking --disable-translate --no-default-browser-check"
        desktop.commands.run(f"google-chrome {chrome_flags} {shlex.quote(start_url)} &", background=True)
        desktop.wait(3000)
        
        # Close dialogs
        desktop.press("enter")
        desktop.wait(1000)
        
        # Add to memory
        state.visited_urls.append(start_url)
        state.add_action({"type": "navigate", "url": start_url})
        
        # Main loop - time based with memory
        while state.should_continue():
            state.step_count += 1
            elapsed = int(state.get_elapsed_time())
            remaining = int(state.get_remaining_time())
            
            yield {"type": "status", "message": f"🔍 Step {state.step_count}: Analyzing... ({elapsed}s / {MAX_TIME_SECONDS}s)"}
            
            # Take screenshot
            screenshot_bytes = desktop.screenshot()
            screenshot_b64 = base64.b64encode(screenshot_bytes).decode('utf-8')
            
            # Get page content
            current_url = state.visited_urls[-1]
            page_content = ""
            
            try:
                result = desktop.commands.run(
                    f"curl -sL --max-time 10 --connect-timeout 5 "
                    f"-A 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0.0.0' "
                    f"{shlex.quote(current_url)} 2>/dev/null | "
                    "sed -e 's/<script[^>]*>.*<\\/script>//g' -e 's/<style[^>]*>.*<\\/style>//g' | "
                    "sed 's/<[^>]*>//g' | "
                    "tr -s ' \\n' ' ' | "
                    "head -c 6000",
                    timeout=15
                )
                page_content = result.stdout.strip() if hasattr(result, 'stdout') else ""
                state.page_content = page_content
            except Exception as e:
                logger.warning(f"Content extraction failed: {e}")
                state.add_error(f"Content extraction failed: {e}")
            
            # Check for Cloudflare block
            is_blocked = is_cloudflare_blocked(page_content) if page_content else False
            
            if is_blocked:
                yield {"type": "status", "message": f"🚫 Cloudflare at {current_url[:40]}..., trying next link..."}
                state.add_error(f"Cloudflare blocked: {current_url}")
            else:
                # Add to memory
                state.extracted_data.append({
                    "url": current_url,
                    "content_length": len(page_content),
                    "preview": page_content[:200]
                })
            
            # Build prompt with memory context
            memory_context = state.get_context_for_llm()
            history_str = "\n".join([f"- {u}" for u in state.visited_urls[-5:]])
            content_preview = page_content[:2000] if page_content else "(empty page)"
            known_str = "\n".join([f"- {f}" for f in state.known_facts[-6:]]) if state.known_facts else "(none yet)"
            missing_str = "\n".join([f"- {m}" for m in state.missing_points[-6:]]) if state.missing_points else "(none)"
            recent_queries_str = "\n".join([f"- {q}" for q in state.last_queries[-6:]]) if state.last_queries else "(none)"
            
            prompt = f"""You are a browser agent with memory. Analyze the page and decide the next action.

TASK: {task}
CURRENT URL: {current_url}
TIME REMAINING: {remaining}s
STEP: {state.step_count}

MEMORY:
{memory_context}

RESEARCH PROGRESS:
KNOWN FACTS:
{known_str}

MISSING INFO:
{missing_str}

RECENT QUERIES:
{recent_queries_str}

VISITED URLS:
{history_str}

PAGE CONTENT (blocked={is_blocked}):
{content_preview}

What should I do? Reply with JSON:
{{
  "action": "SEARCH|NAVIGATE|SCROLL|DONE",
  "value": "search query or URL",
  "reason": "brief reason",
  "known_facts": ["short fact", "..."],
  "missing_points": ["what is still missing", "..."]
}}

- SEARCH: Search for something new (use if current results are insufficient)
- NAVIGATE: Go to a specific URL found on the page (MUST be different from visited URLs)
- SCROLL: Scroll down for more content
- DONE: Task is complete, provide final answer

RULES:
1. Do NOT navigate to already visited URLs
2. If blocked, navigate to a different link immediately
3. If you have enough info, respond with DONE
4. Include "answer" field when action is DONE"""

            response = await generate_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500
            )
            
            # Parse response
            try:
                json_match = response[response.find('{'):response.rfind('}')+1]
                decision = json.loads(json_match)
            except:
                logger.warning(f"Could not parse LLM response: {response[:200]}")
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
            
            # Record action in memory
            state.add_action({"type": action.lower(), "value": value, "reason": reason})
            
            yield {"type": "status", "message": f"🤔 Action: {action} - {reason[:50]}"}
            
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
                        f"Based on this content, answer: {task}\n\n"
                        f"Known facts:\n{known_summary}\n\n"
                        f"Missing points:\n{missing_summary}\n\n"
                        f"Content:\n{all_content}"
                    )
                    final_answer = await generate_completion(
                        messages=[{"role": "user", "content": final_prompt}],
                        max_tokens=1000
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
                
                yield {"type": "complete", "message": f"Completed in {int(state.get_elapsed_time())}s with {state.step_count} steps"}
                return
            
            elif action == "SEARCH":
                search_query = value.replace(' ', '+')
                new_url = f"https://html.duckduckgo.com/html/?q={search_query}"
                
                if new_url not in state.visited_urls:
                    desktop.commands.run(f"google-chrome {shlex.quote(new_url)} &", background=True)
                    desktop.wait(3000)
                    state.visited_urls.append(new_url)
            
            elif action == "NAVIGATE":
                if value and value.startswith("http"):
                    if value in state.visited_urls:
                        yield {"type": "status", "message": f"⏭️ Already visited, skipping..."}
                        state.add_error(f"Tried to revisit: {value}")
                    else:
                        desktop.commands.run(f"google-chrome {shlex.quote(value)} &", background=True)
                        desktop.wait(3000)
                        state.visited_urls.append(value)
            
            elif action == "SCROLL":
                desktop.press("pagedown")
                desktop.wait(1500)
            
            # Small delay
            desktop.wait(1000)
        
        # Timeout - generate from memory
        yield {"type": "status", "message": "⏰ Time limit reached, generating final answer from memory..."}
        
        all_content = "\n\n".join([
            f"Source: {d['url']}\n{d.get('preview', '')}" 
            for d in state.extracted_data[-5:]
        ])
        known_summary = "\n".join([f"- {f}" for f in state.known_facts[-8:]]) or "(none)"
        missing_summary = "\n".join([f"- {m}" for m in state.missing_points[-8:]]) or "(none)"
        final_prompt = (
            f"Based on this content, answer: {task}\n\n"
            f"Known facts:\n{known_summary}\n\n"
            f"Missing points:\n{missing_summary}\n\n"
            f"Content:\n{all_content}"
        )
        final_answer = await generate_completion(
            messages=[{"role": "user", "content": final_prompt}],
            max_tokens=1000
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
        yield {"type": "complete", "message": f"Completed in {MAX_TIME_SECONDS}s (timeout) with {state.step_count} steps"}
        
    except ImportError as e:
        yield {"type": "error", "message": "e2b-desktop not installed"}
    except Exception as e:
        logger.exception("Browser agent error")
        yield {"type": "error", "message": f"Error: {str(e)}"}
    finally:
        if desktop:
            try:
                desktop.stream.stop()
            except:
                pass
            try:
                desktop.kill()
            except:
                pass
