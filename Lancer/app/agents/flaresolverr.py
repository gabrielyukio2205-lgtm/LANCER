"""FlareSolverr client for Cloudflare bypass.

FlareSolverr uses undetected-chromedriver to solve Cloudflare challenges.
Must be running at http://localhost:8191 in the E2B sandbox.
"""

import logging
import json
import shlex
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

FLARESOLVERR_URL = "http://localhost:8191/v1"


async def solve_cloudflare(desktop, url: str, timeout: int = 60) -> Tuple[bool, str]:
    """
    Use FlareSolverr to bypass Cloudflare protection.
    
    Args:
        desktop: E2B desktop instance
        url: URL to fetch through FlareSolverr
        timeout: Max seconds to wait for solution
        
    Returns:
        (success: bool, content: str)
    """
    try:
        # Make request to FlareSolverr - properly escape the JSON payload
        payload = json.dumps({
            "cmd": "request.get",
            "url": url,
            "maxTimeout": timeout * 1000
        })
        
        result = desktop.commands.run(
            f"curl -s -X POST {shlex.quote(FLARESOLVERR_URL)} "
            f"-H 'Content-Type: application/json' "
            f"-d {shlex.quote(payload)} 2>/dev/null",
            timeout=timeout + 10
        )
        
        if not hasattr(result, 'stdout') or not result.stdout:
            return False, ""
        
        response = json.loads(result.stdout)
        
        if response.get("status") == "ok":
            solution = response.get("solution", {})
            html = solution.get("response", "")
            
            # Strip HTML tags - use base64 to safely pass content
            if html:
                import base64
                html_b64 = base64.b64encode(html[:10000].encode()).decode()
                clean_result = desktop.commands.run(
                    f"echo {shlex.quote(html_b64)} | base64 -d | sed 's/<[^>]*>//g' | tr -s ' \\n' ' ' | head -c 6000",
                    timeout=5
                )
                content = clean_result.stdout.strip() if hasattr(clean_result, 'stdout') else html[:6000]
                logger.info(f"FlareSolverr solved: {url[:50]}")
                return True, content
        
        logger.warning(f"FlareSolverr failed: {response.get('message', 'unknown')}")
        return False, ""
        
    except Exception as e:
        logger.warning(f"FlareSolverr error: {e}")
        return False, ""


def is_cloudflare_blocked(content: str) -> bool:
    """Check if page content indicates Cloudflare block.
    
    Only returns True for actual Cloudflare challenge pages,
    not just pages that mention Cloudflare.
    """
    content_lower = content.lower()
    
    # Must have multiple strong indicators to be considered blocked
    strong_indicators = [
        "checking your browser before accessing",
        "please wait while we verify",
        "ray id:",
        "cloudflare ray id",
        "enable javascript and cookies",
        "attention required! | cloudflare",
        "just a moment...",
        "ddos protection by cloudflare",
    ]
    
    # Check for strong indicators (need at least 1)
    has_strong = any(ind in content_lower for ind in strong_indicators)
    
    # Also check if content is suspiciously short (challenge pages are small)
    is_short = len(content) < 500
    
    # Only block if we have strong indicator AND page is short
    # (real content pages that mention cloudflare will be longer)
    if has_strong and is_short:
        return True
    
    # Very specific patterns that are definitely challenge pages
    definite_blocks = [
        "checking if the site connection is secure",
        "please turn javascript on and reload the page",
        "please enable cookies",
    ]
    
    return any(block in content_lower for block in definite_blocks)


def is_login_wall(content: str) -> bool:
    """Check if page requires login."""
    login_indicators = [
        "sign in",
        "log in",
        "login",
        "create account",
        "register",
        "enter your password",
        "authentication required",
    ]
    
    content_lower = content.lower()
    # Check for login indicators but make sure it's not just a login link
    return sum(1 for ind in login_indicators if ind in content_lower) >= 2
