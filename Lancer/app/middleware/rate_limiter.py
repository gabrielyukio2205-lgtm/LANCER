"""Rate limiting middleware using SlowAPI.

Provides IP-based rate limiting for all API endpoints.
"""

from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from fastapi import Request
from fastapi.responses import JSONResponse


# Create limiter instance with IP-based key
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["100/minute"],
    storage_uri="memory://",  # Use memory storage (OK for single instance on HF Spaces)
)


def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    """Custom handler for rate limit exceeded errors."""
    return JSONResponse(
        status_code=429,
        content={
            "error": "rate_limit_exceeded",
            "message": f"Rate limit exceeded: {exc.detail}",
            "retry_after": getattr(exc, "retry_after", 60),
        },
    )


# Rate limit decorators for different endpoints
LIMITS = {
    "search": "30/minute",
    "heavy": "10/minute", 
    "deep": "5/minute",
    "images": "60/minute",
}


def get_limiter():
    """Get the limiter instance for dependency injection."""
    return limiter
