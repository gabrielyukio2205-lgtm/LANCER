"""Lancer API - Main FastAPI application."""

from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.api.routes import search
from app.config import get_settings
from app.middleware.rate_limiter import limiter


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    settings = get_settings()
    print(f"🚀 Lancer API starting...")
    print(f"   LLM Provider: {settings.llm_provider}")
    print(f"   LLM Model: {settings.llm_model}")
    print(f"   Rate limiting: enabled")
    yield
    # Shutdown
    print("👋 Lancer API shutting down...")


app = FastAPI(
    title="Lancer Search API",
    description="Advanced AI-powered search API with temporal intelligence",
    version="0.1.0",
    lifespan=lifespan,
)

# Rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(search.router, prefix="/api/v1", tags=["search"])


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "0.1.0",
    }


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Lancer Search API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
    }
