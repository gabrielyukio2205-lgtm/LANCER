"""Pydantic schemas for API request/response models."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


# === Request Models ===

class SearchRequest(BaseModel):
    """Search request payload."""
    
    query: str = Field(..., min_length=1, max_length=1000, description="Search query")
    max_results: int = Field(default=10, ge=1, le=50, description="Maximum results to return")
    freshness: Literal["day", "week", "month", "year", "any"] = Field(
        default="any",
        description="Filter results by recency"
    )
    include_domains: list[str] | None = Field(
        default=None,
        description="Only include results from these domains"
    )
    exclude_domains: list[str] | None = Field(
        default=None,
        description="Exclude results from these domains"
    )
    include_answer: bool = Field(
        default=True,
        description="Include AI-generated answer"
    )


# === Response Models ===

class Citation(BaseModel):
    """Citation reference for the answer."""
    
    index: int = Field(..., description="Citation index (1-based)")
    url: str = Field(..., description="Source URL")
    title: str = Field(..., description="Source title")


class TemporalContext(BaseModel):
    """Temporal metadata about the search."""
    
    query_temporal_intent: Literal["current", "historical", "neutral"] = Field(
        ...,
        description="Detected temporal intent of the query"
    )
    temporal_urgency: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="How important freshness is for this query (0-1)"
    )
    current_date: str = Field(..., description="Current date for context")


class SearchResult(BaseModel):
    """Individual search result."""
    
    title: str = Field(..., description="Result title")
    url: str = Field(..., description="Result URL")
    content: str = Field(..., description="Result content/snippet")
    score: float = Field(..., ge=0.0, le=1.0, description="Overall relevance score")
    published_date: datetime | None = Field(
        default=None,
        description="Publication date if available"
    )
    freshness_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="How fresh/recent the content is"
    )
    authority_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Domain authority/trust score"
    )


class SearchResponse(BaseModel):
    """Complete search response."""
    
    query: str = Field(..., description="Original query")
    answer: str | None = Field(
        default=None,
        description="AI-generated answer synthesized from results"
    )
    results: list[SearchResult] = Field(
        default_factory=list,
        description="Ranked search results"
    )
    citations: list[Citation] = Field(
        default_factory=list,
        description="Citations referenced in the answer"
    )
    temporal_context: TemporalContext | None = Field(
        default=None,
        description="Temporal analysis metadata"
    )
    processing_time_ms: float = Field(..., description="Total processing time in milliseconds")


class ErrorResponse(BaseModel):
    """Error response model."""
    
    error: str = Field(..., description="Error message")
    detail: str | None = Field(default=None, description="Detailed error information")


# === Deep Research Models ===

class DeepResearchRequest(BaseModel):
    """Deep research request payload."""
    
    query: str = Field(..., min_length=1, max_length=2000, description="Research query")
    max_dimensions: int = Field(
        default=5,
        ge=2,
        le=8,
        description="Maximum research dimensions to explore"
    )
    max_sources_per_dim: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Maximum sources per dimension"
    )
    max_total_searches: int = Field(
        default=20,
        ge=5,
        le=30,
        description="Maximum total API searches"
    )


# === Browser Agent Models ===

class BrowseRequest(BaseModel):
    """Browser agent request payload."""
    
    task: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Task description (e.g., 'Get the top 5 headlines')"
    )
    url: str | None = Field(
        default=None,
        description="URL to navigate to"
    )
    stream_visual: bool = Field(
        default=False,
        description="Use Chrome with live video stream (less stealth, but visual)"
    )
