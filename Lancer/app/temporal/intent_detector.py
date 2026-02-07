"""Temporal intent detection for search queries.

Analyzes queries to determine if they require fresh/current information
or if historical information is acceptable.
"""

import re
from datetime import datetime
from typing import Literal


def _get_dynamic_years() -> set[str]:
    """Get current and previous year dynamically."""
    current_year = datetime.now().year
    return {str(current_year), str(current_year - 1)}


# Keywords that strongly indicate need for current information
FRESHNESS_KEYWORDS = {
    # English
    "latest", "newest", "recent", "current", "today", "now",
    "this week", "this month", "this year", "breaking",
    "update", "updates", "new", "just", "announced",
    *_get_dynamic_years(),  # Dynamic years
    # Portuguese
    "último", "últimos", "recente", "atual", "hoje", "agora",
    "essa semana", "esse mês", "esse ano", "novidade",
    "atualização", "novo", "novos", "anunciado",
}

# Keywords that indicate historical queries (less urgent freshness)
HISTORICAL_KEYWORDS = {
    "history", "historical", "origin", "origins", "invented",
    "founded", "first", "original", "classic", "traditional",
    "história", "histórico", "origem", "inventado", "fundado",
}

# Entity types that typically require fresh information
FRESH_ENTITY_PATTERNS = [
    r"\b(?:price|prices|stock|stocks|market)\b",  # Financial
    r"\b(?:weather|forecast|temperature)\b",  # Weather
    r"\b(?:news|headlines|breaking)\b",  # News
    r"\b(?:score|scores|game|match|vs)\b",  # Sports
    r"\b(?:version|release|update|patch)\b",  # Software
    r"\b(?:gpt-?\d|claude|gemini|llama|mistral)\b",  # AI models
]


def detect_temporal_intent(
    query: str,
) -> tuple[Literal["current", "historical", "neutral"], float]:
    """
    Detect the temporal intent of a search query.
    
    Args:
        query: The search query string
        
    Returns:
        Tuple of (intent, urgency) where:
        - intent: "current", "historical", or "neutral"
        - urgency: float 0-1 indicating how important freshness is
    """
    query_lower = query.lower()
    
    # Count freshness indicators
    freshness_score = 0.0
    historical_score = 0.0
    
    # Check for freshness keywords
    for keyword in FRESHNESS_KEYWORDS:
        if keyword in query_lower:
            freshness_score += 0.3
    
    # Check for historical keywords
    for keyword in HISTORICAL_KEYWORDS:
        if keyword in query_lower:
            historical_score += 0.3
    
    # Check for fresh entity patterns
    for pattern in FRESH_ENTITY_PATTERNS:
        if re.search(pattern, query_lower):
            freshness_score += 0.2
    
    # Question words that often imply current info needed
    if re.search(r"\b(?:what is|who is|how to|where is)\b", query_lower):
        freshness_score += 0.1
    
    # Superlatives often need current info
    if re.search(r"\b(?:best|top|most|fastest|cheapest)\b", query_lower):
        freshness_score += 0.15
    
    # Normalize scores
    freshness_score = min(freshness_score, 1.0)
    historical_score = min(historical_score, 1.0)
    
    # Determine intent
    if freshness_score > historical_score and freshness_score > 0.2:
        intent = "current"
        urgency = min(0.3 + freshness_score, 1.0)
    elif historical_score > freshness_score and historical_score > 0.2:
        intent = "historical"
        urgency = max(0.2 - historical_score * 0.1, 0.1)
    else:
        intent = "neutral"
        urgency = 0.5
    
    return intent, urgency
