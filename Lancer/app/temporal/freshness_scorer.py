"""Freshness scoring for search results.

Calculates how fresh/recent content is using exponential decay.
"""

import math
from datetime import datetime
from typing import Optional

from app.config import get_settings


def calculate_freshness_score(
    published_date: Optional[datetime | str] = None,
    half_life_days: Optional[int] = None,
) -> float:
    """
    Calculate freshness score using exponential decay.
    
    The score decays exponentially based on content age:
    - Just published: ~1.0
    - half_life_days old: ~0.5
    - 2x half_life_days old: ~0.25
    - Very old: approaches 0
    
    Args:
        published_date: When the content was published
        half_life_days: Days until score halves (default from settings)
        
    Returns:
        Freshness score between 0.0 and 1.0
    """
    if published_date is None:
        # Unknown date gets neutral score
        return 0.5
    
    settings = get_settings()
    if half_life_days is None:
        half_life_days = settings.default_freshness_half_life
    
    # Parse string dates if needed
    if isinstance(published_date, str):
        try:
            # Try common formats
            for fmt in ["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ"]:
                try:
                    published_date = datetime.strptime(published_date, fmt)
                    break
                except ValueError:
                    continue
            else:
                return 0.5  # Couldn't parse, neutral score
        except Exception:
            return 0.5
    
    # Calculate age in days
    now = datetime.now()
    if published_date > now:
        # Future date (probably an error), treat as very fresh
        return 1.0
    
    age_days = (now - published_date).days
    
    # Exponential decay: score = e^(-λt) where λ = ln(2) / half_life
    decay_constant = 0.693147 / half_life_days  # ln(2)
    score = math.exp(-decay_constant * age_days)
    
    # Ensure score is in valid range
    return max(0.01, min(1.0, score))


def get_freshness_label(score: float) -> str:
    """
    Get a human-readable label for a freshness score.
    
    Args:
        score: Freshness score 0-1
        
    Returns:
        Label like "Very Fresh", "Recent", "Dated", etc.
    """
    if score >= 0.9:
        return "Very Fresh"
    elif score >= 0.7:
        return "Fresh"
    elif score >= 0.5:
        return "Recent"
    elif score >= 0.3:
        return "Dated"
    elif score >= 0.1:
        return "Old"
    else:
        return "Very Old"


def adjust_score_by_freshness(
    base_score: float,
    freshness_score: float,
    temporal_urgency: float,
) -> float:
    """
    Adjust a result's relevance score based on freshness.
    
    When temporal_urgency is high, freshness matters more.
    When temporal_urgency is low, freshness matters less.
    
    Args:
        base_score: Original relevance score (0-1)
        freshness_score: How fresh the content is (0-1)
        temporal_urgency: How important freshness is for this query (0-1)
        
    Returns:
        Adjusted score (0-1)
    """
    # Weight freshness by temporal urgency
    freshness_weight = temporal_urgency * 0.4  # Max 40% impact from freshness
    base_weight = 1.0 - freshness_weight
    
    adjusted = (base_score * base_weight) + (freshness_score * freshness_weight)
    
    return max(0.0, min(1.0, adjusted))
