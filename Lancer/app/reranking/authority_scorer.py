"""Domain authority scoring.

Assigns trust/authority scores to domains based on known reliable sources.
"""

from urllib.parse import urlparse


# High authority domains (trusted sources)
HIGH_AUTHORITY_DOMAINS = {
    # Academic & Research
    ".edu": 0.9,
    ".gov": 0.9,
    ".ac.uk": 0.85,
    
    # Major tech companies
    "github.com": 0.8,
    "stackoverflow.com": 0.8,
    "docs.python.org": 0.85,
    "developer.mozilla.org": 0.85,
    "arxiv.org": 0.9,
    
    # Major news sources
    "reuters.com": 0.8,
    "bbc.com": 0.75,
    "nytimes.com": 0.75,
    "theguardian.com": 0.75,
    
    # Reference
    "wikipedia.org": 0.7,
    "britannica.com": 0.8,
    
    # AI/ML specific
    "openai.com": 0.85,
    "anthropic.com": 0.85,
    "huggingface.co": 0.8,
    "deepmind.google": 0.85,
    "ai.meta.com": 0.8,
    
    # Tech publications
    "techcrunch.com": 0.7,
    "wired.com": 0.7,
    "arstechnica.com": 0.75,
    "theverge.com": 0.7,
}

# Low authority patterns (less reliable)
LOW_AUTHORITY_PATTERNS = [
    "medium.com",  # User-generated, variable quality
    "reddit.com",  # Forum, variable quality
    "quora.com",   # Q&A, variable quality
    "blogspot.com",
    "wordpress.com",
    "tumblr.com",
]


def calculate_authority_score(url: str) -> float:
    """
    Calculate domain authority score for a URL.
    
    Args:
        url: The URL to score
        
    Returns:
        Authority score between 0.0 and 1.0
    """
    if not url:
        return 0.5
    
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        
        # Remove www. prefix
        if domain.startswith("www."):
            domain = domain[4:]
        
        # Check for exact domain matches
        for known_domain, score in HIGH_AUTHORITY_DOMAINS.items():
            if domain == known_domain or domain.endswith(known_domain):
                return score
        
        # Check for TLD-based authority (.edu, .gov, etc.)
        for tld, score in HIGH_AUTHORITY_DOMAINS.items():
            if tld.startswith(".") and domain.endswith(tld):
                return score
        
        # Check for low authority patterns
        for pattern in LOW_AUTHORITY_PATTERNS:
            if pattern in domain:
                return 0.4
        
        # Default score for unknown domains
        return 0.5
        
    except Exception:
        return 0.5


def get_domain_category(url: str) -> str:
    """
    Get a category label for the domain.
    
    Args:
        url: The URL to categorize
        
    Returns:
        Category string like "Academic", "News", "Tech", etc.
    """
    if not url:
        return "Unknown"
    
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        
        if ".edu" in domain or ".ac.uk" in domain or "arxiv" in domain:
            return "Academic"
        elif ".gov" in domain:
            return "Government"
        elif any(site in domain for site in ["github", "stackoverflow", "docs."]):
            return "Developer"
        elif any(site in domain for site in ["reuters", "bbc", "nytimes", "cnn", "guardian"]):
            return "News"
        elif any(site in domain for site in ["openai", "anthropic", "huggingface", "deepmind"]):
            return "AI/ML"
        elif "wikipedia" in domain:
            return "Reference"
        else:
            return "General"
            
    except Exception:
        return "Unknown"
