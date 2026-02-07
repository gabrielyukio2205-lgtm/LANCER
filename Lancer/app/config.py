"""Application configuration using pydantic-settings."""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
    # API Keys - Search Sources
    tavily_api_key: str = ""
    brave_api_key: str = ""  # 2000 free/month
    
    # SearXNG (self-hosted meta-search - uses your HF Space by default)
    searxng_url: str = "https://madras1-searxng-space.hf.space"
    serper_api_key: str | None = None
    
    # E2B Desktop (cloud browser for browser agent)
    e2b_api_key: str = ""
    
    # API Keys - LLM Providers
    groq_api_key: str | None = None
    openrouter_api_key: str | None = None
    
    # LLM Configuration
    llm_provider: Literal["groq", "openrouter"] = "openrouter"
    llm_model: str = "stepfun/step-3.5-flash:free"
    
    # Reranking Models (lightweight for HF Spaces)
    bi_encoder_model: str = "Madras1/minilm-gooaq-mnr-v5"  # Fine-tuned on GooAQ + NQ
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"  # ~90MB
    
    # Temporal Settings
    default_freshness_half_life: int = 30  # days
    
    # API Settings
    max_search_results: int = 20
    max_final_results: int = 10
    
    # Deep Research Settings
    max_research_dimensions: int = 6
    max_tavily_calls_per_research: int = 20
    deep_research_model: str | None = None  # Use main model if None
    
    @property
    def llm_api_key(self) -> str:
        """Get the appropriate API key based on provider."""
        if self.llm_provider == "groq":
            return self.groq_api_key or ""
        return self.openrouter_api_key or ""


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
