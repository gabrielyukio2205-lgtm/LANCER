"""LLM client abstraction for multiple providers.

Supports Groq and OpenRouter for LLM inference.
"""

import httpx
import json
from typing import Optional, AsyncIterator
import asyncio

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from app.config import get_settings


class RetryableError(Exception):
    """Error that should trigger a retry."""
    pass


async def generate_completion(
    messages: list[dict],
    model: Optional[str] = None,
    temperature: float = 0.3,
    max_tokens: int = 2048,
) -> str:
    """Generate a completion using the configured LLM provider."""
    settings = get_settings()
    provider = settings.llm_provider
    model = model or settings.llm_model
    
    if provider == "groq":
        return await _call_groq(messages, model, temperature, max_tokens)
    elif provider == "openrouter":
        return await _call_openrouter(messages, model, temperature, max_tokens)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(RetryableError),
    reraise=True,
)
async def _call_groq(
    messages: list[dict],
    model: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """Call Groq API with retry logic."""
    settings = get_settings()
    
    if not settings.groq_api_key:
        raise ValueError("GROQ_API_KEY not configured")
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.groq_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
            )
            
            # Retry on rate limit or server errors
            if response.status_code in (429, 502, 503, 504):
                raise RetryableError(f"Groq error {response.status_code}")
            
            response.raise_for_status()
            data = response.json()
            
        return data["choices"][0]["message"]["content"]
    except httpx.TimeoutException as e:
        raise RetryableError(f"Groq timeout: {e}")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(RetryableError),
    reraise=True,
)
async def _call_openrouter(
    messages: list[dict],
    model: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """Call OpenRouter API with retry logic."""
    settings = get_settings()
    
    if not settings.openrouter_api_key:
        raise ValueError("OPENROUTER_API_KEY not configured")
    
    headers = {
        "Authorization": f"Bearer {settings.openrouter_api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://madras1-lancer.hf.space",
        "X-Title": "Lancer Search API",
    }
    
    payload = {
        "model": model,
        "messages": messages,
    }
    
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                content=json.dumps(payload),
            )
            
            # Retry on rate limit or server errors
            if response.status_code in (429, 502, 503, 504):
                raise RetryableError(f"OpenRouter error {response.status_code}")
            
            if response.status_code != 200:
                error_text = response.text
                raise ValueError(f"OpenRouter error {response.status_code}: {error_text}")
            
            data = response.json()
            return data["choices"][0]["message"]["content"]
    except httpx.TimeoutException as e:
        raise RetryableError(f"OpenRouter timeout: {e}")


async def generate_completion_stream(
    messages: list[dict],
    model: Optional[str] = None,
    temperature: float = 0.3,
    max_tokens: int = 2048,
) -> AsyncIterator[str]:
    """Generate a streaming completion using OpenRouter."""
    settings = get_settings()
    model = model or settings.llm_model
    
    if not settings.openrouter_api_key:
        raise ValueError("OPENROUTER_API_KEY not configured")
    
    headers = {
        "Authorization": f"Bearer {settings.openrouter_api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://madras1-lancer.hf.space",
        "X-Title": "Lancer Search API",
    }
    
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
    }
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream(
            "POST",
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            content=json.dumps(payload),
        ) as response:
            if response.status_code != 200:
                error_text = await response.aread()
                raise ValueError(f"OpenRouter streaming error {response.status_code}: {error_text}")
            
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        delta = data.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        continue
