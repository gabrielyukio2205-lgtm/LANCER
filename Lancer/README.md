---
title: Lancer Search API
emoji: 🔍
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
---

# Lancer Search API

🔍 Advanced AI-powered search API with temporal intelligence.

## Features

- **Temporal Intelligence**: Understands when you need fresh vs historical info
- **Multi-Stage Reranking**: Freshness + Authority scoring
- **Multi-Source Search**: Tavily, DuckDuckGo
- **LLM Synthesis**: Groq or OpenRouter

## API Endpoints

```bash
# Search with synthesis
POST /api/v1/search
{
    "query": "What is the latest GPT model?",
    "max_results": 10,
    "freshness": "week"
}

# Health check
GET /health
```

## Environment Variables

Configure these in HuggingFace Space Secrets:

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | Yes* | Groq API key |
| `OPENROUTER_API_KEY` | Yes* | OpenRouter API key |
| `TAVILY_API_KEY` | Yes | Tavily search API key |
| `LLM_PROVIDER` | No | "groq" or "openrouter" |

*At least one LLM provider key required

## Local Development

```bash
pip install -e .
uvicorn app.main:app --reload
```
