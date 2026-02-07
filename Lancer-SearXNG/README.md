---
title: Lancer SearXNG
emoji: 🔍
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
---

# Lancer SearXNG

Versão experimental do Lancer usando apenas SearXNG como fonte de busca.

## Diferenças

- **Sem APIs pagas** (Tavily, Brave)
- **50+ resultados** por query (vs 15-20 das APIs)
- **Embedding reranking** faz sentido aqui!

## Endpoints

```bash
POST /api/v1/search
{
    "query": "python asyncio tutorial",
    "max_results": 10
}
```
