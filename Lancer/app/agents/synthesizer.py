"""Answer synthesizer agent.

Generates a coherent answer from search results with citations.
"""

from datetime import datetime
from typing import Optional, AsyncIterator

from app.api.schemas import SearchResult, TemporalContext, Citation
from app.agents.llm_client import generate_completion, generate_completion_stream


SYNTHESIS_PROMPT = """You are a research assistant that synthesizes information from search results.

CURRENT DATE: {current_date}

USER QUERY: {query}

TEMPORAL CONTEXT:
- Query intent: {temporal_intent} (the user {intent_explanation})
- Temporal urgency: {temporal_urgency:.0%} (how important freshness is)

SEARCH RESULTS:
{formatted_results}

INSTRUCTIONS:
1. Synthesize a comprehensive answer based on the search results
2. ALWAYS cite your sources using [1], [2], etc. format
3. If the query requires current information, prioritize the most recent results
4. If there are conflicting dates or versions mentioned, use the most recent accurate information
5. Be concise but thorough
6. If information seems outdated compared to current date ({current_date}), note this
7. Write in the same language as the query

Generate your answer:"""


async def synthesize_answer(
    query: str,
    results: list[SearchResult],
    temporal_context: Optional[TemporalContext] = None,
) -> tuple[str, list[Citation]]:
    """
    Synthesize an answer from search results.
    
    Args:
        query: Original search query
        results: List of search results to synthesize from
        temporal_context: Temporal analysis context
        
    Returns:
        Tuple of (answer_text, citations_list)
    """
    if not results:
        return "No results found to synthesize an answer.", []
    
    messages = _build_messages(query, results, temporal_context)
    
    try:
        answer = await generate_completion(messages, temperature=0.3)
    except Exception as e:
        # Fallback: return a simple summary without LLM
        answer = f"Error generating synthesis: {e}. Please review the search results directly."
    
    # Build citations list
    citations = _build_citations(results)
    
    return answer, citations


async def synthesize_answer_stream(
    query: str,
    results: list[SearchResult],
    temporal_context: Optional[TemporalContext] = None,
) -> AsyncIterator[str]:
    """
    Synthesize an answer with streaming output.
    
    Yields chunks of the answer as they are generated.
    
    Args:
        query: Original search query
        results: List of search results to synthesize from
        temporal_context: Temporal analysis context
        
    Yields:
        Chunks of the answer text
    """
    if not results:
        yield "No results found to synthesize an answer."
        return
    
    messages = _build_messages(query, results, temporal_context)
    
    try:
        async for chunk in generate_completion_stream(messages, temperature=0.3):
            yield chunk
    except Exception as e:
        yield f"Error generating synthesis: {e}. Please review the search results directly."


def _build_messages(
    query: str,
    results: list[SearchResult],
    temporal_context: Optional[TemporalContext] = None,
) -> list[dict]:
    """Build messages for LLM prompt."""
    # Format results for the prompt
    formatted_results = format_results_for_prompt(results[:10])  # Top 10 only
    
    # Prepare temporal context
    current_date = datetime.now().strftime("%Y-%m-%d")
    temporal_intent = "neutral"
    temporal_urgency = 0.5
    
    if temporal_context:
        temporal_intent = temporal_context.query_temporal_intent
        temporal_urgency = temporal_context.temporal_urgency
        current_date = temporal_context.current_date
    
    # Map intent to explanation
    intent_explanations = {
        "current": "is looking for the most recent/current information",
        "historical": "is interested in historical or background information",
        "neutral": "has no specific temporal preference",
    }
    
    prompt = SYNTHESIS_PROMPT.format(
        current_date=current_date,
        query=query,
        temporal_intent=temporal_intent,
        intent_explanation=intent_explanations.get(temporal_intent, ""),
        temporal_urgency=temporal_urgency,
        formatted_results=formatted_results,
    )
    
    return [
        {"role": "system", "content": "You are a helpful research assistant."},
        {"role": "user", "content": prompt},
    ]


def _build_citations(results: list[SearchResult]) -> list[Citation]:
    """Build citations list from results."""
    citations = []
    for i, result in enumerate(results[:10], 1):
        citations.append(
            Citation(
                index=i,
                url=result.url,
                title=result.title,
            )
        )
    return citations


def format_results_for_prompt(results: list[SearchResult]) -> str:
    """Format search results for inclusion in the LLM prompt."""
    formatted = []
    
    for i, result in enumerate(results, 1):
        date_str = ""
        if result.published_date:
            date_str = f" (Published: {result.published_date.strftime('%Y-%m-%d')})"
        
        formatted.append(
            f"[{i}] {result.title}{date_str}\n"
            f"    URL: {result.url}\n"
            f"    Freshness: {result.freshness_score:.0%} | Authority: {result.authority_score:.0%}\n"
            f"    Content: {result.content[:500]}..."
        )
    
    return "\n\n".join(formatted)
