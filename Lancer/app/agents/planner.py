"""Research Planner Agent.

Decomposes complex queries into multiple research dimensions.
"""

import json
from typing import Optional

from pydantic import BaseModel, Field

from app.agents.llm_client import generate_completion
from app.config import get_settings


class ResearchDimension(BaseModel):
    """A single dimension/aspect to research."""
    
    name: str = Field(..., description="Short name for this dimension")
    description: str = Field(..., description="What this dimension covers")
    search_query: str = Field(..., description="Optimized search query for this dimension")
    priority: int = Field(default=1, ge=1, le=3, description="1=high, 2=medium, 3=low")


class ResearchPlan(BaseModel):
    """Complete research plan with all dimensions."""
    
    original_query: str
    refined_query: str = Field(..., description="Clarified version of the query")
    dimensions: list[ResearchDimension]
    estimated_sources: int = Field(default=20)


PLANNER_PROMPT = """You are a research planning assistant. Your job is to decompose a complex query into multiple research dimensions.

USER QUERY: {query}

INSTRUCTIONS:
1. Analyze the query and identify 2-6 key dimensions/aspects that need to be researched
2. Each dimension should be distinct and cover a different angle
3. Create an optimized search query for each dimension
4. Assign priority (1=high, 2=medium, 3=low) based on relevance to the main query
5. Respond ONLY with valid JSON, no other text

OUTPUT FORMAT:
{{
    "refined_query": "A clearer version of the user's query",
    "dimensions": [
        {{
            "name": "Short name",
            "description": "What this covers",
            "search_query": "Optimized search query",
            "priority": 1
        }}
    ]
}}

Generate the research plan:"""


async def create_research_plan(
    query: str,
    max_dimensions: int = 6,
) -> ResearchPlan:
    """
    Create a research plan by decomposing a query into dimensions.
    
    Args:
        query: The user's research query
        max_dimensions: Maximum number of dimensions to generate
        
    Returns:
        ResearchPlan with dimensions to investigate
    """
    settings = get_settings()
    
    messages = [
        {"role": "system", "content": "You are a research planning assistant. Always respond with valid JSON only."},
        {"role": "user", "content": PLANNER_PROMPT.format(query=query)},
    ]
    
    try:
        response = await generate_completion(messages, temperature=0.3)
        
        # Parse JSON response
        # Try to extract JSON if there's extra text
        json_start = response.find("{")
        json_end = response.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            response = response[json_start:json_end]
        
        data = json.loads(response)
        
        # Build dimensions
        dimensions = []
        for dim_data in data.get("dimensions", [])[:max_dimensions]:
            dimensions.append(ResearchDimension(
                name=dim_data.get("name", "Unknown"),
                description=dim_data.get("description", ""),
                search_query=dim_data.get("search_query", query),
                priority=dim_data.get("priority", 2),
            ))
        
        # Sort by priority
        dimensions.sort(key=lambda d: d.priority)
        
        return ResearchPlan(
            original_query=query,
            refined_query=data.get("refined_query", query),
            dimensions=dimensions,
            estimated_sources=len(dimensions) * 5,
        )
        
    except (json.JSONDecodeError, KeyError) as e:
        # Fallback: create a simple 2-dimension plan
        return ResearchPlan(
            original_query=query,
            refined_query=query,
            dimensions=[
                ResearchDimension(
                    name="Main Research",
                    description=f"Primary research on: {query}",
                    search_query=query,
                    priority=1,
                ),
                ResearchDimension(
                    name="Background",
                    description=f"Background and context for: {query}",
                    search_query=f"{query} background overview",
                    priority=2,
                ),
            ],
            estimated_sources=10,
        )
