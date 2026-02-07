"""Graph nodes for the agent execution.

Each node represents a step in the agent's decision process:
- PlanNode: Decomposes the task into subtasks
- SearchNode: Performs web searches
- NavigateNode: Navigates to URLs
- ExtractNode: Extracts content from pages
- VerifyNode: Verifies if goal is achieved
- RespondNode: Generates final response
"""

import json
import logging
import shlex
import base64
from abc import ABC, abstractmethod
from typing import Tuple

from app.agents.graph.state import AgentState, NodeType
from app.agents.llm_client import generate_completion

logger = logging.getLogger(__name__)


class BaseNode(ABC):
    """Base class for all graph nodes."""
    
    node_type: NodeType = NodeType.START
    
    @abstractmethod
    async def execute(self, state: AgentState) -> Tuple[AgentState, NodeType]:
        """Execute the node logic and return updated state + next node."""
        pass


class PlanNode(BaseNode):
    """Decomposes task into subtasks."""
    
    node_type = NodeType.PLAN
    
    async def execute(self, state: AgentState) -> Tuple[AgentState, NodeType]:
        prompt = f"""Você é um planejador de tarefas. Decomponha a tarefa em passos simples.

TAREFA: {state.task}
URL inicial: {state.url or 'Nenhuma - começar com busca'}

Responda com JSON:
{{
    "goal": "objetivo principal",
    "steps": [
        {{"action": "search", "query": "termos de busca"}},
        {{"action": "navigate", "description": "onde navegar"}},
        {{"action": "extract", "what": "o que extrair"}}
    ],
    "success_criteria": "critério de sucesso"
}}

Responda APENAS o JSON, sem explicação."""

        try:
            response = await generate_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500
            )
            
            # Parse JSON
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
            
            plan = json.loads(response)
            state.plan = plan
            logger.info(f"Plan created: {plan.get('goal', 'No goal')}")
            
            # Decide next node based on plan
            if plan.get("steps") and plan["steps"][0].get("action") == "navigate" and state.url:
                return state, NodeType.NAVIGATE
            return state, NodeType.SEARCH
            
        except Exception as e:
            logger.error(f"Planning failed: {e}")
            state.add_error(f"Planning failed: {e}")
            # Fallback to search
            state.plan = {"goal": state.task, "steps": [{"action": "search", "query": state.task}]}
            return state, NodeType.SEARCH


class SearchNode(BaseNode):
    """Performs web search."""
    
    node_type = NodeType.SEARCH
    
    async def execute(self, state: AgentState) -> Tuple[AgentState, NodeType]:
        desktop = state.desktop
        
        # Determine search query
        query = state.task
        if state.plan.get("steps"):
            for step in state.plan["steps"]:
                if step.get("action") == "search" and step.get("query"):
                    query = step["query"]
                    break
        
        # Execute search
        search_url = f"https://html.duckduckgo.com/html/?q={query.replace(' ', '+')}"
        
        try:
            desktop.commands.run(f"google-chrome {shlex.quote(search_url)} &", background=True)
            state.visited_urls.append(search_url)
            desktop.wait(3000)
            
            state.add_action({"type": "search", "query": query})
            logger.info(f"Searched: {query}")
            
            return state, NodeType.EXTRACT
            
        except Exception as e:
            state.add_error(f"Search failed: {e}")
            return state, NodeType.VERIFY


class NavigateNode(BaseNode):
    """Navigates to a URL."""
    
    node_type = NodeType.NAVIGATE
    
    async def execute(self, state: AgentState) -> Tuple[AgentState, NodeType]:
        desktop = state.desktop
        
        # Get URL to navigate
        url = state.url
        if not url and state.extracted_data:
            # Try to get URL from extracted links
            last_data = state.extracted_data[-1]
            if "links" in last_data.get("data", {}):
                links = last_data["data"]["links"]
                if links:
                    url = links[0]
        
        if not url:
            return state, NodeType.SEARCH
        
        try:
            desktop.commands.run(f"google-chrome {shlex.quote(url)} &", background=True)
            if url not in state.visited_urls:
                state.visited_urls.append(url)
            desktop.wait(3000)
            
            state.add_action({"type": "navigate", "url": url})
            logger.info(f"Navigated to: {url[:50]}")
            
            return state, NodeType.EXTRACT
            
        except Exception as e:
            state.add_error(f"Navigation failed: {e}")
            return state, NodeType.SEARCH


class ExtractNode(BaseNode):
    """Extracts content from current page."""
    
    node_type = NodeType.EXTRACT
    
    async def execute(self, state: AgentState) -> Tuple[AgentState, NodeType]:
        desktop = state.desktop
        current_url = state.visited_urls[-1] if state.visited_urls else ""
        
        try:
            # Get window title
            result = desktop.commands.run("xdotool getactivewindow getwindowname 2>/dev/null", timeout=5)
            state.window_title = result.stdout.strip() if hasattr(result, 'stdout') else ""
            
            # Extract page content via curl
            if current_url.startswith("http"):
                result = desktop.commands.run(
                    f"curl -sL --max-time 10 --connect-timeout 5 "
                    f"-A 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36' "
                    f"'{current_url}' 2>/dev/null | "
                    "sed -e 's/<script[^>]*>.*<\\/script>//g' -e 's/<style[^>]*>.*<\\/style>//g' | "
                    "sed 's/<[^>]*>//g' | "
                    "tr -s ' \\n' ' ' | "
                    "head -c 6000",
                    timeout=15
                )
                state.page_content = result.stdout.strip() if hasattr(result, 'stdout') else ""
            
            state.add_action({"type": "extract", "content_length": len(state.page_content)})
            logger.info(f"Extracted {len(state.page_content)} chars from {current_url[:50]}")
            
            return state, NodeType.VERIFY
            
        except Exception as e:
            state.add_error(f"Extraction failed: {e}")
            return state, NodeType.VERIFY


class VerifyNode(BaseNode):
    """Verifies if goal is achieved and decides next action."""
    
    node_type = NodeType.VERIFY
    
    async def execute(self, state: AgentState) -> Tuple[AgentState, NodeType]:
        context = state.get_context_for_llm()
        page_preview = state.page_content[:4000] if state.page_content else "(No content)"
        
        prompt = f"""Você é um agente de navegação web. Analise o conteúdo e decida o próximo passo.

TAREFA: {state.task}
PLANO: {state.plan.get('goal', 'Nenhum')}
CRITÉRIO DE SUCESSO: {state.plan.get('success_criteria', 'Encontrar a informação pedida')}

HISTÓRICO:
{context}

CONTEÚDO DA PÁGINA ATUAL:
{page_preview}

TEMPO RESTANTE: {int(state.get_remaining_time())}s

Decida:
1. Se encontrou a resposta, retorne: {{"status": "complete", "result": "Sua resposta formatada com **negrito** para valores importantes"}}
2. Se precisa buscar mais, retorne: {{"action": "search", "query": "nova busca"}}
3. Se precisa navegar para um link, retorne: {{"action": "navigate", "url": "https://..."}}
4. Se precisa rolar a página, retorne: {{"action": "scroll"}}

REGRAS:
- Use **negrito** para preços e valores importantes
- Cite as fontes
- Se página pede login, tente outra fonte
- Seja eficiente

Responda APENAS com JSON válido."""

        try:
            response = await generate_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800
            )
            
            # Parse response
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
            
            decision = json.loads(response)
            state.add_action({"type": "verify", "decision": decision})
            
            # Route based on decision
            if decision.get("status") == "complete":
                state.final_result = decision.get("result", "")
                state.success = True
                logger.info("Goal achieved!")
                return state, NodeType.RESPOND
            
            action = decision.get("action", "")
            if action == "search":
                # Update plan with new search
                state.plan["steps"] = [{"action": "search", "query": decision.get("query", state.task)}]
                return state, NodeType.SEARCH
            elif action == "navigate":
                state.url = decision.get("url", "")
                return state, NodeType.NAVIGATE
            elif action == "scroll":
                state.desktop.scroll(-3)
                state.desktop.wait(1000)
                return state, NodeType.EXTRACT
            
            # Default: try another search
            return state, NodeType.SEARCH
            
        except Exception as e:
            logger.error(f"Verify failed: {e}")
            state.add_error(f"Verify failed: {e}")
            
            # If we have some content, try to respond anyway
            if state.get_remaining_time() < 30:
                return state, NodeType.RESPOND
            return state, NodeType.SEARCH


class RespondNode(BaseNode):
    """Generates final response."""
    
    node_type = NodeType.RESPOND
    
    async def execute(self, state: AgentState) -> Tuple[AgentState, NodeType]:
        # If we already have a result, we're done
        if state.final_result:
            state.success = True
            return state, NodeType.RESPOND
        
        # Generate response from collected data
        context = state.get_context_for_llm()
        page_content = state.page_content[:3000] if state.page_content else "(Nenhum conteúdo extraído)"
        
        prompt = f"""Você realizou uma tarefa de navegação web. Sintetize os resultados.

TAREFA: {state.task}

DADOS COLETADOS:
{context}

ÚLTIMO CONTEÚDO DA PÁGINA:
{page_content}

URLs VISITADAS:
{chr(10).join(state.visited_urls[:5]) if state.visited_urls else '(Nenhuma)'}

INSTRUÇÕES:
- Gere uma resposta útil baseada no que foi encontrado
- Use **negrito** para valores importantes (preços, números, nomes)
- Cite as fontes quando possível
- Se não encontrou o que foi pedido, explique o que encontrou ou diga honestamente que não encontrou

Responda em português de forma clara e organizada."""

        try:
            response = await generate_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000
            )
            state.final_result = response.strip()
            state.success = bool(state.final_result)
            logger.info(f"Generated response: {len(state.final_result)} chars")
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            # Fallback: create response from available data
            if state.page_content:
                state.final_result = f"**Informação encontrada:**\n\n{state.page_content[:500]}...\n\n*Fonte: {state.visited_urls[-1] if state.visited_urls else 'desconhecida'}*"
            else:
                state.final_result = f"Não foi possível completar a tarefa. Erro: {e}"
        
        return state, NodeType.RESPOND
