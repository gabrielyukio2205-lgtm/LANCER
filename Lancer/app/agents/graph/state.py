"""Agent state management for graph-based execution.

The state is passed between nodes and accumulates information
throughout the agent's execution.
"""

from dataclasses import dataclass, field
from typing import Optional, Any
from enum import Enum


class NodeType(Enum):
    """Types of nodes in the agent graph."""
    START = "start"
    PLAN = "plan"
    SEARCH = "search"
    NAVIGATE = "navigate"
    EXTRACT = "extract"
    VERIFY = "verify"
    RESPOND = "respond"
    ERROR = "error"


@dataclass
class AgentState:
    """Shared state passed between graph nodes."""
    
    # Task info
    task: str = ""
    url: Optional[str] = None
    
    # Planning
    plan: dict = field(default_factory=dict)
    current_subtask: int = 0
    
    # Execution
    current_node: NodeType = NodeType.START
    step_count: int = 0
    start_time: float = field(default_factory=lambda: 0.0)
    timeout_seconds: float = 300.0  # 5 minutes default
    
    # Memory
    visited_urls: list = field(default_factory=list)
    extracted_data: list = field(default_factory=list)
    page_content: str = ""
    window_title: str = ""
    known_facts: list = field(default_factory=list)
    missing_points: list = field(default_factory=list)
    last_queries: list = field(default_factory=list)
    
    # History
    action_history: list = field(default_factory=list)
    error_history: list = field(default_factory=list)
    
    # Results
    final_result: str = ""
    success: bool = False
    
    # Desktop reference (set at runtime)
    desktop: Any = None
    
    def add_action(self, action: dict):
        """Add action to history."""
        self.action_history.append({
            "step": self.step_count,
            "node": self.current_node.value,
            "action": action
        })
    
    def add_error(self, error: str):
        """Add error to history."""
        self.error_history.append({
            "step": self.step_count,
            "error": error
        })
    
    def add_extracted_data(self, source: str, data: dict):
        """Add extracted data from a source."""
        self.extracted_data.append({
            "source": source,
            "url": self.visited_urls[-1] if self.visited_urls else "",
            "data": data
        })

    def add_query(self, query: str):
        """Track recent search queries used by the agent."""
        query = (query or "").strip()
        if not query:
            return
        if query not in self.last_queries:
            self.last_queries.append(query)
        self.last_queries = self.last_queries[-8:]

    def update_research_progress(self, known_facts: list | None = None, missing_points: list | None = None):
        """Update short-term research memory from LLM output."""
        if known_facts:
            for fact in known_facts:
                text = str(fact).strip()
                if text and text not in self.known_facts:
                    self.known_facts.append(text)
            self.known_facts = self.known_facts[-10:]

        if missing_points:
            cleaned = []
            for point in missing_points:
                text = str(point).strip()
                if text:
                    cleaned.append(text)
            # Keep latest gaps as "active" missing info.
            self.missing_points = cleaned[-8:]
    
    def get_context_for_llm(self) -> str:
        """Get formatted context for LLM prompts."""
        context_parts = []
        
        if self.action_history:
            recent = self.action_history[-5:]
            context_parts.append("Recent actions:")
            for h in recent:
                action_str = h.get('action', h)
                node_str = h.get('node', 'action')
                context_parts.append(f"  - {node_str}: {action_str}")
        
        if self.extracted_data:
            context_parts.append("\nExtracted data:")
            for d in self.extracted_data[-5:]:
                # Support both old format (source/data) and new format (url/preview)
                source = d.get('source') or d.get('url', 'unknown')
                data = d.get('data') or d.get('preview', '')[:100]
                context_parts.append(f"  - {source[:50]}: {data[:100]}...")

        if self.known_facts:
            context_parts.append("\nKnown facts:")
            for fact in self.known_facts[-5:]:
                context_parts.append(f"  - {fact}")

        if self.missing_points:
            context_parts.append("\nMissing points:")
            for point in self.missing_points[-5:]:
                context_parts.append(f"  - {point}")

        if self.last_queries:
            context_parts.append("\nRecent queries:")
            for query in self.last_queries[-5:]:
                context_parts.append(f"  - {query}")
        
        if self.error_history:
            context_parts.append("\nErrors encountered:")
            for e in self.error_history[-3:]:
                context_parts.append(f"  - {e.get('error', str(e))}")
        
        return "\n".join(context_parts)
    
    def should_continue(self) -> bool:
        """Check if agent should continue execution based on timeout."""
        import time
        if self.start_time == 0:
            self.start_time = time.time()
        
        elapsed = time.time() - self.start_time
        time_ok = elapsed < self.timeout_seconds
        
        return (
            not self.success and
            time_ok and
            self.current_node != NodeType.ERROR
        )
    
    def get_elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        import time
        if self.start_time == 0:
            return 0.0
        return time.time() - self.start_time
    
    def get_remaining_time(self) -> float:
        """Get remaining time in seconds."""
        return max(0, self.timeout_seconds - self.get_elapsed_time())
