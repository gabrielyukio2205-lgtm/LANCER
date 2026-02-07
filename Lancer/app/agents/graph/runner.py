"""Graph runner - executes the agent graph.

The runner orchestrates node execution, manages state transitions,
and yields status updates for streaming.

Uses timeout-based execution instead of fixed iteration count.
"""

import logging
import time
from typing import AsyncGenerator, Dict, Type

from app.agents.graph.state import AgentState, NodeType
from app.agents.graph.nodes import (
    BaseNode,
    PlanNode,
    SearchNode,
    NavigateNode,
    ExtractNode,
    VerifyNode,
    RespondNode,
)

logger = logging.getLogger(__name__)

# Node registry
NODE_REGISTRY: Dict[NodeType, Type[BaseNode]] = {
    NodeType.PLAN: PlanNode,
    NodeType.SEARCH: SearchNode,
    NodeType.NAVIGATE: NavigateNode,
    NodeType.EXTRACT: ExtractNode,
    NodeType.VERIFY: VerifyNode,
    NodeType.RESPOND: RespondNode,
}

# Status messages with emojis
STATUS_MESSAGES = {
    NodeType.PLAN: "🎯 Planning task...",
    NodeType.SEARCH: "🔍 Searching...",
    NodeType.NAVIGATE: "🌐 Navigating...",
    NodeType.EXTRACT: "📊 Extracting content...",
    NodeType.VERIFY: "🤔 Analyzing...",
    NodeType.RESPOND: "✅ Generating response...",
}


async def run_graph(state: AgentState) -> AsyncGenerator[dict, None]:
    """Run the agent graph and yield status updates.
    
    Args:
        state: Initial agent state with task, url, and desktop
        
    Yields:
        Status updates and final result
    """
    # Initialize timing
    state.start_time = time.time()
    current_node_type = NodeType.PLAN
    state.current_node = current_node_type
    
    logger.info(f"Starting graph execution for task: {state.task[:50]}, timeout: {state.timeout_seconds}s")
    
    while state.should_continue():
        state.step_count += 1
        state.current_node = current_node_type
        
        # Get node instance
        node_class = NODE_REGISTRY.get(current_node_type)
        if not node_class:
            logger.error(f"Unknown node type: {current_node_type}")
            break
        
        node = node_class()
        
        # Calculate remaining time
        remaining = int(state.get_remaining_time())
        elapsed = int(state.get_elapsed_time())
        
        # Yield status update
        status_msg = STATUS_MESSAGES.get(current_node_type, "Processing...")
        if current_node_type == NodeType.SEARCH and state.plan.get("steps"):
            for step in state.plan["steps"]:
                if step.get("action") == "search":
                    status_msg = f"🔍 Searching: {step.get('query', state.task)[:40]}..."
                    break
        elif current_node_type == NodeType.NAVIGATE and state.url:
            status_msg = f"🌐 Navigating to {state.url[:40]}..."
        
        yield {
            "type": "status",
            "message": f"{status_msg} (step {state.step_count}, {remaining}s remaining)"
        }
        
        # Execute node
        try:
            state, next_node_type = await node.execute(state)
            logger.info(f"Step {state.step_count}: {current_node_type.value} -> {next_node_type.value} ({elapsed}s elapsed)")
            
            # Check if we're done
            if current_node_type == NodeType.RESPOND:
                break
            
            # Transition to next node
            current_node_type = next_node_type
            
        except Exception as e:
            logger.exception(f"Node execution failed: {e}")
            state.add_error(str(e))
            
            # If running low on time, try to respond
            if state.get_remaining_time() < 30:
                current_node_type = NodeType.RESPOND
            else:
                current_node_type = NodeType.SEARCH
    
    # If we timed out without a result, generate one from what we have
    if not state.final_result and not state.success:
        logger.warning("Timeout reached, forcing response generation")
        respond_node = RespondNode()
        state, _ = await respond_node.execute(state)
    
    # Yield final result
    yield {
        "type": "result",
        "content": state.final_result,
        "links": state.visited_urls[:10],
        "success": state.success
    }
    
    yield {"type": "complete", "message": f"Task completed in {int(state.get_elapsed_time())}s"}
    
    logger.info(f"Graph execution complete. Success: {state.success}, Steps: {state.step_count}, Time: {state.get_elapsed_time():.1f}s")

