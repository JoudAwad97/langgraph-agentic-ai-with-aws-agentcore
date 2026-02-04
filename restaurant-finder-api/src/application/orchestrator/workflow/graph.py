from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from loguru import logger

from src.application.orchestrator.workflow.state import OrchestratorState
from src.application.orchestrator.workflow.nodes import (
    orchestrator_node,
    reflector_node,
    memory_post_hook,
)
from src.application.orchestrator.workflow.edges import (
    should_continue_orchestrator,
    should_refine_or_end,
)
from src.application.orchestrator.workflow.tools import get_orchestrator_tools
from src.infrastructure.memory import ShortTermMemory


# Module-level graph instance (created lazily)
_graph_instance = None


def create_orchestrator_graph(force_recreate: bool = False):
    """
    Create the multi-agent orchestrator workflow graph with Reflection pattern.

    Uses a module-level singleton pattern instead of @lru_cache to support
    dynamic tool loading based on configuration changes.

    Note: Guardrails are applied at the API layer (utils.py), not in the graph.
    This keeps the graph focused on orchestration logic.

    Args:
        force_recreate: If True, recreates the graph even if one exists.
                       Useful when configuration changes at runtime.

    Architecture (Reflection Pattern):

        START
          │
          ▼
    ┌───────────────┐
    │  Orchestrator │◄────────────────────┐
    │     Node      │                     │
    └───────┬───────┘                     │
            │                             │
            ▼                             │
    [has tool calls?]                     │
       │         │                        │
    yes│         │no                      │
       ▼         │                        │
    ┌─────────┐  │                        │
    │ToolNode │──┘                        │
    └────┬────┘                           │
         │                                │
         └───► back to orchestrator       │
                    │                     │
                    ▼ (when no tool calls)│
            ┌───────────────┐             │
            │   Reflector   │             │
            │   (Critic)    │             │
            └───────┬───────┘             │
                    │                     │
            [is satisfactory?]            │
               │         │                │
             no│         │yes             │
               │         │                │
               └─────────┼────────────────┘
                         │ (refine loop)
                         ▼
            ┌───────────────┐
            │ Memory Post-  │
            │    Hook       │
            └───────┬───────┘
                    │
                    ▼
                   END

    The orchestrator decides which sub-agent tools to call:
    - restaurant_data_tool: MCP Gateway for structured restaurant data (always available)
    - restaurant_explorer_tool: Browser-based web search (if ENABLE_BROWSER_TOOLS=True)
    - restaurant_research_tool: Detailed restaurant research (if ENABLE_BROWSER_TOOLS=True)
    - memory_retrieval_tool: On-demand memory retrieval (always available)

    Reflection:
    - After orchestrator finishes, reflector evaluates response quality
    - If not satisfactory, loops back to orchestrator with feedback
    - Maximum 2 refinement iterations to prevent infinite loops

    Memory:
    - Retrieval: On-demand via memory_retrieval_tool (agent calls when needed)
    - Post-hook: Saves the conversation turn to trigger memory strategies
    """
    global _graph_instance

    if _graph_instance is not None and not force_recreate:
        return _graph_instance

    logger.info("Creating orchestrator graph...")

    graph_builder = StateGraph(OrchestratorState)

    # Add the orchestrator node
    graph_builder.add_node("orchestrator_node", orchestrator_node)

    # Get tools dynamically based on current config (respects ENABLE_BROWSER_TOOLS)
    tools = get_orchestrator_tools()
    tool_node = ToolNode(tools)
    graph_builder.add_node("tool_node", tool_node)

    # Add the reflector node (quality evaluation)
    graph_builder.add_node("reflector_node", reflector_node)

    # Add the memory post-hook node (saves conversation turn after response)
    graph_builder.add_node("memory_post_hook", memory_post_hook)

    # Define edges
    # START -> Orchestrator (memory is now retrieved on-demand via tool)
    graph_builder.add_edge(START, "orchestrator_node")

    # Conditional edge from orchestrator - check for tool calls
    graph_builder.add_conditional_edges(
        "orchestrator_node",
        should_continue_orchestrator,
        {
            "tools": "tool_node",
            "reflector": "reflector_node",  # Go to reflector when no more tool calls
            "memory_post_hook": "memory_post_hook",  # Skip reflector for pure conversational responses
        },
    )

    # After tools, return to orchestrator for further processing
    graph_builder.add_edge("tool_node", "orchestrator_node")

    # Conditional edge from reflector - check if response is satisfactory
    graph_builder.add_conditional_edges(
        "reflector_node",
        should_refine_or_end,
        {
            "refine": "orchestrator_node",  # Loop back for improvement
            "end": "memory_post_hook",  # Proceed to save and end
        },
    )

    # Memory Post-Hook -> END
    graph_builder.add_edge("memory_post_hook", END)

    # Setup Short-Term Memory (STM) checkpointer
    checkpointer = ShortTermMemory().get_memory()

    _graph_instance = graph_builder.compile(checkpointer=checkpointer)
    logger.info("Orchestrator graph created successfully")

    return _graph_instance


def reset_graph():
    """Reset the cached graph instance. Call this if configuration changes."""
    global _graph_instance
    _graph_instance = None
    logger.info("Graph instance reset - will be recreated on next use")
