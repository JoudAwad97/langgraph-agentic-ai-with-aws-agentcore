from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from loguru import logger

from src.application.orchestrator.workflow.state import OrchestratorState
from src.application.orchestrator.workflow.nodes import (
    orchestrator_node,
    memory_post_hook,
)
from src.application.orchestrator.workflow.edges import should_continue_orchestrator
from src.application.orchestrator.workflow.tools import get_orchestrator_tools
from src.infrastructure.memory import ShortTermMemory


# Module-level graph instance (created lazily)
_graph_instance = None


def create_orchestrator_graph(force_recreate: bool = False):
    """
    Create the multi-agent orchestrator workflow graph with ReAct pattern.

    Uses a module-level singleton pattern instead of @lru_cache to support
    dynamic tool loading based on configuration changes.

    Note: Guardrails are applied at the API layer (utils.py), not in the graph.
    This keeps the graph focused on orchestration logic.

    Args:
        force_recreate: If True, recreates the graph even if one exists.
                       Useful when configuration changes at runtime.

    Architecture (ReAct Pattern):

        START
          │
          ▼
    ┌───────────────┐
    │  Orchestrator │◄──────────────┐
    │   (ReAct)     │               │
    │               │               │
    │ Thought:      │               │
    │ Action:       │               │
    └───────┬───────┘               │
            │                       │
            ▼                       │
    [has tool calls?]               │
       │         │                  │
    yes│         │no                │
       │         │(Final Answer)    │
       ▼         │                  │
    ┌─────────┐  │                  │
    │ToolNode │  │                  │
    │         │  │                  │
    │Observa- │  │                  │
    │tion:    │──┘                  │
    └────┬────┘                     │
         │                          │
         └──────────────────────────┘
                 │
                 ▼ (when no tool calls)
        ┌───────────────┐
        │ Memory Post-  │
        │    Hook       │
        └───────┬───────┘
                │
                ▼
               END

    ReAct Loop (Reasoning + Acting):
    1. Orchestrator outputs: Thought (reasoning) + Action (tool call or Final Answer)
    2. If Action is a tool call → ToolNode executes → Observation returned → loop back
    3. If Action is Final Answer → proceed to memory hook → END

    The orchestrator decides which sub-agent tools to call:
    - restaurant_data_tool: MCP Gateway for structured restaurant data (always available)
    - restaurant_explorer_tool: Browser-based web search (if ENABLE_BROWSER_TOOLS=True)
    - restaurant_research_tool: Detailed restaurant research (if ENABLE_BROWSER_TOOLS=True)
    - memory_retrieval_tool: On-demand memory retrieval (always available)

    Memory:
    - Retrieval: On-demand via memory_retrieval_tool (agent calls when needed)
    - Post-hook: Saves the conversation turn to trigger memory strategies
    """
    global _graph_instance

    if _graph_instance is not None and not force_recreate:
        return _graph_instance

    logger.info("Creating orchestrator graph (ReAct pattern)...")

    graph_builder = StateGraph(OrchestratorState)

    # Add the orchestrator node (implements ReAct reasoning)
    graph_builder.add_node("orchestrator_node", orchestrator_node)

    # Get tools dynamically based on current config (respects ENABLE_BROWSER_TOOLS)
    tools = get_orchestrator_tools()
    tool_node = ToolNode(tools)
    graph_builder.add_node("tool_node", tool_node)

    # Add the memory post-hook node (saves conversation turn after response)
    graph_builder.add_node("memory_post_hook", memory_post_hook)

    # Define edges
    # START -> Orchestrator
    graph_builder.add_edge(START, "orchestrator_node")

    # Conditional edge from orchestrator - ReAct loop
    # - If tool calls: route to tools, then back to orchestrator
    # - If no tool calls (Final Answer): route to memory hook, then END
    graph_builder.add_conditional_edges(
        "orchestrator_node",
        should_continue_orchestrator,
        {
            "tools": "tool_node",
            "end": "memory_post_hook",
        },
    )

    # After tools (Observation), return to orchestrator for next Thought/Action
    graph_builder.add_edge("tool_node", "orchestrator_node")

    # Memory Post-Hook -> END
    graph_builder.add_edge("memory_post_hook", END)

    # Setup Short-Term Memory (STM) checkpointer
    checkpointer = ShortTermMemory().get_memory()

    _graph_instance = graph_builder.compile(checkpointer=checkpointer)
    logger.info("Orchestrator graph (ReAct) created successfully")

    return _graph_instance


def reset_graph():
    """Reset the cached graph instance. Call this if configuration changes."""
    global _graph_instance
    _graph_instance = None
    logger.info("Graph instance reset - will be recreated on next use")
