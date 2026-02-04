from typing import Literal

from langchain_core.messages import AIMessage
from loguru import logger

from src.application.orchestrator.workflow.state import OrchestratorState


# Maximum tool calls per turn to prevent excessive latency
MAX_TOOL_CALLS_PER_TURN = 4


def should_continue_orchestrator(
    state: OrchestratorState,
) -> Literal["tools", "reflector", "memory_post_hook"]:
    """
    Determine if the orchestrator should continue to tools, reflector, or skip to end.

    This edge condition checks if the last message from the orchestrator
    contains tool calls. If so, route to the tools node.

    If no tool calls and no tools were used this turn, skip reflector (pure conversation).
    Otherwise, go to reflector for quality evaluation.

    Args:
        state: The current orchestrator state.

    Returns:
        "tools" if there are tool calls to process
        "memory_post_hook" if no tool calls and this was a pure conversational response
        "reflector" if tool calls were made and we need quality evaluation
    """
    messages = state.get("messages", [])
    tool_call_count = state.get("tool_call_count", 0)
    made_tool_calls = state.get("made_tool_calls", False)

    if not messages:
        return "memory_post_hook"

    last_message = messages[-1]

    # Check if the last message is an AIMessage with tool calls
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        # Check if we've hit the tool call limit
        if tool_call_count >= MAX_TOOL_CALLS_PER_TURN:
            logger.warning(f"Tool call limit ({MAX_TOOL_CALLS_PER_TURN}) reached, skipping to reflector")
            return "reflector"

        logger.debug(f"Orchestrator has tool calls (count: {tool_call_count + 1}), routing to tools")
        return "tools"

    # No tool calls - check if we should skip reflector
    if not made_tool_calls:
        # Pure conversational response (no tools used this turn)
        logger.debug("Pure conversational response, skipping reflector")
        return "memory_post_hook"

    logger.debug("Orchestrator finished with tool usage, routing to reflector for evaluation")
    return "reflector"


def should_refine_or_end(
    state: OrchestratorState,
) -> Literal["refine", "end"]:
    """
    Determine if the response needs refinement or can be finalized.

    This edge condition is called after the reflector evaluates the response.
    If the response is satisfactory, proceed to end (memory post-hook).
    If not satisfactory, loop back to the orchestrator for refinement.

    Args:
        state: The current orchestrator state with reflection results.

    Returns:
        "refine" if the response needs improvement, "end" if satisfactory.
    """
    is_satisfactory = state.get("is_satisfactory", True)
    reflection_count = state.get("reflection_count", 0)

    logger.info(f"Reflection decision: satisfactory={is_satisfactory}, count={reflection_count}")

    if is_satisfactory:
        logger.info("Response is satisfactory, proceeding to end")
        return "end"
    else:
        logger.info("Response needs refinement, looping back to orchestrator")
        return "refine"
