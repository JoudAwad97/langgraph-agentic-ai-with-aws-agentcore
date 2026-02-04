from typing import Literal

from langchain_core.messages import AIMessage
from loguru import logger

from src.application.orchestrator.workflow.state import OrchestratorState


# Maximum tool calls per turn to prevent excessive latency
MAX_TOOL_CALLS_PER_TURN = 4


def should_continue_orchestrator(
    state: OrchestratorState,
) -> Literal["tools", "end"]:
    """
    Determine if the orchestrator should continue to tools or end (ReAct pattern).

    This edge condition implements the ReAct loop:
    - If the agent has tool calls → route to tools → back to orchestrator
    - If no tool calls (Final Answer) → route to memory_post_hook → END

    The ReAct cycle continues until:
    1. The agent produces a Final Answer (no tool calls)
    2. The tool call limit is reached

    Args:
        state: The current orchestrator state.

    Returns:
        "tools" if there are tool calls to process (continue ReAct loop)
        "end" if no tool calls (Final Answer reached)
    """
    messages = state.get("messages", [])
    tool_call_count = state.get("tool_call_count", 0)

    if not messages:
        logger.debug("No messages in state, ending ReAct loop")
        return "end"

    last_message = messages[-1]

    # Check if the last message is an AIMessage with tool calls
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        # Check if we've hit the tool call limit
        if tool_call_count >= MAX_TOOL_CALLS_PER_TURN:
            logger.warning(
                f"Tool call limit ({MAX_TOOL_CALLS_PER_TURN}) reached, "
                "forcing end of ReAct loop"
            )
            return "end"

        logger.debug(
            f"ReAct loop: tool calls detected (count: {tool_call_count + 1}), "
            "routing to tools"
        )
        return "tools"

    # No tool calls - agent has produced Final Answer
    logger.debug("ReAct loop: no tool calls (Final Answer), ending")
    return "end"
