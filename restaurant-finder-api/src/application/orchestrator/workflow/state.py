from typing import Annotated, TypedDict

from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage


class OrchestratorState(TypedDict):
    """
    State for the Orchestrator Agent with Reflection pattern.

    The reflection pattern adds a critic step that evaluates the orchestrator's
    response before finalizing. If the response doesn't meet quality criteria,
    it loops back for refinement.

    Architecture:
        Orchestrator → Tools → Reflector → (loop back or END)
    """

    # Messages with reducer that appends new messages
    messages: Annotated[list[BaseMessage], add_messages]

    # Customer name for personalization
    customer_name: str

    # Reflection tracking
    reflection_count: int  # Number of reflection iterations completed
    reflection_feedback: str  # Feedback from reflector for improvement
    is_satisfactory: bool  # Whether the response meets quality criteria

    # Tool call tracking (for efficiency limits)
    tool_call_count: int  # Number of tool calls in current turn
    made_tool_calls: bool  # Whether any tool calls were made this turn
