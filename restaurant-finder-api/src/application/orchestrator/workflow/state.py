from typing import Annotated, TypedDict

from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage


class OrchestratorState(TypedDict):
    """
    State for the Orchestrator Agent with ReAct pattern.

    The ReAct (Reasoning + Acting) pattern interleaves reasoning and action:
    - Thought: Agent reasons about what to do next
    - Action: Agent calls a tool or provides final answer
    - Observation: System provides tool results

    Architecture:
        START → Orchestrator → Tools → Orchestrator → ... → Memory Hook → END
    """

    # Messages with reducer that appends new messages
    messages: Annotated[list[BaseMessage], add_messages]

    # Customer name for personalization
    customer_name: str

    # Tool call tracking (for efficiency limits)
    tool_call_count: int  # Number of tool calls in current turn
    made_tool_calls: bool  # Whether any tool calls were made this turn
