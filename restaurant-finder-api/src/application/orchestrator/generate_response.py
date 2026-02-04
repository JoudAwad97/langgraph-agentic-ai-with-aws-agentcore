import re
from typing import AsyncGenerator, Any, Union

from langchain_core.messages import AIMessage, HumanMessage
from loguru import logger

from src.application.orchestrator.workflow.graph import create_orchestrator_graph


def _is_malformed_tool_content(content: str) -> bool:
    """
    Check if content appears to be malformed tool call XML/syntax.

    Sometimes the model streams partial tool call formatting that should
    not be shown to the user. This function detects such cases.

    Args:
        content: The content string to check

    Returns:
        True if content looks like malformed tool syntax, False otherwise
    """
    if not content:
        return False

    # Patterns that indicate malformed tool call content
    malformed_patterns = [
        "<function",
        "</function",
        "<antml",
        "</antml",
        "antml::",
        "antml:invoke",
        "antml:parameter",
        "<tml",
        "</tml",
        'name="restaurant_',
        'name="memory_',
    ]

    content_lower = content.lower()
    return any(pattern.lower() in content_lower for pattern in malformed_patterns)


def _sanitize_actor_id(name: str) -> str:
    """
    Sanitize and format the actor ID to match AgentCore's required pattern.

    The pattern requires: [a-zA-Z0-9][a-zA-Z0-9-/]*(?::[a-zA-Z0-9-/]+)[a-zA-Z0-9-_/]
    Format: prefix:identifier (e.g., "user:john-doe")

    Args:
        name: The customer name or identifier to sanitize

    Returns:
        A properly formatted actor ID like "user:john-doe"
    """
    # Remove any characters that aren't alphanumeric, dash, underscore, or space
    sanitized = re.sub(r'[^a-zA-Z0-9\-_ ]', '', name)
    # Replace spaces with dashes and convert to lowercase
    sanitized = sanitized.replace(' ', '-').lower()
    # Ensure it's not empty
    if not sanitized:
        sanitized = "guest"
    # Format as user:identifier
    return f"user:{sanitized}"


def _extract_text_from_content(content: Any) -> str:
    """
    Extract text from message content.

    Handles both string content and Bedrock's list-of-blocks format.

    Args:
        content: Message content (string or list of content blocks)

    Returns:
        Extracted text as a string
    """
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict):
                text_parts.append(block.get("text", ""))
            elif hasattr(block, "text"):
                text_parts.append(block.text)
            else:
                text_parts.append(str(block))
        return "".join(text_parts)

    return str(content) if content else ""


async def get_streaming_response(
    messages: str | list[str],
    customer_name: str = "Guest",
    conversation_id: str | None = None,
) -> AsyncGenerator[str, None]:
    """
    Get response from the orchestrator workflow using buffer mode.

    Buffer Mode:
    - Runs the full ReAct workflow (Reasoning + Acting loop)
    - Agent reasons, calls tools, observes results, and continues until ready
    - Only yields the FINAL response after the ReAct loop completes

    This ensures users only see the final response, avoiding confusing
    intermediate outputs or partial content during tool execution.

    Args:
        messages: User message(s) to process
        customer_name: Name of the customer for personalization
        conversation_id: Unique ID for the conversation thread (for memory persistence)

    Yields:
        The final response content after ReAct loop completes
    """
    graph = create_orchestrator_graph()

    try:
        thread_id = conversation_id or "default-thread"
        actor_id = _sanitize_actor_id(customer_name)

        config = {
            "configurable": {
                "thread_id": thread_id,
                "customer_name": customer_name,
                "actor_id": actor_id,
            }
        }

        logger.info(f"Starting ReAct workflow execution (thread_id={thread_id})")

        # Run the full ReAct workflow - waits for completion before returning
        result = await graph.ainvoke(
            input={
                "messages": __format_messages(messages=messages),
                "customer_name": customer_name,
            },
            config=config,
        )

        logger.info("ReAct workflow complete, extracting final response")

        # Extract the final AI message from the result
        final_response = ""
        messages_list = result.get("messages", [])

        # Find the last AI message that doesn't have tool calls (the actual response)
        for msg in reversed(messages_list):
            if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
                content = _extract_text_from_content(msg.content)
                if content and not _is_malformed_tool_content(content):
                    final_response = content
                    break

        if not final_response:
            logger.warning("No valid final response found in workflow result")
            final_response = "I apologize, but I wasn't able to generate a response. Please try again."

        # Log ReAct completion metrics
        tool_call_count = result.get("tool_call_count", 0)
        logger.info(f"Final response ready: tool_calls={tool_call_count}")

        # Yield the final response
        # We yield in chunks to maintain SSE compatibility and allow frontend to process
        chunk_size = 500  # Characters per chunk for smooth streaming effect
        for i in range(0, len(final_response), chunk_size):
            chunk = final_response[i:i + chunk_size]
            yield chunk

        logger.info(f"Response delivered: {len(final_response)} characters")

    except Exception as e:
        logger.error(f"Error in get_streaming_response: {type(e).__name__}: {str(e)}")
        logger.exception("Full traceback:")
        raise RuntimeError(f"Error running conversation workflow: {str(e)}") from e


def __format_messages(
    messages: Union[str, list[dict[str, Any]]],
) -> list[Union[HumanMessage, AIMessage]]:
    """Convert various message formats to a list of LangChain message objects.

    Args:
        messages: Can be one of:
            - A single string message
            - A list of string messages
            - A list of dictionaries with 'role' and 'content' keys

    Returns:
        List[Union[HumanMessage, AIMessage]]: A list of LangChain message objects
    """
    if isinstance(messages, str):
        return [HumanMessage(content=messages)]

    if isinstance(messages, list):
        if not messages:
            return []

        if (
            isinstance(messages[0], dict)
            and "role" in messages[0]
            and "content" in messages[0]
        ):
            result = []
            for msg in messages:
                if msg["role"] == "user":
                    result.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    result.append(AIMessage(content=msg["content"]))
            return result

        return [HumanMessage(content=message) for message in messages]

    return []
