import re
from typing import AsyncGenerator, Any, Union

from langchain_core.messages import AIMessage, HumanMessage, AIMessageChunk
from loguru import logger

from src.application.orchestrator.workflow.graph import create_orchestrator_graph


# Nodes that produce user-facing responses (for streaming)
RESPONSE_NODES = {"search_agent_node", "simple_response_node"}

# Nodes that should NOT have their output streamed (internal nodes)
INTERNAL_NODES = {"router_node", "memory_post_hook"}

# Event types to stream from
STREAMABLE_EVENTS = {"on_chat_model_stream"}


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
        "<tool_call>",
        "</tool_call>",
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
    enable_true_streaming: bool = True,
) -> AsyncGenerator[str, None]:
    """
    Get response from the workflow with Router + Search Agent pattern.

    Supports two modes:
    1. True Streaming (default): Streams tokens as they're generated
       - Best for simple responses (greetings, etc.) - immediate feedback
       - For search agent: streams final response tokens after tool calls complete
    2. Buffer Mode: Waits for full completion, then streams in chunks
       - Fallback mode if true streaming has issues

    Architecture:
        Router → [Simple Response | Search Agent] → Memory Hook → END

    The router classifies intent and routes to:
    - simple_response_node: Direct response (streams immediately)
    - search_agent_node: ReAct loop with tools (streams final response)

    Args:
        messages: User message(s) to process
        customer_name: Name of the customer for personalization
        conversation_id: Unique ID for the conversation thread (for memory persistence)
        enable_true_streaming: Use token streaming (True) or buffer mode (False)

    Yields:
        Response content tokens/chunks
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

        input_data = {
            "messages": __format_messages(messages=messages),
            "customer_name": customer_name,
        }

        logger.info(f"Starting workflow execution (thread_id={thread_id}, streaming={enable_true_streaming})")

        if enable_true_streaming:
            # True streaming mode - stream tokens as they're generated
            async for chunk in _stream_with_events(graph, input_data, config):
                yield chunk
        else:
            # Buffer mode - wait for completion, then stream
            async for chunk in _stream_buffered(graph, input_data, config):
                yield chunk

    except Exception as e:
        logger.error(f"Error in get_streaming_response: {type(e).__name__}: {str(e)}")
        logger.exception("Full traceback:")
        raise RuntimeError(f"Error running conversation workflow: {str(e)}") from e


async def _stream_with_events(
    graph,
    input_data: dict,
    config: dict,
) -> AsyncGenerator[str, None]:
    """
    Stream response tokens using LangGraph's astream_events.

    This provides true token-by-token streaming for better UX.
    Only streams from response nodes (search_agent_node, simple_response_node).
    Explicitly excludes internal nodes (router_node, memory_post_hook).

    For the search agent, we only stream the FINAL response (after tool calls),
    not intermediate reasoning or tool call formatting.

    Args:
        graph: The compiled LangGraph workflow
        input_data: Input state for the graph
        config: Runtime configuration

    Yields:
        Response tokens as they're generated
    """
    # Track state for smart streaming
    current_node = None
    has_tool_calls_pending = False
    streamed_content = []
    final_state = None  # Track final state for fallback extraction
    in_internal_node = False  # Track if we're in router or other internal nodes

    try:
        async for event in graph.astream_events(
            input=input_data,
            config=config,
            version="v2",
        ):
            event_type = event.get("event")
            event_name = event.get("name", "")
            event_data = event.get("data", {})
            tags = event.get("tags", [])

            # Track which node we're in (both response and internal nodes)
            if event_type == "on_chain_start":
                if event_name in RESPONSE_NODES:
                    current_node = event_name
                    in_internal_node = False
                    logger.debug(f"Entered response node: {current_node}")
                elif event_name in INTERNAL_NODES:
                    in_internal_node = True
                    current_node = None
                    logger.debug(f"Entered internal node: {event_name} (not streaming)")

            elif event_type == "on_chain_end":
                if event_name in RESPONSE_NODES:
                    logger.debug(f"Exited response node: {event_name}")
                    current_node = None
                elif event_name in INTERNAL_NODES:
                    in_internal_node = False
                    logger.debug(f"Exited internal node: {event_name}")

                # Capture final state when the main graph completes
                output = event_data.get("output")
                if output and isinstance(output, dict) and "messages" in output:
                    final_state = output

            # Stream tokens from the chat model
            elif event_type == "on_chat_model_stream":
                # Skip if we're in an internal node (router, memory_post_hook)
                if in_internal_node:
                    continue

                chunk = event_data.get("chunk")

                if chunk and isinstance(chunk, AIMessageChunk):
                    # Check if this chunk has tool calls (don't stream tool call content)
                    if chunk.tool_calls or chunk.tool_call_chunks:
                        has_tool_calls_pending = True
                        continue

                    # Extract text content
                    content = _extract_text_from_content(chunk.content)

                    if content:
                        # Filter out malformed tool content
                        if _is_malformed_tool_content(content):
                            continue

                        # Check if we're in a response node via tags or current_node
                        in_response_node = any(
                            node in str(tags) for node in RESPONSE_NODES
                        ) or current_node in RESPONSE_NODES

                        # Also check tags don't contain internal nodes
                        in_internal_via_tags = any(
                            node in str(tags) for node in INTERNAL_NODES
                        )

                        # Only stream if:
                        # 1. We're in a response node, AND
                        # 2. We're not in an internal node, AND
                        # 3. For search_agent, no pending tool calls (final response)
                        should_stream = (
                            in_response_node
                            and not in_internal_via_tags
                            and (current_node == "simple_response_node" or not has_tool_calls_pending)
                        )

                        if should_stream:
                            streamed_content.append(content)
                            yield content

            # Reset tool call flag when tools complete
            elif event_type == "on_chain_end" and "tool" in event_name.lower():
                # Tool execution completed, next response from search_agent is final
                has_tool_calls_pending = False

        total_chars = sum(len(c) for c in streamed_content)
        logger.info(f"Streaming complete: {total_chars} characters streamed")

        # If nothing was streamed, extract from final state (avoid re-running workflow)
        if not streamed_content and final_state:
            logger.warning("No content streamed, extracting from final state")
            final_response = _extract_final_response(final_state)
            if final_response:
                # Yield the extracted response in chunks
                chunk_size = 500
                for i in range(0, len(final_response), chunk_size):
                    yield final_response[i:i + chunk_size]
                logger.info(f"Fallback response delivered: {len(final_response)} characters")

    except Exception as e:
        logger.error(f"Error in streaming: {type(e).__name__}: {str(e)}")
        # Fall back to buffer mode on streaming errors
        logger.info("Falling back to buffer mode due to streaming error")
        async for chunk in _stream_buffered(graph, input_data, config):
            yield chunk


def _extract_final_response(state: dict) -> str:
    """
    Extract the final response from a workflow state.

    Args:
        state: The workflow state containing messages

    Returns:
        The final response text, or empty string if not found
    """
    messages_list = state.get("messages", [])

    # Find the last AI message that doesn't have tool calls
    for msg in reversed(messages_list):
        if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
            content = _extract_text_from_content(msg.content)
            if content and not _is_malformed_tool_content(content):
                return content

    return ""


async def _stream_buffered(
    graph,
    input_data: dict,
    config: dict,
) -> AsyncGenerator[str, None]:
    """
    Buffer mode: Run workflow to completion, then stream the final response.

    This is the fallback mode that ensures we always return a response.

    Args:
        graph: The compiled LangGraph workflow
        input_data: Input state for the graph
        config: Runtime configuration

    Yields:
        Response content in chunks
    """
    logger.info("Running workflow in buffer mode")

    # Run the full workflow - waits for completion
    result = await graph.ainvoke(input=input_data, config=config)

    logger.info("Workflow complete, extracting final response")

    # Extract the final response using shared helper
    final_response = _extract_final_response(result)

    if not final_response:
        logger.warning("No valid final response found in workflow result")
        final_response = "I apologize, but I wasn't able to generate a response. Please try again."

    # Log completion metrics
    tool_call_count = result.get("tool_call_count", 0)
    intent = result.get("intent", "unknown")
    logger.info(f"Final response ready: intent={intent}, tool_calls={tool_call_count}")

    # Yield in chunks for SSE compatibility
    chunk_size = 500
    for i in range(0, len(final_response), chunk_size):
        chunk = final_response[i:i + chunk_size]
        yield chunk

    logger.info(f"Response delivered: {len(final_response)} characters")


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
