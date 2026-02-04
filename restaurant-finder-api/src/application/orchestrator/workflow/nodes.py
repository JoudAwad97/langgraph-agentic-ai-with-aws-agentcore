import time
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from loguru import logger

from src.application.orchestrator.workflow.state import OrchestratorState
from src.application.orchestrator.workflow.chains import get_orchestrator_chain
from src.domain.models import ReflectionResult
from src.domain.prompts import REFLECTOR_PROMPT
from src.infrastructure.memory import ShortTermMemory
from src.infrastructure.model import get_model, ModelType
from src.infrastructure.observability import get_observability_manager


# Maximum number of reflection iterations to prevent infinite loops
MAX_REFLECTION_ITERATIONS = 2

# Shared memory instance for post-hook
_memory_instance: ShortTermMemory | None = None


def _extract_text_content(content) -> str:
    """
    Extract text content from a message content field.

    Bedrock models return content as a list of blocks like:
    [{'type': 'text', 'text': '...', 'index': 0}]

    This helper handles both string and list formats.

    Args:
        content: Either a string or a list of content blocks

    Returns:
        The extracted text as a string
    """
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif isinstance(block, str):
                text_parts.append(block)
        return "".join(text_parts)

    return str(content) if content else ""


def _get_memory_instance() -> ShortTermMemory:
    """Get or create the shared memory instance."""
    global _memory_instance
    if _memory_instance is None:
        _memory_instance = ShortTermMemory()
    return _memory_instance


async def orchestrator_node(
    state: OrchestratorState,
    config: RunnableConfig,
) -> dict:
    """
    Orchestrator node that coordinates sub-agents via tools.

    This node uses the orchestrator chain which has sub-agent tools bound,
    including the memory_retrieval_tool for on-demand memory access.

    The LLM decides which tools to call based on the user's request:
    - memory_retrieval_tool: For fetching user preferences/facts/summaries
    - restaurant_data_tool: For MCP Gateway searches
    - restaurant_explorer_tool: For web-based searches

    When reflection feedback is present, the orchestrator will attempt to
    improve its response based on the feedback.

    Args:
        state: The orchestrator state containing messages.
        config: Runtime configuration with customer context.

    Returns:
        Updated state with the orchestrator's response.
    """
    observability = get_observability_manager()
    start_time = time.time()

    configurable = config.get("configurable", {})
    customer_name = configurable.get("customer_name", "Guest")

    # Check if there's reflection feedback to incorporate
    reflection_feedback = state.get("reflection_feedback", "")
    reflection_count = state.get("reflection_count", 0)
    tool_call_count = state.get("tool_call_count", 0)

    messages = list(state["messages"])

    # If we have reflection feedback, add it as a system instruction
    is_refinement = reflection_feedback and reflection_count > 0
    if is_refinement:
        logger.info(f"=== ORCHESTRATOR REFINEMENT (iteration {reflection_count}) ===")
        logger.info(f"  Feedback to address: {reflection_feedback}")

        # Add a refinement instruction to guide improvement
        refinement_msg = HumanMessage(
            content=f"""[INTERNAL FEEDBACK - IMPROVE YOUR RESPONSE]

Your previous response needs improvement. Please address the following feedback:

{reflection_feedback}

Guidelines for improvement:
- If this is a restaurant search response, include at least 6 restaurants with full details
- If this is a conversational follow-up, respond appropriately to the user's question
- Keep responses focused and relevant to what the user actually asked
- Format clearly and maintain a friendly, helpful tone

Provide your improved response now."""
        )
        messages.append(refinement_msg)

    with observability.create_span(
        "orchestrator.invoke",
        attributes={
            "customer.name": customer_name,
            "reflection.count": reflection_count,
            "is_refinement": is_refinement,
            "message.count": len(messages),
        }
    ):
        orchestrator_chain = get_orchestrator_chain(customer_name=customer_name)

        response = await orchestrator_chain.ainvoke(
            {"messages": messages},
            config,
        )

    # Record workflow step completion
    duration_ms = (time.time() - start_time) * 1000
    observability.record_workflow_step(
        step_name="orchestrator",
        step_type="node",
        duration_ms=duration_ms,
        success=True,
        metadata={
            "reflection_count": reflection_count,
            "is_refinement": str(is_refinement),
        }
    )

    # Track tool calls for efficiency limiting
    has_tool_calls = hasattr(response, "tool_calls") and response.tool_calls
    new_tool_count = tool_call_count + (len(response.tool_calls) if has_tool_calls else 0)

    return {
        "messages": response,
        "tool_call_count": new_tool_count,
        "made_tool_calls": state.get("made_tool_calls", False) or has_tool_calls,
    }


async def memory_post_hook(
    state: OrchestratorState,
    config: RunnableConfig,
) -> dict:
    """
    Post-hook node: Save the conversation turn to memory after processing.

    This triggers all memory strategies configured in the CDK stack:
    - Extracts and stores user preferences
    - Extracts semantic facts from the conversation
    - Updates conversation summaries

    Args:
        state: The orchestrator state containing messages.
        config: Runtime configuration with actor/session identifiers.

    Returns:
        Empty dict (no state changes, just side effects).
    """
    observability = get_observability_manager()
    start_time = time.time()

    configurable = config.get("configurable", {})
    actor_id = configurable.get("actor_id", "user:default")
    session_id = configurable.get("thread_id", "default_session")

    messages = state.get("messages", [])

    # Find the latest user input and agent response
    user_input = ""
    agent_response = ""

    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not agent_response:
            # Skip tool calls, get the actual response
            if msg.content and not msg.tool_calls:
                # Extract text from content (handles both string and list formats)
                agent_response = _extract_text_content(msg.content)
        elif isinstance(msg, HumanMessage) and not user_input:
            user_input = _extract_text_content(msg.content)

        if user_input and agent_response:
            break

    if not user_input or not agent_response:
        logger.debug("Missing user input or agent response, skipping memory save")
        observability.add_span_event(
            "memory.skipped",
            attributes={"reason": "missing_input_or_response"}
        )
        return {}

    memory = _get_memory_instance()

    try:
        with observability.create_span(
            "memory.process_turn",
            attributes={
                "actor.id": actor_id,
                "session.id": session_id,
            }
        ):
            result = memory.process_turn(
                actor_id=actor_id,
                session_id=session_id,
                user_input=user_input,
                agent_response=agent_response,
            )

        if result.get("success"):
            logger.info(f"Saved conversation turn to memory for actor={actor_id}")
            duration_ms = (time.time() - start_time) * 1000
            observability.record_workflow_step(
                step_name="memory_post_hook",
                step_type="node",
                duration_ms=duration_ms,
                success=True,
                metadata={"actor_id": actor_id, "session_id": session_id}
            )
        else:
            logger.warning(f"Memory save returned error: {result.get('error')}")
            observability.add_span_event(
                "memory.error",
                attributes={"error": result.get("error", "unknown")}
            )

    except Exception as e:
        logger.error(f"Memory post-hook failed: {e}")
        observability.add_span_event(
            "memory.exception",
            attributes={"error.type": type(e).__name__, "error.message": str(e)}
        )

    return {}


async def reflector_node(
    state: OrchestratorState,
    config: RunnableConfig,
) -> dict:
    """
    Reflector node that evaluates the quality of the orchestrator's response.

    This node acts as a critic, evaluating whether the response meets quality
    criteria before finalizing. If the response is not satisfactory, it provides
    feedback for improvement.

    Evaluation criteria:
    - Completeness: At least 6 restaurants with full details
    - Relevance: Matches user's stated criteria (cuisine, location, price)
    - Quality: Well-formatted, logical order, friendly tone
    - Actionability: User can take action (addresses, contact info)

    Args:
        state: The orchestrator state containing messages.
        config: Runtime configuration.

    Returns:
        Updated state with reflection results:
        - is_satisfactory: Whether the response passes quality check
        - reflection_feedback: Feedback for improvement if not satisfactory
        - reflection_count: Incremented count of reflection iterations
    """
    observability = get_observability_manager()
    start_time = time.time()

    logger.info("=== REFLECTOR NODE INVOKED ===")

    messages = state.get("messages", [])
    reflection_count = state.get("reflection_count", 0)

    # Find the latest user input and agent response
    user_input = ""
    agent_response = ""

    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not agent_response:
            if msg.content and not getattr(msg, "tool_calls", None):
                agent_response = _extract_text_content(msg.content)
        elif isinstance(msg, HumanMessage) and not user_input:
            user_input = _extract_text_content(msg.content)

        if user_input and agent_response:
            break

    # If no response to evaluate, skip reflection
    if not agent_response:
        logger.info("  No agent response to evaluate, skipping reflection")
        observability.add_span_event(
            "reflector.skipped",
            attributes={"reason": "no_response_to_evaluate"}
        )
        return {
            "is_satisfactory": True,
            "reflection_feedback": "",
            "reflection_count": reflection_count,
        }

    # Check if we've reached max iterations
    if reflection_count >= MAX_REFLECTION_ITERATIONS:
        logger.info(f"  Max reflection iterations ({MAX_REFLECTION_ITERATIONS}) reached, accepting response")
        observability.add_span_event(
            "reflector.max_iterations",
            attributes={"max_iterations": MAX_REFLECTION_ITERATIONS}
        )
        return {
            "is_satisfactory": True,
            "reflection_feedback": "",
            "reflection_count": reflection_count,
        }

    logger.info(f"  Evaluating response (iteration {reflection_count + 1}/{MAX_REFLECTION_ITERATIONS})")
    logger.debug(f"  User input: {user_input[:200]}...")
    logger.debug(f"  Agent response: {agent_response[:200]}...")

    try:
        with observability.create_span(
            "reflector.evaluate",
            attributes={
                "reflection.iteration": reflection_count + 1,
                "response.length": len(agent_response),
            }
        ):
            # Create the reflector with structured output
            model = get_model(temperature=0.1, streaming=False, model_type=ModelType.REFLECTOR)
            structured_model = model.with_structured_output(ReflectionResult)

            evaluation_request = f"""
## User's Original Request
{user_input}

## Agent's Response to Evaluate
{agent_response}

Evaluate this response against the quality criteria.
"""

            eval_messages = [
                SystemMessage(content=REFLECTOR_PROMPT.prompt),
                HumanMessage(content=evaluation_request),
            ]

            # Get structured output directly - no JSON parsing needed
            evaluation: ReflectionResult = await structured_model.ainvoke(eval_messages)

        logger.info(f"  Reflector structured output received")

        is_satisfactory = evaluation.is_satisfactory
        score = evaluation.score
        issues = evaluation.issues
        feedback = evaluation.feedback

        logger.info(f"  Reflection result: satisfactory={is_satisfactory}, score={score}")
        if issues:
            logger.info(f"  Issues found: {issues}")
        if feedback:
            logger.info(f"  Feedback: {feedback}")

        logger.info("=== REFLECTOR NODE COMPLETE ===")

        # Record workflow step completion
        duration_ms = (time.time() - start_time) * 1000
        observability.record_workflow_step(
            step_name="reflector",
            step_type="node",
            duration_ms=duration_ms,
            success=True,
            metadata={
                "is_satisfactory": str(is_satisfactory),
                "score": str(score),
                "iteration": str(reflection_count + 1),
            }
        )

        return {
            "is_satisfactory": is_satisfactory,
            "reflection_feedback": feedback if not is_satisfactory else "",
            "reflection_count": reflection_count + 1,
        }

    except Exception as e:
        logger.error(f"  Reflector evaluation failed: {type(e).__name__}: {e}")
        logger.exception("  Full reflector traceback:")

        # Record error in observability
        duration_ms = (time.time() - start_time) * 1000
        observability.record_workflow_step(
            step_name="reflector",
            step_type="node",
            duration_ms=duration_ms,
            success=False,
            metadata={
                "error.type": type(e).__name__,
                "error.message": str(e),
            }
        )

        # On error, accept the response to avoid blocking
        return {
            "is_satisfactory": True,
            "reflection_feedback": "",
            "reflection_count": reflection_count + 1,
        }
