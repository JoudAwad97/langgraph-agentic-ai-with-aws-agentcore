from typing import NamedTuple

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from loguru import logger

from src.config import settings
from src.domain.prompts import ORCHESTRATOR_AGENT_PROMPT
from src.application.orchestrator.workflow.tools import get_orchestrator_tools
from src.infrastructure.model import get_model, ModelType


class PromptMetadata(NamedTuple):
    """Metadata about the prompt used for observability tracing."""
    name: str
    version: str | None
    id: str | None
    arn: str | None


class OrchestratorChainResult(NamedTuple):
    """Result containing the chain and its prompt metadata for tracing."""
    chain: Runnable
    prompt_metadata: PromptMetadata


def _escape_braces(text: str) -> str:
    """
    Escape curly braces to prevent ChatPromptTemplate from interpreting them as variables.

    Args:
        text: The text to escape

    Returns:
        Text with { replaced by {{ and } replaced by }}
    """
    return text.replace("{", "{{").replace("}", "}}")


def get_prompt_metadata() -> PromptMetadata:
    """
    Get metadata about the orchestrator prompt for observability tracing.

    Returns:
        PromptMetadata with name, version, id, and arn from Bedrock sync.
    """
    bedrock_meta = ORCHESTRATOR_AGENT_PROMPT.bedrock_metadata
    if bedrock_meta:
        return PromptMetadata(
            name=bedrock_meta.get("name", ORCHESTRATOR_AGENT_PROMPT.name),
            version=bedrock_meta.get("version"),
            id=bedrock_meta.get("id"),
            arn=bedrock_meta.get("arn"),
        )
    return PromptMetadata(
        name=ORCHESTRATOR_AGENT_PROMPT.name,
        version=None,
        id=None,
        arn=None,
    )


def get_orchestrator_chain(
    customer_name: str = "Guest",
    include_browser_tools: bool | None = None,
) -> OrchestratorChainResult:
    """
    Create the orchestrator chain with bound sub-agent tools.

    The orchestrator uses tools for:
    - memory_retrieval_tool: On-demand memory retrieval (preferences, facts, summaries)
    - restaurant_data_tool: MCP Gateway for structured restaurant data
    - restaurant_explorer_tool: For searching/exploring restaurants via browser (optional)
    - restaurant_research_tool: For detailed restaurant research (optional)

    Tool availability depends on the ENABLE_BROWSER_TOOLS config setting:
    - When False (default): Only MCP-based tools are available (faster)
    - When True: Browser-based tools are also available (more comprehensive)

    Memory is retrieved ON-DEMAND via the memory_retrieval_tool rather than
    being pre-loaded into the prompt. This gives the agent control over when
    and what type of memory to retrieve.

    Args:
        customer_name: The customer's name for personalization.
        include_browser_tools: Override for browser tools inclusion.
                              If None, uses ENABLE_BROWSER_TOOLS from config.

    Returns:
        OrchestratorChainResult with:
        - chain: A runnable chain (prompt | model) with tools bound
        - prompt_metadata: Metadata about the prompt for tracing
    """
    # Determine browser tools setting
    use_browser = include_browser_tools if include_browser_tools is not None else settings.ENABLE_BROWSER_TOOLS

    logger.info(f"Creating orchestrator chain (browser_tools={'enabled' if use_browser else 'disabled'})")

    model = get_model(temperature=0.5, model_type=ModelType.ORCHESTRATOR)

    # Get the appropriate tools based on config
    tools = get_orchestrator_tools(include_browser_tools=use_browser)
    tool_names = [t.name for t in tools]

    # Bind the tools to the model
    model = model.bind_tools(tools)

    # Escape braces in dynamic content to prevent ChatPromptTemplate from
    # interpreting them as template variables
    safe_customer_name = _escape_braces(customer_name)

    # Get the orchestrator prompt and its metadata
    system_message = ORCHESTRATOR_AGENT_PROMPT.format(
        customer_name=safe_customer_name,
    )
    prompt_metadata = get_prompt_metadata()

    logger.debug(
        f"Orchestrator prompt: name={prompt_metadata.name}, "
        f"version={prompt_metadata.version}, tools={tool_names}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    return OrchestratorChainResult(
        chain=prompt | model,
        prompt_metadata=prompt_metadata,
    )
