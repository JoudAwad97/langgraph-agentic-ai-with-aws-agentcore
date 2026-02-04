"""
Model infrastructure for Bedrock LLM access.

Provides centralized model loading with configurable parameters.
Guardrails are applied at the graph level (input/output boundaries),
not at individual model calls.
"""

from enum import Enum

from langchain_aws import ChatBedrockConverse

from src.config import settings


class ModelType(str, Enum):
    """Model types for different components of the system."""

    DEFAULT = "default"  # General purpose, backwards compatible
    ORCHESTRATOR = "orchestrator"  # Main agent, tool selection, conversations
    EXTRACTION = "extraction"  # Data extraction, JSON parsing


def _get_model_id_for_type(model_type: ModelType) -> str:
    """Get the configured model ID for a given model type."""
    model_map = {
        ModelType.DEFAULT: settings.CONVERSATION_CHAT_MODEL_ID,
        ModelType.ORCHESTRATOR: settings.ORCHESTRATOR_MODEL_ID,
        ModelType.EXTRACTION: settings.EXTRACTION_MODEL_ID,
    }
    return model_map.get(model_type, settings.CONVERSATION_CHAT_MODEL_ID)


def get_model(
    temperature: float = 0.7,
    streaming: bool = True,
    model_id: str | None = None,
    model_type: ModelType = ModelType.DEFAULT,
) -> ChatBedrockConverse:
    """
    Get a ChatBedrockConverse model instance.

    Note: Guardrails are applied at the graph level (input/output boundaries),
    not at individual model calls. This reduces cost and latency.

    Args:
        temperature: Model temperature (0.0-1.0).
            - 0.1: For extraction/factual tasks
            - 0.5: For analysis/reasoning tasks
            - 0.7: For conversational tasks
        streaming: Enable streaming responses.
        model_id: Model ID override. Takes precedence over model_type.
        model_type: The type of model to use (orchestrator, extraction).
            Defaults to DEFAULT for backwards compatibility.

    Returns:
        ChatBedrockConverse: Configured model instance.
    """
    # Explicit model_id takes precedence, then model_type, then default
    resolved_model_id = model_id or _get_model_id_for_type(model_type)

    return ChatBedrockConverse(
        model=resolved_model_id,
        temperature=temperature,
    )
