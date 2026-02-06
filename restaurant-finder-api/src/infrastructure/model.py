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

    ORCHESTRATOR = "orchestrator"
    EXTRACTION = "extraction"
    ROUTER = "router"


def _get_model_id_for_type(model_type: ModelType) -> str:
    """Get the configured model ID for a given model type."""
    model_map = {
        ModelType.ORCHESTRATOR: settings.ORCHESTRATOR_MODEL_ID,
        ModelType.EXTRACTION: settings.EXTRACTION_MODEL_ID,
        ModelType.ROUTER: settings.ROUTER_MODEL_ID,
    }
    return model_map.get(model_type, settings.ORCHESTRATOR_MODEL_ID)


def get_model(
    temperature: float = 0.7,
    model_id: str | None = None,
    model_type: ModelType = ModelType.ORCHESTRATOR,
) -> ChatBedrockConverse:
    """
    Get a ChatBedrockConverse model instance.

    Args:
        temperature: Model temperature (0.0-1.0).
        model_id: Model ID override. Takes precedence over model_type.
        model_type: The type of model to use (orchestrator, extraction, router).

    Returns:
        ChatBedrockConverse: Configured model instance.
    """
    resolved_model_id = model_id or _get_model_id_for_type(model_type)

    return ChatBedrockConverse(
        model=resolved_model_id,
        temperature=temperature,
    )
