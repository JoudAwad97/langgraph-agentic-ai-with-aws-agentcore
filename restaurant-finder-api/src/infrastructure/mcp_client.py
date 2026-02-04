"""
MCP Client for AgentCore Gateway connection.

Provides MCP client for communicating with AgentCore Gateway,
which routes tool calls to Lambda functions.

⚠️ WARNING: This client is configured WITHOUT authentication.
This is NOT RECOMMENDED for production environments.
We are using this setup for demo/testing purposes only.
For production, implement one of the following:
  - Bearer token authentication with Cognito/OIDC
  - IAM-based authentication
See: https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway-authorization.html
"""

from langchain_mcp_adapters.client import MultiServerMCPClient
from loguru import logger

from src.config import settings


# =============================================================================
# MCP Client Factory
# =============================================================================

def get_mcp_client() -> MultiServerMCPClient:
    """
    Create and return an MCP Client for AgentCore Gateway.

    The client is configured with:
    - HTTP transport to the Gateway URL
    - No authentication (gateway configured with authorizerType: NONE)
    - Compatible with LangGraph async context manager pattern

    Returns:
        MultiServerMCPClient configured for AgentCore Gateway.

    Raises:
        RuntimeError: If GATEWAY_URL is not set.

    Usage:
        async with get_mcp_client() as client:
            tools = await client.get_tools()
            result = await client.call_tool("tool_name", {"arg": "value"})
    """
    gateway_url = settings.GATEWAY_URL
    if not gateway_url:
        raise RuntimeError("Missing required configuration: GATEWAY_URL")

    logger.debug(f"Creating MCP client for gateway: {gateway_url}")

    # NOTE: No authentication headers since gateway has authorizerType: NONE
    # TODO: Add authentication before deploying to production
    return MultiServerMCPClient(
        {
            "agentcore_gateway": {
                "transport": "streamable_http",
                "url": gateway_url,
            }
        }
    )


def is_mcp_configured() -> bool:
    """
    Check if MCP Gateway is configured.

    Returns:
        True if GATEWAY_URL is set.
    """
    return bool(settings.GATEWAY_URL)
