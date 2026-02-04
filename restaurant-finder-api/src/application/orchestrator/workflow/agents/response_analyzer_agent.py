"""
Response Analyzer Agent - Deduplication and formatting of restaurant results.

This agent takes combined restaurant data from multiple sources (database + web),
deduplicates entries, and formats them into user-friendly responses.
"""

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from src.domain.prompts import RESPONSE_ANALYZER_PROMPT
from src.infrastructure.model import get_model, ModelType


def _extract_text_content(content: Any) -> str:
    """Extract text from LLM response content."""
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


async def run_response_analyzer(
    data: str,
    customer_name: str = "Guest",
) -> str:
    """
    Analyze, deduplicate, and format restaurant data for user presentation.

    Takes raw restaurant data (potentially from multiple sources), removes
    duplicates, and creates a user-friendly formatted response.

    Args:
        data: JSON string containing restaurant search results.
              Can be a single result or combined results from multiple tools.
        customer_name: Customer name for personalization.

    Returns:
        Formatted, user-friendly response ready for presentation.
    """
    logger.info(f"Running response analyzer for {customer_name}")

    try:
        # Get the prompt with customer name substituted
        system_prompt = RESPONSE_ANALYZER_PROMPT.format(
            customer_name=customer_name
        )

        model = get_model(temperature=0.7, streaming=False, model_type=ModelType.EXTRACTION)

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=f"Please analyze, deduplicate, and format the following restaurant data:\n\n{data}"
            ),
        ]

        response = await model.ainvoke(messages)
        result = _extract_text_content(response.content)

        logger.info(f"Response analyzer completed, output length: {len(result)}")
        return result

    except Exception as e:
        logger.error(f"Response analyzer failed: {e}")
        return f"I found some restaurant options but encountered an issue formatting them. Here's the raw data:\n\n{data[:2000]}"
