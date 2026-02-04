import json
import uuid
from typing import Annotated, Literal

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool, InjectedToolArg
from loguru import logger

from src.application.orchestrator.workflow.agents.restaurant_explorer_agent import (
    run_restaurant_explorer,
)
from src.application.orchestrator.workflow.agents.restaurant_data_agent import (
    run_restaurant_data_agent,
)
from src.application.orchestrator.workflow.agents.restaurant_research_agent import (
    run_restaurant_research,
)
from src.config import settings
from src.domain.models import RestaurantSearchResult
from src.infrastructure.memory import ShortTermMemory

# Shared memory instance for the memory tool
_memory_instance: ShortTermMemory | None = None


def _get_memory_instance() -> ShortTermMemory:
    """Get or create the shared memory instance."""
    global _memory_instance
    if _memory_instance is None:
        _memory_instance = ShortTermMemory()
    return _memory_instance


@tool
async def restaurant_explorer_tool(
    query: str,
    config: Annotated[RunnableConfig, InjectedToolArg],
) -> str:
    """
    Use the Restaurant Explorer agent to search for restaurants and gather information.

    This tool searches for restaurants using web browsing and external APIs.
    Use it to find restaurants based on criteria like cuisine type, location,
    price range, and dietary restrictions.

    Args:
        query: The search request describing what kind of restaurants to find.
               Include details like cuisine, location, price range, and any
               special requirements.

    Returns:
        JSON structured restaurant data with the following schema:
        - query: The original search query
        - total_results: Number of restaurants found
        - restaurants: List of restaurant objects containing:
          - name: Restaurant name
          - cuisine_type: Type of cuisine
          - rating: Rating out of 5
          - review_count: Number of reviews
          - price_range: $, $$, $$$, or $$$$
          - address: Street address
          - city: City location
          - features: List of features (outdoor seating, etc.)
          - dietary_options: List of dietary accommodations
          - operating_hours: Business hours
          - reservation_available: Whether reservations are available
        - search_location: Location used for search
        - data_source: Source of the data

    Example queries:
        - "Find Italian restaurants near downtown under $30"
        - "Search for vegetarian-friendly Japanese restaurants in San Francisco"
        - "Get details for highly-rated Thai restaurants with outdoor seating"
    """
    # Extract thread_id from config for browser session isolation
    # Generate a unique UUID as fallback to avoid session conflicts
    configurable = config.get("configurable", {}) if config else {}
    thread_id = configurable.get("thread_id") or str(uuid.uuid4())

    result: RestaurantSearchResult = await run_restaurant_explorer(
        query=query,
        thread_id=thread_id,
    )
    return result.model_dump_json(indent=2)


@tool
async def restaurant_data_tool(
    query: str,
    cuisine: str = "",
    location: str = "",
    price_range: str = "$$",
    dietary_restrictions: list[str] = None,
    limit: int = 5,
) -> str:
    """
    Use the Restaurant Data agent to search for restaurants via MCP Gateway.

    This tool connects to an AgentCore Gateway using MCP protocol to fetch
    restaurant data from a backend Lambda function. Use this for structured
    restaurant searches with specific filters.

    Args:
        query: Natural language description of what restaurants to find.
        cuisine: Type of cuisine (e.g., "Italian", "Japanese", "Mexican").
        location: City or area to search (e.g., "New York", "San Francisco").
        price_range: Price level - "$" (budget), "$$" (moderate),
                    "$$$" (upscale), "$$$$" (fine dining).
        dietary_restrictions: List of dietary requirements
                             (e.g., ["Vegetarian", "Gluten-Free"]).
        limit: Maximum number of results to return (1-10).

    Returns:
        JSON structured restaurant data with detailed information including
        ratings, features, dietary options, and operating hours.

    Example usage:
        - restaurant_data_tool("Find Italian food", cuisine="Italian", location="NYC")
        - restaurant_data_tool("Vegan options", dietary_restrictions=["Vegan"], price_range="$$")
    """
    result: RestaurantSearchResult = await run_restaurant_data_agent(
        query=query,
        cuisine=cuisine,
        location=location,
        price_range=price_range,
        dietary_restrictions=dietary_restrictions or [],
        limit=limit,
    )
    return result.model_dump_json(indent=2)


@tool
async def memory_retrieval_tool(
    query: str,
    memory_types: list[Literal["preferences", "facts", "summaries"]],
    config: Annotated[RunnableConfig, InjectedToolArg],
) -> str:
    """
    Retrieve relevant memories from the user's long-term memory.

    Use this tool to fetch personalized context about the user before making
    recommendations. Call this tool with the appropriate memory types based
    on what information you need.

    IMPORTANT: Memory types are retrieved in PARALLEL for speed optimization.
    You can request multiple types in a single call.

    Args:
        query: The search query to find relevant memories. Use the user's
               current request or intent as the query for semantic matching.
        memory_types: List of memory types to retrieve. Options:
            - "preferences": User preferences (dietary restrictions, favorite
                            cuisines, price preferences, location preferences)
            - "facts": Semantic facts from previous conversations (important
                      details the user has shared before)
            - "summaries": Conversation summaries from the current session

    Returns:
        JSON object with retrieved memories organized by type:
        {
            "preferences": [...],  // User preference items
            "facts": [...],        // Conversation facts
            "summaries": [...]     // Session summaries
        }

    When to use each memory type:
        - "preferences": When you need to know dietary restrictions, cuisine
                        preferences, budget, or location preferences
        - "facts": When you want to recall specific details from past conversations
        - "summaries": When you need context about what was discussed earlier
                      in the current session

    Example usage:
        - memory_retrieval_tool(query="Italian food", memory_types=["preferences"])
        - memory_retrieval_tool(query="restaurant", memory_types=["preferences", "facts"])
        - memory_retrieval_tool(query="recommendations", memory_types=["preferences", "facts", "summaries"])
    """
    configurable = config.get("configurable", {}) if config else {}
    actor_id = configurable.get("actor_id", "user:default")
    session_id = configurable.get("thread_id", "default_session")

    logger.info(f"=== MEMORY RETRIEVAL TOOL INVOKED ===")
    logger.info(f"  Query: '{query}'")
    logger.info(f"  Memory types requested: {memory_types}")
    logger.info(f"  Actor ID: {actor_id}")
    logger.info(f"  Session ID: {session_id}")

    try:
        memory = _get_memory_instance()
        retrieved = memory.retrieve_specific_memories(
            query=query,
            actor_id=actor_id,
            session_id=session_id,
            memory_types=memory_types,
            top_k=5,
        )

        # Format results for the agent
        formatted_results = {}
        for mem_type, items in retrieved.items():
            formatted_results[mem_type] = [
                item.get("content", str(item)) for item in items
            ]
            logger.info(f"  Retrieved {len(items)} items for '{mem_type}'")

        result_json = json.dumps(formatted_results, indent=2)

        logger.info(f"=== MEMORY RETRIEVAL COMPLETE ===")
        logger.debug(f"Memory retrieval results: {result_json}")
        return result_json

    except Exception as e:
        logger.error(f"=== MEMORY RETRIEVAL FAILED ===")
        logger.error(f"  Error: {e}")
        return json.dumps({"error": str(e), "preferences": [], "facts": [], "summaries": []})


@tool
async def restaurant_research_tool(
    restaurant_name: str,
    location: str,
    research_topics: list[str] = None,
    config: Annotated[RunnableConfig, InjectedToolArg] = None,
) -> str:
    """
    Research detailed information about a specific restaurant using web browsing.

    Use this tool when the user asks for MORE DETAILS about a restaurant that was
    already mentioned, or when you need to look up specific information like:
    - Full menu details
    - Customer reviews and ratings from multiple sources
    - Reservation policies
    - Parking information
    - Special events or happy hours
    - Photos or ambiance descriptions
    - Contact information and directions

    This is different from restaurant_explorer_tool which searches for MULTIPLE
    restaurants. Use this tool for deep research on ONE specific restaurant.

    Args:
        restaurant_name: The name of the restaurant to research.
        location: The city or area where the restaurant is located.
        research_topics: Optional list of specific topics to research. Options:
            - "menu": Full menu with prices
            - "reviews": Customer reviews and ratings
            - "reservations": Booking policies and availability
            - "parking": Parking options nearby
            - "events": Special events, happy hours, live music
            - "contact": Phone, email, website
            - "directions": Address and how to get there
            If not specified, will do a general research covering key topics.

    Returns:
        JSON object with detailed research findings:
        {
            "restaurant_name": "...",
            "location": "...",
            "research_summary": "...",
            "details": {
                "menu_highlights": [...],
                "reviews_summary": "...",
                "contact_info": {...},
                "special_features": [...],
                ...
            },
            "sources": [...]
        }

    Example usage:
        - restaurant_research_tool("The Grill House", "Hamburg", ["menu", "reviews"])
        - restaurant_research_tool("Bella Italia", "San Francisco", ["reservations", "parking"])
        - restaurant_research_tool("Tokyo Garden", "NYC")  # General research
    """
    # Extract thread_id from config for browser session isolation
    configurable = config.get("configurable", {}) if config else {}
    thread_id = configurable.get("thread_id") or str(uuid.uuid4())

    logger.info(f"=== RESTAURANT RESEARCH TOOL INVOKED ===")
    logger.info(f"  Restaurant: '{restaurant_name}'")
    logger.info(f"  Location: '{location}'")
    logger.info(f"  Topics: {research_topics}")

    try:
        result = await run_restaurant_research(
            restaurant_name=restaurant_name,
            location=location,
            research_topics=research_topics,
            thread_id=thread_id,
        )

        logger.info(f"=== RESTAURANT RESEARCH COMPLETE ===")
        return json.dumps(result, indent=2)

    except Exception as e:
        logger.error(f"=== RESTAURANT RESEARCH FAILED ===")
        logger.error(f"  Error: {e}")
        return json.dumps({
            "restaurant_name": restaurant_name,
            "location": location,
            "error": str(e),
            "research_summary": f"Unable to research {restaurant_name}. Please try again.",
        })


# Core tools (always available)
_CORE_TOOLS = [
    restaurant_data_tool,       # MCP Gateway to Lambda (structured data)
    memory_retrieval_tool,      # On-demand memory retrieval
]

# Browser-based tools (optional)
_BROWSER_TOOLS = [
    restaurant_explorer_tool,   # Browser-based web search for finding restaurants
    restaurant_research_tool,   # Browser-based detailed research on specific restaurant
]


def get_orchestrator_tools(include_browser_tools: bool | None = None) -> list:
    """
    Get the list of tools available to the orchestrator.

    Args:
        include_browser_tools: Override for browser tools inclusion.
                              If None, uses ENABLE_BROWSER_TOOLS from config.

    Returns:
        List of tools for the orchestrator to use.
    """
    use_browser = include_browser_tools if include_browser_tools is not None else settings.ENABLE_BROWSER_TOOLS

    tools = list(_CORE_TOOLS)

    if use_browser:
        tools.extend(_BROWSER_TOOLS)
        logger.info("Browser tools enabled")
    else:
        logger.info("Browser tools disabled")

    return tools


# Legacy export for backwards compatibility
ORCHESTRATOR_TOOLS = get_orchestrator_tools()
