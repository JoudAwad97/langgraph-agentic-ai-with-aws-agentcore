import json
import os
import urllib.request
import urllib.parse
import urllib.error
from typing import Any, Dict, List

import boto3
from botocore.exceptions import ClientError

# SearchAPI configuration
SEARCHAPI_BASE_URL = "https://www.searchapi.io/api/v1/search"

# Cache for the secret to avoid repeated API calls
_cached_searchapi_key: str | None = None


def _get_searchapi_key() -> str:
    """
    Retrieve the SearchAPI key from AWS Secrets Manager.

    The secret is cached after the first retrieval to avoid repeated API calls
    during the Lambda's warm start period.

    Returns:
        The SearchAPI key string.

    Raises:
        RuntimeError: If the secret cannot be retrieved.
    """
    global _cached_searchapi_key

    if _cached_searchapi_key is not None:
        return _cached_searchapi_key

    secret_name = os.environ.get("SEARCHAPI_SECRET_NAME")

    if not secret_name:
        raise RuntimeError(
            "SEARCHAPI_SECRET_NAME environment variable not set. "
            "Please configure the secret in AWS Secrets Manager."
        )

    try:
        client = boto3.client("secretsmanager")
        response = client.get_secret_value(SecretId=secret_name)

        # Parse the secret - supports both plain string and JSON format
        secret_string = response["SecretString"]

        # Try to parse as JSON first (e.g., {"api_key": "xxx"})
        try:
            secret_data = json.loads(secret_string)
            if isinstance(secret_data, dict):
                # Look for common key names
                _cached_searchapi_key = (
                    secret_data.get("api_key")
                    or secret_data.get("SEARCHAPI_KEY")
                    or secret_data.get("key")
                    or secret_string  # Fallback to raw string
                )
            else:
                _cached_searchapi_key = secret_string
        except json.JSONDecodeError:
            # Plain text secret
            _cached_searchapi_key = secret_string

        return _cached_searchapi_key

    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "Unknown")
        raise RuntimeError(
            f"Failed to retrieve SearchAPI key from Secrets Manager: {error_code}. "
            f"Ensure the secret '{secret_name}' exists and Lambda has permission to read it."
        )


def lambda_handler(event, context):
    """
    Lambda handler for Bedrock AgentCore Gateway tools.

    Handles tool invocations from the AgentCore Gateway, routing to the
    appropriate tool function based on the tool name.

    Expected input:
        event: Tool-specific parameters
        context.client_context.custom["bedrockAgentCoreToolName"]: Tool identifier

    Tool naming convention from AgentCore Gateway:
        "LambdaTarget___<tool_name>"
    """
    try:
        extended_name = context.client_context.custom.get("bedrockAgentCoreToolName")
        tool_name = None

        # Handle AgentCore Gateway tool naming convention
        # https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway-tool-naming.html
        if extended_name and "___" in extended_name:
            tool_name = extended_name.split("___", 1)[1]

        if not tool_name:
            return _response(400, {"error": "Missing tool name"})

        # Route to appropriate tool
        if tool_name == "search_restaurants":
            result = search_restaurants(event)
            return _response(200, {"result": result})
        elif tool_name == "placeholder_tool":
            result = placeholder_tool(event)
            return _response(200, {"result": result})
        else:
            return _response(400, {"error": f"Unknown tool '{tool_name}'"})

    except Exception as e:
        return _response(500, {"system_error": str(e)})


def _response(status_code: int, body: Dict[str, Any]):
    """Consistent JSON response wrapper."""
    return {"statusCode": status_code, "body": json.dumps(body)}


# =============================================================================
# SearchAPI Integration
# =============================================================================

def _call_searchapi_local(query: str, location: str = "", num_results: int = 10) -> Dict[str, Any]:
    """
    Call SearchAPI with google_local engine for structured local business data.

    This returns rich restaurant data with ratings, addresses, hours, etc.

    Args:
        query: Search query string (e.g., "Italian restaurants").
        location: Location for geo-targeting (e.g., "New York, NY").
        num_results: Number of results to request.

    Returns:
        SearchAPI response with local_results containing business data.
    """
    api_key = _get_searchapi_key()

    params = {
        "api_key": api_key,
        "engine": "google_local",
        "q": query,
        "num": str(num_results),
    }

    # Add location if provided for better geo-targeting
    if location and location.strip():
        params["location"] = location.strip()

    url = f"{SEARCHAPI_BASE_URL}?{urllib.parse.urlencode(params)}"

    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8") if e.fp else ""
        raise RuntimeError(f"SearchAPI HTTP error {e.code}: {error_body}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"SearchAPI connection error: {e.reason}")


def _call_searchapi_web(query: str, num_results: int = 10) -> Dict[str, Any]:
    """
    Call SearchAPI with google engine for general web search (fallback).

    Args:
        query: Search query string.
        num_results: Number of results to request.

    Returns:
        SearchAPI response with organic_results.
    """
    api_key = _get_searchapi_key()

    params = {
        "api_key": api_key,
        "engine": "google",
        "q": query,
        "num": str(num_results),
    }

    url = f"{SEARCHAPI_BASE_URL}?{urllib.parse.urlencode(params)}"

    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8") if e.fp else ""
        raise RuntimeError(f"SearchAPI HTTP error {e.code}: {error_body}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"SearchAPI connection error: {e.reason}")


def _build_search_query(
    query: str = "",
    cuisine: str = "",
    location: str = "",
    price_range: str = "",
    dietary_restrictions: List[str] = None,
) -> str:
    """
    Build a search query string for restaurant searches.

    Args:
        query: Free-form search query.
        cuisine: Type of cuisine.
        location: City or area.
        price_range: Price level indicator.
        dietary_restrictions: List of dietary requirements.

    Returns:
        Formatted search query string.
    """
    parts = []

    # Add the main query if provided
    if query and query.strip():
        parts.append(query.strip())

    # Add cuisine
    if cuisine and cuisine.strip():
        parts.append(f"{cuisine.strip()} restaurants")
    elif not query:
        parts.append("restaurants")

    # Add location
    if location and location.strip():
        parts.append(f"in {location.strip()}")

    # Add price range context
    price_descriptions = {
        "$": "budget-friendly cheap",
        "$$": "moderate mid-range",
        "$$$": "upscale high-end",
        "$$$$": "fine dining luxury",
    }
    if price_range and price_range in price_descriptions:
        parts.append(price_descriptions[price_range])

    # Add dietary restrictions
    if dietary_restrictions:
        if isinstance(dietary_restrictions, str):
            dietary_restrictions = [d.strip() for d in dietary_restrictions.split(",") if d.strip()]
        if dietary_restrictions:
            parts.append(" ".join(dietary_restrictions))

    return " ".join(parts)


def _parse_local_results(
    api_response: Dict[str, Any],
    location: str,
    cuisine: str,
    price_range: str,
    limit: int,
) -> List[Dict[str, Any]]:
    """
    Parse SearchAPI google_local response into restaurant objects.

    The google_local engine returns structured local business data with:
    - title, rating, reviews, address, phone, hours, type, price, etc.

    Args:
        api_response: Response from SearchAPI google_local engine.
        location: Search location for context.
        cuisine: Cuisine type for context.
        price_range: Price range for context.
        limit: Maximum results to return.

    Returns:
        List of parsed restaurant dictionaries.
    """
    restaurants = []

    # google_local returns results in "local_results" array
    local_results = api_response.get("local_results", [])

    for idx, result in enumerate(local_results[:limit]):
        name = result.get("title") or result.get("name", f"Restaurant {idx + 1}")

        # Extract rating (google_local provides numeric ratings)
        rating = result.get("rating", 0.0)
        if isinstance(rating, str):
            try:
                rating = float(rating)
            except ValueError:
                rating = 0.0

        # Extract review count
        reviews = result.get("reviews", 0)
        if isinstance(reviews, str):
            reviews = int("".join(filter(str.isdigit, reviews)) or "0")

        # Extract type/cuisine from the result
        type_info = result.get("type", result.get("types", ""))
        if isinstance(type_info, list):
            cuisine_type = ", ".join(type_info[:3]) if type_info else cuisine or "Restaurant"
        else:
            cuisine_type = type_info if type_info else cuisine or "Restaurant"

        # Extract address components
        address = result.get("address", "")

        # Extract service options (dine-in, takeout, delivery)
        service_options = result.get("service_options", {})
        features = []
        if isinstance(service_options, dict):
            if service_options.get("dine_in"):
                features.append("Dine-in")
            if service_options.get("takeout"):
                features.append("Takeout")
            if service_options.get("delivery"):
                features.append("Delivery")
        elif isinstance(service_options, list):
            features = service_options

        # Extract hours
        hours = result.get("hours", result.get("operating_hours", ""))
        if isinstance(hours, dict):
            hours = hours.get("today", "")

        restaurant = {
            "name": name,
            "cuisine_type": cuisine_type,
            "rating": round(float(rating), 1) if rating else 0.0,
            "review_count": int(reviews) if reviews else 0,
            "price_range": result.get("price", price_range or "$$"),
            "address": address[:200] if address else "",
            "city": location.title() if location else "",
            "neighborhood": result.get("neighborhood", ""),
            "features": features,
            "dietary_options": [],
            "operating_hours": hours if isinstance(hours, str) else "",
            "reservation_available": "reservations" in str(service_options).lower(),
            "phone": result.get("phone", ""),
            "website": result.get("website", result.get("link", "")),
            "thumbnail": result.get("thumbnail", ""),
            "gps_coordinates": result.get("gps_coordinates", {}),
            "place_id": result.get("place_id", ""),
            "source": "google_local",
        }
        restaurants.append(restaurant)

    return restaurants


def _parse_web_results(
    api_response: Dict[str, Any],
    location: str,
    cuisine: str,
    price_range: str,
    limit: int,
) -> List[Dict[str, Any]]:
    """
    Parse SearchAPI google (web) response into restaurant objects (fallback).

    Web results have less structured data, so we extract what we can.

    Args:
        api_response: Response from SearchAPI google engine.
        location: Search location for context.
        cuisine: Cuisine type for context.
        price_range: Price range for context.
        limit: Maximum results to return.

    Returns:
        List of parsed restaurant dictionaries.
    """
    restaurants = []

    # Web search returns organic_results
    organic_results = api_response.get("organic_results", [])

    for idx, result in enumerate(organic_results[:limit]):
        title = result.get("title", f"Restaurant {idx + 1}")
        snippet = result.get("snippet", "")

        # Try to extract rating from snippet if present
        rating = 0.0
        reviews = 0

        restaurant = {
            "name": title,
            "cuisine_type": cuisine or "Restaurant",
            "rating": rating,
            "review_count": reviews,
            "price_range": price_range or "$$",
            "address": snippet[:200] if snippet else "",
            "city": location.title() if location else "",
            "neighborhood": "",
            "features": [],
            "dietary_options": [],
            "operating_hours": "",
            "reservation_available": False,
            "phone": "",
            "website": result.get("link", ""),
            "source": "web_search",
        }
        restaurants.append(restaurant)

    return restaurants


# =============================================================================
# Restaurant Search Tool
# =============================================================================

def search_restaurants(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Search for restaurants using SearchAPI with google_local engine.

    Uses google_local engine for structured local business data (ratings,
    addresses, hours, etc.). Falls back to web search if local returns no results.

    Args:
        event: Search parameters
            - query: Free-form search query
            - cuisine: Type of cuisine (e.g., "Italian", "Japanese")
            - location: City or area to search
            - price_range: Price level ("$", "$$", "$$$", "$$$$")
            - dietary_restrictions: List of dietary requirements
            - limit: Maximum number of results (default: 5)

    Returns:
        Dictionary containing:
            - restaurants: List of restaurant objects with rich metadata
            - total_found: Number of results
            - search_params: Echo of search parameters used
            - data_source: "google_local" or "web_search"
    """
    # Extract parameters with defaults
    query = event.get("query", "").strip()
    cuisine = event.get("cuisine", "").strip()
    location = event.get("location", "").strip()
    price_range = event.get("price_range", "$$")
    dietary_restrictions = event.get("dietary_restrictions", [])
    limit = min(int(event.get("limit", 5)), 10)  # Cap at 10

    # Normalize dietary restrictions
    if isinstance(dietary_restrictions, str):
        dietary_restrictions = [d.strip() for d in dietary_restrictions.split(",") if d.strip()]

    # Build the search query for local search
    search_query = _build_search_query(
        query=query,
        cuisine=cuisine,
        location=location,
        price_range=price_range,
        dietary_restrictions=dietary_restrictions,
    )

    restaurants = []
    data_source = "google_local"
    error_message = None

    try:
        # Try google_local first for structured restaurant data
        api_response = _call_searchapi_local(
            query=search_query,
            location=location,
            num_results=limit * 2,
        )

        restaurants = _parse_local_results(
            api_response=api_response,
            location=location,
            cuisine=cuisine,
            price_range=price_range,
            limit=limit,
        )

        # Fallback to web search if local returns no results
        if not restaurants:
            data_source = "web_search"
            api_response = _call_searchapi_web(search_query, num_results=limit * 2)

            restaurants = _parse_web_results(
                api_response=api_response,
                location=location,
                cuisine=cuisine,
                price_range=price_range,
                limit=limit,
            )

    except Exception as e:
        error_message = str(e)
        # Try web search as fallback on local search error
        try:
            data_source = "web_search"
            api_response = _call_searchapi_web(search_query, num_results=limit * 2)

            restaurants = _parse_web_results(
                api_response=api_response,
                location=location,
                cuisine=cuisine,
                price_range=price_range,
                limit=limit,
            )
        except Exception as fallback_error:
            error_message = f"Local: {error_message}, Web: {str(fallback_error)}"

    # Sort by rating (higher first)
    restaurants.sort(key=lambda x: x.get("rating", 0), reverse=True)

    result = {
        "restaurants": restaurants,
        "total_found": len(restaurants),
        "search_params": {
            "query": query,
            "cuisine": cuisine or "any",
            "location": location or "any",
            "price_range": price_range,
            "dietary_restrictions": dietary_restrictions,
            "limit": limit,
        },
        "search_query_used": search_query,
        "data_source": data_source,
        "message": f"Found {len(restaurants)} restaurants via {data_source}.",
    }

    if error_message and not restaurants:
        result["error"] = error_message
        result["message"] = f"Search failed: {error_message}"

    return result


# =============================================================================
# Placeholder Tool (for testing)
# =============================================================================

def placeholder_tool(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    No-op placeholder tool for testing gateway connectivity.

    Demonstrates argument passing from AgentCore Gateway.
    """
    return {
        "message": "Placeholder tool executed.",
        "string_param": event.get("string_param"),
        "int_param": event.get("int_param"),
        "float_array_param": event.get("float_array_param"),
        "event_args_received": event,
    }
