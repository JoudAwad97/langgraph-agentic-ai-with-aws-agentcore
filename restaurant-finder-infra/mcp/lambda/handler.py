import json
import random
from typing import Any, Dict, List


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
# Restaurant Search Tool
# =============================================================================

# Dummy data for generating restaurant responses
CUISINES = [
    "Italian", "Japanese", "Mexican", "Indian", "Thai", "Chinese",
    "French", "Mediterranean", "American", "Korean", "Vietnamese", "Greek"
]

RESTAURANT_NAMES = {
    "Italian": ["Bella Italia", "La Trattoria", "Pasta Paradise", "Roma Kitchen", "Olive Garden"],
    "Japanese": ["Sakura Sushi", "Tokyo Ramen", "Zen Garden", "Wasabi House", "Ninja Grill"],
    "Mexican": ["Casa del Sol", "El Mariachi", "Taco Fiesta", "Cantina Verde", "Aztec Kitchen"],
    "Indian": ["Taj Mahal", "Curry House", "Spice Route", "Bombay Bistro", "Masala Magic"],
    "Thai": ["Thai Orchid", "Bangkok Street", "Golden Temple", "Lotus Thai", "Siam Palace"],
    "Chinese": ["Golden Dragon", "Wok & Roll", "Jade Palace", "Lucky Fortune", "Silk Road"],
    "French": ["Le Petit Bistro", "Café Parisien", "Maison Rouge", "L'Escargot", "Brasserie Lyon"],
    "Mediterranean": ["Olive Branch", "Sea Breeze", "Aegean Grill", "Mezze House", "Blue Coast"],
    "American": ["The Grill House", "Liberty Diner", "Main Street Café", "Eagle's Nest", "Route 66"],
    "Korean": ["Seoul Kitchen", "K-BBQ House", "Kimchi Corner", "Han River", "Bibimbap Bowl"],
    "Vietnamese": ["Pho Paradise", "Saigon Street", "Lotus Leaf", "Mekong Kitchen", "Hanoi House"],
    "Greek": ["Santorini Taverna", "Olympus Grill", "Acropolis Kitchen", "Athena's Table", "Mykonos Blue"],
}

FEATURES = [
    "Outdoor Seating", "Private Dining", "Live Music", "Happy Hour",
    "Delivery", "Takeout", "Reservations", "Full Bar", "Kid-Friendly",
    "Pet-Friendly", "Wheelchair Accessible", "Free Wi-Fi", "Parking Available"
]

DIETARY_OPTIONS = [
    "Vegetarian", "Vegan", "Gluten-Free", "Halal", "Kosher",
    "Dairy-Free", "Nut-Free", "Low-Carb", "Organic"
]

CITIES = {
    "new york": ["Manhattan", "Brooklyn", "Queens", "Bronx"],
    "san francisco": ["Downtown", "Mission District", "North Beach", "SOMA"],
    "los angeles": ["Hollywood", "Santa Monica", "Downtown LA", "Beverly Hills"],
    "chicago": ["River North", "Wicker Park", "Lincoln Park", "Loop"],
    "seattle": ["Capitol Hill", "Ballard", "Fremont", "Downtown"],
    "default": ["Downtown", "Midtown", "Uptown", "Waterfront"]
}


def search_restaurants(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Search for restaurants based on provided criteria.

    Generates dummy restaurant data matching the search parameters.
    In production, this would query a real database or external API.

    Args:
        event: Search parameters
            - cuisine: Type of cuisine (e.g., "Italian", "Japanese")
            - location: City or area to search
            - price_range: Price level ("$", "$$", "$$$", "$$$$")
            - dietary_restrictions: List of dietary requirements
            - limit: Maximum number of results (default: 5)

    Returns:
        Dictionary containing:
            - restaurants: List of restaurant objects
            - total_found: Number of results
            - search_params: Echo of search parameters used
    """
    # Extract parameters with defaults
    cuisine = event.get("cuisine", "").strip()
    location = event.get("location", "").strip().lower()
    price_range = event.get("price_range", "$$")
    dietary_restrictions = event.get("dietary_restrictions", [])
    limit = min(int(event.get("limit", 5)), 10)  # Cap at 10

    # Normalize dietary restrictions
    if isinstance(dietary_restrictions, str):
        dietary_restrictions = [d.strip() for d in dietary_restrictions.split(",") if d.strip()]

    # Determine cuisine to use
    if cuisine and cuisine in RESTAURANT_NAMES:
        selected_cuisines = [cuisine]
    elif cuisine:
        # Fuzzy match cuisine
        matched = [c for c in CUISINES if cuisine.lower() in c.lower()]
        selected_cuisines = matched if matched else random.sample(CUISINES, min(3, len(CUISINES)))
    else:
        selected_cuisines = random.sample(CUISINES, min(3, len(CUISINES)))

    # Get neighborhoods for location
    neighborhoods = CITIES.get(location, CITIES["default"])

    # Generate restaurants
    restaurants = []
    for i in range(limit):
        cuisine_type = random.choice(selected_cuisines)
        names = RESTAURANT_NAMES.get(cuisine_type, RESTAURANT_NAMES["American"])

        # Generate realistic rating (weighted toward higher ratings)
        rating = round(random.uniform(3.5, 5.0), 1)
        review_count = random.randint(50, 2000)

        # Select random features and dietary options
        num_features = random.randint(3, 6)
        num_dietary = random.randint(2, 4)

        # Include requested dietary restrictions if any
        restaurant_dietary = list(set(
            dietary_restrictions[:2] +  # Include requested restrictions
            random.sample(DIETARY_OPTIONS, min(num_dietary, len(DIETARY_OPTIONS)))
        ))

        restaurant = {
            "name": random.choice(names) + (f" {random.choice(['East', 'West', 'North', 'Express', ''])}".strip() if random.random() > 0.5 else ""),
            "cuisine_type": cuisine_type,
            "rating": rating,
            "review_count": review_count,
            "price_range": price_range,
            "address": f"{random.randint(100, 9999)} {random.choice(['Main', 'Oak', 'Maple', 'Broadway', 'First', 'Second'])} {random.choice(['St', 'Ave', 'Blvd'])}",
            "city": location.title() if location else "New York",
            "neighborhood": random.choice(neighborhoods),
            "features": random.sample(FEATURES, min(num_features, len(FEATURES))),
            "dietary_options": restaurant_dietary[:4],
            "operating_hours": _generate_hours(),
            "reservation_available": random.choice([True, True, True, False]),  # 75% have reservations
            "phone": f"+1-{random.randint(200,999)}-{random.randint(200,999)}-{random.randint(1000,9999)}",
            "distance_miles": round(random.uniform(0.1, 5.0), 1),
        }
        restaurants.append(restaurant)

    # Sort by rating (highest first)
    restaurants.sort(key=lambda x: x["rating"], reverse=True)

    return {
        "restaurants": restaurants,
        "total_found": len(restaurants),
        "search_params": {
            "cuisine": cuisine or "any",
            "location": location or "default",
            "price_range": price_range,
            "dietary_restrictions": dietary_restrictions,
            "limit": limit,
        },
        "message": f"Found {len(restaurants)} restaurants matching your criteria.",
    }


def _generate_hours() -> str:
    """Generate realistic operating hours string."""
    patterns = [
        "Mon-Fri 11:00 AM - 10:00 PM, Sat-Sun 10:00 AM - 11:00 PM",
        "Daily 11:30 AM - 9:30 PM",
        "Mon-Thu 12:00 PM - 10:00 PM, Fri-Sat 12:00 PM - 11:00 PM, Sun 12:00 PM - 9:00 PM",
        "Tue-Sun 5:00 PM - 10:00 PM, Closed Mon",
        "Daily 10:00 AM - 10:00 PM",
        "Mon-Sat 11:00 AM - 11:00 PM, Sun 11:00 AM - 9:00 PM",
    ]
    return random.choice(patterns)


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
