import json
import re
import uuid
import os

import aiohttp
import chainlit as cl
from chainlit.input_widget import TextInput


def strip_thinking_tags(text: str) -> str:
    """Remove <thinking>...</thinking> tags and their content from text."""
    return re.sub(r'<thinking>.*?</thinking>\s*', '', text, flags=re.DOTALL)

AGENTCORE_API_URL = os.environ.get("AGENTCORE_API_URL", "http://localhost:8080/invocations")


@cl.on_settings_update
async def settings_update(settings):
    """Handle settings updates."""
    cl.user_session.set("customer_name", settings.get("customer_name", "Guest"))

    await cl.Message(
        content=f"Settings updated! Welcome, {settings.get('customer_name', 'Guest')}! Ready to find your perfect restaurant."
    ).send()


@cl.on_chat_start
async def on_chat_start():
    """Initialize the chat session with settings."""
    # Set up chat settings UI
    settings = await cl.ChatSettings(
        [
            TextInput(
                id="customer_name",
                label="Your Name",
                placeholder="Enter your name",
                initial="Guest"
            ),
        ]
    ).send()

    # Store initial settings in session
    cl.user_session.set("customer_name", settings.get("customer_name", "Guest"))

    # Generate unique conversation ID for this session
    conversation_id = str(uuid.uuid4())
    cl.user_session.set("conversation_id", conversation_id)

    # Send welcome message with customer name
    customer_name = settings.get("customer_name", "Guest")
    await cl.Message(
        content=f"Welcome, {customer_name}! I'm your restaurant finder assistant. What kind of dining experience are you looking for today?\n\n*Tip: Click the settings icon to update your profile.*"
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming messages by calling the AgentCore API."""
    # Get customer context from session
    customer_name = cl.user_session.get("customer_name", "Guest")
    conversation_id = cl.user_session.get("conversation_id")

    # Create a message placeholder for streaming
    msg = cl.Message(content="")
    await msg.send()

    try:
        # Build the payload for the AgentCore API
        payload = {
            "prompt": message.content,
            "customer_name": customer_name,
            "conversation_id": conversation_id,
        }

        full_response = ""
        thinking_buffer = ""
        in_thinking = False
        line_buffer = ""

        # Stream response from AgentCore API
        timeout = aiohttp.ClientTimeout(total=120)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                AGENTCORE_API_URL,
                json=payload,
                headers={"Content-Type": "application/json"},
            ) as response:
                response.raise_for_status()

                # Stream chunks as they arrive
                async for chunk_bytes in response.content.iter_any():
                    line_buffer += chunk_bytes.decode("utf-8", errors="replace")

                    # Process complete lines
                    while "\n" in line_buffer:
                        line, line_buffer = line_buffer.split("\n", 1)
                        line = line.strip()

                        if not line or not line.startswith("data: "):
                            continue

                        json_str = line[6:]  # Remove outer "data: " prefix

                        try:
                            # First parse: get the inner SSE string
                            inner_data = json.loads(json_str)

                            # If it's a string starting with "data: ", parse again
                            if isinstance(inner_data, str) and inner_data.startswith("data: "):
                                inner_json = inner_data[6:].strip().rstrip()
                                data = json.loads(inner_json)
                            else:
                                data = inner_data

                            if isinstance(data, dict):
                                if "chunk" in data:
                                    chunk = data["chunk"]
                                    thinking_buffer += chunk

                                    # Track thinking tags
                                    if "<thinking" in thinking_buffer and not in_thinking:
                                        in_thinking = True
                                    if "</thinking>" in thinking_buffer and in_thinking:
                                        in_thinking = False
                                        thinking_buffer = re.sub(
                                            r'<thinking>.*?</thinking>\s*', '',
                                            thinking_buffer, flags=re.DOTALL
                                        )

                                    # Stream content when not in thinking block
                                    if not in_thinking and not thinking_buffer.startswith("<thinking"):
                                        if thinking_buffer:
                                            await msg.stream_token(thinking_buffer)
                                            full_response += thinking_buffer
                                            thinking_buffer = ""

                                elif "error" in data:
                                    msg.content = f"Error: {data['error']}"
                                    await msg.update()
                                    return

                        except json.JSONDecodeError:
                            continue

        # Handle remaining buffer
        if thinking_buffer:
            thinking_buffer = strip_thinking_tags(thinking_buffer)
            if thinking_buffer:
                await msg.stream_token(thinking_buffer)
                full_response += thinking_buffer

        # Final update
        final_content = strip_thinking_tags(full_response)
        msg.content = final_content if final_content else "No response received."
        await msg.update()

    except aiohttp.ClientResponseError as e:
        error_msg = f"API Error: {e.status}"
        msg.content = error_msg
        await msg.update()

    except aiohttp.ClientError:
        msg.content = "Connection Error: Could not connect to the API. Please try again later."
        await msg.update()

    except Exception:
        msg.content = "An unexpected error occurred. Please try again."
        await msg.update()


if __name__ == "__main__":
    from chainlit.cli import run_chainlit
    run_chainlit(__file__)
