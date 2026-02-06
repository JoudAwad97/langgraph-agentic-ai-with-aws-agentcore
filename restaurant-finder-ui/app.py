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


# --- Connection mode configuration ---
# Set AGENT_CONNECTION_MODE to "aws" to invoke the deployed AgentCore Runtime on AWS.
# Set to "local" (or leave unset) to call the local API server.
AGENT_CONNECTION_MODE = os.environ.get("AGENT_CONNECTION_MODE", "local").lower()

# Local mode settings
AGENTCORE_API_URL = os.environ.get("AGENTCORE_API_URL", "http://localhost:8080/invocations")

# AWS mode settings
AGENT_RUNTIME_ARN = os.environ.get("AGENT_RUNTIME_ARN", "")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-2")

# Lazily initialized boto3 client for AWS mode
_agentcore_client = None


def _get_agentcore_client():
    """Get or create the boto3 bedrock-agentcore client (lazy init)."""
    global _agentcore_client
    if _agentcore_client is None:
        import boto3
        _agentcore_client = boto3.client("bedrock-agentcore", region_name=AWS_REGION)
    return _agentcore_client


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
    """Handle incoming messages by calling the AgentCore API (local or AWS)."""
    # Get customer context from session
    customer_name = cl.user_session.get("customer_name", "Guest")
    conversation_id = cl.user_session.get("conversation_id")

    # Create a message placeholder for streaming
    msg = cl.Message(content="")
    await msg.send()

    if AGENT_CONNECTION_MODE == "aws":
        await _invoke_aws_runtime(msg, message.content, customer_name, conversation_id)
    else:
        await _invoke_local_api(msg, message.content, customer_name, conversation_id)


async def _invoke_local_api(
    msg: cl.Message,
    user_input: str,
    customer_name: str,
    conversation_id: str,
):
    """Invoke the agent via the local HTTP API (localhost)."""
    try:
        payload = {
            "prompt": user_input,
            "customer_name": customer_name,
            "conversation_id": conversation_id,
        }

        full_response = ""
        thinking_buffer = ""
        in_thinking = False
        line_buffer = ""

        timeout = aiohttp.ClientTimeout(total=120)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                AGENTCORE_API_URL,
                json=payload,
                headers={"Content-Type": "application/json"},
            ) as response:
                response.raise_for_status()

                async for chunk_bytes in response.content.iter_any():
                    line_buffer += chunk_bytes.decode("utf-8", errors="replace")

                    while "\n" in line_buffer:
                        line, line_buffer = line_buffer.split("\n", 1)
                        line = line.strip()

                        if not line or not line.startswith("data: "):
                            continue

                        json_str = line[6:]

                        try:
                            inner_data = json.loads(json_str)

                            if isinstance(inner_data, str) and inner_data.startswith("data: "):
                                inner_json = inner_data[6:].strip().rstrip()
                                data = json.loads(inner_json)
                            else:
                                data = inner_data

                            if isinstance(data, dict):
                                if "chunk" in data:
                                    chunk = data["chunk"]
                                    thinking_buffer += chunk

                                    if "<thinking" in thinking_buffer and not in_thinking:
                                        in_thinking = True
                                    if "</thinking>" in thinking_buffer and in_thinking:
                                        in_thinking = False
                                        thinking_buffer = re.sub(
                                            r'<thinking>.*?</thinking>\s*', '',
                                            thinking_buffer, flags=re.DOTALL
                                        )

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

        if thinking_buffer:
            thinking_buffer = strip_thinking_tags(thinking_buffer)
            if thinking_buffer:
                await msg.stream_token(thinking_buffer)
                full_response += thinking_buffer

        final_content = strip_thinking_tags(full_response)
        msg.content = final_content if final_content else "No response received."
        await msg.update()

    except aiohttp.ClientResponseError as e:
        msg.content = f"API Error: {e.status}"
        await msg.update()

    except aiohttp.ClientError:
        msg.content = "Connection Error: Could not connect to the local API. Please ensure the API server is running."
        await msg.update()

    except Exception:
        msg.content = "An unexpected error occurred. Please try again."
        await msg.update()


async def _invoke_aws_runtime(
    msg: cl.Message,
    user_input: str,
    customer_name: str,
    conversation_id: str,
):
    """Invoke the agent via the AWS Bedrock AgentCore Runtime."""
    import asyncio

    if not AGENT_RUNTIME_ARN:
        msg.content = "Configuration Error: AGENT_RUNTIME_ARN is required when AGENT_CONNECTION_MODE=aws."
        await msg.update()
        return

    try:
        client = _get_agentcore_client()

        payload = json.dumps({
            "prompt": user_input,
            "customer_name": customer_name,
            "conversation_id": conversation_id,
        })

        # Run the synchronous boto3 call in a thread to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.invoke_agent_runtime(
                agentRuntimeArn=AGENT_RUNTIME_ARN,
                qualifier="DEFAULT",
                runtimeSessionId=conversation_id,
                payload=payload,
            ),
        )

        full_response = ""
        thinking_buffer = ""
        in_thinking = False
        content_type = response.get("contentType", "")

        if "text/event-stream" in content_type:
            # Process the streaming response from the runtime
            for line in response["response"].iter_lines(chunk_size=1):
                if not line:
                    continue
                line = line.decode("utf-8")
                if not line.startswith("data: "):
                    continue

                json_str = line[6:]
                try:
                    data = json.loads(json_str)

                    # Handle nested SSE format
                    if isinstance(data, str) and data.startswith("data: "):
                        data = json.loads(data[6:].strip())

                    if isinstance(data, dict) and "chunk" in data:
                        chunk = data["chunk"]
                        thinking_buffer += chunk

                        if "<thinking" in thinking_buffer and not in_thinking:
                            in_thinking = True
                        if "</thinking>" in thinking_buffer and in_thinking:
                            in_thinking = False
                            thinking_buffer = re.sub(
                                r'<thinking>.*?</thinking>\s*', '',
                                thinking_buffer, flags=re.DOTALL
                            )

                        if not in_thinking and not thinking_buffer.startswith("<thinking"):
                            if thinking_buffer:
                                await msg.stream_token(thinking_buffer)
                                full_response += thinking_buffer
                                thinking_buffer = ""

                    elif isinstance(data, dict) and "error" in data:
                        msg.content = f"Error: {data['error']}"
                        await msg.update()
                        return

                except json.JSONDecodeError:
                    continue
        else:
            # Non-streaming response: collect all chunks
            try:
                for event in response.get("response", []):
                    chunk = event.decode("utf-8") if isinstance(event, bytes) else str(event)
                    full_response += chunk
            except Exception:
                pass

        # Handle remaining thinking buffer
        if thinking_buffer:
            thinking_buffer = strip_thinking_tags(thinking_buffer)
            if thinking_buffer:
                await msg.stream_token(thinking_buffer)
                full_response += thinking_buffer

        final_content = strip_thinking_tags(full_response)
        msg.content = final_content if final_content else "No response received."
        await msg.update()

    except Exception as e:
        error_name = type(e).__name__
        msg.content = f"AWS Runtime Error ({error_name}): {str(e)}"
        await msg.update()


if __name__ == "__main__":
    from chainlit.cli import run_chainlit
    run_chainlit(__file__)
