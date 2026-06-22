"""
Anthropic-to-LiteLLM Proxy Server

Routes Claude Code requests to custom OpenAI-compatible endpoints.
Supports CUSTOM_MODEL override to route ALL requests to a specific model.
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, field_validator
from typing import Any, Dict, List, Literal, Optional, Union

import json
import logging
import os
import sys
import time
import uuid

import traceback
import litellm
from litellm import token_counter
import uvicorn
from dotenv import load_dotenv

# ─── Configuration ────────────────────────────────────────────────────────────
# Load environment variables from .env file
load_dotenv()

# ════════════════════════════════════════════════════════════════════════════════
# PROVIDER API KEYS
# ════════════════════════════════════════════════════════════════════════════════
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
OPENAI_API_KEY    = os.environ.get("OPENAI_API_KEY")
GEMINI_API_KEY    = os.environ.get("GEMINI_API_KEY")

# ════════════════════════════════════════════════════════════════════════════════
# CUSTOM PROVIDER (overrides ALL requests when set)
# ════════════════════════════════════════════════════════════════════════════════
# Model name - routes ALL Anthropic requests to this model
CUSTOM_MODEL         = os.environ.get("CUSTOM_MODEL")
# Credentials and endpoint for the custom provider
CUSTOM_API_KEY       = os.environ.get("CUSTOM_API_KEY")
CUSTOM_BASE_URL      = os.environ.get("CUSTOM_BASE_URL")   # e.g. https://api.autonaisol.xyz/api/v1/gateway/v1

# ════════════════════════════════════════════════════════════════════════════════
# AZURE OPENAI (when using Azure-hosted models)
# ════════════════════════════════════════════════════════════════════════════════
AZURE_API_KEY    = os.environ.get("AZURE_API_KEY")
AZURE_BASE_URL   = os.environ.get("AZURE_BASE_URL")
AZURE_API_VERSION = os.environ.get("AZURE_API_VERSION", "2024-06-01")

# ════════════════════════════════════════════════════════════════════════════════
# VERTEX AI (Google Cloud) - for Gemini models on GCP
# ════════════════════════════════════════════════════════════════════════════════
VERTEX_PROJECT  = os.environ.get("VERTEX_PROJECT", "unset")
VERTEX_LOCATION = os.environ.get("VERTEX_LOCATION", "unset")
# Use GCP Application Default Credentials instead of API key
USE_VERTEX_AUTH = os.environ.get("USE_VERTEX_AUTH", "False").lower() == "true"

# ════════════════════════════════════════════════════════════════════════════════
# ROUTING & MODEL MAPPING
# ════════════════════════════════════════════════════════════════════════════════
# Auto-detect effective provider based on configured credentials
# Priority: CUSTOM > AZURE > VERTEX > explicit PREFERRED_PROVIDER
def _resolve_provider() -> str:
    if CUSTOM_API_KEY and CUSTOM_BASE_URL:
        return "custom"
    if AZURE_API_KEY and AZURE_BASE_URL:
        return "azure"
    if VERTEX_PROJECT != "unset" and USE_VERTEX_AUTH:
        return "vertex"
    return os.environ.get("PREFERRED_PROVIDER", "openai").lower()

PREFERRED_PROVIDER = _resolve_provider()

# Model tier mapping for Claude model names
BIG_MODEL   = os.environ.get("BIG_MODEL", "gpt-4.1")
SMALL_MODEL = os.environ.get("SMALL_MODEL", "gpt-4.1-mini")
# Override OpenAI base URL (e.g. for proxies or Azure endpoints)
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL")

# ════════════════════════════════════════════════════════════════════════════════
# KNOWN MODEL LISTS
# ════════════════════════════════════════════════════════════════════════════════
OPENAI_MODELS = {"o3-mini", "o1", "o1-mini", "o1-pro", "gpt-4.5-preview", "gpt-4o",
                 "gpt-4o-audio-preview", "chatgpt-4o-latest", "gpt-4o-mini",
                 "gpt-4o-mini-audio-preview", "gpt-4.1", "gpt-4.1-mini", "gpt-5.5",
                 "gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano"}
GEMINI_MODELS = {"gemini-2.5-flash", "gemini-2.5-pro", "gemini-3.1-pro", "gemini-3.5-pro",
                 "gemini-3.5-flash", "gemini-3-flash", "gemini-3.1-flash-lite"}

# ─── Constants ─────────────────────────────────────────────────────────────────

STOP_REASON_MAP = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use"}
BLOCKED_LOG_PHRASES = frozenset({"LiteLLM completion()", "HTTP Request:",
                                  "selected model name for cost calculation",
                                  "utils.py", "cost_calculator"})

# ─── Logging Setup ────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
for name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
    logging.getLogger(name).setLevel(logging.WARNING)


class MessageFilter(logging.Filter):
    def filter(self, record):
        return not any(p in str(record.msg) for p in BLOCKED_LOG_PHRASES)


logging.getLogger().addFilter(MessageFilter())

# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI()

# ─── Pydantic Models ──────────────────────────────────────────────────────────

class ContentBlockText(BaseModel):
    type: Literal["text"]
    text: str


class ContentBlockImage(BaseModel):
    type: Literal["image"]
    source: Dict[str, Any]


class ContentBlockToolUse(BaseModel):
    type: Literal["tool_use"]
    id: str
    name: str
    input: Dict[str, Any]


class ContentBlockToolResult(BaseModel):
    type: Literal["tool_result"]
    tool_use_id: str
    content: Union[str, List[Dict[str, Any]], Dict[str, Any], List[Any], Any]


class SystemContent(BaseModel):
    type: Literal["text"]
    text: str


class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: Union[str, List[Union[ContentBlockText, ContentBlockImage, ContentBlockToolUse, ContentBlockToolResult]]]


class Tool(BaseModel):
    name: str
    description: Optional[str] = None
    input_schema: Dict[str, Any]


class ThinkingConfig(BaseModel):
    enabled: bool = True


# ─── Model Mapping Helper ─────────────────────────────────────────────────────

def _strip_prefix(model: str) -> str:
    """Remove provider prefix from model name."""
    for prefix in ("anthropic/", "openai/", "gemini/"):
        if model.startswith(prefix):
            return model[len(prefix):]
    return model


def _strip_model(model: str) -> str:
    """Remove provider prefix for display (last segment after /)."""
    return model.split("/")[-1] if "/" in model else model


def _apply_custom_override(model: str) -> str:
    """Strip openai/ prefix from model when using CUSTOM_BASE_URL."""
    if CUSTOM_BASE_URL and model.startswith("openai/"):
        return model[7:]
    return model


def _map_model_name(model: str) -> str:
    """
    Map Anthropic model names to provider-specific models.
    CUSTOM_MODEL overrides everything when set.
    PREFERRED_PROVIDER determines routing (auto-detected if CUSTOM/AZURE/VERTEX vars are set).
    Returns the provider-prefixed model name.
    """
    clean = _strip_prefix(model)

    # CUSTOM_MODEL overrides ALL requests
    if CUSTOM_MODEL:
        return CUSTOM_MODEL

    # No prefix for custom provider → routes via CUSTOM_BASE_URL
    if PREFERRED_PROVIDER == "custom":
        return clean

    # Azure → azure/ prefix
    if PREFERRED_PROVIDER == "azure":
        return f"azure/{clean}"

    # Vertex (GCP) → gemini/ prefix
    if PREFERRED_PROVIDER == "vertex":
        return f"gemini/{clean}"

    # Provider-specific routing
    if PREFERRED_PROVIDER == "anthropic":
        return f"anthropic/{clean}"

    # Haiku → small model
    if "haiku" in clean.lower():
        if PREFERRED_PROVIDER == "google" and SMALL_MODEL in GEMINI_MODELS:
            return f"gemini/{SMALL_MODEL}"
        return f"openai/{SMALL_MODEL}"

    # Sonnet → big model
    if "sonnet" in clean.lower():
        if PREFERRED_PROVIDER == "google" and BIG_MODEL in GEMINI_MODELS:
            return f"gemini/{BIG_MODEL}"
        return f"openai/{BIG_MODEL}"

    # Opus → gpt-5.5 equivalent
    if "opus" in clean.lower():
        if PREFERRED_PROVIDER == "google":
            return f"gemini/{BIG_MODEL}"
        return "openai/gpt-5.5"

    # Add prefix to known models that don't have one
    if clean in GEMINI_MODELS and not model.startswith("gemini/"):
        return f"gemini/{clean}"
    if clean in OPENAI_MODELS and not model.startswith("openai/"):
        return f"openai/{clean}"

    # Default to preferred provider
    if not model.startswith(("openai/", "gemini/", "anthropic/")):
        return f"{PREFERRED_PROVIDER}/{clean}"

    return model


def _is_custom_model(model: str) -> bool:
    """Check if a model should be routed to the custom provider."""
    if CUSTOM_MODEL:
        return True
    if PREFERRED_PROVIDER == "custom":
        return True
    return False


class MessagesRequest(BaseModel):
    model: str
    max_tokens: int
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    stop_sequences: Optional[List[str]] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Dict[str, Any]] = None
    thinking: Optional[ThinkingConfig] = None
    original_model: Optional[str] = None

    @field_validator("model")
    def validate_model_field(cls, v, info):
        original = v
        mapped = _map_model_name(v)
        if mapped != v or v.startswith(("openai/", "gemini/", "anthropic/")):
            logger.debug(f"MODEL MAPPING: '{original}' → '{mapped}'")
        info.data["original_model"] = original
        return mapped


class TokenCountRequest(BaseModel):
    model: str
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    tools: Optional[List[Tool]] = None
    thinking: Optional[ThinkingConfig] = None
    tool_choice: Optional[Dict[str, Any]] = None
    original_model: Optional[str] = None

    @field_validator("model")
    def validate_model_field(cls, v, info):
        original = v
        mapped = _map_model_name(v)
        if mapped != v or v.startswith(("openai/", "gemini/", "anthropic/")):
            logger.debug(f"TOKEN COUNT MODEL MAPPING: '{original}' → '{mapped}'")
        info.data["original_model"] = original
        return mapped


class TokenCountResponse(BaseModel):
    input_tokens: int


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


class MessagesResponse(BaseModel):
    id: str
    model: str
    role: Literal["assistant"] = "assistant"
    content: List[Union[ContentBlockText, ContentBlockToolUse]]
    type: Literal["message"] = "message"
    stop_reason: Optional[Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"]] = None
    stop_sequence: Optional[str] = None
    usage: Usage


# ─── Request Logging ──────────────────────────────────────────────────────────

class Colors:
    CYAN = "\033[96m"; BLUE = "\033[94m"; GREEN = "\033[92m"; YELLOW = "\033[93m"
    RED = "\033[91m"; MAGENTA = "\033[95m"; RESET = "\033[0m"; BOLD = "\033[1m"


def log_request_beautifully(method, path, claude_model, routed_model, num_messages, num_tools, status_code):
    """Log requests showing Claude → routed model mapping."""
    status = (f"{Colors.GREEN}✓ {status_code} OK{Colors.RESET}" if status_code == 200
              else f"{Colors.RED}✗ {status_code}{Colors.RESET}")

    print(f"{Colors.BOLD}{method} {path}{Colors.RESET} {status}")
    print(f"{Colors.CYAN}{_strip_model(claude_model)}{Colors.RESET} → "
          f"{Colors.GREEN}{_strip_model(routed_model)}{Colors.RESET} "
          f"{Colors.MAGENTA}{num_tools} tools{Colors.RESET} "
          f"{Colors.BLUE}{num_messages} messages{Colors.RESET}")
    sys.stdout.flush()


# ─── Helper Functions ──────────────────────────────────────────────────────────

def clean_gemini_schema(schema: Any) -> Any:
    """Remove unsupported fields from JSON schema for Gemini."""
    if isinstance(schema, dict):
        schema.pop("additionalProperties", None)
        schema.pop("default", None)
        if schema.get("type") == "string" and "format" in schema:
            allowed = {"enum", "date-time"}
            if schema["format"] not in allowed:
                schema.pop("format")
        for key, value in list(schema.items()):
            schema[key] = clean_gemini_schema(value)
    elif isinstance(schema, list):
        return [clean_gemini_schema(item) for item in schema]
    return schema


def parse_tool_result_content(content):
    """Normalize tool result content to string."""
    if content is None:
        return "No content provided"
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        result = ""
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                result += item.get("text", "") + "\n"
            elif isinstance(item, str):
                result += item + "\n"
            elif isinstance(item, dict):
                result += (item.get("text") or json.dumps(item)) + "\n"
            else:
                result += str(item) + "\n"
        return result.strip()
    if isinstance(content, dict):
        if content.get("type") == "text":
            return content.get("text", "")
        try:
            return json.dumps(content)
        except:
            return str(content)
    return str(content)


def _is_anthropic_model(model: str) -> bool:
    """Check if model is Anthropic (has anthropic/ prefix or is a known Claude model)."""
    if model.startswith("anthropic/"):
        return True
    stripped = _strip_prefix(model)
    return stripped.startswith("claude-") or stripped in ("opus", "sonnet", "haiku", "claude-3-5-sonnet-20240620")


def _is_openai_model(model: str) -> bool:
    """Check if model should be treated as OpenAI (openai/ prefix or CUSTOM_MODEL without anthropic prefix)."""
    if model.startswith("openai/"):
        return True
    if model.startswith("anthropic/") or model.startswith("gemini/"):
        return False
    if PREFERRED_PROVIDER == "custom":
        return False
    # CUSTOM_MODEL or other non-prefixed models default to openai-compatible
    stripped = _strip_prefix(model)
    return stripped in OPENAI_MODELS or "/" in model or CUSTOM_MODEL


def _is_gemini_model(model: str) -> bool:
    """Check if model is Gemini (gemini/ prefix or in GEMINI_MODELS)."""
    if model.startswith("gemini/"):
        return True
    stripped = _strip_prefix(model)
    return stripped in GEMINI_MODELS


def convert_anthropic_to_litellm(anthropic_request: MessagesRequest) -> Dict[str, Any]:
    """Convert Anthropic API request format to LiteLLM format."""

    messages = []
    system_content = None

    # Extract system message
    if anthropic_request.system:
        if isinstance(anthropic_request.system, str):
            system_content = anthropic_request.system
        elif isinstance(anthropic_request.system, list):
            texts = []
            for block in anthropic_request.system:
                if hasattr(block, "type") and block.type == "text":
                    texts.append(block.text)
                elif isinstance(block, dict) and block.get("type") == "text":
                    texts.append(block.get("text", ""))
            system_content = "\n\n".join(texts).strip() if texts else None

    # Process messages
    for msg in anthropic_request.messages:
        content = msg.content

        # Extract system messages
        if msg.role == "system":
            if isinstance(content, str):
                system_content = (system_content + "\n\n" + content) if system_content else content
            elif isinstance(content, list):
                for block in content:
                    if hasattr(block, "type") and block.type == "text":
                        system_content = (system_content + "\n\n" + block.text) if system_content else block.text
                    elif isinstance(block, dict) and block.get("type") == "text":
                        system_content = (system_content + "\n\n" + block.get("text", "")) if system_content else block.get("text", "")
            continue

        # Build message content
        if isinstance(content, str):
            messages.append({"role": msg.role, "content": content})
        else:
            has_tool_result = msg.role == "user" and any(
                isinstance(b, ContentBlockToolResult) for b in content
            )

            if has_tool_result:
                text_content = ""
                for block in content:
                    if isinstance(block, ContentBlockText):
                        text_content += block.text + "\n"
                    elif isinstance(block, ContentBlockToolResult):
                        tool_id = getattr(block, "tool_use_id", "")
                        result_content = parse_tool_result_content(getattr(block, "content", ""))
                        text_content += f"Tool result for {tool_id}:\n{result_content}\n"
                messages.append({"role": "user", "content": text_content.strip()})
            else:
                processed = []
                for block in content:
                    if isinstance(block, ContentBlockText):
                        processed.append({"type": "text", "text": block.text})
                    elif isinstance(block, ContentBlockImage):
                        processed.append({"type": "image", "source": block.source})
                    elif isinstance(block, ContentBlockToolUse):
                        processed.append({
                            "type": "tool_use", "id": block.id, "name": block.name, "input": block.input
                        })
                    elif isinstance(block, ContentBlockToolResult):
                        content_val = getattr(block, "content", "")
                        if isinstance(content_val, str):
                            content_out = [{"type": "text", "text": content_val}]
                        else:
                            content_out = content_val if isinstance(content_val, list) else [{"type": "text", "text": str(content_val)}]
                        processed.append({
                            "type": "tool_result",
                            "tool_use_id": getattr(block, "tool_use_id", ""),
                            "content": content_out
                        })
                messages.append({"role": msg.role, "content": processed})

    # Cap max_tokens for OpenAI/Gemini
    model = anthropic_request.model
    max_tokens = anthropic_request.max_tokens
    if _is_openai_model(model) or _is_gemini_model(model):
        max_tokens = min(max_tokens, 16384)

    # Ensure model has provider prefix for LiteLLM
    if not model.startswith(("anthropic/", "openai/", "gemini/")):
        if CUSTOM_MODEL:
            model = CUSTOM_MODEL
        else:
            model = f"anthropic/{model}"

    litellm_req = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": max_tokens,
        "temperature": anthropic_request.temperature,
        "stream": anthropic_request.stream,
    }

    if system_content:
        litellm_req["system"] = system_content

    # Only add thinking for Anthropic models
    if anthropic_request.thinking and _is_anthropic_model(model):
        litellm_req["thinking"] = anthropic_request.thinking

    if anthropic_request.stop_sequences:
        litellm_req["stop"] = anthropic_request.stop_sequences
    if anthropic_request.top_p:
        litellm_req["top_p"] = anthropic_request.top_p
    if anthropic_request.top_k:
        litellm_req["top_k"] = anthropic_request.top_k

    # Convert tools
    if anthropic_request.tools:
        openai_tools = []
        is_gemini = _is_gemini_model(model)

        for tool in anthropic_request.tools:
            tool_dict = dict(tool) if hasattr(tool, "dict") else (tool if isinstance(tool, dict) else {})
            if not tool_dict:
                continue

            input_schema = tool_dict.get("input_schema", {})
            if is_gemini:
                input_schema = clean_gemini_schema(input_schema)

            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool_dict["name"],
                    "description": tool_dict.get("description", ""),
                    "parameters": input_schema
                }
            })

        if openai_tools:
            litellm_req["tools"] = openai_tools

    # Convert tool_choice
    if anthropic_request.tool_choice:
        choice = dict(anthropic_request.tool_choice) if hasattr(anthropic_request.tool_choice, "dict") else anthropic_request.tool_choice
        choice_type = choice.get("type") if isinstance(choice, dict) else None
        if choice_type == "auto":
            litellm_req["tool_choice"] = "auto"
        elif choice_type == "any":
            litellm_req["tool_choice"] = "any"
        elif choice_type == "tool" and isinstance(choice, dict) and "name" in choice:
            litellm_req["tool_choice"] = {"type": "function", "function": {"name": choice["name"]}}
        else:
            litellm_req["tool_choice"] = "auto"

    return litellm_req


def convert_litellm_to_anthropic(litellm_response: Union[Dict[str, Any], Any], original_request: MessagesRequest) -> MessagesResponse:
    """Convert LiteLLM response to Anthropic API format."""

    try:
        clean_model = _strip_prefix(original_request.model)
        is_claude = clean_model.startswith("claude-")

        # Extract from ModelResponse object or dict
        if hasattr(litellm_response, "choices") and hasattr(litellm_response, "usage"):
            choices = litellm_response.choices
            message = choices[0].message if choices else None
            content_text = message.content if message and hasattr(message, "content") else ""
            tool_calls = message.tool_calls if message and hasattr(message, "tool_calls") else None
            finish_reason = choices[0].finish_reason if choices else "stop"
            usage_info = litellm_response.usage
            response_id = getattr(litellm_response, "id", f"msg_{uuid.uuid4()}")
        else:
            resp_dict = litellm_response
            if not isinstance(resp_dict, dict):
                resp_dict = getattr(litellm_response, "dict", lambda: {})() or {}
            choices = resp_dict.get("choices", [{}])
            message = choices[0].get("message", {}) if choices else {}
            content_text = message.get("content", "")
            tool_calls = message.get("tool_calls")
            finish_reason = choices[0].get("finish_reason", "stop") if choices else "stop"
            usage_info = resp_dict.get("usage", {})
            response_id = resp_dict.get("id", f"msg_{uuid.uuid4()}")

        content = []
        if content_text:
            content.append({"type": "text", "text": content_text})

        # Handle tool calls
        if tool_calls and is_claude:
            if not isinstance(tool_calls, list):
                tool_calls = [tool_calls]
            for tc in tool_calls:
                if isinstance(tc, dict):
                    func = tc.get("function", {})
                    tool_id = tc.get("id", f"tool_{uuid.uuid4()}")
                    name = func.get("name", "")
                    arguments = func.get("arguments", "{}")
                else:
                    func = getattr(tc, "function", None)
                    tool_id = getattr(tc, "id", f"tool_{uuid.uuid4()}")
                    name = getattr(func, "name", "") if func else ""
                    arguments = getattr(func, "arguments", "{}") if func else "{}"

                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {"raw": arguments}

                content.append({"type": "tool_use", "id": tool_id, "name": name, "input": arguments})

        elif tool_calls and not is_claude:
            tool_text = "\n\nTool usage:\n"
            if not isinstance(tool_calls, list):
                tool_calls = [tool_calls]
            for tc in tool_calls:
                if isinstance(tc, dict):
                    func = tc.get("function", {})
                    name = func.get("name", "")
                    arguments = func.get("arguments", "{}")
                else:
                    func = getattr(tc, "function", None)
                    name = getattr(func, "name", "") if func else ""
                    arguments = getattr(func, "arguments", "{}") if func else "{}"
                try:
                    args_str = json.dumps(json.loads(arguments) if isinstance(arguments, str) else arguments, indent=2)
                except:
                    args_str = str(arguments)
                tool_text += f"Tool: {name}\nArguments: {args_str}\n\n"
            if content and content[0]["type"] == "text":
                content[0]["text"] += tool_text
            else:
                content.append({"type": "text", "text": tool_text})

        # Extract usage
        if isinstance(usage_info, dict):
            prompt_tokens = usage_info.get("prompt_tokens", 0)
            completion_tokens = usage_info.get("completion_tokens", 0)
        else:
            prompt_tokens = getattr(usage_info, "prompt_tokens", 0)
            completion_tokens = getattr(usage_info, "completion_tokens", 0)

        # Map finish_reason
        stop_reason = STOP_REASON_MAP.get(finish_reason, "end_turn")

        if not content:
            content.append({"type": "text", "text": ""})

        return MessagesResponse(
            id=response_id,
            model=original_request.model,
            role="assistant",
            content=content,
            stop_reason=stop_reason,
            usage=Usage(input_tokens=prompt_tokens, output_tokens=completion_tokens)
        )

    except Exception as e:
        logger.error(f"Error converting response: {str(e)}\n{traceback.format_exc()}")
        return MessagesResponse(
            id=f"msg_{uuid.uuid4()}",
            model=original_request.model,
            role="assistant",
            content=[{"type": "text", "text": f"Error converting response: {str(e)}"}],
            stop_reason="end_turn",
            usage=Usage(input_tokens=0, output_tokens=0)
        )


async def handle_streaming(response_generator, original_request: MessagesRequest):
    """Handle streaming responses from LiteLLM and convert to Anthropic format."""
    try:
        message_id = f"msg_{uuid.uuid4().hex[:24]}"

        yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': message_id, 'type': 'message', 'role': 'assistant', 'model': original_request.model, 'content': [], 'stop_reason': None, 'stop_sequence': None, 'usage': {'input_tokens': 0, 'cache_creation_input_tokens': 0, 'cache_read_input_tokens': 0, 'output_tokens': 0}}})}\n\n"
        yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
        yield f"event: ping\ndata: {json.dumps({'type': 'ping'})}\n\n"

        tool_index = None
        accumulated_text = ""
        text_sent = False
        text_block_closed = False
        output_tokens = 0
        has_sent_stop_reason = False
        last_tool_index = 0

        async for chunk in response_generator:
            try:
                if hasattr(chunk, "usage") and chunk.usage:
                    output_tokens = getattr(chunk.usage, "completion_tokens", 0)

                if hasattr(chunk, "choices") and chunk.choices:
                    choice = chunk.choices[0]
                    delta = getattr(choice, "delta", None) or getattr(choice, "message", {})
                    finish_reason = getattr(choice, "finish_reason", None)

                    delta_content = getattr(delta, "content", None)
                    if delta_content is None and isinstance(delta, dict):
                        delta_content = delta.get("content")

                    if delta_content:
                        accumulated_text += delta_content
                        if tool_index is None and not text_block_closed:
                            text_sent = True
                            yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': delta_content}})}\n\n"

                    # Process tool calls
                    delta_tool_calls = getattr(delta, "tool_calls", None)
                    if delta_tool_calls is None and isinstance(delta, dict):
                        delta_tool_calls = delta.get("tool_calls")

                    if delta_tool_calls:
                        if not isinstance(delta_tool_calls, list):
                            delta_tool_calls = [delta_tool_calls]

                        for tool_call in delta_tool_calls:
                            current_index = tool_call.get("index") if isinstance(tool_call, dict) else getattr(tool_call, "index", 0)

                            if tool_index is None or current_index != tool_index:
                                if tool_index is None and not text_block_closed:
                                    if text_sent:
                                        text_block_closed = True
                                        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
                                    elif accumulated_text:
                                        text_sent = True
                                        yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': accumulated_text}})}\n\n"
                                        text_block_closed = True
                                        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
                                    else:
                                        text_block_closed = True
                                        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"

                                tool_index = current_index
                                last_tool_index += 1
                                anthropic_tool_index = last_tool_index

                                func = tool_call.get("function", {}) if isinstance(tool_call, dict) else getattr(tool_call, "function", None)
                                name = func.get("name", "") if isinstance(func, dict) else getattr(func, "name", "") if func else ""
                                tool_id = tool_call.get("id", f"toolu_{uuid.uuid4().hex[:24]}") if isinstance(tool_call, dict) else getattr(tool_call, "id", f"toolu_{uuid.uuid4().hex[:24]}")

                                yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': anthropic_tool_index, 'content_block': {'type': 'tool_use', 'id': tool_id, 'name': name, 'input': {}}})}\n\n"

                            func = tool_call.get("function", {}) if isinstance(tool_call, dict) else getattr(tool_call, "function", None)
                            arguments = func.get("arguments", "") if isinstance(func, dict) else getattr(func, "arguments", "") if func else ""

                            if arguments:
                                try:
                                    json.loads(arguments)
                                    args_json = arguments
                                except json.JSONDecodeError:
                                    args_json = arguments

                                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': anthropic_tool_index, 'delta': {'type': 'input_json_delta', 'partial_json': args_json}})}\n\n"

                    if finish_reason and not has_sent_stop_reason:
                        has_sent_stop_reason = True
                        stop_reason = STOP_REASON_MAP.get(finish_reason, "end_turn")

                        if tool_index is not None:
                            for i in range(1, last_tool_index + 1):
                                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': i})}\n\n"

                        if not text_block_closed:
                            if accumulated_text and not text_sent:
                                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': accumulated_text}})}\n\n"
                            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"

                        yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': stop_reason, 'stop_sequence': None}, 'usage': {'output_tokens': output_tokens}})}\n\n"
                        yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
                        yield "data: [DONE]\n\n"
                        return

            except Exception as e:
                logger.error(f"Error processing chunk: {str(e)}")
                continue

        # Handle case where we didn't get a finish reason
        if not has_sent_stop_reason:
            if tool_index is not None:
                for i in range(1, last_tool_index + 1):
                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': i})}\n\n"
            if not text_block_closed:
                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
            yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'usage': {'output_tokens': output_tokens}})}\n\n"
            yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
            yield "data: [DONE]\n\n"

    except Exception as e:
        logger.error(f"Error in streaming: {str(e)}\n{traceback.format_exc()}")
        yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'error', 'stop_sequence': None}, 'usage': {'output_tokens': 0}})}\n\n"
        yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
        yield "data: [DONE]\n\n"


# ─── Routes ────────────────────────────────────────────────────────────────────

@app.post("/v1/messages")
async def create_message(request: MessagesRequest, raw_request: Request):
    """Main proxy endpoint - converts Anthropic requests to LiteLLM."""
    try:
        client_api_key = raw_request.headers.get("x-api-key") or raw_request.headers.get("authorization", "").replace("Bearer ", "")

        # Re-parse body to get original model name (Pydantic already mapped it)
        body = await raw_request.body()
        body_json = json.loads(body.decode("utf-8"))
        original_model = body_json.get("model", request.model)

        logger.debug(f"PROCESSING: Model={request.model}, Stream={request.stream}")

        litellm_req = convert_anthropic_to_litellm(request)
        model_for_routing = litellm_req["model"]

        # Apply CUSTOM_MODEL override - strip openai/ prefix if using custom api_base
        if CUSTOM_MODEL:
            litellm_req["model"] = _apply_custom_override(CUSTOM_MODEL)
            logger.debug(f"CUSTOM_MODEL override: '{litellm_req['model']}'")

        # Route based on PREFERRED_PROVIDER (auto-detected from env)
        if PREFERRED_PROVIDER == "custom":
            litellm_req["api_key"] = CUSTOM_API_KEY or OPENAI_API_KEY
            if CUSTOM_BASE_URL:
                litellm_req["api_base"] = CUSTOM_BASE_URL
            litellm_req["custom_llm_provider"] = "openai"
            logger.debug(f"Using custom provider: base={CUSTOM_BASE_URL}, model={litellm_req['model']}")
        elif PREFERRED_PROVIDER == "azure":
            litellm_req["api_key"] = AZURE_API_KEY
            litellm_req["api_base"] = AZURE_BASE_URL
            litellm_req["api_version"] = AZURE_API_VERSION
            litellm_req["custom_llm_provider"] = "azure"
            logger.debug(f"Using Azure: base={AZURE_BASE_URL}, model={litellm_req['model']}")
        elif PREFERRED_PROVIDER == "vertex":
            litellm_req["vertex_project"] = VERTEX_PROJECT
            litellm_req["vertex_location"] = VERTEX_LOCATION
            litellm_req["custom_llm_provider"] = "vertex_ai"
            logger.debug(f"Using Vertex AI: project={VERTEX_PROJECT}, model={litellm_req['model']}")
        elif PREFERRED_PROVIDER == "openai":
            if OPENAI_BASE_URL:
                litellm_req["api_key"] = OPENAI_API_KEY
                litellm_req["api_base"] = OPENAI_BASE_URL
                logger.debug(f"Using OpenAI with base URL: {OPENAI_BASE_URL}")
            else:
                litellm_req["api_key"] = OPENAI_API_KEY
                logger.debug(f"Using OpenAI API key: {model_for_routing}")
        elif PREFERRED_PROVIDER == "anthropic":
            if client_api_key:
                litellm_req["api_key"] = client_api_key
                os.environ["ANTHROPIC_API_KEY"] = client_api_key
            elif ANTHROPIC_API_KEY:
                litellm_req["api_key"] = ANTHROPIC_API_KEY
                os.environ["ANTHROPIC_API_KEY"] = ANTHROPIC_API_KEY
            logger.debug(f"Using Anthropic API key: {model_for_routing}")
        else:
            # google / gemini
            litellm_req["api_key"] = GEMINI_API_KEY
            logger.debug(f"Using Gemini API key: {model_for_routing}")

        # Simplify content for OpenAI models
        if _is_openai_model(model_for_routing) and "messages" in litellm_req:
            for i, msg in enumerate(litellm_req["messages"]):
                if "content" in msg and isinstance(msg["content"], list):
                    text_content = ""
                    for block in msg["content"]:
                        if not isinstance(block, dict):
                            continue
                        if block.get("type") == "text":
                            text_content += block.get("text", "") + "\n"
                        elif block.get("type") == "tool_result":
                            tool_id = block.get("tool_use_id", "unknown")
                            text_content += f"[Tool Result ID: {tool_id}]\n"
                            result_content = block.get("content", [])
                            if isinstance(result_content, list):
                                for item in result_content:
                                    if isinstance(item, dict) and item.get("type") == "text":
                                        text_content += item.get("text", "") + "\n"
                                    elif isinstance(item, dict):
                                        text_content += (item.get("text") or json.dumps(item)) + "\n"
                            elif isinstance(result_content, str):
                                text_content += result_content + "\n"
                            else:
                                text_content += json.dumps(result_content) + "\n"
                        elif block.get("type") == "tool_use":
                            tool_name = block.get("name", "unknown")
                            tool_id = block.get("id", "unknown")
                            tool_input = json.dumps(block.get("input", {}))
                            text_content += f"[Tool: {tool_name} (ID: {tool_id})]\nInput: {tool_input}\n\n"
                        elif block.get("type") == "image":
                            text_content += "[Image content]\n"

                    litellm_req["messages"][i]["content"] = text_content.strip() or "..."

                # Remove unsupported fields
                for key in list(msg.keys()):
                    if key not in ("role", "content", "name", "tool_call_id", "tool_calls"):
                        del msg[key]

                # Validate content
                if isinstance(msg.get("content"), list):
                    litellm_req["messages"][i]["content"] = f"Content as JSON: {json.dumps(msg.get('content'))}"
                elif msg.get("content") is None:
                    litellm_req["messages"][i]["content"] = "..."

        num_tools = len(request.tools) if request.tools else 0
        display_model = _strip_model(original_model)

        log_request_beautifully(
            "POST", raw_request.url.path, display_model,
            litellm_req.get("model", ""),
            len(litellm_req.get("messages", [])),
            num_tools, 200
        )

        if request.stream:
            response_generator = await litellm.acompletion(**litellm_req)
            return StreamingResponse(
                handle_streaming(response_generator, request),
                media_type="text/event-stream"
            )
        else:
            start_time = time.time()
            litellm_response = litellm.completion(**litellm_req)
            logger.debug(f"RESPONSE: Model={litellm_req.get('model')}, Time={time.time() - start_time:.2f}s")

            anthropic_response = convert_litellm_to_anthropic(litellm_response, request)
            return anthropic_response

    except Exception as e:
        import traceback
        error_details = {"error": str(e), "type": type(e).__name__, "traceback": traceback.format_exc()}
        for attr in ("message", "status_code", "response", "llm_provider", "model"):
            if hasattr(e, attr):
                error_details[attr] = getattr(e, attr)

        def sanitize(obj):
            if isinstance(obj, dict):
                return {k: sanitize(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [sanitize(i) for i in obj]
            elif hasattr(obj, "__dict__"):
                return sanitize(obj.__dict__)
            elif hasattr(obj, "text"):
                return str(obj.text)
            else:
                try:
                    json.dumps(obj)
                    return obj
                except:
                    return str(obj)

        logger.error(f"Error: {json.dumps(sanitize(error_details), indent=2)}")
        status_code = error_details.get("status_code", 500)
        raise HTTPException(status_code=status_code, detail=str(e))


@app.post("/v1/messages/count_tokens")
async def count_tokens(request: TokenCountRequest, raw_request: Request):
    """Count tokens for a request - applies CUSTOM_MODEL override."""
    try:
        display_model = _strip_model(request.original_model or request.model)

        # Apply CUSTOM_MODEL override for token counting
        model_for_count = _apply_custom_override(_map_model_name(request.model))

        converted = convert_anthropic_to_litellm(
            MessagesRequest(
                model=request.model,
                max_tokens=100,
                messages=request.messages,
                system=request.system,
                tools=request.tools,
                tool_choice=request.tool_choice,
                thinking=request.thinking
            )
        )

        num_tools = len(request.tools) if request.tools else 0

        log_request_beautifully(
            "POST", raw_request.url.path, display_model,
            model_for_count,
            len(converted["messages"]),
            num_tools, 200
        )

        token_counter_args = {"model": model_for_count, "messages": converted["messages"]}

        if PREFERRED_PROVIDER == "custom" and CUSTOM_BASE_URL:
            token_counter_args["api_base"] = CUSTOM_BASE_URL
        elif PREFERRED_PROVIDER == "azure" and AZURE_BASE_URL:
            token_counter_args["api_base"] = AZURE_BASE_URL
            token_counter_args["api_version"] = AZURE_API_VERSION
        elif PREFERRED_PROVIDER == "openai" and OPENAI_BASE_URL:
            token_counter_args["api_base"] = OPENAI_BASE_URL

        token_count = token_counter(**token_counter_args)
        return TokenCountResponse(input_tokens=token_count)

    except ImportError:
        logger.error("Could not import token_counter from litellm")
        return TokenCountResponse(input_tokens=1000)
    except Exception as e:
        logger.error(f"Error counting tokens: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error counting tokens: {str(e)}")


@app.get("/")
async def root():
    return {"message": "Anthropic Proxy for LiteLLM"}


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Run with: uvicorn server:app --reload --host 0.0.0.0 --port 8082")
        sys.exit(0)
    uvicorn.run(app, host="0.0.0.0", port=8082, log_level="error")
