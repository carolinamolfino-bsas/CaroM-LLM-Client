"""Unified Async LLM Client: OpenAI y Anthropic bajo una interfaz común."""

from .anthropic_client import AnthropicClient, anthropic_supports_sampling
from .base import BaseLLMClient
from .manager import (
    API_KEY_ENV,
    DEFAULT_MODELS,
    MODEL_ENV,
    AsyncLLMManager,
    ManagerSettings,
    MissingAPIKeyError,
)
from .openai_client import OpenAIClient
from .schemas import (
    ChatMessage,
    ChatRequest,
    ErrorType,
    LLMError,
    ModelConfig,
    ModelResponse,
    Provider,
    StreamChunk,
    Usage,
)

__version__ = "1.1.0"

__all__ = [
    "API_KEY_ENV",
    "DEFAULT_MODELS",
    "MODEL_ENV",
    "AnthropicClient",
    "AsyncLLMManager",
    "BaseLLMClient",
    "ChatMessage",
    "ChatRequest",
    "ErrorType",
    "LLMError",
    "ManagerSettings",
    "MissingAPIKeyError",
    "ModelConfig",
    "ModelResponse",
    "OpenAIClient",
    "Provider",
    "StreamChunk",
    "Usage",
    "__version__",
    "anthropic_supports_sampling",
]
