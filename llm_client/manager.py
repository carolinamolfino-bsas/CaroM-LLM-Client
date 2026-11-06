"""AsyncLLMManager: elige el proveedor según configuración y expone una única API."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Sequence

from pydantic import BaseModel, ConfigDict, Field, SecretStr

from .anthropic_client import AnthropicClient
from .base import BaseLLMClient
from .openai_client import OpenAIClient
from .schemas import ChatMessage, ModelConfig, ModelResponse, Provider, StreamChunk

DEFAULT_MODELS: dict[Provider, str] = {
    Provider.OPENAI: "gpt-4o-mini",
    Provider.ANTHROPIC: "claude-haiku-4-5",
}

_REGISTRY: dict[Provider, type[BaseLLMClient]] = {
    Provider.OPENAI: OpenAIClient,
    Provider.ANTHROPIC: AnthropicClient,
}

_KEY_ENV = {Provider.OPENAI: "OPENAI_API_KEY", Provider.ANTHROPIC: "ANTHROPIC_API_KEY"}
_MODEL_ENV = {Provider.OPENAI: "OPENAI_MODEL", Provider.ANTHROPIC: "ANTHROPIC_MODEL"}


class MissingAPIKeyError(ValueError):
    """La API key del proveedor seleccionado no está configurada."""


class ManagerSettings(BaseModel):
    """Configuración del manager (validada con Pydantic)."""

    model_config = ConfigDict(extra="forbid")

    provider: Provider
    api_key: SecretStr | None = None  # SecretStr evita que la key aparezca en logs
    model: ModelConfig

    @classmethod
    def from_env(cls, provider: str | Provider | None = None) -> "ManagerSettings":
        """Construye la configuración desde variables de entorno (.env)."""
        prov = Provider((provider or os.getenv("LLM_PROVIDER", "openai")).lower())
        key = os.getenv(_KEY_ENV[prov])
        model_kwargs: dict = {
            "model": os.getenv(_MODEL_ENV[prov]) or DEFAULT_MODELS[prov],
        }
        for env, field in (
            ("LLM_TEMPERATURE", "temperature"),
            ("LLM_MAX_TOKENS", "max_tokens"),
            ("LLM_TIMEOUT", "timeout"),
            ("LLM_MAX_RETRIES", "max_retries"),
        ):
            if (value := os.getenv(env)) not in (None, ""):
                model_kwargs[field] = value  # Pydantic convierte y valida tipos/rangos
        return cls(
            provider=prov,
            api_key=SecretStr(key) if key else None,
            model=ModelConfig(**model_kwargs),
        )


class AsyncLLMManager:
    """Punto de entrada único. Ejemplo:

        async with AsyncLLMManager.from_env("anthropic") as llm:
            resp = await llm.generate([ChatMessage(role="user", content="Hola")])
            async for chunk in llm.stream(...):
                print(chunk.text, end="")
    """

    def __init__(self, settings: ManagerSettings) -> None:
        if settings.api_key is None:
            raise MissingAPIKeyError(
                f"Falta {_KEY_ENV[settings.provider]} para el proveedor "
                f"'{settings.provider}'. Configurala en el archivo .env"
            )
        self.settings = settings
        client_cls = _REGISTRY[settings.provider]
        self.client: BaseLLMClient = client_cls(
            settings.model, api_key=settings.api_key.get_secret_value()
        )

    @classmethod
    def from_env(cls, provider: str | Provider | None = None) -> "AsyncLLMManager":
        return cls(ManagerSettings.from_env(provider))

    @property
    def provider(self) -> Provider:
        return self.settings.provider

    async def generate(self, messages: Sequence[ChatMessage]) -> ModelResponse:
        return await self.client.generate(messages)

    async def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[StreamChunk]:
        async for chunk in self.client.stream(messages):
            yield chunk

    async def aclose(self) -> None:
        await self.client.aclose()

    async def __aenter__(self) -> "AsyncLLMManager":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()
