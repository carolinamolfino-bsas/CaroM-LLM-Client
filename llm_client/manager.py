"""AsyncLLMManager: elige el proveedor según configuración y expone una única API."""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Sequence

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

from .anthropic_client import AnthropicClient
from .base import BaseLLMClient
from .openai_client import OpenAIClient
from .schemas import ChatMessage, ModelConfig, ModelResponse, Provider, StreamChunk

DEFAULT_MODELS: dict[Provider, str] = {
    Provider.OPENAI: "gpt-4o-mini",
    Provider.ANTHROPIC: "claude-haiku-4-5",
}

API_KEY_ENV: dict[Provider, str] = {
    Provider.OPENAI: "OPENAI_API_KEY",
    Provider.ANTHROPIC: "ANTHROPIC_API_KEY",
}
# Sólo OpenAI: permite apuntar a servidores compatibles (Ollama, LM Studio...).
BASE_URL_ENV: dict[Provider, str] = {Provider.OPENAI: "OPENAI_BASE_URL"}

MODEL_ENV: dict[Provider, str] = {
    Provider.OPENAI: "OPENAI_MODEL",
    Provider.ANTHROPIC: "ANTHROPIC_MODEL",
}

_REGISTRY: dict[Provider, type[BaseLLMClient]] = {
    Provider.OPENAI: OpenAIClient,
    Provider.ANTHROPIC: AnthropicClient,
}


class MissingAPIKeyError(ValueError):
    """La API key del proveedor seleccionado no está configurada."""


class ManagerSettings(BaseModel):
    """Configuración del manager (validada con Pydantic).

    Es reutilizable desde otros proyectos (ej. un pipeline de LangChain):
    `ManagerSettings.from_env()` devuelve proveedor, API key y ModelConfig
    validados, sin crear ningún cliente.
    """

    model_config = ConfigDict(extra="forbid")

    provider: Provider
    api_key: SecretStr | None = None  # SecretStr evita que la key aparezca en logs
    base_url: str | None = Field(
        default=None,
        description="URL de un servidor compatible con OpenAI (ej. http://localhost:11434/v1).",
    )
    model: ModelConfig
    max_concurrency: int = Field(
        default=5, ge=1, le=100,
        description="Máximo de llamadas simultáneas al proveedor (evita el rate limit).",
    )

    @field_validator("base_url")
    @classmethod
    def _validar_base_url(cls, value: str | None) -> str | None:
        if value is None or not value.strip():
            return None
        value = value.strip().rstrip("/")
        if not value.startswith(("http://", "https://")):
            raise ValueError("base_url debe empezar con http:// o https://")
        return value

    @classmethod
    def from_env(cls, provider: str | Provider | None = None) -> ManagerSettings:
        """Construye la configuración desde variables de entorno (.env)."""
        prov = Provider(str(provider or os.getenv("LLM_PROVIDER") or "openai").lower())
        key = os.getenv(API_KEY_ENV[prov])
        model_kwargs: dict = {
            "model": os.getenv(MODEL_ENV[prov]) or DEFAULT_MODELS[prov],
        }
        for env, field in (
            ("LLM_TEMPERATURE", "temperature"),
            ("LLM_MAX_TOKENS", "max_tokens"),
            ("LLM_TIMEOUT", "timeout"),
            ("LLM_MAX_RETRIES", "max_retries"),
        ):
            if (value := os.getenv(env)) not in (None, ""):
                model_kwargs[field] = value  # Pydantic convierte y valida tipos/rangos
        extra: dict = {}
        if prov in BASE_URL_ENV and (value := os.getenv(BASE_URL_ENV[prov])):
            extra["base_url"] = value
        if (value := os.getenv("LLM_MAX_CONCURRENCY")) not in (None, ""):
            extra["max_concurrency"] = value
        return cls(
            provider=prov,
            api_key=SecretStr(key) if key else None,
            model=ModelConfig(**model_kwargs),
            **extra,
        )

    def require_api_key(self) -> str:
        """Devuelve la API key o lanza MissingAPIKeyError con un mensaje claro."""
        if self.api_key is None:
            raise MissingAPIKeyError(
                f"Falta {API_KEY_ENV[self.provider]} para el proveedor "
                f"'{self.provider}'. Configurala en el archivo .env"
            )
        return self.api_key.get_secret_value()


class AsyncLLMManager:
    """Punto de entrada único. Ejemplo:

        async with AsyncLLMManager.from_env("anthropic") as llm:
            resp = await llm.generate([ChatMessage(role="user", content="Hola")])
            async for chunk in llm.stream(...):
                print(chunk.text, end="")
    """

    def __init__(self, settings: ManagerSettings) -> None:
        key = settings.require_api_key()
        self.settings = settings
        self.client: BaseLLMClient = _REGISTRY[settings.provider](
            settings.model, api_key=key, base_url=settings.base_url
        )
        # Limita las llamadas simultáneas: con muchas tareas en paralelo
        # (asyncio.gather) se evita disparar el rate limit del proveedor.
        self._semaphore = asyncio.Semaphore(settings.max_concurrency)

    @classmethod
    def from_env(cls, provider: str | Provider | None = None) -> AsyncLLMManager:
        return cls(ManagerSettings.from_env(provider))

    @property
    def provider(self) -> Provider:
        return self.settings.provider

    async def generate(self, messages: Sequence[ChatMessage]) -> ModelResponse:
        async with self._semaphore:
            return await self.client.generate(messages)

    async def generate_many(
        self, conversations: Sequence[Sequence[ChatMessage]]
    ) -> list[ModelResponse]:
        """Procesa varias conversaciones en paralelo, respetando max_concurrency.

        El orden de las respuestas coincide con el de las conversaciones.
        """
        return list(await asyncio.gather(*(self.generate(c) for c in conversations)))

    async def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[StreamChunk]:
        async with self._semaphore:  # el cupo se mantiene mientras dura el stream
            async for chunk in self.client.stream(messages):
                yield chunk

    async def aclose(self) -> None:
        await self.client.aclose()

    async def __aenter__(self) -> AsyncLLMManager:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()
