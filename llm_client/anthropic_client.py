"""Cliente asíncrono para Anthropic (SDK oficial, AsyncAnthropic)."""

from __future__ import annotations

import inspect
import logging
from collections.abc import AsyncIterator

import anthropic
from anthropic import AsyncAnthropic

from .base import BaseLLMClient
from .schemas import (
    ChatRequest,
    ErrorType,
    LLMError,
    ModelConfig,
    ModelResponse,
    Provider,
    StreamChunk,
    Usage,
)

logger = logging.getLogger("llm_client")

ANTHROPIC_MAX_TEMPERATURE = 1.0

# Desde la versión de API 2026-10-21 (SDK `anthropic` >= 2.0) la Messages API
# rechaza `temperature`, `top_p` y `top_k`, y el SDK los quitó de su firma.
# Se detecta en tiempo de ejecución para funcionar con SDK 1.x y 2.x.
SUPPORTS_SAMPLING_PARAMS = "temperature" in inspect.signature(
    anthropic.resources.messages.AsyncMessages.create
).parameters


class AnthropicClient(BaseLLMClient):
    provider = Provider.ANTHROPIC

    def __init__(
        self,
        config: ModelConfig,
        api_key: str | None = None,
        *,
        send_sampling_params: bool | None = None,
    ) -> None:
        super().__init__(config)
        self._client = AsyncAnthropic(
            api_key=api_key, timeout=config.timeout, max_retries=0
        )
        self._send_sampling = (
            SUPPORTS_SAMPLING_PARAMS if send_sampling_params is None else send_sampling_params
        )
        if not self._send_sampling:
            logger.info(
                "SDK anthropic >= 2.0: temperature/top_p no se envían "
                "(la API 2026-10-21 los rechaza; se usa el muestreo por defecto)."
            )

    def _params(self, request: ChatRequest) -> dict:
        # Anthropic recibe el system prompt aparte, no dentro de `messages`.
        system = "\n\n".join(m.content for m in request.messages if m.role == "system")
        messages = [
            {"role": m.role, "content": m.content}
            for m in request.messages
            if m.role != "system"
        ]
        params: dict = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": self.config.max_tokens,  # obligatorio en Anthropic
        }
        if system:
            params["system"] = system
        if self._send_sampling:
            # La API legacy acepta temperature en [0, 1]; la interfaz común
            # admite [0, 2], así que se ajusta al rango del proveedor.
            temperature = self.config.temperature
            if temperature > ANTHROPIC_MAX_TEMPERATURE:
                logger.warning(
                    "temperature=%.2f fuera de rango para Anthropic; se usa 1.0", temperature
                )
                temperature = ANTHROPIC_MAX_TEMPERATURE
            params["temperature"] = temperature
            if self.config.top_p is not None:
                params["top_p"] = self.config.top_p
        return params

    async def _generate_raw(self, request: ChatRequest) -> ModelResponse:
        message = await self._client.messages.create(**self._params(request))
        text = "".join(
            block.text for block in message.content if getattr(block, "type", "") == "text"
        )
        return ModelResponse(
            provider=self.provider,
            model=message.model,
            content=text,
            finish_reason=message.stop_reason,
            usage=Usage(
                input_tokens=message.usage.input_tokens,
                output_tokens=message.usage.output_tokens,
            ),
        )

    async def _stream_raw(self, request: ChatRequest) -> AsyncIterator[StreamChunk]:
        async with self._client.messages.stream(**self._params(request)) as stream:
            async for text in stream.text_stream:
                if text:
                    yield StreamChunk(text=text)
            final = await stream.get_final_message()
        yield StreamChunk(
            done=True,
            usage=Usage(
                input_tokens=final.usage.input_tokens,
                output_tokens=final.usage.output_tokens,
            ),
        )

    def _classify_error(self, exc: Exception) -> LLMError:
        status = getattr(exc, "status_code", None)
        if isinstance(exc, anthropic.APITimeoutError):
            etype, retry = ErrorType.TIMEOUT, True
        elif isinstance(exc, anthropic.APIConnectionError):
            etype, retry = ErrorType.CONNECTION, True
        elif isinstance(exc, anthropic.AuthenticationError | anthropic.PermissionDeniedError):
            etype, retry = ErrorType.AUTHENTICATION, False
        elif isinstance(exc, anthropic.RateLimitError):
            etype, retry = ErrorType.RATE_LIMIT, True
        elif isinstance(exc, anthropic.InternalServerError) or status == 529:
            # 529 = "overloaded" en Anthropic: vale la pena reintentar.
            etype, retry = ErrorType.SERVER, True
        elif isinstance(exc, anthropic.BadRequestError | anthropic.NotFoundError
                        | anthropic.UnprocessableEntityError):
            etype, retry = ErrorType.BAD_REQUEST, False
        else:
            etype, retry = ErrorType.UNKNOWN, False
        return LLMError(
            type=etype,
            message=str(exc) or exc.__class__.__name__,
            provider=self.provider,
            status_code=status,
            retryable=retry,
        )

    async def aclose(self) -> None:
        await self._client.close()
