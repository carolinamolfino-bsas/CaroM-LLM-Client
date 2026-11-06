"""Clase base abstracta: la interfaz común que comparten todos los proveedores."""

from __future__ import annotations

import asyncio
import logging
import random
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence

from .schemas import (
    ChatMessage,
    ChatRequest,
    LLMError,
    ModelConfig,
    ModelResponse,
    Provider,
    StreamChunk,
)

logger = logging.getLogger("llm_client")


class BaseLLMClient(ABC):
    """Interfaz común.

    Las subclases sólo implementan:
      * `_generate_raw`  -> una llamada no-streaming al SDK
      * `_stream_raw`    -> el stream del SDK, traducido a StreamChunk
      * `_classify_error`-> mapear excepciones del SDK a LLMError

    El manejo de reintentos, validación y errores controlados vive acá,
    así no se duplica entre proveedores.
    """

    provider: Provider

    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    # ---------- API pública ----------

    async def generate(self, messages: Sequence[ChatMessage]) -> ModelResponse:
        """Genera una respuesta completa. Nunca lanza excepciones de API:
        ante un fallo devuelve un ModelResponse con `error` cargado."""
        request = ChatRequest(messages=list(messages))
        attempt = 0
        while True:
            attempt += 1
            start = time.perf_counter()
            try:
                response = await self._generate_raw(request)
                response.latency_ms = round((time.perf_counter() - start) * 1000, 1)
                return response
            except asyncio.CancelledError:
                raise  # nunca tragarse una cancelación del event loop
            except Exception as exc:  # noqa: BLE001 - se clasifica abajo
                error = self._classify_error(exc)
                error.attempts = attempt
                if error.retryable and attempt <= self.config.max_retries:
                    await self._backoff(attempt, error)
                    continue
                logger.error("[%s] %s: %s", self.provider, error.type, error.message)
                return ModelResponse(
                    provider=self.provider, model=self.config.model, error=error
                )

    async def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[StreamChunk]:
        """Generador asíncrono: emite fragmentos de texto a medida que llegan.

        Reintenta sólo si el error ocurre ANTES del primer fragmento (si ya se
        emitió texto, reintentar duplicaría la salida). Siempre termina con un
        chunk `done=True`, con o sin `error`.
        """
        request = ChatRequest(messages=list(messages))
        attempt = 0
        while True:
            attempt += 1
            emitted = False
            try:
                async for chunk in self._stream_raw(request):
                    if chunk.text:
                        emitted = True
                    yield chunk
                return
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                error = self._classify_error(exc)
                error.attempts = attempt
                if error.retryable and not emitted and attempt <= self.config.max_retries:
                    await self._backoff(attempt, error)
                    continue
                logger.error("[%s] stream %s: %s", self.provider, error.type, error.message)
                yield StreamChunk(done=True, error=error)
                return

    async def aclose(self) -> None:
        """Cierra el cliente HTTP subyacente."""

    async def __aenter__(self) -> "BaseLLMClient":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    # ---------- a implementar por cada proveedor ----------

    @abstractmethod
    async def _generate_raw(self, request: ChatRequest) -> ModelResponse: ...

    @abstractmethod
    def _stream_raw(self, request: ChatRequest) -> AsyncIterator[StreamChunk]: ...

    @abstractmethod
    def _classify_error(self, exc: Exception) -> LLMError: ...

    # ---------- utilidades ----------

    async def _backoff(self, attempt: int, error: LLMError) -> None:
        """Backoff exponencial con jitter (no bloquea el event loop)."""
        delay = min(2 ** (attempt - 1), 20) + random.uniform(0, 0.5)
        logger.warning(
            "[%s] %s (intento %d/%d). Reintentando en %.1fs...",
            self.provider, error.type, attempt, self.config.max_retries + 1, delay,
        )
        await asyncio.sleep(delay)
