"""Clase base abstracta: la interfaz común que comparten todos los proveedores."""

from __future__ import annotations

import asyncio
import logging
import random
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from types import ModuleType

from .schemas import (
    ChatMessage,
    ChatRequest,
    ErrorType,
    LLMError,
    ModelConfig,
    ModelResponse,
    Provider,
    StreamChunk,
)

logger = logging.getLogger("llm_client")


class BaseLLMClient(ABC):
    """Interfaz común.

    Las subclases implementan:
      * `sdk`           -> el módulo del SDK oficial (`openai` o `anthropic`)
      * `_generate_raw` -> una llamada no-streaming al SDK
      * `_stream_raw`   -> el stream del SDK, traducido a StreamChunk

    Reintentos, validación, clasificación de errores y avisos viven acá, así no
    se duplican entre proveedores. Los dos SDKs oficiales exponen la misma
    jerarquía de excepciones (APITimeoutError, RateLimitError, ...), por eso
    una única función de clasificación sirve para ambos.
    """

    provider: Provider
    sdk: ModuleType

    def __init__(
        self, config: ModelConfig, api_key: str | None = None, base_url: str | None = None
    ) -> None:
        self.config = config

    # ---------- API pública ----------

    async def generate(self, messages: Sequence[ChatMessage]) -> ModelResponse:
        """Genera una respuesta completa. Nunca lanza excepciones de API:
        ante un fallo devuelve un ModelResponse con `error` cargado."""
        request = ChatRequest(messages=list(messages))
        warnings = self.param_warnings()
        attempt = 0
        while True:
            attempt += 1
            start = time.perf_counter()
            try:
                response = await self._generate_raw(request)
                response.latency_ms = round((time.perf_counter() - start) * 1000, 1)
                response.warnings = warnings
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
                    provider=self.provider, model=self.config.model,
                    error=error, warnings=warnings,
                )

    async def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[StreamChunk]:
        """Generador asíncrono: emite fragmentos de texto a medida que llegan.

        Reintenta sólo si el error ocurre ANTES del primer fragmento (si ya se
        emitió texto, reintentar duplicaría la salida). Siempre termina con un
        chunk `done=True`, con o sin `error`.
        """
        request = ChatRequest(messages=list(messages))
        warnings = self.param_warnings()
        attempt = 0
        while True:
            attempt += 1
            emitted = False
            try:
                async for chunk in self._stream_raw(request):
                    if chunk.text:
                        emitted = True
                    if chunk.done:
                        chunk.warnings = warnings
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
                yield StreamChunk(done=True, error=error, warnings=warnings)
                return

    def param_warnings(self) -> list[str]:
        """Parámetros pedidos por el usuario que este proveedor no aplicará."""
        return []

    async def aclose(self) -> None:  # noqa: B027 - hook opcional, no abstracto
        """Cierra el cliente HTTP subyacente."""

    async def __aenter__(self) -> BaseLLMClient:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    # ---------- a implementar por cada proveedor ----------

    @abstractmethod
    async def _generate_raw(self, request: ChatRequest) -> ModelResponse: ...

    @abstractmethod
    def _stream_raw(self, request: ChatRequest) -> AsyncIterator[StreamChunk]: ...

    # ---------- clasificación de errores (común) ----------

    def _is_permanent_rate_limit(self, exc: Exception) -> bool:
        """429 que NO se arregla reintentando (ej. cuota agotada). Hook opcional."""
        return False

    def _classify_error(self, exc: Exception) -> LLMError:
        sdk = self.sdk
        status: int | None = getattr(exc, "status_code", None)
        retry = False
        # El orden importa: APITimeoutError hereda de APIConnectionError.
        if isinstance(exc, sdk.APITimeoutError):
            etype, retry = ErrorType.TIMEOUT, True
        elif isinstance(exc, sdk.APIConnectionError):
            etype, retry = ErrorType.CONNECTION, True
        elif isinstance(exc, sdk.AuthenticationError | sdk.PermissionDeniedError):
            etype = ErrorType.AUTHENTICATION
        elif isinstance(exc, sdk.RateLimitError):
            etype, retry = ErrorType.RATE_LIMIT, not self._is_permanent_rate_limit(exc)
        elif isinstance(exc, sdk.InternalServerError) or (status is not None and status >= 500):
            # Incluye el 529 "overloaded" de Anthropic.
            etype, retry = ErrorType.SERVER, True
        elif isinstance(exc, sdk.BadRequestError | sdk.NotFoundError | sdk.UnprocessableEntityError):
            etype = ErrorType.BAD_REQUEST
        else:
            etype = ErrorType.UNKNOWN
        return LLMError(
            type=etype,
            message=str(exc) or exc.__class__.__name__,
            provider=self.provider,
            status_code=status,
            retryable=retry,
        )

    # ---------- utilidades ----------

    async def _backoff(self, attempt: int, error: LLMError) -> None:
        """Backoff exponencial con jitter (no bloquea el event loop)."""
        delay = min(2 ** (attempt - 1), 20) + random.uniform(0, 0.5)
        logger.warning(
            "[%s] %s (intento %d/%d). Reintentando en %.1fs...",
            self.provider, error.type, attempt, self.config.max_retries + 1, delay,
        )
        await asyncio.sleep(delay)
