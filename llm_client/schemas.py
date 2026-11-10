"""Modelos Pydantic: mensajes, configuración del modelo, respuestas y errores.

Definir todo con Pydantic evita el "error de diccionarios anidados": cada dato
que entra o sale del cliente tiene una forma validada y tipada.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Provider(StrEnum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class ChatMessage(BaseModel):
    """Un mensaje de la conversación."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class ModelConfig(BaseModel):
    """Parámetros de generación, validados antes de llamar a la API."""

    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1, description="Nombre del modelo del proveedor")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, gt=0, le=32_000)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    timeout: float = Field(default=60.0, gt=0, description="Timeout por request (s)")
    max_retries: int = Field(default=3, ge=0, le=10)


class ChatRequest(BaseModel):
    """Entrada completa de una generación."""

    messages: list[ChatMessage] = Field(min_length=1)

    @field_validator("messages")
    @classmethod
    def _needs_user_message(cls, msgs: list[ChatMessage]) -> list[ChatMessage]:
        if not any(m.role == "user" for m in msgs):
            raise ValueError("Debe haber al menos un mensaje con role='user'")
        return msgs


class ErrorType(StrEnum):
    AUTHENTICATION = "authentication"   # API key inválida
    RATE_LIMIT = "rate_limit"           # 429 / cuota agotada
    CONNECTION = "connection"           # red caída, DNS, etc.
    TIMEOUT = "timeout"
    SERVER = "server"                   # 5xx del proveedor
    BAD_REQUEST = "bad_request"         # 400/404/422: modelo inexistente, parámetros
    UNKNOWN = "unknown"


class LLMError(BaseModel):
    """Error controlado: se devuelve en lugar de propagar la excepción."""

    type: ErrorType
    message: str
    provider: Provider
    status_code: int | None = None
    retryable: bool = False
    attempts: int = 1


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class ModelResponse(BaseModel):
    """Respuesta unificada, igual para cualquier proveedor."""

    provider: Provider
    model: str
    content: str = ""
    finish_reason: str | None = None
    usage: Usage | None = None
    latency_ms: float | None = None
    error: LLMError | None = None
    warnings: list[str] = Field(
        default_factory=list,
        description="Avisos no fatales, ej. parámetros que el proveedor ignoró.",
    )

    @property
    def ok(self) -> bool:
        return self.error is None


class StreamChunk(BaseModel):
    """Fragmento emitido por el generador de streaming.

    - Fragmento normal: `text` con el delta de tokens.
    - Fin del stream: `done=True` (puede traer `usage`).
    - Falla: `error` con el detalle y `done=True`.
    - `warnings` viaja en el chunk final (igual que en ModelResponse).
    """

    text: str = ""
    done: bool = False
    usage: Usage | None = None
    error: LLMError | None = None
    warnings: list[str] = Field(default_factory=list)
