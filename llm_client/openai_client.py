"""Cliente asíncrono para OpenAI (SDK oficial, AsyncOpenAI)."""

from __future__ import annotations

from collections.abc import AsyncIterator

import openai
from openai import AsyncOpenAI

from .base import BaseLLMClient
from .schemas import (
    ChatRequest,
    ModelConfig,
    ModelResponse,
    Provider,
    StreamChunk,
    Usage,
)


class OpenAIClient(BaseLLMClient):
    provider = Provider.OPENAI
    sdk = openai

    def __init__(self, config: ModelConfig, api_key: str | None = None) -> None:
        super().__init__(config)
        # max_retries=0: los reintentos los controla BaseLLMClient.
        self._client = AsyncOpenAI(
            api_key=api_key, timeout=config.timeout, max_retries=0
        )

    def _params(self, request: ChatRequest) -> dict:
        params: dict = {
            "model": self.config.model,
            "messages": [m.model_dump() for m in request.messages],
            "temperature": self.config.temperature,
            "max_completion_tokens": self.config.max_tokens,
        }
        if self.config.top_p is not None:
            params["top_p"] = self.config.top_p
        return params

    async def _generate_raw(self, request: ChatRequest) -> ModelResponse:
        # await sobre el cliente ASÍNCRONO: no bloquea el event loop.
        completion = await self._client.chat.completions.create(**self._params(request))
        choice = completion.choices[0]
        usage = (
            Usage(
                input_tokens=completion.usage.prompt_tokens,
                output_tokens=completion.usage.completion_tokens,
            )
            if completion.usage
            else None
        )
        return ModelResponse(
            provider=self.provider,
            model=completion.model,
            content=choice.message.content or "",
            finish_reason=choice.finish_reason,
            usage=usage,
        )

    async def _stream_raw(self, request: ChatRequest) -> AsyncIterator[StreamChunk]:
        stream = await self._client.chat.completions.create(
            **self._params(request),
            stream=True,
            stream_options={"include_usage": True},
        )
        usage: Usage | None = None
        async for event in stream:
            if event.usage:
                usage = Usage(
                    input_tokens=event.usage.prompt_tokens,
                    output_tokens=event.usage.completion_tokens,
                )
            if event.choices:
                delta = event.choices[0].delta.content
                if delta:
                    yield StreamChunk(text=delta)
        yield StreamChunk(done=True, usage=usage)

    def _is_permanent_rate_limit(self, exc: Exception) -> bool:
        # "insufficient_quota" es un 429 que NO se arregla reintentando.
        return getattr(exc, "code", None) == "insufficient_quota"

    async def aclose(self) -> None:
        await self._client.close()
