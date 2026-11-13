"""Tests offline (sin API keys reales): se simulan los SDKs con dobles de prueba.

Ejecutar:  python -m pytest -q
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import anthropic
import openai

try:  # SDKs recientes usan httpx2; versiones anteriores, httpx
    import httpx2 as httpx
except ImportError:  # pragma: no cover
    import httpx
import pytest
from pydantic import ValidationError

from llm_client import (
    AnthropicClient,
    AsyncLLMManager,
    ChatMessage,
    ErrorType,
    ManagerSettings,
    ModelConfig,
    OpenAIClient,
    Provider,
)

MSGS = [ChatMessage(role="user", content="¿Qué es la entropía?")]
REQ = httpx.Request("POST", "https://example.test")


def run(coro):
    return asyncio.run(coro)


async def _no_sleep(*_a, **_k):
    return None


def http_error(cls, status: int, body=None):
    return cls("simulated", response=httpx.Response(status, request=REQ), body=body)


# ---------------------------------------------------------------- schemas

def test_temperature_range():
    ModelConfig(model="x", temperature=0)
    ModelConfig(model="x", temperature=2)
    with pytest.raises(ValidationError):
        ModelConfig(model="x", temperature=2.1)
    with pytest.raises(ValidationError):
        ModelConfig(model="x", temperature=-0.1)


def test_max_tokens_positive():
    with pytest.raises(ValidationError):
        ModelConfig(model="x", max_tokens=0)


def test_message_role_and_content():
    with pytest.raises(ValidationError):
        ChatMessage(role="robot", content="hola")
    with pytest.raises(ValidationError):
        ChatMessage(role="user", content="   ")


def test_unknown_config_field_rejected():
    with pytest.raises(ValidationError):
        ModelConfig(model="x", temprature=1)  # typo detectado


# ---------------------------------------------------------------- manager

def test_manager_selects_provider(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    assert isinstance(AsyncLLMManager.from_env("openai").client, OpenAIClient)
    assert isinstance(AsyncLLMManager.from_env("anthropic").client, AnthropicClient)


def test_missing_key_raises_clear_error(monkeypatch):
    from llm_client import MissingAPIKeyError

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(MissingAPIKeyError):
        AsyncLLMManager.from_env("openai")


def test_settings_from_env_validates(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.3")
    s = ManagerSettings.from_env()
    assert s.provider is Provider.ANTHROPIC and s.model.temperature == 0.3
    monkeypatch.setenv("LLM_TEMPERATURE", "5")
    with pytest.raises(ValidationError):
        ManagerSettings.from_env()


def test_api_key_not_leaked_in_repr(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-super-secret")
    assert "sk-super-secret" not in repr(ManagerSettings.from_env("openai"))


# ---------------------------------------------------------------- OpenAI

def _openai_client(fake_create, retries=2) -> OpenAIClient:
    c = OpenAIClient(ModelConfig(model="gpt-test", max_retries=retries), api_key="sk-test")
    c._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))
    )
    c._backoff = _no_sleep
    return c


def test_openai_generate_ok():
    async def create(**kw):
        assert kw["max_completion_tokens"] == 512
        return SimpleNamespace(
            model="gpt-test",
            choices=[SimpleNamespace(message=SimpleNamespace(content="Medida del desorden."),
                                     finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=5, completion_tokens=4),
        )

    r = run(_openai_client(create).generate(MSGS))
    assert r.ok and r.content == "Medida del desorden." and r.usage.output_tokens == 4


def test_openai_stream_yields_fragments():
    async def gen():
        for t in ["La ", "entropía ", "mide..."]:
            yield SimpleNamespace(usage=None,
                                  choices=[SimpleNamespace(delta=SimpleNamespace(content=t))])
        yield SimpleNamespace(usage=SimpleNamespace(prompt_tokens=3, completion_tokens=3),
                              choices=[])

    async def create(**kw):
        assert kw["stream"] is True
        return gen()

    async def collect():
        return [c async for c in _openai_client(create).stream(MSGS)]

    chunks = run(collect())
    assert "".join(c.text for c in chunks) == "La entropía mide..."
    assert chunks[-1].done and chunks[-1].usage.output_tokens == 3


def test_openai_invalid_key_returns_controlled_error():
    calls = 0

    async def create(**kw):
        nonlocal calls
        calls += 1
        raise http_error(openai.AuthenticationError, 401)

    r = run(_openai_client(create).generate(MSGS))
    assert not r.ok and r.error.type is ErrorType.AUTHENTICATION
    assert calls == 1  # no se reintenta un error de autenticación


def test_openai_rate_limit_retries_then_succeeds():
    calls = 0

    async def create(**kw):
        nonlocal calls
        calls += 1
        if calls < 3:
            raise http_error(openai.RateLimitError, 429)
        return SimpleNamespace(
            model="gpt-test",
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"), finish_reason="stop")],
            usage=None,
        )

    r = run(_openai_client(create, retries=3).generate(MSGS))
    assert r.ok and calls == 3


def test_openai_rate_limit_exhausts_retries():
    async def create(**kw):
        raise http_error(openai.RateLimitError, 429)

    r = run(_openai_client(create, retries=2).generate(MSGS))
    assert r.error.type is ErrorType.RATE_LIMIT and r.error.attempts == 3


def test_openai_insufficient_quota_not_retried():
    calls = 0

    async def create(**kw):
        nonlocal calls
        calls += 1
        raise http_error(openai.RateLimitError, 429,
                         body={"code": "insufficient_quota", "message": "quota"})

    r = run(_openai_client(create, retries=3).generate(MSGS))
    assert r.error.type is ErrorType.RATE_LIMIT and calls == 1


def test_openai_network_error_in_stream_is_controlled():
    async def create(**kw):
        raise openai.APIConnectionError(request=REQ)

    async def collect():
        return [c async for c in _openai_client(create, retries=1).stream(MSGS)]

    chunks = run(collect())
    assert len(chunks) == 1 and chunks[0].done
    assert chunks[0].error.type is ErrorType.CONNECTION and chunks[0].error.attempts == 2


# ---------------------------------------------------------------- Anthropic

class _FakeAnthropicStream:
    def __init__(self, parts):
        self._parts = parts

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    @property
    def text_stream(self):
        async def g():
            for p in self._parts:
                yield p
        return g()

    async def get_final_message(self):
        return SimpleNamespace(usage=SimpleNamespace(input_tokens=7, output_tokens=2))


def _anthropic_client(create=None, stream=None, temperature=0.7, retries=2,
                      sampling=True) -> AnthropicClient:
    c = AnthropicClient(ModelConfig(model="claude-test", temperature=temperature,
                                    max_retries=retries), api_key="sk-ant-test",
                        send_sampling_params=sampling)
    c._client = SimpleNamespace(messages=SimpleNamespace(create=create, stream=stream))
    c._backoff = _no_sleep
    return c


def test_anthropic_system_prompt_and_temperature_clamp():
    seen = {}

    async def create(**kw):
        seen.update(kw)
        return SimpleNamespace(
            model="claude-test", stop_reason="end_turn",
            content=[SimpleNamespace(type="text", text="Desorden.")],
            usage=SimpleNamespace(input_tokens=10, output_tokens=2),
        )

    msgs = [ChatMessage(role="system", content="Sé breve."), *MSGS]
    r = run(_anthropic_client(create=create, temperature=1.8).generate(msgs))
    assert r.ok and r.content == "Desorden."
    assert seen["system"] == "Sé breve."
    assert all(m["role"] != "system" for m in seen["messages"])
    assert seen["temperature"] == 1.0


def test_anthropic_sdk2_omits_sampling_params():
    seen = {}

    async def create(**kw):
        seen.update(kw)
        return SimpleNamespace(
            model="claude-test", stop_reason="end_turn",
            content=[SimpleNamespace(type="text", text="ok")],
            usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        )

    r = run(_anthropic_client(create=create, sampling=False).generate(MSGS))
    assert r.ok and "temperature" not in seen and "top_p" not in seen


def test_anthropic_stream():
    def stream(**kw):
        return _FakeAnthropicStream(["Hola", " mundo"])

    async def collect():
        return [c async for c in _anthropic_client(stream=stream).stream(MSGS)]

    chunks = run(collect())
    assert "".join(c.text for c in chunks) == "Hola mundo"
    assert chunks[-1].done and chunks[-1].usage.input_tokens == 7


def test_anthropic_overloaded_529_is_retryable():
    async def create(**kw):
        raise anthropic.APIStatusError(
            "overloaded", response=httpx.Response(529, request=REQ), body=None
        )

    r = run(_anthropic_client(create=create, retries=1).generate(MSGS))
    assert r.error.type is ErrorType.SERVER and r.error.attempts == 2


def test_event_loop_not_blocked():
    """Dos llamadas 'lentas' en paralelo tardan ~1x, no ~2x."""
    async def create(**kw):
        await asyncio.sleep(0.3)
        return SimpleNamespace(
            model="gpt-test",
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"), finish_reason="stop")],
            usage=None,
        )

    async def both():
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        c = _openai_client(create)
        await asyncio.gather(c.generate(MSGS), c.generate(MSGS))
        return loop.time() - t0

    assert run(both()) < 0.5

def test_invalid_provider_is_handled():
    """Un proveedor inválido debe generar un error controlado."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "main.py", "--provider", "google"],
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode != 0
    assert "invalid choice" in result.stderr or "no reconocido" in result.stdout
    assert "Traceback" not in result.stderr


# ---------------------------------------------------------------- v1.1

def test_main_runs_providers_without_keys():
    """Regresión: main.py debe llegar a ejecutar los proveedores (no cortar antes)."""
    import os
    import subprocess
    import sys

    env = {**os.environ, "OPENAI_API_KEY": "", "ANTHROPIC_API_KEY": ""}
    result = subprocess.run(
        [sys.executable, "main.py", "--provider", "all"],
        capture_output=True, text=True, timeout=30, env=env,
    )
    assert "Falta OPENAI_API_KEY" in result.stdout
    assert "Falta ANTHROPIC_API_KEY" in result.stdout
    assert "Resumen" in result.stdout
    assert "Traceback" not in result.stderr


def test_missing_key_message_is_not_provider_error(monkeypatch):
    """MissingAPIKeyError hereda de ValueError: no debe confundirse con proveedor inválido."""
    from llm_client import MissingAPIKeyError

    assert issubclass(MissingAPIKeyError, ValueError)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(MissingAPIKeyError, match="ANTHROPIC_API_KEY"):
        AsyncLLMManager.from_env("anthropic")


def test_semaphore_limits_concurrency(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_MAX_CONCURRENCY", "2")
    manager = AsyncLLMManager.from_env("openai")
    active = peak = 0

    async def create(**kw):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.05)
        active -= 1
        return SimpleNamespace(
            model="gpt-test",
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"), finish_reason="stop")],
            usage=None,
        )

    manager.client._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    responses = run(manager.generate_many([MSGS] * 6))
    assert len(responses) == 6 and all(r.ok for r in responses)
    assert peak == 2


def test_anthropic_warns_when_temperature_ignored():
    async def create(**kw):
        assert "temperature" not in kw
        return SimpleNamespace(
            model="claude-test", stop_reason="end_turn",
            content=[SimpleNamespace(type="text", text="ok")],
            usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        )

    c = AnthropicClient(ModelConfig(model="claude-test", temperature=0.2),
                        api_key="sk-ant-test", send_sampling_params=False)
    c._client = SimpleNamespace(messages=SimpleNamespace(create=create))
    r = run(c.generate(MSGS))
    assert r.ok and any("temperature" in w for w in r.warnings)


def test_no_warning_when_temperature_not_requested():
    c = AnthropicClient(ModelConfig(model="claude-test"), api_key="k", send_sampling_params=False)
    assert c.param_warnings() == []


def test_shared_error_classification_for_both_sdks():
    oc = OpenAIClient(ModelConfig(model="m"), api_key="k")
    ac = AnthropicClient(ModelConfig(model="m"), api_key="k")
    assert oc._classify_error(http_error(openai.InternalServerError, 503)).retryable
    assert ac._classify_error(http_error(anthropic.RateLimitError, 429)).type is ErrorType.RATE_LIMIT
    assert not ac._classify_error(http_error(anthropic.AuthenticationError, 401)).retryable


# ---------------------------------------------------------------- v1.2: OPENAI_BASE_URL

def test_base_url_from_env_and_validation(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1/")
    assert ManagerSettings.from_env("openai").base_url == "http://localhost:11434/v1"
    assert ManagerSettings.from_env("anthropic").base_url is None  # sólo aplica a OpenAI
    monkeypatch.setenv("OPENAI_BASE_URL", "localhost:11434")
    with pytest.raises(ValidationError):
        ManagerSettings.from_env("openai")


def test_openai_official_uses_max_completion_tokens():
    from llm_client import ChatRequest

    c = OpenAIClient(ModelConfig(model="gpt-test", max_tokens=99), api_key="k")
    params = c._params(ChatRequest(messages=MSGS))
    assert params["max_completion_tokens"] == 99 and "max_tokens" not in params


def test_compatible_server_end_to_end_over_http(monkeypatch):
    """Llamada HTTP real a un servidor local compatible con OpenAI (como Ollama)."""
    from tests.fake_openai_server import RESPUESTA, FakeOpenAIServer

    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    with FakeOpenAIServer() as server:
        monkeypatch.setenv("OPENAI_BASE_URL", server.base_url)
        monkeypatch.setenv("OPENAI_API_KEY", "ollama")
        monkeypatch.setenv("OPENAI_MODEL", "qwen2.5:3b")
        monkeypatch.setenv("LLM_MAX_TOKENS", "200")

        async def flujo():
            async with AsyncLLMManager.from_env("openai") as llm:
                resp = await llm.generate(MSGS)
                chunks = [c async for c in llm.stream(MSGS)]
            return resp, chunks

        resp, chunks = run(flujo())

    assert resp.ok and resp.content == RESPUESTA and resp.usage.output_tokens == 9
    assert "".join(c.text for c in chunks).strip() == RESPUESTA
    assert chunks[-1].done and chunks[-1].usage.input_tokens == 12
    assert server.paths == ["/v1/chat/completions"] * 2
    assert server.auth_headers[0] == "Bearer ollama"
    normal, streaming = server.requests
    assert normal["model"] == "qwen2.5:3b"
    assert normal["max_tokens"] == 200 and "max_completion_tokens" not in normal
    assert streaming["stream"] is True and streaming["stream_options"] == {"include_usage": True}
