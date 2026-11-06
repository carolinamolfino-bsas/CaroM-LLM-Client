# CaroM-LLM-Client | Unified Async LLM Client
[![Python Tests](https://github.com/carolinamolfino-bsas/CaroM-LLM-Client/actions/workflows/tests.yml/badge.svg)](https://github.com/carolinamolfino-bsas/CaroM-LLM-Client/actions/workflows/tests.yml)
Cliente asíncrono unificado para **OpenAI** y **Anthropic** en Python 3.12: una única interfaz (`AsyncLLMManager`) con generación normal, **streaming** token a token, validación con **Pydantic** y manejo de errores controlado (sin crashes por API key inválida, rate limit o caída de red).

## Estructura

```
unified-llm-client/
├── llm_client/
│   ├── schemas.py           # Pydantic: ChatMessage, ModelConfig, ModelResponse, StreamChunk, LLMError
│   ├── base.py              # BaseLLMClient (ABC): generate(), stream(), reintentos con backoff
│   ├── openai_client.py     # OpenAIClient    -> AsyncOpenAI
│   ├── anthropic_client.py  # AnthropicClient -> AsyncAnthropic
│   └── manager.py           # AsyncLLMManager: elige proveedor según configuración
├── tests/test_client.py     # 20 tests offline (SDKs simulados, no gastan tokens)
├── main.py                  # Script de validación: modo normal + streaming
├── .env.example
├── requirements.txt
└── README.md
```

## Instalación

```bash
git clone https://github.com/carolinamolfino-bsas/CaroM-LLM-Client.git
cd CaroM-LLM-Client

python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt    # openai, anthropic, pydantic, python-dotenv (+ pytest)

cp .env.example .env               # Windows: copy .env.example .env
# editar .env y completar las API keys
```

## Variables de entorno

| Variable | Obligatoria | Default | Descripción |
|---|---|---|---|
| `LLM_PROVIDER` | No | `openai` | Proveedor por defecto: `openai` o `anthropic` |
| `OPENAI_API_KEY` | Si usás OpenAI | — | Clave de https://platform.openai.com |
| `ANTHROPIC_API_KEY` | Si usás Anthropic | — | Clave de https://console.anthropic.com |
| `OPENAI_MODEL` | No | `gpt-4o-mini` | Modelo de OpenAI |
| `ANTHROPIC_MODEL` | No | `claude-haiku-4-5` | Modelo de Anthropic |
| `LLM_TEMPERATURE` | No | `0.7` | 0 a 2 (validado) |
| `LLM_MAX_TOKENS` | No | `512` | Tokens máximos de salida (> 0) |
| `LLM_TIMEOUT` | No | `60` | Timeout por request, en segundos |
| `LLM_MAX_RETRIES` | No | `3` | Reintentos ante rate limit / red / 5xx |
| `LOG_LEVEL` | No | `WARNING` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |

Si un valor está fuera de rango (por ejemplo `LLM_TEMPERATURE=5`), Pydantic lo rechaza con un mensaje claro antes de llamar a la API. El archivo `.env` está en `.gitignore`: **nunca subas tus claves**.

## Ejecutar el script de prueba

```bash
python main.py                        # usa LLM_PROVIDER del .env
python main.py --provider anthropic   # fuerza un proveedor
python main.py --provider all         # prueba los dos
```

Pregunta *"¿Qué es la entropía?"* primero en modo normal (respuesta completa + tokens + latencia) y después en streaming (el texto aparece a medida que llega). Termina con un resumen OK/FALLÓ por proveedor (código de salida 0 si todo funcionó).

## Ejecutar los tests

```bash
python -m pytest -q
```

No necesitan API keys: simulan los SDKs para verificar validación, streaming, reintentos, errores controlados y que el event loop no se bloquee.

### Resultados de validación

El proyecto fue validado mediante 20 pruebas automatizadas utilizando `pytest`.

| Entorno | Versión de Python | Resultado | Tiempo |
|---|---|---|---|
| Windows (local) | 3.12.10 | 20/20 PASSED | 4.53 s |
| Ubuntu (GitHub Actions) | 3.12.16 | 20/20 PASSED | 1.54 s |

**Integración continua (CI):** GitHub Actions ejecuta automáticamente las pruebas con cada `push` o `pull request` a la rama `main`.

[Consultar las ejecuciones de GitHub Actions](https://github.com/carolinamolfino-bsas/CaroM-LLM-Client/actions)

**Alcance de la validación:** las pruebas automatizadas utilizan mocks de los SDK de OpenAI y Anthropic. Verifican la lógica de validación, generación, streaming, manejo de errores, reintentos y concurrencia sin consumir créditos de API.

**Limitación:** las llamadas reales a las APIs de OpenAI y Anthropic no fueron verificadas por no disponer de créditos de API.

## Uso como librería

```python
import asyncio
from dotenv import load_dotenv
from llm_client import AsyncLLMManager, ChatMessage

async def demo():
    msgs = [ChatMessage(role="user", content="¿Qué es la entropía?")]
    async with AsyncLLMManager.from_env("anthropic") as llm:
        resp = await llm.generate(msgs)
        print(resp.content if resp.ok else resp.error)

        async for chunk in llm.stream(msgs):
            print(chunk.text, end="", flush=True)
            if chunk.error:
                print("\nError:", chunk.error.type, chunk.error.message)

load_dotenv()
asyncio.run(demo())
```

También se puede configurar en código, sin `.env`:

```python
from llm_client import AsyncLLMManager, ManagerSettings, ModelConfig

settings = ManagerSettings(
    provider="openai",
    api_key="sk-...",
    model=ModelConfig(model="gpt-4o-mini", temperature=0.2, max_tokens=300),
)
llm = AsyncLLMManager(settings)
```

## Decisiones de diseño

- **Asincronía real.** Se usan `AsyncOpenAI` y `AsyncAnthropic` con `await`; nunca el cliente síncrono dentro de una corrutina. Un test verifica que dos llamadas en paralelo tardan lo mismo que una.
- **Streaming con `yield`.** `stream()` es un generador asíncrono que recorre el stream del SDK con `async for` y emite `StreamChunk` (`text`, y al final `done=True` con `usage` o `error`).
- **Errores controlados.** Las excepciones de los SDKs se clasifican en un `LLMError` (`authentication`, `rate_limit`, `connection`, `timeout`, `server`, `bad_request`, `unknown`). `generate()` devuelve `ModelResponse` con `error` cargado; `stream()` emite un chunk final con `error`. El programa nunca se cae.
- **Reintentos inteligentes.** Backoff exponencial con jitter (`asyncio.sleep`, no bloqueante) sólo para errores recuperables (429, red, timeout, 5xx, 529 *overloaded* de Anthropic). No se reintenta una API key inválida ni `insufficient_quota` de OpenAI (cuota agotada: reintentar no sirve). En streaming sólo se reintenta si todavía no se emitió texto, para no duplicar la salida. Los reintentos internos de los SDKs están desactivados (`max_retries=0`) para que la política sea una sola.
- **API key protegida.** Se guarda como `SecretStr`: no aparece en logs ni en `repr`. Si falta, se lanza `MissingAPIKeyError` con el nombre de la variable a configurar.
- **Diferencias entre proveedores** resueltas dentro de cada cliente: Anthropic recibe el `system` aparte y exige `max_tokens`; OpenAI usa `max_completion_tokens`.

### Nota sobre `temperature` en Anthropic

Desde la versión de API `2026-10-21` (SDK `anthropic` >= 2.0), la Messages API **rechaza** `temperature`, `top_p` y `top_k`, y el SDK los quitó de su firma. El cliente lo detecta automáticamente:

- **SDK `anthropic` 2.x:** no envía esos parámetros (se usa el muestreo por defecto del modelo). `LLM_TEMPERATURE` sigue validándose y se aplica a OpenAI.
- **SDK `anthropic` 1.x:** los envía, ajustando `temperature` al rango [0, 1] que acepta Anthropic.

**Compatibilidad:** la disponibilidad de los parámetros `temperature`, `top_p` y `top_k` depende de la versión de la API y del modelo de Anthropic utilizado. El cliente comprueba la firma del SDK para determinar qué parámetros puede enviar. Esta comprobación no garantiza por sí sola la compatibilidad con todos los modelos o versiones de la API. Se recomienda utilizar versiones actualizadas del SDK y verificar los parámetros admitidos antes de realizar llamadas reales.
