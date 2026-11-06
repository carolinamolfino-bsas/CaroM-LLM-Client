"""Script de validación: pregunta "¿Qué es la entropía?" en modo normal y streaming.

Uso:
    python main.py                      # usa LLM_PROVIDER del .env (default: openai)
    python main.py --provider anthropic
    python main.py --provider all       # prueba ambos proveedores
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

from dotenv import load_dotenv
from pydantic import ValidationError

from llm_client import AsyncLLMManager, ChatMessage, MissingAPIKeyError, Provider

QUESTION = "¿Qué es la entropía?"

MESSAGES = [
    ChatMessage(role="system", content="Respondé en español, en no más de 3 oraciones."),
    ChatMessage(role="user", content=QUESTION),
]


def banner(text: str) -> None:
    print(f"\n{'=' * 64}\n{text}\n{'=' * 64}")


async def run_provider(provider: Provider) -> bool:
    """Corre ambos modos para un proveedor. Devuelve True si los dos funcionaron."""
    try:
        manager = AsyncLLMManager.from_env(provider)
    except ValidationError as exc:
        print(f"[{provider}] Configuración inválida en .env:\n{exc}")
        return False
    except MissingAPIKeyError as exc:
        print(f"[{provider}] {exc}. Se omite este proveedor.")
        return False

    ok = True
    async with manager:
        model = manager.settings.model.model

        # ---- Modo normal ----
        banner(f"{provider.upper()} · {model} · modo normal")
        resp = await manager.generate(MESSAGES)
        if resp.ok:
            print(resp.content)
            if resp.usage:
                print(
                    f"\n[tokens in={resp.usage.input_tokens} "
                    f"out={resp.usage.output_tokens} · {resp.latency_ms} ms]"
                )
        else:
            ok = False
            e = resp.error
            print(f"ERROR CONTROLADO -> {e.type} (HTTP {e.status_code}, "
                  f"intentos={e.attempts}): {e.message[:200]}")

        # ---- Modo streaming ----
        banner(f"{provider.upper()} · {model} · modo streaming")
        async for chunk in manager.stream(MESSAGES):
            if chunk.text:
                print(chunk.text, end="", flush=True)
            if chunk.error:
                ok = False
                e = chunk.error
                print(f"\nERROR CONTROLADO -> {e.type} (HTTP {e.status_code}, "
                      f"intentos={e.attempts}): {e.message[:200]}")
            elif chunk.done and chunk.usage:
                print(f"\n\n[tokens in={chunk.usage.input_tokens} "
                      f"out={chunk.usage.output_tokens}]")
    return ok


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--provider",
        choices=[p.value for p in Provider] + ["all"],
        default=None,
        help="Proveedor a probar (default: LLM_PROVIDER del .env)",
    )
    args = parser.parse_args()

    selected = args.provider or os.getenv("LLM_PROVIDER", "openai").lower()
    providers = list(Provider) if selected == "all" else [Provider(selected)]

    results = {p: await run_provider(p) for p in providers}

    banner("Resumen")
    for p, ok in results.items():
        print(f"  {p.value:<10} {'OK' if ok else 'FALLÓ (ver arriba)'}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    load_dotenv()
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "WARNING"),
        format="%(levelname)s %(name)s: %(message)s",
    )
    sys.exit(asyncio.run(main()))
