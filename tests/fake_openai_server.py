"""Servidor HTTP mínimo compatible con la API de OpenAI (para tests).

Simula lo que haría Ollama en http://localhost:11434/v1: responde
/v1/chat/completions en modo normal y streaming (SSE) y guarda cada pedido
recibido, para verificar qué parámetros envía el cliente.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

RESPUESTA = "La entropía mide el desorden de un sistema."


class FakeOpenAIServer:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.paths: list[str] = []
        self.auth_headers: list[str | None] = []
        servidor = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:  # silencio en los tests
                pass

            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                servidor.requests.append(body)
                servidor.paths.append(self.path)
                servidor.auth_headers.append(self.headers.get("Authorization"))
                if body.get("stream"):
                    self._stream(body)
                else:
                    self._json(body)

            def _json(self, body: dict) -> None:
                data = json.dumps({
                    "id": "chatcmpl-1", "object": "chat.completion", "created": 0,
                    "model": body["model"],
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant", "content": RESPUESTA}}],
                    "usage": {"prompt_tokens": 12, "completion_tokens": 9, "total_tokens": 21},
                }).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _stream(self, body: dict) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                base = {"id": "c", "object": "chat.completion.chunk", "created": 0,
                        "model": body["model"]}
                for palabra in RESPUESTA.split(" "):
                    chunk = {**base, "choices": [{"index": 0, "delta": {"content": palabra + " "},
                                                  "finish_reason": None}]}
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                fin = {**base, "choices": [],
                       "usage": {"prompt_tokens": 12, "completion_tokens": 9, "total_tokens": 21}}
                self.wfile.write(f"data: {json.dumps(fin)}\n\ndata: [DONE]\n\n".encode())

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base_url = f"http://127.0.0.1:{self._httpd.server_address[1]}/v1"
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def __enter__(self) -> FakeOpenAIServer:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
