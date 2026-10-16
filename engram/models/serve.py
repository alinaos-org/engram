"""A local OpenAI-compatible endpoint in front of any engram ``ChatModel``.

Third-party memory systems (Mem0, Letta, Zep/Graphiti) talk to an OpenAI-style API. This lets them
use the same models as the rest of the benchmark: subscription CLIs (``claude:``/``codex:``), a vLLM
endpoint, or a mock. It binds to 127.0.0.1 only and implements just enough of the API:

    GET  /v1/models
    POST /v1/chat/completions   (non-streaming; ``response_format`` json_object is honoured by
                                 instruction, and the reply is checked to be JSON when requested)

    engram-serve --model claude:claude-opus-5-5 --port 8100
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from engram.models.chat import ChatModel, Message

JSON_INSTRUCTION = (
    "Respond with a single valid JSON object only: no prose, no markdown code fences."
)


def _content(message: dict) -> str:
    """OpenAI content may be a string or a list of parts; keep the text parts."""
    c = message.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "\n".join(p.get("text", "") for p in c if isinstance(p, dict))
    return ""


def _extract_json(text: str) -> str:
    """Strip code fences and surrounding prose; return the first JSON object if there is one."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fenced:
        return fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass
    return text


def complete_openai(model: ChatModel, body: dict) -> dict:
    """Turn an OpenAI chat-completions request into a ``ChatModel`` call and back."""
    messages: list[Message] = [
        {"role": m.get("role", "user"), "content": _content(m)}
        for m in body.get("messages", [])
        if m.get("role") in {"system", "user", "assistant", "developer"}
    ]
    for m in messages:
        if m["role"] == "developer":
            m["role"] = "system"
    wants_json = (body.get("response_format") or {}).get("type") in {"json_object", "json_schema"}
    if wants_json:
        messages.insert(0, {"role": "system", "content": JSON_INSTRUCTION})
    max_tokens = int(body.get("max_tokens") or body.get("max_completion_tokens") or 1024)
    text = model.complete(messages, max_tokens=max_tokens)
    if wants_json:
        text = _extract_json(text)
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.get("model") or model.name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def make_server(model: ChatModel, port: int = 8100, host: str = "127.0.0.1") -> ThreadingHTTPServer:
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("engram-serve only binds to localhost")

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, payload: dict) -> None:
            data = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802
            if self.path.rstrip("/").endswith("/models"):
                self._send(200, {"object": "list", "data": [{"id": model.name, "object": "model"}]})
            else:
                self._send(404, {"error": {"message": "not found"}})

        def do_POST(self) -> None:  # noqa: N802
            if not self.path.rstrip("/").endswith("/chat/completions"):
                self._send(404, {"error": {"message": "only /v1/chat/completions is supported"}})
                return
            length = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
                if body.get("stream"):
                    self._send(400, {"error": {"message": "streaming is not supported"}})
                    return
                self._send(200, complete_openai(model, body))
            except Exception as e:  # report, don't crash the server
                self._send(500, {"error": {"message": f"{type(e).__name__}: {e}"}})

        def log_message(self, fmt: str, *args) -> None:  # quiet by default
            pass

    return ThreadingHTTPServer((host, port), Handler)


class BackgroundServer:
    """Run ``make_server`` in a daemon thread (used by tests and by in-process baselines)."""

    def __init__(self, model: ChatModel, port: int = 0) -> None:
        self.server = make_server(model, port)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        host, port = self.server.server_address[:2]
        return f"http://{host}:{port}/v1"

    def __enter__(self) -> BackgroundServer:
        self.thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.server.shutdown()
        self.server.server_close()


def main(argv: list[str] | None = None) -> None:
    from engram.cli import _model

    ap = argparse.ArgumentParser(prog="engram-serve", description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", help="model spec, e.g. claude:claude-opus-5-5")
    ap.add_argument("--mock", choices=["null", "parrot"])
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--api-key-env")
    ap.add_argument("--effort", default="low")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--port", type=int, default=8100)
    args = ap.parse_args(argv)
    model = _model(args)
    server = make_server(model, args.port)
    print(f"serving {model.name} at http://127.0.0.1:{args.port}/v1", flush=True)
    with contextlib.suppress(KeyboardInterrupt):
        server.serve_forever()


if __name__ == "__main__":
    main()
