"""OpenAI-compatible /v1/chat/completions over a local VL adapter.

Used by `make serve` and as the request handler for Modal's web endpoint.
Loads 4-bit Qwen3.8-27B + PEFT; needs a GPU.
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .infer import generate_from_png, load_vl_adapter, openai_compat_reply, parse_chat_request

_MODEL = None
_PROCESSOR = None
_ADAPTER = ""


class Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        if self.path.rstrip("/") not in ("/v1/chat/completions", "/chat/completions"):
            self.send_error(404)
            return
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n).decode())
        parsed = parse_chat_request(body)
        content = generate_from_png(_MODEL, _PROCESSOR, parsed["image"])
        payload = json.dumps(openai_compat_reply(parsed["model"], content)).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt, *args) -> None:
        return


def main() -> None:
    global _MODEL, _PROCESSOR, _ADAPTER
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter-path", default="adapters/champion")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    _ADAPTER = args.adapter_path
    _MODEL, _PROCESSOR = load_vl_adapter(args.adapter_path)
    print(f"serving {args.adapter_path} on http://{args.host}:{args.port}/v1")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
