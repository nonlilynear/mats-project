#!/usr/bin/env python3
"""Expose the BBF template contract and proxy OpenAI requests to vLLM."""

from __future__ import annotations

import argparse
import hashlib
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib import error, request


HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}


def _json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")


class AttestedProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _contract(self) -> None:
        payload = _json_bytes(self.server.template_contract)  # type: ignore[attr-defined]
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(payload)

    def _forward(self) -> None:
        upstream = self.server.upstream.rstrip("/")  # type: ignore[attr-defined]
        body_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(body_length) if body_length else None
        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in {"host", "content-length", *HOP_BY_HOP}
        }
        upstream_request = request.Request(
            f"{upstream}{self.path}",
            data=body,
            headers=headers,
            method=self.command,
        )
        try:
            with request.urlopen(upstream_request, timeout=600) as response:
                status = response.status
                response_headers = dict(response.headers.items())
                response_body = response.read()
        except error.HTTPError as exc:
            status = exc.code
            response_headers = dict(exc.headers.items()) if exc.headers else {}
            response_body = exc.read()
        except (OSError, error.URLError) as exc:
            status = 502
            response_headers = {"Content-Type": "text/plain; charset=utf-8"}
            response_body = f"upstream vLLM request failed: {exc}\n".encode("utf-8")
        self.send_response(status)
        for key, value in response_headers.items():
            if key.lower() not in {"content-length", *HOP_BY_HOP}:
                self.send_header(key, value)
        self.send_header("Content-Length", str(len(response_body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(response_body)

    def _dispatch(self) -> None:
        if self.path.split("?", 1)[0] == "/bbf/template-contract" and self.command == "GET":
            self._contract()
        else:
            self._forward()

    do_GET = _dispatch
    do_POST = _dispatch
    do_OPTIONS = _dispatch

    def log_message(self, format: str, *args: object) -> None:
        print(f"bbf-attested-proxy: {self.address_string()} - {format % args}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen", default="127.0.0.1:18080")
    parser.add_argument("--upstream", default="http://127.0.0.1:18000")
    parser.add_argument("--template-path", type=Path, required=True)
    parser.add_argument("--contract-version", default="qwen36-upstream-v1")
    args = parser.parse_args()
    host, port_text = args.listen.rsplit(":", 1)
    template_sha256 = hashlib.sha256(args.template_path.read_bytes()).hexdigest()
    server = ThreadingHTTPServer((host, int(port_text)), AttestedProxyHandler)
    server.upstream = args.upstream
    server.template_contract = {
        "contract_version": args.contract_version,
        "template_path": args.template_path.name,
        "template_sha256": template_sha256,
        "verification_method": "endpoint_contract",
    }
    print(json.dumps(server.template_contract, sort_keys=True), flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
