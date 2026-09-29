"""Web endpoint: upload a menu photo, review the agent's plan, approve, see the rebuilt page.

Runs as an AWS Lambda function behind a Function URL (`handler`) or locally
(`python webapp.py`, http://localhost:8080). It is stateless: the site config travels
with each request, so nothing a visitor uploads is stored. Every run logs its trace as one
JSON line, which CloudWatch keeps.

    POST /plan   {"photo": <base64 JPEG/PNG> | null for the demo photo, "config": <yaml> | null}
    POST /apply  {"config": <yaml> | null, "approved": [<change>, ...]}
"""
from __future__ import annotations

import base64
import importlib.util
import json
import os
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2 as cv
import numpy as np
import yaml

from menuvision import agent
from menuvision.ocr import Reader

HERE = Path(__file__).resolve().parent
DEMO = HERE / "demo"
# A standalone checkout vendors the builder in sitebuilder/; the monorepo keeps it in products/.
BUILDER = next(p for p in (HERE / "sitebuilder/build.py",
                           HERE.parents[1] / "products/static-site/template/build.py") if p.exists())
MAX_PHOTO = 4_000_000  # bytes after decoding; a Function URL caps the request at 6 MB
MAX_CONFIG = 50_000

_reader: Reader | None = None


def reader() -> Reader:
    global _reader  # the models load once per warm Lambda container
    if _reader is None:
        _reader = Reader()
    return _reader


def builder():
    spec = importlib.util.spec_from_file_location("site_build", BUILDER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BadRequest(Exception):
    pass


def load_config(text: str | None) -> dict[str, Any]:
    if text is None:
        text = (DEMO / "site.yaml").read_text(encoding="utf-8")
    if len(text) > MAX_CONFIG:
        raise BadRequest("config too large")
    try:
        cfg = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise BadRequest(f"config is not valid YAML: {error}") from None
    if not isinstance(cfg, dict) or not isinstance(cfg.get("pages"), list):
        raise BadRequest("config needs a pages list")
    return cfg


def decode_photo(data: str | None) -> np.ndarray:
    raw = (DEMO / "menu.jpg").read_bytes() if data is None else base64.b64decode(data, validate=True)
    if len(raw) > MAX_PHOTO:
        raise BadRequest("photo larger than 4 MB")
    photo = cv.imdecode(np.frombuffer(raw, np.uint8), cv.IMREAD_COLOR)
    if photo is None:
        raise BadRequest("not a JPEG or PNG image")
    longest = max(photo.shape[:2])
    if longest > 2400:  # phone photos are large; the pipeline needs far less
        scale = 2400 / longest
        photo = cv.resize(photo, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)
    return photo


def plan(body: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    cfg = load_config(body.get("config"))
    outcome = agent.run(decode_photo(body.get("photo")), cfg, reader())
    result = asdict(outcome)
    result["seconds"] = round(time.perf_counter() - started, 2)
    print(json.dumps({"event": "plan", "status": outcome.status, "trace": outcome.trace,
                      "proposals": len(outcome.proposals), "questions": len(outcome.questions),
                      "seconds": result["seconds"]}))
    return result


def apply(body: dict[str, Any]) -> dict[str, Any]:
    text = body.get("config")
    load_config(text)  # validate before anything is written
    try:
        approved = [agent.Change(**c) for c in body.get("approved", [])]
    except TypeError:
        raise BadRequest("approved changes are malformed") from None
    with tempfile.TemporaryDirectory() as tmp:
        config = Path(tmp) / "site.yaml"
        config.write_text(text or (DEMO / "site.yaml").read_text(encoding="utf-8"), encoding="utf-8")
        cfg = agent.apply_approved(config, approved)
        build = builder()
        out = Path(tmp) / "dist"
        build.build(config, out)
        page = (out / "index.html").read_text(encoding="utf-8")
        css = (out / "assets" / "base.css").read_text(encoding="utf-8")
    page = page.replace('<link rel="stylesheet" href="assets/base.css">', f"<style>{css}</style>")
    print(json.dumps({"event": "apply", "approved": len(approved)}))
    return {"config": yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), "page": page}


ROUTES = {"/plan": plan, "/apply": apply}


def respond(method: str, path: str, raw_body: str | None) -> tuple[int, str, str]:
    if method == "GET" and path in ("/", "/index.html"):
        return 200, "text/html; charset=utf-8", (HERE / "web" / "index.html").read_text(encoding="utf-8")
    if method == "GET" and path == "/site.yaml":
        return 200, "text/plain; charset=utf-8", (DEMO / "site.yaml").read_text(encoding="utf-8")
    if method == "GET" and path == "/demo.jpg":
        return 200, "image/jpeg", base64.b64encode((DEMO / "menu.jpg").read_bytes()).decode()
    if method == "POST" and path in ROUTES:
        try:
            body = json.loads(raw_body or "{}")
            if not isinstance(body, dict):
                raise BadRequest("body must be a JSON object")
            return 200, "application/json", json.dumps(ROUTES[path](body), ensure_ascii=False)
        except (BadRequest, json.JSONDecodeError, ValueError) as error:
            return 400, "application/json", json.dumps({"error": str(error)})
    return 404, "application/json", json.dumps({"error": "not found"})


def handler(event: dict[str, Any], _context: Any) -> dict[str, Any]:
    """Lambda Function URL entry point (payload format 2.0)."""
    http = event.get("requestContext", {}).get("http", {})
    body = event.get("body")
    if body and event.get("isBase64Encoded"):
        body = base64.b64decode(body).decode("utf-8")
    status, content_type, payload = respond(http.get("method", "GET"), event.get("rawPath", "/"), body)
    binary = content_type == "image/jpeg"
    return {"statusCode": status, "headers": {"content-type": content_type},
            "body": payload, "isBase64Encoded": binary}


def serve(port: int = 8080) -> None:
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Local(BaseHTTPRequestHandler):
        def _send(self, method: str) -> None:
            length = int(self.headers.get("content-length") or 0)
            body = self.rfile.read(length).decode("utf-8") if length else None
            status, content_type, payload = respond(method, self.path, body)
            data = base64.b64decode(payload) if content_type == "image/jpeg" else payload.encode("utf-8")
            self.send_response(status)
            self.send_header("content-type", content_type)
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            self._send("GET")

        def do_POST(self) -> None:
            self._send("POST")

    print(f"http://localhost:{port}")
    HTTPServer(("127.0.0.1", port), Local).serve_forever()


if __name__ == "__main__":
    serve(int(os.environ.get("PORT", 8080)))
