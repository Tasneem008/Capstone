"""Secure gateway that lets the online website use the AI model running on the team PC.

Runs on the PC next to Ollama. It accepts OpenAI-compatible requests only when they carry the right key
and forwards them to Ollama, streaming included. Expose this gateway (never Ollama itself) through a
tunnel such as ngrok, then point the website at it.

On the PC (PowerShell), from the repository root:
    $env:SARP_GATEWAY_KEY = "<long random key>"       # e.g. python -c "import secrets; print(secrets.token_urlsafe(32))"
    .venv\\Scripts\\python.exe -m agent.llm_gateway      # listens on 127.0.0.1:8787
    ngrok http 8787 --url https://<your-static-domain>.ngrok-free.app
On Vercel:
    SARP_LLM_BASE_URL = https://<your-static-domain>.ngrok-free.app/v1
    SARP_LLM_API_KEY  = <the same key>
agent/start_gateway.ps1 does the PC part in one step.

Settings: SARP_GATEWAY_KEY (required), SARP_OLLAMA_URL (default http://127.0.0.1:11434),
SARP_GATEWAY_PORT (default 8787), SARP_GATEWAY_MODELS (optional comma list of models callers may use),
SARP_LLM_MODEL (model loaded into the GPU at start, default gemma3:4b).
"""
from __future__ import annotations

import hmac
import json
import os

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

MAX_BODY_BYTES = 256_000

app = FastAPI(title="SARP-Agent model gateway", docs_url=None, redoc_url=None, openapi_url=None)


def _ollama() -> str:
    return os.environ.get("SARP_OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")


def _check_key(authorization: str | None) -> None:
    key = os.environ.get("SARP_GATEWAY_KEY", "")
    if not key:
        raise HTTPException(503, "Gateway key is not set on the PC")
    if not authorization or not hmac.compare_digest(authorization.encode(), f"Bearer {key}".encode()):
        raise HTTPException(401, "Invalid or missing key")


def _allowed_models() -> set[str]:
    return {m.strip() for m in os.environ.get("SARP_GATEWAY_MODELS", "").split(",") if m.strip()}


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/v1/models")
def models(authorization: str | None = Header(default=None)):
    _check_key(authorization)
    try:
        response = httpx.get(f"{_ollama()}/v1/models", timeout=5)
    except httpx.HTTPError:
        raise HTTPException(502, "Ollama is not running on the PC")
    data = response.json()
    allowed = _allowed_models()
    if allowed:
        data["data"] = [m for m in data.get("data", []) if m.get("id") in allowed]
    return JSONResponse(data, status_code=response.status_code)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request, authorization: str | None = Header(default=None)):
    _check_key(authorization)
    body = await request.body()
    if len(body) > MAX_BODY_BYTES:
        raise HTTPException(413, "Request too large")
    try:
        model = json.loads(body).get("model")
    except ValueError:
        raise HTTPException(400, "Body must be JSON")
    allowed = _allowed_models()
    if allowed and model not in allowed:
        raise HTTPException(403, f"Model {model!r} is not offered by this gateway")

    client = httpx.AsyncClient(timeout=httpx.Timeout(180, connect=5))
    try:
        upstream = await client.send(
            client.build_request("POST", f"{_ollama()}/v1/chat/completions", content=body,
                                 headers={"Content-Type": "application/json"}),
            stream=True)
    except httpx.HTTPError:
        await client.aclose()
        raise HTTPException(502, "Ollama is not running on the PC")

    async def relay():
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        finally:
            await upstream.aclose()
            await client.aclose()

    return StreamingResponse(relay(), status_code=upstream.status_code,
                             media_type=upstream.headers.get("content-type", "application/json"))


def main() -> None:
    import uvicorn

    if not os.environ.get("SARP_GATEWAY_KEY"):
        raise SystemExit("Set SARP_GATEWAY_KEY first (use the same value as SARP_LLM_API_KEY on Vercel).")
    model = os.environ.get("SARP_LLM_MODEL", "gemma3:4b")
    try:  # load the model into GPU memory so the first student does not wait
        httpx.post(f"{_ollama()}/api/generate", json={"model": model, "prompt": "", "keep_alive": "60m"}, timeout=120)
    except httpx.HTTPError:
        print("Warning: Ollama did not respond; start Ollama before students use the site.")
    port = int(os.environ.get("SARP_GATEWAY_PORT", "8787"))
    print(f"SARP-Agent model gateway on http://127.0.0.1:{port} (model {model}). Expose it with: "
          f"ngrok http {port} --url https://<your-static-domain>")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
