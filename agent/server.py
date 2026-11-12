"""SARP-Agent website: student chat at / and counsellor insights at /counsellor.

Run from the repository root (Ollama should be running for AI replies):
    .venv/Scripts/python.exe -m agent.server          then open http://127.0.0.1:8000
Online, Vercel loads `app` from this file (see pyproject.toml and agent/DEPLOY.md).

Settings (environment variables):
    SARP_LLM_BASE_URL, SARP_LLM_API_KEY, SARP_LLM_MODEL    AI model (default local Ollama, gemma3:4b)
    SARP_LLM_FALLBACK_BASE_URL, _API_KEY, _MODEL           optional backup AI (see agent/llm.py)
    DATABASE_URL          Postgres for online use; without it chats go to ../agent_data/sarp_agent.db
    SARP_POLITICAL_PHASE  phase used for SARP-Net predictions (default Newly Elected)
    SARP_BDT_PER_USD      taka per US dollar for budgets (default 122)
    SARP_DATA_DIR         folder with splits/train.jsonl for counsellor examples (default ../data)
    SARP_COUNSELLOR_PIN   PIN for /counsellor (required online; optional on the PC)
    SARP_RATE_BROWSER_10MIN / _DAY, SARP_RATE_IP_10MIN / _DAY   message limits (0 turns a limit off)
    SARP_HOST, SARP_PORT  where to serve on the PC (default 127.0.0.1:8000)
"""
from __future__ import annotations

import hmac
import json
import os
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

if __package__ in (None, ""):  # started as a script (python agent/server.py): make `agent` importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agent.chat import ChatEngine, Turn

WEB_DIR = Path(__file__).resolve().parent / "web"
_engine: ChatEngine | None = None
_lock = threading.Lock()
# (setting, default, window in seconds, counted per) — protects the free AI quota from abuse.
RATE_LIMITS = [
    ("SARP_RATE_BROWSER_10MIN", 20, 600, "browser"),
    ("SARP_RATE_BROWSER_DAY", 200, 86_400, "browser"),
    ("SARP_RATE_IP_10MIN", 60, 600, "ip"),     # higher: a campus or home network shares one address
    ("SARP_RATE_IP_DAY", 600, 86_400, "ip"),
]


def on_vercel() -> bool:
    return bool(os.environ.get("VERCEL"))


def engine() -> ChatEngine:
    global _engine
    with _lock:
        if _engine is None:
            _engine = ChatEngine()
    return _engine


@asynccontextmanager
async def lifespan(_app: FastAPI):
    eng = engine()  # load the SARP-Net model and counsellor examples before the first student arrives
    if not on_vercel() and eng.llm.available(max_age=0):  # online, skip checks to keep cold starts short
        threading.Thread(target=eng.llm.warm_up, daemon=True).start()
    yield


app = FastAPI(title="SARP-Agent", docs_url=None, redoc_url=None, lifespan=lifespan)
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


# ---------------------------------------------------------------------- pages
@app.get("/", include_in_schema=False)
def student_page():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/counsellor", include_in_schema=False)
def counsellor_page():
    return FileResponse(WEB_DIR / "counsellor.html")


# ---------------------------------------------------------------------- student API
class ChatIn(BaseModel):
    client_id: str = Field(min_length=8, max_length=64)
    text: str = Field(min_length=1, max_length=4000)
    conversation_id: str | None = Field(default=None, max_length=32)
    quick_reply: dict | None = None
    language: str | None = Field(default=None, max_length=16)


class NewConversationIn(BaseModel):
    client_id: str = Field(min_length=8, max_length=64)


class RenameIn(BaseModel):
    title: str = Field(min_length=1, max_length=80)


class FeedbackIn(BaseModel):
    client_id: str
    value: int = Field(ge=-1, le=1)


class CallbackIn(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    contact: str = Field(min_length=5, max_length=80)
    preferred_time: str | None = Field(default=None, max_length=80)


def _owned(conversation_id: str, client_id: str) -> dict:
    conv = engine().store.get_conversation(conversation_id)
    if conv is None or conv["client_id"] != client_id:
        raise HTTPException(404, "Conversation not found")
    return conv


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/status")
def status():
    return engine().status()


@app.get("/api/conversations")
def list_conversations(client_id: str):
    return [{"id": c["id"], "title": c["title"] or "New chat", "updated": c["updated"]}
            for c in engine().store.list_conversations(client_id) if c["message_count"]]


@app.post("/api/conversations")
def create_conversation(body: NewConversationIn):
    return {"id": engine().store.create_conversation(body.client_id)["id"]}


@app.get("/api/conversations/{conversation_id}")
def get_conversation(conversation_id: str, client_id: str):
    eng = engine()
    conv = _owned(conversation_id, client_id)
    messages = [{"id": m["id"], "role": m["role"], "content": m["content"], "feedback": m["feedback"],
                 "chips": m["meta"].get("chips", []) if m["role"] == "assistant" else []}
                for m in eng.store.messages(conversation_id) if m["content"]]
    state = conv["state"] or {}
    return {"id": conv["id"], "title": conv["title"] or "New chat", "messages": messages,
            "language": state.get("language"),
            "plan": eng.plan_card(state, state.get("language", "english")),
            "callback_requested": bool(state.get("handoff"))}


@app.patch("/api/conversations/{conversation_id}")
def rename_conversation(conversation_id: str, client_id: str, body: RenameIn):
    _owned(conversation_id, client_id)
    engine().store.update_conversation(conversation_id, title=body.title.strip())
    return {"ok": True}


@app.delete("/api/conversations/{conversation_id}")
def delete_conversation(conversation_id: str, client_id: str):
    _owned(conversation_id, client_id)
    engine().store.delete_conversation(conversation_id)
    return {"ok": True}


@app.post("/api/conversations/{conversation_id}/plan/reset")
def reset_plan(conversation_id: str, client_id: str):
    _owned(conversation_id, client_id)
    return engine().reset_plan(conversation_id)


@app.post("/api/conversations/{conversation_id}/callback")
def request_callback(conversation_id: str, client_id: str, body: CallbackIn):
    _owned(conversation_id, client_id)
    return engine().request_callback(conversation_id, body.name.strip(), body.contact.strip(),
                                     (body.preferred_time or "").strip() or None)


@app.post("/api/messages/{message_id}/feedback")
def feedback(message_id: int, body: FeedbackIn):
    owner = engine().store.message_owner(message_id)
    if owner is None or owner[1] != body.client_id:
        raise HTTPException(404, "Message not found")
    engine().store.set_feedback(message_id, body.value)
    return {"ok": True}


def client_ip(request: Request) -> str | None:
    # Vercel sets X-Forwarded-For to the visitor's address; on the PC it is the direct client.
    forwarded = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
    return forwarded or request.headers.get("x-real-ip") or (request.client.host if request.client else None)


def rate_limited(client_id: str, ip: str | None) -> bool:
    store, now = engine().store, time.time()
    for setting, default, window, per in RATE_LIMITS:
        limit = int(os.environ.get(setting, default))
        if limit <= 0 or (per == "ip" and not ip):
            continue
        count = store.count_user_messages(now - window, client_id=client_id) if per == "browser" \
            else store.count_user_messages(now - window, client_ip=ip)
        if count >= limit:
            return True
    return False


@app.post("/api/chat")
def chat(body: ChatIn, request: Request):
    ip = client_ip(request)
    if rate_limited(body.client_id, ip):
        return JSONResponse({"detail": "rate_limited"}, status_code=429, headers={"Retry-After": "600"})
    turn = Turn(client_id=body.client_id, text=body.text, conversation_id=body.conversation_id,
                quick_reply=body.quick_reply, language=body.language, client_ip=ip)

    def events():
        try:
            for event in engine().handle(turn):
                yield f"event: {event['type']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as error:  # surface the failure to the page instead of a silent stop
            payload = {"type": "error", "text": f"Something went wrong on the server ({type(error).__name__})."}
            yield f"event: error\ndata: {json.dumps(payload)}\n\n"
            raise

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ---------------------------------------------------------------------- counsellor API
def _check_pin(pin: str | None) -> None:
    expected = os.environ.get("SARP_COUNSELLOR_PIN", "")
    if not expected:
        if on_vercel():  # never serve student data publicly without a PIN
            raise HTTPException(503, "The counsellor page is turned off until SARP_COUNSELLOR_PIN is set")
        return
    if not hmac.compare_digest((pin or "").encode(), expected.encode()):
        raise HTTPException(401, "Counsellor PIN required")


@app.get("/api/counsellor/conversations")
def counsellor_list(x_counsellor_pin: str | None = Header(default=None)):
    _check_pin(x_counsellor_pin)
    eng = engine()
    return {"status": eng.detailed_status(), "political_phase": eng.tool.political_phase,
            "model_card": eng.tool.findings.model_card(),
            "conversations": [eng.counsellor_summary(c) for c in eng.store.list_conversations() if c["message_count"]]}


@app.get("/api/counsellor/conversations/{conversation_id}")
def counsellor_detail(conversation_id: str, x_counsellor_pin: str | None = Header(default=None)):
    _check_pin(x_counsellor_pin)
    eng = engine()
    conv = eng.store.get_conversation(conversation_id)
    if conv is None:
        raise HTTPException(404, "Conversation not found")
    state = conv["state"] or {}
    messages = [{"id": m["id"], "role": m["role"], "content": m["content"], "feedback": m["feedback"],
                 "created": m["created"]} for m in eng.store.messages(conversation_id) if m["content"]]
    return {"id": conv["id"], "title": conv["title"] or "New chat", "updated": conv["updated"],
            "language": state.get("language"), "profile": state.get("profile", {}),
            "plan": eng.plan_card(state, "english"), "assessment": state.get("assessment"),
            "handoff": state.get("handoff"), "messages": messages}


def main() -> None:
    import uvicorn

    host = os.environ.get("SARP_HOST", "127.0.0.1")
    port = int(os.environ.get("SARP_PORT", "8000"))
    print(f"SARP-Agent: open http://{host}:{port}  (counsellor view: http://{host}:{port}/counsellor)")
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
