"""Storage for conversations, messages, feedback, call-back requests and counsellor examples.

Two backends behind the same methods, chosen automatically:
* On your PC: an SQLite file at <capstone folder>/agent_data/sarp_agent.db (override with SARP_DB_PATH).
* Online: Postgres when DATABASE_URL is set (the Neon integration on Vercel sets it).
SQLAlchemy Core writes the SQL for each, so tests on SQLite run the same queries Postgres will.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path

from sqlalchemy import (Column, Float, Integer, MetaData, String, Table, Text, create_engine, delete, func, insert,
                        inspect, select, text, update)

from agent.risk_tool import REPO_DIR

DEFAULT_DB_PATH = Path(os.environ.get("SARP_DB_PATH", REPO_DIR.parent / "agent_data" / "sarp_agent.db"))

metadata = MetaData()
conversations_t = Table(
    "conversations", metadata,
    Column("id", String(32), primary_key=True),
    Column("client_id", String(64), nullable=False),
    Column("title", Text),
    Column("created", Float, nullable=False),
    Column("updated", Float, nullable=False),
    Column("state", Text, nullable=False),
)
messages_t = Table(
    "messages", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("conversation_id", String(32), nullable=False),
    Column("role", String(16), nullable=False),
    Column("content", Text, nullable=False),
    Column("created", Float, nullable=False),
    Column("meta", Text, nullable=False),
    Column("feedback", Integer, nullable=False, default=0),
    Column("client_id", String(64)),   # who sent it, for rate limits
    Column("client_ip", String(64)),
)
examples_t = Table(
    "examples", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("question", Text, nullable=False),
    Column("answer", Text, nullable=False),
    Column("conversation", String(32)),
    Column("language", String(16)),
    Column("destination", String(64)),
)
INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_conversations_client ON conversations (client_id, updated)",
    "CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages (conversation_id, id)",
    "CREATE INDEX IF NOT EXISTS idx_messages_client_rate ON messages (client_id, created)",
    "CREATE INDEX IF NOT EXISTS idx_messages_ip_rate ON messages (client_ip, created)",
]


def database_url(path: Path | str | None = None) -> str:
    """Postgres from DATABASE_URL when no file path is given, otherwise an SQLite file."""
    url = os.environ.get("DATABASE_URL") or os.environ.get("POSTGRES_URL")
    if url and path is None:
        for prefix in ("postgres://", "postgresql://"):
            if url.startswith(prefix):
                return "postgresql+psycopg://" + url[len(prefix):]
        return url
    file = Path(path or DEFAULT_DB_PATH)
    file.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{file.as_posix()}"


class Store:
    def __init__(self, path: Path | str | None = None, url: str | None = None):
        self.url = url or database_url(path)
        if self.url.startswith("postgresql"):
            # Small pool for serverless; no prepared statements so Neon's connection pooler works.
            self.engine = create_engine(self.url, pool_pre_ping=True, pool_size=2, max_overflow=3, pool_recycle=300,
                                        connect_args={"prepare_threshold": None})
        else:
            self.engine = create_engine(self.url, connect_args={"timeout": 10, "check_same_thread": False})
        metadata.create_all(self.engine)
        self._migrate()

    @property
    def backend(self) -> str:
        return self.engine.dialect.name  # "sqlite" or "postgresql"

    def describe(self) -> str:
        """Where data is stored, without the password."""
        return self.engine.url.render_as_string(hide_password=True)

    def _migrate(self) -> None:
        """Add columns and indexes introduced after the first local databases were made."""
        have = {c["name"] for c in inspect(self.engine).get_columns("messages")}
        with self.engine.begin() as db:
            for name in ("client_id", "client_ip"):
                if name not in have:
                    db.execute(text(f"ALTER TABLE messages ADD COLUMN {name} VARCHAR(64)"))
            for statement in INDEXES:
                db.execute(text(statement))

    @staticmethod
    def _conversation(row) -> dict | None:
        if row is None:
            return None
        out = dict(row._mapping)
        out["state"] = json.loads(out.get("state") or "{}")
        return out

    # ------------------------------------------------------------------ conversations
    def create_conversation(self, client_id: str) -> dict:
        now = time.time()
        conv = {"id": uuid.uuid4().hex[:16], "client_id": client_id, "title": None, "created": now, "updated": now}
        with self.engine.begin() as db:
            db.execute(insert(conversations_t).values(**conv, state="{}"))
        return {**conv, "state": {}}

    def get_conversation(self, conversation_id: str) -> dict | None:
        with self.engine.connect() as db:
            row = db.execute(select(conversations_t).where(conversations_t.c.id == conversation_id)).first()
        return self._conversation(row)

    def list_conversations(self, client_id: str | None = None, limit: int = 200) -> list[dict]:
        count = (select(func.count()).select_from(messages_t)
                 .where(messages_t.c.conversation_id == conversations_t.c.id).scalar_subquery())
        query = select(conversations_t, count.label("message_count"))
        if client_id is not None:
            query = query.where(conversations_t.c.client_id == client_id)
        query = query.order_by(conversations_t.c.updated.desc()).limit(limit)
        with self.engine.connect() as db:
            return [self._conversation(r) for r in db.execute(query)]

    def update_conversation(self, conversation_id: str, *, title: str | None = None, state: dict | None = None) -> None:
        values: dict = {"updated": time.time()}
        if title is not None:
            values["title"] = title
        if state is not None:
            values["state"] = json.dumps(state, ensure_ascii=False)
        with self.engine.begin() as db:
            db.execute(update(conversations_t).where(conversations_t.c.id == conversation_id).values(**values))

    def delete_conversation(self, conversation_id: str) -> None:
        with self.engine.begin() as db:
            db.execute(delete(messages_t).where(messages_t.c.conversation_id == conversation_id))
            db.execute(delete(conversations_t).where(conversations_t.c.id == conversation_id))

    # ------------------------------------------------------------------ messages
    def add_message(self, conversation_id: str, role: str, content: str, meta: dict | None = None,
                    client_id: str | None = None, client_ip: str | None = None) -> int:
        with self.engine.begin() as db:
            result = db.execute(insert(messages_t).values(
                conversation_id=conversation_id, role=role, content=content, created=time.time(),
                meta=json.dumps(meta or {}, ensure_ascii=False), feedback=0, client_id=client_id, client_ip=client_ip))
            return int(result.inserted_primary_key[0])

    def update_message(self, message_id: int, content: str, meta: dict | None = None) -> None:
        values: dict = {"content": content}
        if meta is not None:
            values["meta"] = json.dumps(meta, ensure_ascii=False)
        with self.engine.begin() as db:
            db.execute(update(messages_t).where(messages_t.c.id == message_id).values(**values))

    def messages(self, conversation_id: str) -> list[dict]:
        query = select(messages_t).where(messages_t.c.conversation_id == conversation_id).order_by(messages_t.c.id)
        with self.engine.connect() as db:
            rows = [dict(r._mapping) for r in db.execute(query)]
        for m in rows:
            m["meta"] = json.loads(m["meta"] or "{}")
        return rows

    def message_owner(self, message_id: int) -> tuple[str, str] | None:
        """(conversation id, client id) for a message, or None."""
        query = (select(conversations_t.c.id, conversations_t.c.client_id)
                 .join(messages_t, messages_t.c.conversation_id == conversations_t.c.id)
                 .where(messages_t.c.id == message_id))
        with self.engine.connect() as db:
            row = db.execute(query).first()
        return (row.id, row.client_id) if row else None

    def set_feedback(self, message_id: int, value: int) -> None:
        with self.engine.begin() as db:
            db.execute(update(messages_t).where(messages_t.c.id == message_id, messages_t.c.role == "assistant")
                       .values(feedback=max(-1, min(1, value))))

    def count_user_messages(self, since: float, client_id: str | None = None, client_ip: str | None = None) -> int:
        """Student messages sent since a time, by one browser and/or one IP address (for rate limits)."""
        query = (select(func.count()).select_from(messages_t)
                 .where(messages_t.c.role == "user", messages_t.c.created >= since))
        if client_id is not None:
            query = query.where(messages_t.c.client_id == client_id)
        if client_ip is not None:
            query = query.where(messages_t.c.client_ip == client_ip)
        with self.engine.connect() as db:
            return int(db.execute(query).scalar_one())

    # ------------------------------------------------------------------ counsellor examples
    def replace_examples(self, rows: list[dict]) -> int:
        keys = ("question", "answer", "conversation", "language", "destination")
        clean = [{k: r.get(k) for k in keys} for r in rows]
        with self.engine.begin() as db:
            db.execute(delete(examples_t))
            if clean:
                db.execute(insert(examples_t), clean)
        return len(clean)

    def load_examples(self) -> list[dict]:
        with self.engine.connect() as db:
            return [dict(r._mapping) for r in db.execute(select(examples_t).order_by(examples_t.c.id))]
