"""Tests for the online setup: any-provider AI with backup, database storage, server protection, gateway.

Run from the repository root:
    .venv/Scripts/python.exe -m unittest agent.test_deploy -v
Set TEST_DATABASE_URL (a Postgres database you can wipe) to also run the storage tests on Postgres.
"""
import json
import os
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import httpx
from fastapi.testclient import TestClient

from agent import llm_gateway, prompts as P, server
from agent.chat import ChatEngine, Turn
from agent.llm import Endpoint, LLMRouter
from agent.retriever import Retriever
from agent.store import Store, database_url
from agent.test_chat import NoExamples, tool


def sse(*pieces: str) -> bytes:
    lines = [f"data: {json.dumps({'choices': [{'delta': {'content': p}}]})}\n\n" for p in pieces]
    return ("".join(lines) + "data: [DONE]\n\n").encode()


def fake_api(model="gemma3:4b", reply=("Hello ", "student"), reject_system=False, fail_chat=False, seen=None):
    """A mock OpenAI-compatible server for httpx.MockTransport."""
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": model}]})
        body = json.loads(request.content)
        if fail_chat:
            return httpx.Response(500, json={"error": "boom"})
        if reject_system and body["messages"][0]["role"] == "system":
            return httpx.Response(400, json={"error": "Developer instruction is not enabled"})
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=sse(*reply))
    return httpx.MockTransport(handler)


def down():
    def handler(request):
        raise httpx.ConnectError("unreachable", request=request)
    return httpx.MockTransport(handler)


MESSAGES = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}]


class LLMTest(unittest.TestCase):
    def test_streams_openai_format_with_key(self):
        seen = []
        ep = Endpoint("primary", "https://pc.example/v1", "gemma3:4b", "secret", transport=fake_api(seen=seen))
        self.assertTrue(ep.available())
        self.assertEqual("".join(ep.stream_chat(MESSAGES)), "Hello student")
        self.assertEqual(seen[-1].headers["authorization"], "Bearer secret")
        self.assertEqual(seen[-1].headers["ngrok-skip-browser-warning"], "1")

    def test_missing_model_counts_as_unavailable(self):
        ep = Endpoint("primary", "https://pc.example/v1", "sarp-agent", transport=fake_api(model="gemma3:4b"))
        self.assertFalse(ep.available())

    def test_system_message_folded_when_rejected(self):
        seen = []
        ep = Endpoint("backup", "https://api.example/v1", "gemma-3-27b-it",
                      transport=fake_api(model="models/gemma-3-27b-it", reject_system=True, seen=seen))
        self.assertTrue(ep.available())
        self.assertEqual("".join(ep.stream_chat(MESSAGES)), "Hello student")
        retried = json.loads(seen[-1].content)["messages"]
        self.assertEqual(retried[0]["role"], "user")
        self.assertIn("Be brief.", retried[0]["content"])

    def test_backup_answers_when_pc_is_off(self):
        router = LLMRouter([Endpoint("primary", "https://pc.example/v1", "gemma3:4b", transport=down()),
                            Endpoint("backup", "https://api.example/v1", "llama", transport=fake_api(model="llama"))])
        self.assertEqual("".join(router.stream_chat(MESSAGES)), "Hello student")
        self.assertEqual(router.last_used, "backup")

    def test_backup_answers_when_pc_fails_before_first_word(self):
        primary = Endpoint("primary", "https://pc.example/v1", "gemma3:4b", transport=fake_api(fail_chat=True))
        router = LLMRouter([primary, Endpoint("backup", "https://api.example/v1", "llama", transport=fake_api(model="llama"))])
        self.assertEqual("".join(router.stream_chat(MESSAGES)), "Hello student")
        self.assertFalse(primary.available())  # marked down for a while

    def test_offline_answers_when_everything_is_down(self):
        router = LLMRouter([Endpoint("primary", "https://pc.example/v1", "gemma3:4b", transport=down())])
        engine = ChatEngine(store=Store(Path(tempfile.mkdtemp()) / "t.db"), tool=tool(), llm=router,
                            retriever=NoExamples())
        engine.type_delay = 0
        events = list(engine.handle(Turn(client_id="offline-client", text="What documents do I need for a UK visa?")))
        reply = "".join(e["text"] for e in events if e["type"] == "token")
        self.assertIn(P.GENERAL_REPLY["english"], reply)

    def test_settings_from_environment(self):
        env = {"SARP_LLM_BASE_URL": "https://pc.example/v1", "SARP_LLM_API_KEY": "k", "SARP_LLM_MODEL": "sarp-agent",
               "SARP_LLM_FALLBACK_BASE_URL": "https://api.groq.com/openai/v1", "SARP_LLM_FALLBACK_MODEL": "llama"}
        with mock.patch.dict(os.environ, env):
            router = LLMRouter.from_env()
        self.assertEqual([(e.name, e.model) for e in router.endpoints], [("primary", "sarp-agent"), ("backup", "llama")])
        self.assertFalse(router.endpoints[0].is_local_ollama)


class StoreTests:
    """Shared storage tests, run on SQLite always and on Postgres when TEST_DATABASE_URL is set."""

    def make_store(self) -> Store:
        raise NotImplementedError

    def test_conversation_round_trip_and_counts(self):
        store = self.make_store()
        conv = store.create_conversation("browser-1")
        store.add_message(conv["id"], "user", "আমি ইউকে যেতে চাই", client_id="browser-1", client_ip="1.2.3.4")
        mid = store.add_message(conv["id"], "assistant", "ঠিক আছে", {"chips": [{"label": "x"}]})
        store.update_conversation(conv["id"], title="Plan", state={"profile": {"country": "UK"}})
        store.set_feedback(mid, 1)
        listed = [c for c in store.list_conversations("browser-1") if c["id"] == conv["id"]][0]
        self.assertEqual((listed["title"], listed["message_count"], listed["state"]["profile"]["country"]), ("Plan", 2, "UK"))
        msgs = store.messages(conv["id"])
        self.assertEqual(msgs[0]["content"], "আমি ইউকে যেতে চাই")
        self.assertEqual((msgs[1]["feedback"], msgs[1]["meta"]["chips"][0]["label"]), (1, "x"))
        self.assertEqual(store.message_owner(mid), (conv["id"], "browser-1"))
        since = time.time() - 60
        self.assertEqual(store.count_user_messages(since, client_id="browser-1"), 1)
        self.assertEqual(store.count_user_messages(since, client_ip="1.2.3.4"), 1)
        self.assertEqual(store.count_user_messages(since, client_id="someone-else"), 0)
        store.delete_conversation(conv["id"])
        self.assertIsNone(store.get_conversation(conv["id"]))

    def test_examples_and_retriever_from_database(self):
        store = self.make_store()
        rows = [{"question": "Can I study in Malaysia without IELTS?", "answer": "Some universities accept other "
                 "English evidence; check each programme's requirement first.", "conversation": "c2-001",
                 "language": "english", "destination": "Malaysia"}]
        self.assertEqual(store.replace_examples(rows), 1)
        retriever = Retriever(data_dir=Path(tempfile.mkdtemp()), store=store)
        self.assertEqual((retriever.source, len(retriever)), ("database", 1))
        self.assertEqual(retriever.search("Malaysia without IELTS?")[0]["conversation"], "c2-001")


class SQLiteStoreTest(StoreTests, unittest.TestCase):
    def make_store(self):
        return Store(Path(tempfile.mkdtemp()) / "test.db")

    def test_upgrades_existing_local_database(self):
        path = Path(tempfile.mkdtemp()) / "old.db"
        with sqlite3.connect(path) as db:  # schema used before the online version
            db.executescript("""
                CREATE TABLE conversations (id TEXT PRIMARY KEY, client_id TEXT NOT NULL, title TEXT,
                    created REAL NOT NULL, updated REAL NOT NULL, state TEXT NOT NULL DEFAULT '{}');
                CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL, content TEXT NOT NULL, created REAL NOT NULL,
                    meta TEXT NOT NULL DEFAULT '{}', feedback INTEGER NOT NULL DEFAULT 0);
                INSERT INTO conversations VALUES ('abc', 'b1', 'Old chat', 1, 1, '{}');
                INSERT INTO messages (conversation_id, role, content, created) VALUES ('abc', 'user', 'hello', 1);
            """)
        store = Store(path)
        self.assertEqual(store.messages("abc")[0]["content"], "hello")
        store.add_message("abc", "user", "again", client_id="b1")
        self.assertEqual(store.count_user_messages(time.time() - 60, client_id="b1"), 1)

    def test_database_url_prefers_postgres_setting(self):
        with mock.patch.dict(os.environ, {"DATABASE_URL": "postgres://u:p@host/db?sslmode=require"}):
            self.assertEqual(database_url(), "postgresql+psycopg://u:p@host/db?sslmode=require")
            self.assertTrue(database_url("x.db").startswith("sqlite:///"))  # an explicit file path wins


@unittest.skipUnless(os.environ.get("TEST_DATABASE_URL"), "set TEST_DATABASE_URL to test on Postgres")
class PostgresStoreTest(StoreTests, unittest.TestCase):
    def make_store(self):
        with mock.patch.dict(os.environ, {"DATABASE_URL": os.environ["TEST_DATABASE_URL"]}):
            return Store(url=database_url())


class ServerProtectionTest(unittest.TestCase):
    def setUp(self):
        engine = ChatEngine(store=Store(Path(tempfile.mkdtemp()) / "t.db"), tool=tool(),
                            llm=LLMRouter([Endpoint("primary", "https://pc.example/v1", "m", transport=down())]),
                            retriever=NoExamples())
        engine.type_delay = 0
        server._engine = engine
        self.client = TestClient(server.app)

    def test_counsellor_page_needs_pin_online(self):
        with mock.patch.dict(os.environ, {"VERCEL": "1"}, clear=False):
            os.environ.pop("SARP_COUNSELLOR_PIN", None)
            self.assertEqual(self.client.get("/api/counsellor/conversations").status_code, 503)
        with mock.patch.dict(os.environ, {"VERCEL": "1", "SARP_COUNSELLOR_PIN": "4321"}):
            self.assertEqual(self.client.get("/api/counsellor/conversations").status_code, 401)
            ok = self.client.get("/api/counsellor/conversations", headers={"X-Counsellor-Pin": "4321"})
            self.assertEqual(ok.status_code, 200)
            self.assertEqual(ok.json()["status"]["providers"][0]["name"], "primary")
            self.assertNotIn("pc.example/v1", json.dumps(ok.json()))  # the tunnel address is not exposed

    def test_rate_limit_per_browser(self):
        body = {"client_id": "rate-browser-1", "text": "I want to study in the UK"}
        with mock.patch.dict(os.environ, {"SARP_RATE_BROWSER_10MIN": "2"}):
            codes = [self.client.post("/api/chat", json=body).status_code for _ in range(3)]
            other = self.client.post("/api/chat", json={**body, "client_id": "rate-browser-2"}).status_code
        self.assertEqual(codes, [200, 200, 429])
        self.assertEqual(other, 200)

    def test_health(self):
        self.assertEqual(self.client.get("/api/health").json(), {"ok": True})


class GatewayTest(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(llm_gateway.app)

    def test_key_required(self):
        with mock.patch.dict(os.environ, {"SARP_GATEWAY_KEY": ""}):
            self.assertEqual(self.client.get("/v1/models").status_code, 503)
        with mock.patch.dict(os.environ, {"SARP_GATEWAY_KEY": "right-key"}):
            self.assertEqual(self.client.get("/v1/models").status_code, 401)
            bad = self.client.post("/v1/chat/completions", headers={"Authorization": "Bearer wrong"},
                                   json={"model": "gemma3:4b", "messages": []})
            self.assertEqual(bad.status_code, 401)
        self.assertEqual(self.client.get("/health").json(), {"ok": True})

    def test_model_allow_list(self):
        env = {"SARP_GATEWAY_KEY": "k", "SARP_GATEWAY_MODELS": "sarp-agent"}
        with mock.patch.dict(os.environ, env):
            response = self.client.post("/v1/chat/completions", headers={"Authorization": "Bearer k"},
                                        json={"model": "some-other-model", "messages": []})
        self.assertEqual(response.status_code, 403)


if __name__ == "__main__":
    unittest.main()
