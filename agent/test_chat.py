"""Tests for the chat website's engine, text extraction and API.

Run from the repository root:
    .venv/Scripts/python.exe -m unittest agent.test_chat -v
"""
import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

from fastapi.testclient import TestClient

from agent import prompts as P
from agent import server
from agent.chat import ChatEngine, Turn
from agent.extract import detect_intent, extract
from agent.language import detect_language
from agent.risk_tool import RiskTool
from agent.store import Store

TODAY = date(2026, 11, 13)
TOOL = None


def tool() -> RiskTool:
    global TOOL
    if TOOL is None:
        TOOL = RiskTool(political_phase="Newly Elected", bdt_per_usd=122)
    return TOOL


class FakeLLM:
    model = "fake-model"

    def __init__(self, available=True, reply="You will need your transcripts and passport."):
        self._available, self.reply, self.calls = available, reply, []

    def available(self, max_age=15.0):
        return self._available

    def stream_chat(self, messages, temperature=0.4):
        self.calls.append(messages)
        yield self.reply[:12]
        yield self.reply[12:]


class NoExamples:
    def search(self, *args, **kwargs):
        return []

    def __len__(self):
        return 0


def make_engine(llm=None) -> ChatEngine:
    db = Path(tempfile.mkdtemp()) / "test.db"
    engine = ChatEngine(store=Store(db), tool=tool(), llm=llm or FakeLLM(available=False), retriever=NoExamples())
    engine.type_delay = 0
    return engine


def run(engine, client, text, cid=None, qr=None, language=None):
    events = list(engine.handle(Turn(client_id=client, text=text, conversation_id=cid, quick_reply=qr,
                                     language=language)))
    reply = "".join(e["text"] for e in events if e["type"] == "token")
    chips = next((e["chips"] for e in events if e["type"] == "chips"), [])
    cid = next(e["conversation_id"] for e in events if e["type"] == "meta")
    return cid, reply, chips, events


class ExtractTest(unittest.TestCase):
    CASES = [
        ("I want to study Computer Science in Canada, my CGPA is 3.25 out of 4", None,
         {"country": "Canada", "course": "Computer Science", "gpa": 3.25, "gpa_scale": 4.0}),
        ("Ami UK te porte chai, budget 18 lakh er moto", None,
         {"country": "UK", "budget_amount": 18.0, "budget_unit": "lakh_taka"}),
        ("আমি অস্ট্রেলিয়ায় পড়তে চাই, বাজেট ২০ লাখ টাকা", None,
         {"country": "Australia", "budget_amount": 20.0, "budget_unit": "lakh_taka"}),
        ("আমি মালয়েশিয়ায় বিবিএ পড়তে চাই। কি কি ডকুমেন্ট লাগবে?", None,
         {"country": "Malaysia", "course": "Business Administration"}),  # no false "May" intake
        ("Ami Canada te CSE porte chai, CGPA 3.2. Kivabe shuru korbo?", None,
         {"country": "Canada", "course": "Computer Science", "gpa": 3.2, "gpa_scale": None}),
        ("January 2027 intake", None, {"intake": "2027-01"}),
        ("I may apply next year", None, {}),
        ("IELTS 6.5 and HSC GPA 5.00", None, {"gpa": 5.0, "gpa_scale": 5.0}),
        ("IELTS result 5.5", None, {}),
        ("3.4", "gpa", {"gpa": 3.4, "gpa_scale": None}),
        ("september", "intake", {"intake": "2027-09"}),
        ("মে মাসে ভর্তি হতে চাই", None, {"intake": "2027-05"}),
        ("I've shortlisted Monash and Deakin", "due_diligence",
         {"institution": "Monash University", "due_diligence": "Yes Medium"}),
        ("$15k per year", None, {"budget_amount": 15000.0, "budget_unit": "usd"}),
    ]

    def test_cases(self):
        for text, awaiting, expected in self.CASES:
            with self.subTest(text=text):
                self.assertEqual(extract(text, awaiting, TODAY)[0], expected)

    def test_skip_when_unsure(self):
        self.assertEqual(extract("not sure yet", "intake", TODAY), ({}, ["intake"]))

    def test_intents(self):
        self.assertEqual(detect_intent("Can I edit my bank statement to show more money?"), "integrity")
        self.assertEqual(detect_intent("I want to talk to a counsellor"), "handoff")
        self.assertEqual(detect_intent("Hello bhaiya"), "greeting")
        self.assertIsNone(detect_intent("What documents do I need?"))

    def test_language(self):
        self.assertEqual(detect_language("Ami UK te porte chai"), "banglish")
        self.assertEqual(detect_language("আমি ইউকে যেতে চাই"), "bengali")
        self.assertEqual(detect_language("I want to study in the UK"), "english")
        self.assertIsNone(detect_language("ok"))


class EngineTest(unittest.TestCase):
    def test_collects_profile_then_shares_plan_with_findings(self):
        engine = make_engine()
        cid, reply, chips, _ = run(engine, "client-0001", "Ami UK te porte chai")
        self.assertIn(P.QUESTIONS["intake"]["banglish"], reply)
        self.assertEqual(chips[-1]["skip"], True)
        cid, reply, chips, _ = run(engine, "client-0001", "January 2027", cid, {"field": "intake", "value": "2027-01"})
        self.assertIn(P.QUESTIONS["budget"]["banglish"], reply)
        cid, reply, chips, _ = run(engine, "client-0001", "18 lakh", cid)
        self.assertIn(P.QUESTIONS["gpa"]["banglish"], reply)
        cid, reply, chips, _ = run(engine, "client-0001", "3.1", cid)
        self.assertEqual([c["value"] for c in chips if not c.get("skip")], P.DUE_DILIGENCE_VALUES)
        cid, reply, chips, events = run(engine, "client-0001", "Ekhono kichu dekhini", cid,
                                        {"field": "due_diligence", "value": "No"})
        # The plan quotes the project's records for the current phase (Newly Elected: 18% vs up to 99%).
        self.assertIn(P.STUDENT_TOPICS["help_research"]["banglish"], reply)
        self.assertIn("(18%)", reply)
        self.assertIn("99%", reply)
        plan = next(e["plan"] for e in events if e["type"] == "plan")
        self.assertTrue(plan["complete"])
        state = engine.store.get_conversation(cid)["state"]
        view = state["assessment"]["counsellor_view"]
        self.assertEqual(view["risk_level"], "high")
        # The student never sees their probability or risk tier.
        transcript = " ".join(m["content"] for m in engine.store.messages(cid)).lower()
        self.assertNotIn("risk", transcript)
        self.assertNotIn(f"{view['probability'] * 100:.1f}", transcript)

    def test_skip_is_not_asked_again(self):
        engine = make_engine()
        cid, _, _, _ = run(engine, "client-0002", "I want to study in Canada")
        cid, reply, _, _ = run(engine, "client-0002", "Skip for now", cid, {"field": "intake", "value": None, "skip": True})
        self.assertIn(P.QUESTIONS["budget"]["english"], reply)
        self.assertIn("intake", engine.store.get_conversation(cid)["state"]["skipped"])

    def test_document_tampering_is_refused(self):
        engine = make_engine(FakeLLM(available=True))
        _, reply, chips, _ = run(engine, "client-0003", "Can I edit my bank statement to show more money?")
        self.assertIn(P.INTEGRITY_REPLY["english"], reply)
        self.assertEqual(chips, [])
        self.assertEqual(engine.llm.calls, [])  # fixed reply, the model is not asked

    def test_model_answer_is_grounded_and_amounts_get_a_caveat(self):
        llm = FakeLLM(available=True, reply="Canada usually needs about CAD 20,000 in your account.")
        engine = make_engine(llm)
        _, reply, _, _ = run(engine, "client-0004", "How much money do I need to show for a Canada study visa?")
        self.assertTrue(reply.startswith("Canada usually needs"))
        self.assertIn(P.AMOUNT_NOTE["english"], reply)
        system = llm.calls[0][0]["content"]
        self.assertIn(P.SYSTEM_PROMPT, system)
        self.assertIn("Never mention risk", system)

    def test_other_scripts_are_removed_from_model_replies(self):
        engine = make_engine(FakeLLM(available=True, reply="வணக்கம்! আপনার পাসপোর্ট আর মার্কশিট লাগবে।"))
        _, reply, _, _ = run(engine, "client-0007", "মালয়েশিয়ায় পড়তে কি কি ডকুমেন্ট লাগবে?")
        self.assertTrue(reply.startswith("আপনার পাসপোর্ট"), reply)

    def test_offline_mode_still_answers(self):
        engine = make_engine(FakeLLM(available=False))
        _, reply, _, _ = run(engine, "client-0005", "What documents do I need for a UK student visa?")
        self.assertIn(P.GENERAL_REPLY["english"], reply)

    def test_reply_language_follows_choice(self):
        engine = make_engine()
        _, reply, _, _ = run(engine, "client-0006", "I want to study in the UK", language="bengali")
        self.assertIn(P.QUESTIONS["intake"]["bengali"], reply)


class ApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        server._engine = make_engine()
        cls.client = TestClient(server.app)

    def test_chat_stream_and_privacy(self):
        body = {"client_id": "api-client-01", "text": "Ami Malaysia te porte chai, budget 10 lakh"}
        with self.client.stream("POST", "/api/chat", json=body) as response:
            self.assertEqual(response.status_code, 200)
            raw = "".join(response.iter_text())
        events = [json.loads(line[6:]) for line in raw.splitlines() if line.startswith("data: ")]
        types = [e["type"] for e in events]
        self.assertEqual(types[0], "meta")
        self.assertIn("plan", types)
        self.assertEqual(types[-1], "done")
        cid = events[0]["conversation_id"]

        student = self.client.get(f"/api/conversations/{cid}", params={"client_id": "api-client-01"}).json()
        self.assertNotIn("assessment", json.dumps(student))
        self.assertEqual(student["plan"]["items"]["country"], "Malaysia")
        # Another browser cannot read this chat.
        self.assertEqual(self.client.get(f"/api/conversations/{cid}", params={"client_id": "someone-else"}).status_code, 404)
        counsellor = self.client.get(f"/api/counsellor/conversations/{cid}").json()
        self.assertEqual(counsellor["assessment"]["status"], "incomplete")

    def test_callback_and_listing(self):
        cid = self.client.post("/api/conversations", json={"client_id": "api-client-02"}).json()["id"]
        result = self.client.post(f"/api/conversations/{cid}/callback", params={"client_id": "api-client-02"},
                                  json={"name": "Rahim", "contact": "01700000000", "preferred_time": None}).json()
        self.assertIn("Rahim", result["content"])
        listed = self.client.get("/api/conversations", params={"client_id": "api-client-02"}).json()
        self.assertEqual([c["id"] for c in listed], [cid])
        summary = self.client.get("/api/counsellor/conversations").json()
        self.assertTrue(any(c["handoff"] for c in summary["conversations"]))


if __name__ == "__main__":
    unittest.main()
