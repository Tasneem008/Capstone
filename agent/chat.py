"""The SARP-Agent conversation engine: one student message in, a stream of events out.

Each turn:
 1. Pick the reply language: the student's choice, or detected from their text.
 2. Update the student's profile from a quick-reply button or from their text (agent/extract.py).
 3. Score the profile with the SARP-Net model (agent/risk_tool.py). The full result is stored for
    counsellors; the student never sees a probability or risk label.
 4. Reply with
    * fixed, translated text for profile answers, greetings, call-back requests and document-tampering
      questions (instant and exactly worded);
    * the language model for open questions, given real counsellor answers to similar questions;
    * once the profile is complete, a plan built from the SHAP drivers and the agency records, so it
      states the project's findings exactly.
 5. Ask for the next missing detail, one at a time, with quick-reply buttons.

Events yielded: meta, plan, token, chips, handoff, done, error.
"""
from __future__ import annotations

import re
import time
from collections.abc import Iterator
from dataclasses import dataclass

from agent import prompts as P
from agent.extract import detect_intent, extract
from agent.language import LANGUAGES, detect_language
from agent.llm import LLMRouter
from agent.retriever import Retriever
from agent.risk_tool import (COUNTRY_ALIASES, COURSE_ALIASES, INSTITUTION_ALIASES, REQUIRED_FIELDS, RiskTool,
                             match_name)
from agent.store import Store

PROFILE_KEYS = ["gpa", "gpa_scale", "course", "country", "institution", "budget_amount", "budget_unit",
                "budget_band", "due_diligence", "intake"]
QUICK_REPLY_KEYS = {"country", "intake", "budget_band", "due_diligence"}
FIELD_OF_KEY = {"budget_amount": "budget", "budget_band": "budget", "budget_unit": "budget", "gpa_scale": "gpa"}
LANG_KEYS = ("english", "banglish", "bengali")
# Money amounts in a model reply ("CAD 10,000", "£1,334", "20 lakh", "৳৫০,০০০")
AMOUNT_RE = re.compile(r"(?:[$£€৳]|\b(?:CAD|AUD|NZD|USD|GBP|EUR|RM|MYR|BDT|Tk)\.?)\s?[\d০-৯][\d০-৯,.]*|"
                       r"[\d০-৯][\d০-৯,.]*\s?(?:lakh|lac|crore|লাখ|কোটি|টাকা|taka|dollars?|pounds?|ডলার)", re.I)
DIGITS_RE = re.compile(r"[\d০-৯][\d০-৯,.]*")
# Small models sometimes slip into another Indic or Asian script (e.g. a Tamil greeting). Bengali
# (U+0980-U+09FF) and Latin are kept; Devanagari, Gurmukhi to Sinhala, Thai, Japanese, Chinese and
# Korean are removed while streaming.
FOREIGN_SCRIPT_RE = re.compile(r"[ऀ-ॿ਀-෿฀-๿぀-ヿ一-鿿가-힯]+")


@dataclass
class Turn:
    client_id: str
    text: str
    conversation_id: str | None = None
    quick_reply: dict | None = None
    language: str | None = None  # "auto" or one of LANGUAGES
    client_ip: str | None = None


def make_title(text: str, limit: int = 48) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return (cut or text[:limit]) + "…"


def _pct(x: float, lang: str) -> str:
    s = f"{x * 100:.0f}%"
    return P.bn_digits(s) if lang == "bengali" else s


def _num(x: float) -> str:
    return f"{x:g}" if float(x) != int(x) else f"{int(x)}"


class ChatEngine:
    def __init__(self, store: Store | None = None, tool: RiskTool | None = None,
                 llm: LLMRouter | None = None, retriever: Retriever | None = None):
        self.store = store or Store()
        self.tool = tool or RiskTool()
        self.llm = llm or LLMRouter.from_env()
        self.retriever = retriever if retriever is not None else Retriever(store=self.store)

    # ------------------------------------------------------------------ status and summaries
    def status(self) -> dict:
        return {"mode": "ai" if self.llm.available() else "offline", "model": self.llm.model,
                "examples": len(self.retriever), "bdt_per_usd": self.tool.bdt_per_usd}

    def detailed_status(self) -> dict:
        """For counsellors: which AI endpoints are up and where data is stored (no secrets)."""
        describe = getattr(self.llm, "describe", None)
        return {**self.status(), "providers": describe() if describe else [],
                "last_used": getattr(self.llm, "last_used", None), "storage": self.store.backend,
                "examples_source": getattr(self.retriever, "source", None)}

    def plan_card(self, state: dict, lang: str) -> dict:
        """The student's 'Your plan' card: what the agent knows so far, never any risk output."""
        p, t = state.get("profile", {}), self.tool
        lang = lang if lang in LANG_KEYS else "english"

        def canon(value, options, aliases):
            return (match_name(value, options, aliases) or value) if value else None

        intake = None
        if p.get("intake") and re.match(r"^\d{4}-\d{2}", str(p["intake"])):
            year, month = int(p["intake"][:4]), int(p["intake"][5:7])
            intake = f"{P.MONTHS_BN[month - 1]} {P.bn_digits(str(year))}" if lang == "bengali" \
                else f"{P.MONTHS_EN[month - 1]} {year}"
        budget = None
        if p.get("budget_band"):
            labels = {c["value"]: c["label"] for c in P.chips_for("budget", lang, t.bdt_per_usd) if not c.get("skip")}
            budget = labels.get(p["budget_band"])
        elif p.get("budget_amount") is not None:
            amount, unit = float(p["budget_amount"]), p.get("budget_unit")
            if unit == "lakh_taka" or (unit is None and amount < 1000):
                budget = f"{P.bn_digits(_num(amount))} লাখ টাকা" if lang == "bengali" else f"{_num(amount)} lakh taka"
            elif unit == "taka" or (unit is None and amount >= 100_000):
                budget = f"{P.bn_digits(f'{amount:,.0f}')} টাকা" if lang == "bengali" else f"{amount:,.0f} taka"
            else:
                budget = f"${amount:,.0f}"
        result = None
        if p.get("gpa") is not None:
            gpa = float(p["gpa"])
            scale = float(p.get("gpa_scale") or (5 if gpa > 4 else 4))
            result = f"{gpa:.2f} / {scale:g}"
            result = P.bn_digits(result) if lang == "bengali" else result
        research = P.RESEARCH_LABELS[p["due_diligence"]][lang] if p.get("due_diligence") in P.RESEARCH_LABELS else None
        country = canon(p.get("country"), t.countries, COUNTRY_ALIASES)
        if country and lang == "bengali":
            country = P.COUNTRY_BN.get(country, country)
        items = {"country": country, "course": canon(p.get("course"), t.courses, COURSE_ALIASES),
                 "institution": canon(p.get("institution"), t.institutions, INSTITUTION_ALIASES),
                 "intake": intake, "budget": budget, "gpa": result, "research": research}
        have = {"due_diligence": research, "intake": intake, "budget": budget, "gpa": result, "country": country}
        return {"items": items, "skipped": state.get("skipped", []),
                "complete": all(have[f] for f in REQUIRED_FIELDS)}

    @staticmethod
    def counsellor_summary(conv: dict) -> dict:
        state = conv.get("state") or {}
        assessment = state.get("assessment") or {}
        view = assessment.get("counsellor_view") or {}
        return {"id": conv["id"], "title": conv.get("title") or "New chat", "updated": conv["updated"],
                "language": state.get("language"), "message_count": conv.get("message_count"),
                "status": assessment.get("status"), "risk_level": view.get("risk_level"),
                "probability": view.get("probability"), "missing": assessment.get("missing", []),
                "handoff": state.get("handoff")}

    # ------------------------------------------------------------------ one turn
    def handle(self, turn: Turn) -> Iterator[dict]:
        conv = self.store.get_conversation(turn.conversation_id) if turn.conversation_id else None
        if conv is None or conv["client_id"] != turn.client_id:
            conv = self.store.create_conversation(turn.client_id)
        cid = conv["id"]
        state = conv["state"] or {}
        for key, default in (("profile", {}), ("skipped", []), ("asked", {})):
            state.setdefault(key, default)
        yield {"type": "meta", "conversation_id": cid, "title": conv["title"]}

        text = turn.text.strip()
        self.store.add_message(cid, "user", text, {"quick_reply": turn.quick_reply} if turn.quick_reply else None,
                               client_id=turn.client_id, client_ip=turn.client_ip)
        detected = None if turn.quick_reply else detect_language(text)
        lang = turn.language if turn.language in LANGUAGES else (detected or state.get("language") or "english")
        state["language"] = lang

        # Profile
        awaiting = state.get("awaiting")
        updates, skipped = self._read_profile(turn, text, awaiting)
        profile = state["profile"]
        if "budget_band" in updates:
            profile.pop("budget_amount", None)
            profile.pop("budget_unit", None)
        if "budget_amount" in updates:
            profile.pop("budget_band", None)
        if "gpa" in updates:
            profile.pop("gpa_scale", None)
        profile.update({k: v for k, v in updates.items() if v is not None})
        learned = {FIELD_OF_KEY.get(k, k) for k, v in updates.items() if v is not None}
        state["skipped"] = [f for f in dict.fromkeys(state["skipped"] + skipped) if f not in learned]

        intent = None if turn.quick_reply else detect_intent(text)
        words = len(text.split())
        answered = bool(awaiting) and (awaiting in learned or awaiting in skipped)
        data_only = bool(turn.quick_reply) or (answered and words <= 8 and "?" not in text) or \
            (bool(learned) and intent is None and words <= 6 and "?" not in text)

        # Score with SARP-Net (stored for counsellors only)
        result = self.tool.assess({k: profile[k] for k in PROFILE_KEYS if k in profile})
        state["assessment"] = result
        missing = result.get("missing", []) if result["status"] == "incomplete" else []
        next_field = None
        if intent not in ("handoff", "integrity"):
            next_field = next((f for f in P.CONVERSATION_ORDER if f in missing and f not in state["skipped"]
                               and state["asked"].get(f, 0) < 2), None)
        share_plan = result["status"] == "ok" and (not state.get("plan_shared") or bool(learned & set(REQUIRED_FIELDS)))
        yield {"type": "plan", "plan": self.plan_card(state, lang)}

        after = []  # fixed text that follows the main answer
        if share_plan:
            after.append(self._plan_text(result, lang))
        if next_field:
            after.append(P.QUESTIONS[next_field][lang])

        if intent == "integrity":
            head = P.INTEGRITY_REPLY[lang]
        elif intent == "handoff":
            head = P.HANDOFF_REPLY[lang]
        elif intent == "greeting":
            head = P.GREETING_REPLY[lang]
        elif data_only:
            head = "" if share_plan else P.ack(lang)
        else:
            head = None  # open question: the language model answers

        message_id = self.store.add_message(cid, "assistant", "")
        out: list[str] = []
        chips = P.chips_for(next_field, lang, self.tool.bdt_per_usd) if next_field else []
        try:
            if head is None:
                yield from self._model_answer(cid, text, lang, state, result, share_plan, out)
                if after:
                    yield from self._typewriter("\n\n" + "\n\n".join(after), out)
            else:
                yield from self._typewriter("\n\n".join(x for x in [head, *after] if x), out)
            if chips:
                yield {"type": "chips", "chips": chips}
            if intent == "handoff":
                yield {"type": "handoff"}
        finally:
            # Saved even if the student presses stop mid-reply.
            self.store.update_message(message_id, "".join(out).strip(), {"chips": chips})
            if next_field:
                state["asked"][next_field] = state["asked"].get(next_field, 0) + 1
            state["awaiting"] = next_field
            if share_plan:
                state["plan_shared"] = True
            title = conv["title"] or make_title(text)
            self.store.update_conversation(cid, title=title, state=state)
        if not conv["title"]:
            yield {"type": "meta", "conversation_id": cid, "title": title}
        yield {"type": "done", "message_id": message_id}

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _read_profile(turn: Turn, text: str, awaiting: str | None) -> tuple[dict, list[str]]:
        qr = turn.quick_reply
        if qr:
            key = qr.get("field")
            if qr.get("skip"):
                field = FIELD_OF_KEY.get(key, key)
                return {}, [field] if field in REQUIRED_FIELDS else []
            if key in QUICK_REPLY_KEYS and qr.get("value") is not None:
                return {key: qr["value"]}, []
            return {}, []
        return extract(text, awaiting)

    type_delay = 0.012  # seconds per word for fixed text (tests set 0)

    def _typewriter(self, text: str, out: list[str]) -> Iterator[dict]:
        """Stream fixed text word by word so every reply feels the same to the student."""
        for piece in re.findall(r"\S+\s*|\s+", text):
            out.append(piece)
            yield {"type": "token", "text": piece}
            if self.type_delay:
                time.sleep(self.type_delay)

    def _plan_text(self, result: dict, lang: str) -> str:
        focus = result["student_guidance"]["focus"]
        if not focus:
            return P.PLAN_ALL_GOOD[lang]
        lines = [P.PLAN_INTRO[lang], ""] + [f"- {P.STUDENT_TOPICS[f][lang]}" for f in focus]
        if "help_research" in focus:
            phase = result["counsellor_view"]["model_profile"]["Political_Phase"]
            rates = self.tool.findings.due_diligence_by_phase.get(phase)
            if rates:
                high = max(rates[level] for level in ("Yes Low", "Yes Medium", "Yes High"))
                lines += ["", P.RESEARCH_FACT[lang].format(hi=_pct(high, lang), lo=_pct(rates["No"], lang))]
        return "\n".join(lines)

    def _known_details(self, state: dict) -> str:
        items = self.plan_card(state, "english")["items"]
        names = {"country": "Destination", "course": "Course", "institution": "University", "intake": "Intake",
                 "budget": "Yearly budget", "gpa": "Latest result", "research": "Research done"}
        return "; ".join(f"{names[k]}: {v}" for k, v in items.items() if v) or "none yet"

    def _model_answer(self, cid: str, text: str, lang: str, state: dict, result: dict, share_plan: bool,
                      out: list[str]) -> Iterator[dict]:
        examples = self.retriever.search(text, lang, k=2, min_score=0.22)
        if self.llm.available():
            context = [P.SYSTEM_PROMPT, P.REPLY_RULES.format(language=P.LANGUAGE_NAMES[lang]),
                       f"Details the student has already shared: {self._known_details(state)}."]
            if result["status"] == "ok" and result["student_guidance"]["topics"]:
                context.append("Where this student most needs support (bring in naturally only when relevant): "
                               + " ".join(result["student_guidance"]["topics"]))
            context.append("The app adds its own follow-up question after your reply"
                           + (" and a short personal plan" if share_plan else "")
                           + ", so answer only what the student asked and do not ask for their details.")
            if examples:
                context.append("Real answers from our counsellors to similar questions. Use one only if it is about "
                               "the same topic; ignore personal details that do not apply to this student:\n\n"
                               + "\n\n".join(f"Student: {e['question']}\nCounsellor: {e['answer']}" for e in examples))
            system = "\n\n".join(context)
            messages = [{"role": "system", "content": system}]
            history = [m for m in self.store.messages(cid) if m["content"].strip()]
            messages += [{"role": m["role"], "content": m["content"]} for m in history[-12:]]
            try:
                for chunk in self.llm.stream_chat(messages):
                    chunk = FOREIGN_SCRIPT_RE.sub("", chunk)
                    if not "".join(out).strip():
                        chunk = chunk.lstrip(" !.,:;-–—\n")  # drop punctuation left by a removed word
                    if not chunk:
                        continue
                    out.append(chunk)
                    yield {"type": "token", "text": chunk}
                # Guard: amounts the model states that are not in the counsellor examples get a caveat.
                grounded = {d.translate(str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")).replace(",", "")
                            for d in DIGITS_RE.findall(system)}
                stated = [DIGITS_RE.search(a).group(0) for a in AMOUNT_RE.findall("".join(out))]
                if any(s.translate(str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")).replace(",", "").rstrip(".")
                       not in grounded for s in stated):
                    yield from self._typewriter("\n\n" + P.AMOUNT_NOTE[lang], out)
                return
            except Exception as error:  # model stopped or unreachable: fall back below
                if out:
                    note = "\n\n_(The reply was cut short. Please try again.)_"
                    out.append(note)
                    yield {"type": "token", "text": note}
                    return
                yield {"type": "notice", "text": f"AI model unavailable ({type(error).__name__}); "
                                                 "showing answers from past counsellor conversations."}
        if examples and examples[0]["score"] >= 0.3:
            reply = f"{P.RETRIEVED_PREFIX[lang]}\n\n> " + examples[0]["answer"].replace("\n", "\n> ")
        else:
            reply = P.GENERAL_REPLY[lang]
        yield from self._typewriter(reply, out)

    # ------------------------------------------------------------------ other actions
    def request_callback(self, conversation_id: str, name: str, contact: str, preferred_time: str | None) -> dict:
        conv = self.store.get_conversation(conversation_id)
        state = conv["state"] or {}
        state["handoff"] = {"name": name, "contact": contact, "preferred_time": preferred_time,
                            "requested_at": time.time()}
        lang = state.get("language", "english")
        when = P.HANDOFF_WHEN[lang].format(time=preferred_time) if preferred_time else ""
        text = P.HANDOFF_CONFIRM[lang].format(name=name, contact=contact, when=when)
        message_id = self.store.add_message(conversation_id, "assistant", text)
        self.store.update_conversation(conversation_id, state=state)
        return {"message_id": message_id, "content": text}

    def reset_plan(self, conversation_id: str) -> dict:
        conv = self.store.get_conversation(conversation_id)
        state = conv["state"] or {}
        for key in ("profile", "skipped", "asked", "awaiting", "plan_shared", "assessment"):
            state.pop(key, None)
        self.store.update_conversation(conversation_id, state=state)
        return self.plan_card(state, state.get("language", "english"))
