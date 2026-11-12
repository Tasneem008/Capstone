"""Read profile details from what a student types, in English, Banglish or Bengali.

Rule-based on purpose: it is fast, predictable and easy to explain, and the website's quick-reply buttons
cover the cases rules cannot (such as how much research the student has done). Every value found is
re-checked by the risk tool's normaliser before it reaches the model.
"""
from __future__ import annotations

import re
import unicodedata
from datetime import date

from agent.prompts import MONTHS_BN, MONTHS_EN

BN_DIGITS_TO_ASCII = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def _compile(patterns: list[str]) -> list[re.Pattern]:
    return [re.compile(_nfc(p), re.I) for p in patterns]


COUNTRY_PATTERNS = {name: _compile(pats) for name, pats in {
    "UK": [r"\buk\b", r"united kingdom", r"\bengland\b", r"\bbritain\b", r"\bscotland\b", r"\blondon\b",
           "যুক্তরাজ্য", "ইউকে", "লন্ডন"],
    "Canada": [r"\bcanada\b", "কানাডা"],
    "Australia": [r"\baustralia\b", "অস্ট্রেলিয়া"],
    "New Zealand": [r"new zealand", r"\bnz\b", "নিউজিল্যান্ড"],
    "Malaysia": [r"\bmalaysia\b", "মালয়েশিয়া"],
    "USA": [r"\busa\b", r"\bamerica\b", r"united states", "আমেরিকা", "যুক্তরাষ্ট্র"],
    "Germany": [r"\bgermany\b", "জার্মানি"],
    "Ireland": [r"\bireland\b", "আয়ারল্যান্ড"],
    "Italy": [r"\bitaly\b", "ইতালি"],
    "Japan": [r"\bjapan\b", "জাপান"],
    "South Korea": [r"\bkorea\b", "কোরিয়া"],
    "China": [r"\bchina\b", "চীন"],
    "Finland": [r"\bfinland\b"],
    "Sweden": [r"\bsweden\b"],
    "Netherlands": [r"\bnetherlands\b", r"\bholland\b"],
    "France": [r"\bfrance\b"],
    "Hungary": [r"\bhungary\b"],
    "Cyprus": [r"\bcyprus\b"],
    "UAE": [r"\bdubai\b", r"\buae\b"],
}.items()}

COURSE_PATTERNS = [(name, re.compile(_nfc(p), flags)) for name, p, flags in [
    ("Computer Science", r"computer science|\bcse\b|কম্পিউটার সায়েন্স|সিএসই", re.I),
    ("Data Science", r"data science|ডেটা সায়েন্স", re.I),
    ("Business Analytics", r"business analytics", re.I),
    ("Business Administration", r"business administration|\bmba\b|\bbba\b|বিজনেস|বিবিএ|এমবিএ", re.I),
    ("Software Engineering", r"software engineering", re.I),
    ("Cyber Security", r"cyber ?security", re.I),
    ("Electrical Engineering", r"electrical|\beee\b", re.I),
    ("Mechanical Engineering", r"mechanical", re.I),
    ("Civil Engineering", r"\bcivil\b", re.I),
    ("Automotive Engineering", r"automotive", re.I),
    ("Supply Chain Management", r"supply chain", re.I),
    ("Hospitality Management", r"hospitality|tourism", re.I),
    ("Environmental Science", r"environmental", re.I),
    ("Public Health", r"public health|\bmph\b", re.I),
    ("Nursing", r"\bnursing\b|নার্সিং", re.I),
    ("Psychology", r"psychology", re.I),
    ("Accounting", r"accounting", re.I),
    ("Finance", r"\bfinance\b", re.I),
    ("Marketing", r"\bmarketing\b", re.I),
    ("Agriculture", r"agricultur", re.I),
    ("Law", r"\blaw\b|\bllb\b|\bllm\b", re.I),
    ("AI", r"artificial intelligence|\bAI\b", 0),          # 'AI' only in capitals
    ("IT", r"information technology|\bIT\b|আইটি", 0),        # 'IT' only in capitals
]]

INSTITUTION_PATTERNS = {name: _compile(pats) for name, pats in {
    "University of Toronto": [r"university of toronto", r"toronto university", r"\buoft\b", r"\bu of t\b"],
    "University of British Columbia": [r"\bubc\b", r"british columbia"],
    "York University": [r"(?<!new )york university", r"\byorku\b"],
    "Monash University": [r"\bmonash\b"],
    "RMIT University": [r"\brmit\b"],
    "Deakin University": [r"\bdeakin\b"],
    "Torrens University": [r"\btorrens\b"],
    "University of Melbourne": [r"university of melbourne", r"melbourne university", r"\bunimelb\b"],
    "University of Sydney": [r"university of sydney", r"sydney university", r"\busyd\b"],
    "University of Auckland": [r"university of auckland", r"auckland university"],
    "University of Malaya": [r"university of malaya", r"universiti malaya", r"malaya university"],
    "Coventry University": [r"\bcoventry\b"],
    "King's College London": [r"king'?s college", r"\bkcl\b"],
    "University of Birmingham": [r"university of birmingham", r"birmingham university"],
    "University of Leeds": [r"university of leeds", r"leeds university"],
    "University of Manchester": [r"university of manchester", r"manchester university"],
    "Arizona State University": [r"arizona state", r"\basu\b"],
    "Technical University of Munich": [r"technical university of munich", r"\btum\b", r"tu munich"],
    "University College Dublin": [r"university college dublin", r"\bucd\b"],
}.items()}
OTHER_UNIVERSITY_RE = re.compile(
    r"\b(University of [A-Z][a-zA-Z]+(?: [A-Z][a-zA-Z]+)?|(?:[A-Z][a-zA-Z]+ ){1,3}University)\b")
NOT_A_NAME = {"which", "a", "the", "any", "good", "best", "private", "public", "top", "my", "your", "that",
              "this", "one", "same", "other", "another", "what", "which", "cheap", "new"}

MONTHS = {m.lower(): i + 1 for i, m in enumerate(MONTHS_EN)}
MONTHS.update({m[:3].lower(): i + 1 for i, m in enumerate(MONTHS_EN)})
MONTHS["sept"] = 9
SEASONS = {"fall": 9, "autumn": 9, "spring": 1, "winter": 1, "summer": 5}
MONTH_RE = re.compile(r"\b(" + "|".join(sorted(list(MONTHS) + list(SEASONS), key=len, reverse=True)) + r")\b\.?"
                      r"(?:\s*(?:intake|session|semester|term))?\s*,?\s*((?:19|20)\d{2})?", re.I)
YEAR_FIRST_RE = re.compile(r"\b((?:19|20)\d{2})\s+(" + "|".join(sorted(MONTHS, key=len, reverse=True)) + r")\b", re.I)
# Whole Bengali words only: "মে" (May) also appears inside words such as "ডকুমেন্ট".
BN_MONTH_RE = re.compile(r"(?<![ঀ-৿])(" + "|".join(_nfc(m) for m in MONTHS_BN) + r")(?![ঀ-৿])"
                         r"\s*((?:19|20)\d{2})?")
ISO_INTAKE_RE = re.compile(r"\b(20\d{2})[-/](0?[1-9]|1[0-2])\b|\b(0?[1-9]|1[0-2])/(20\d{2})\b")
INTAKE_WORD_RE = re.compile(r"intake|session|semester|start|begin|join|admission|jete|jabo|vorti|bhorti|"
                            r"ইনটেক|সেশন|যেতে|ভর্তি|শুরু", re.I)

NUM = r"(\d+(?:[.,]\d+)*)"
LAKH = r"(?:lakh|lakhs|lac|lacs|laakh|লাখ|লক্ষ)"
LAKH_RANGE_RE = re.compile(NUM + r"\s*(?:-|–|to|theke|থেকে)\s*" + NUM + r"\s*" + LAKH, re.I)
LAKH_RE = re.compile(NUM + r"\s*" + LAKH, re.I)
CRORE_RE = re.compile(NUM + r"\s*(?:crore|koti|কোটি)", re.I)
USD_RE = re.compile(r"(?:\$|us\$|usd)\s*" + NUM + r"\s*(k)?\b|\b" + NUM + r"\s*(k)?\s*(?:usd|dollars?|ডলার)", re.I)
TAKA_RE = re.compile(NUM + r"\s*(?:taka|tk|bdt|টাকা)", re.I)

GPA_RE = re.compile(
    r"(?:c\.?g\.?p\.?a|g\.?p\.?a|result|grade|সিজিপিএ|জিপিএ|রেজাল্ট)\s*(?:is|was|of|holo|hocche|ache|:|=|-)?\s*"
    r"(?:about|around|approx\.?|prai|প্রায়)?\s*([0-5](?:\.\d{1,2})?)(?!\.?\d)"
    r"(?:\s*(?:/|out of|er moddhe|এর মধ্যে)\s*([45])(?:\.0+)?)?", re.I)
# (?!\.?\d) lets a number end a sentence ("CGPA 3.2.") while still rejecting "3.25" read as "3.2".
OUT_OF_RE = re.compile(r"(?<![\d.])([0-5]\.\d{1,2})\s*(?:/|out of)\s*([45])(?:\.0+)?(?!\.?\d)", re.I)
BARE_NUMBER_RE = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)(?!\.?\d)")
EXPLICIT_GPA_RE = re.compile(r"\s*(?:c\.?g\.?p\.?a|g\.?p\.?a|সিজিপিএ|জিপিএ)", re.I)
TEST_SCORE_RE = re.compile(r"ielts|toefl|pte|duolingo|band|sat\b|gre\b|gmat", re.I)
FIVE_SCALE_RE = re.compile(r"\bhsc\b|\bssc\b|a.?level|এইচএসসি|এসএসসি", re.I)

DUE_DILIGENCE_RULES = [  # checked only when the agent has just asked about research
    ("Yes High", re.compile(r"(know|jani|জানি).{0,40}(requirement|cost|khoroch|visa|ভিসা|খরচ)|"
                            r"(done|did|finished) (my|all the) research|fully researched", re.I)),
    ("Yes Medium", re.compile(r"shortlist|শর্টলিস্ট|compared|kichu (course|university) dekhechi|few (universities|courses)", re.I)),
    ("Yes Low", re.compile(r"(decided|thik korechi|ঠিক করেছি|chosen).{0,30}(country|subject|desh|দেশ|বিষয়)|"
                           r"(country|subject|desh).{0,20}(decided|thik)", re.I)),
    ("No", re.compile(r"not yet|no idea|haven'?t|have not|kichu(i)? jani na|ekhono (kichu )?dekhini|"
                      r"kono idea nai|জানি না|এখনো|nothing yet|just started", re.I)),
]
SKIP_RE = re.compile(r"\b(skip|not sure|don'?t know|pore bolbo|jani na|ekhono thik kori nai|decide kori nai)\b|"
                     r"পরে বলব|এখনো ঠিক করিনি", re.I)

GREETING_RE = re.compile(r"^\s*(hi|hello|hey|salam|assalamu ?alaikum|assalamualaikum|good (morning|afternoon|evening)|"
                         r"আসসালামু আলাইকুম|হ্যালো)\b[\s!.,]*(\w+[\s!.,]*){0,2}$", re.I)
HANDOFF_RE = re.compile(r"(talk|speak|kotha|connect|call|contact|meet|chat)\w*.{0,25}(counsel+or|human|real person|"
                        r"someone|agent|staff)|কাউন্সেলরের সাথে|মানুষের সাথে কথা", re.I)
INTEGRITY_RE = re.compile(r"(fake|forg|photoshop|edit|alter|modify|tamper|bodl|banai|বানা|বদল|এডিট).{0,40}"
                          r"(certificate|transcript|document|statement|passport|result|marksheet|sertificate|"
                          r"সার্টিফিকেট|ডকুমেন্ট|স্টেটমেন্ট)|fake (bank|document|certificate)", re.I)


def _num(text: str) -> float:
    return float(text.replace(",", ""))


def _first(patterns: dict[str, list[re.Pattern]], text: str) -> str | None:
    best = None
    for name, pats in patterns.items():
        for p in pats:
            m = p.search(text)
            if m and (best is None or m.start() < best[0]):
                best = (m.start(), name)
    return best[1] if best else None


def _intake(text: str, awaiting: bool, today: date) -> str | None:
    def resolve(month: int, year: str | None) -> str:
        y = int(year) if year else (today.year if month > today.month else today.year + 1)
        return f"{y}-{month:02d}"

    m = ISO_INTAKE_RE.search(text)
    if m:
        year, month = (m.group(1), m.group(2)) if m.group(1) else (m.group(4), m.group(3))
        return f"{int(year)}-{int(month):02d}"
    m = YEAR_FIRST_RE.search(text)
    if m:
        return resolve(MONTHS[m.group(2).lower()], m.group(1))
    for m in BN_MONTH_RE.finditer(text):
        month = MONTHS_BN.index(m.group(1)) + 1
        context = text[max(0, m.start() - 25): m.end() + 25]
        if m.group(2) or INTAKE_WORD_RE.search(context) or (awaiting and (month != 5 or len(text.split()) <= 3)):
            return resolve(month, m.group(2))
    for m in MONTH_RE.finditer(text):
        word, year = m.group(1).lower(), m.group(2)
        month = MONTHS.get(word) or SEASONS.get(word)
        context = text[max(0, m.start() - 25): m.end() + 25]
        sure = bool(year) or bool(INTAKE_WORD_RE.search(context)) or awaiting
        if word == "may" and not (year or INTAKE_WORD_RE.search(context) or (awaiting and len(text.split()) <= 3)):
            continue  # "may" is usually the verb
        if sure:
            return resolve(month, year)
    return None


def _budget(text: str, awaiting: bool) -> dict:
    m = LAKH_RANGE_RE.search(text)
    if m:
        return {"budget_amount": (_num(m.group(1)) + _num(m.group(2))) / 2, "budget_unit": "lakh_taka"}
    m = LAKH_RE.search(text)
    if m:
        return {"budget_amount": _num(m.group(1)), "budget_unit": "lakh_taka"}
    m = CRORE_RE.search(text)
    if m:
        return {"budget_amount": _num(m.group(1)) * 100, "budget_unit": "lakh_taka"}
    m = USD_RE.search(text)
    if m:
        value, kilo = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
        return {"budget_amount": _num(value) * (1000 if kilo else 1), "budget_unit": "usd"}
    m = TAKA_RE.search(text)
    if m and _num(m.group(1)) >= 10_000:
        return {"budget_amount": _num(m.group(1)), "budget_unit": "taka"}
    if awaiting:
        m = BARE_NUMBER_RE.search(text)
        if m:
            return {"budget_amount": _num(m.group(1)), "budget_unit": None}
    return {}


def _gpa(text: str, awaiting: bool) -> dict:
    candidates = []
    for regex in (OUT_OF_RE, GPA_RE):
        for m in regex.finditer(text):
            # "GPA 5.00" is a GPA even after "IELTS 6.5"; "IELTS result 5.5" is not.
            labelled = regex is GPA_RE and EXPLICIT_GPA_RE.match(m.group(0))
            before = text[max(0, m.start() - 20): m.start()]
            if not labelled and (TEST_SCORE_RE.search(before) or TEST_SCORE_RE.search(m.group(0))):
                continue
            candidates.append(m)
        if candidates:
            break
    if not candidates and awaiting:
        m = BARE_NUMBER_RE.search(text)
        if m and 0 < float(m.group(1)) <= 5 and not TEST_SCORE_RE.search(text):
            candidates.append(m)
    if not candidates:
        return {}
    m = candidates[0]
    value = float(m.group(1))
    scale = float(m.group(2)) if m.re is not BARE_NUMBER_RE and m.lastindex and m.lastindex >= 2 and m.group(2) else None
    if scale is None and FIVE_SCALE_RE.search(text):
        scale = 5.0
    if not 0 < value <= 5:
        return {}
    return {"gpa": value, "gpa_scale": scale}


def _institution(text: str) -> str | None:
    known = _first(INSTITUTION_PATTERNS, text)
    if known:
        return known
    for m in OTHER_UNIVERSITY_RE.finditer(text):
        name = m.group(1).strip()
        if name.split()[0].lower() not in NOT_A_NAME:
            return name
    return None


def extract(text: str, awaiting: str | None = None, today: date | None = None) -> tuple[dict, list[str]]:
    """Return (profile updates, fields the student chose to skip)."""
    today = today or date.today()
    text = _nfc(text or "").translate(BN_DIGITS_TO_ASCII)
    updates: dict = {}

    country = _first(COUNTRY_PATTERNS, text)
    if country:
        updates["country"] = country
    for name, pattern in COURSE_PATTERNS:
        if pattern.search(text):
            updates["course"] = name
            break
    institution = _institution(text)
    if institution:
        updates["institution"] = institution
    intake = _intake(text, awaiting == "intake", today)
    if intake:
        updates["intake"] = intake
    budget = _budget(text, awaiting == "budget")
    if budget:
        updates.update(budget)
    # GPA only when no budget number was taken from the same text, unless clearly labelled.
    gpa = _gpa(text, awaiting == "gpa" and not budget)
    if gpa:
        updates.update(gpa)
    if awaiting == "due_diligence":
        for level, regex in DUE_DILIGENCE_RULES:
            if regex.search(text):
                updates["due_diligence"] = level
                break

    skipped = []
    awaited_key = {"budget": "budget_amount"}.get(awaiting, awaiting)
    if awaiting and awaited_key not in updates and SKIP_RE.search(text):
        skipped.append(awaiting)
    return updates, skipped


def detect_intent(text: str) -> str | None:
    if INTEGRITY_RE.search(text or ""):
        return "integrity"
    if HANDOFF_RE.search(text or ""):
        return "handoff"
    if GREETING_RE.match(text or ""):
        return "greeting"
    return None
