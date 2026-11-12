"""Detect whether a student writes in Bengali script, Banglish (romanised Bengali) or English."""
from __future__ import annotations

import re

BENGALI_CHAR_RE = re.compile(r"[ঀ-৿]")
BANGLISH_WORDS = set("""
ami apni apnar amar amra amader apnake amake tader uni unar tumi tomar ki ji ache nei hobe korte kore korbo
korben korechi korechen chai chacchen jonno sathe tahole kintu ebong ekta ekhon bhaiya bhai apu thik accha
keno koto shob onek bolte dekhte parben pari jabe jete dorkar proyojon bishoy mane hocche chilo theke niye
diye kono onujayi obosshoi hoye hoyeche bujhte pabo paben din nin ektu shudhu hole lagbe bolchi dekhbo kothay
porte porbo jabo kivabe kemon hoy naki bolun bolen janen jani parbo chacchi korchi korbe khoroch desh
""".split())
LANGUAGES = ("english", "banglish", "bengali")


def detect_language(text: str) -> str | None:
    """Return 'bengali', 'banglish' or 'english', or None when the text is too short to tell."""
    bengali = len(BENGALI_CHAR_RE.findall(text or ""))
    latin = len(re.findall(r"[A-Za-z]", text or ""))
    if bengali and bengali >= 0.3 * (bengali + latin):
        return "bengali"
    words = re.findall(r"[a-z]+", (text or "").lower())
    hits = sum(w in BANGLISH_WORDS for w in words)
    if hits and hits / max(1, len(words)) >= 0.08:
        return "banglish"
    if len(words) >= 3:
        return "english"
    return None
