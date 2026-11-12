"""Find how the agency's counsellors answered similar questions in the real recorded conversations.

The matches are given to the language model as examples (retrieval-augmented generation), so replies
follow the agency's own advice and tone. Only the train and validation splits are used; the test split
stays untouched for evaluation.

Where the examples come from:
* On the team PC: ../data/splits/{train,val}.jsonl (made by data_prep/parse_conversations.py).
* Online: the `examples` table in the database, filled once by `python -m agent.seed_examples`.
The conversations are confidential, so they are never committed to the (public) repository.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from agent.risk_tool import REPO_DIR

DEFAULT_DATA_DIR = Path(os.environ.get("SARP_DATA_DIR", REPO_DIR.parent / "data"))
SPLITS = ("train", "val")


def exchanges_from_conversation(conv: dict) -> list[dict]:
    """Student question + counsellor answer pairs that are long enough to be useful examples."""
    out = []
    turns = conv.get("turns", [])
    for prev, turn in zip(turns, turns[1:]):
        if prev["role"] == "user" and turn["role"] == "assistant":
            q, a = prev["text"].strip(), turn["text"].strip()
            if len(q.split()) >= 4 and len(a.split()) >= 8:
                out.append({"question": q, "answer": a, "conversation": conv["id"],
                            "language": conv.get("language_detected"), "destination": conv.get("destination")})
    return out


def exchanges_from_files(data_dir: Path = DEFAULT_DATA_DIR) -> list[dict]:
    exchanges: list[dict] = []
    for split in SPLITS:
        path = Path(data_dir) / "splits" / f"{split}.jsonl"
        if path.exists():
            with path.open(encoding="utf-8") as f:
                for line in f:
                    exchanges += exchanges_from_conversation(json.loads(line))
    return exchanges


class Retriever:
    def __init__(self, data_dir: Path = DEFAULT_DATA_DIR, store=None):
        self.exchanges = exchanges_from_files(data_dir)
        self.source = "files" if self.exchanges else None
        if not self.exchanges and store is not None:
            self.exchanges = store.load_examples()
            self.source = "database" if self.exchanges else None
        self.vectorizer = None
        if self.exchanges:
            # Character n-grams cope with Banglish spelling variation and Bengali script alike.
            self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
            self.matrix = self.vectorizer.fit_transform([e["question"] for e in self.exchanges])

    def __len__(self) -> int:
        return len(self.exchanges)

    def search(self, text: str, language: str | None = None, k: int = 3, min_score: float = 0.12) -> list[dict]:
        if not self.vectorizer or not text.strip():
            return []
        scores = linear_kernel(self.vectorizer.transform([text]), self.matrix)[0]
        ranked = []
        for i in scores.argsort()[::-1][: k * 6]:
            score = float(scores[i]) + (0.05 if language and self.exchanges[i]["language"] == language else 0.0)
            if score >= min_score:
                ranked.append({**self.exchanges[i], "score": round(score, 3)})
        ranked.sort(key=lambda e: -e["score"])
        seen, out = set(), []
        for e in ranked:  # one example per conversation
            if e["conversation"] not in seen:
                seen.add(e["conversation"])
                out.append(e)
            if len(out) == k:
                break
        return out
