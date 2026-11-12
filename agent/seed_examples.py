"""Upload the counsellor example exchanges to the database the online website uses.

Run once (and again after re-parsing the conversations) from the PC that has the private data folder,
with DATABASE_URL set to the online database (copy it from Vercel: Project, Storage, Neon, .env.local):

    $env:DATABASE_URL = "<Neon connection string>"
    .venv\\Scripts\\python.exe -m agent.seed_examples

Only the train and validation splits are uploaded; the test split stays private for evaluation.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agent.retriever import DEFAULT_DATA_DIR, exchanges_from_files
from agent.store import Store


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_DIR, help="folder holding splits/ (default ../data)")
    args = parser.parse_args()

    exchanges = exchanges_from_files(args.data)
    if not exchanges:
        sys.exit(f"No conversations found in {args.data / 'splits'}. Run data_prep/parse_conversations.py first.")
    store = Store()
    if store.backend != "postgresql":
        print("Note: DATABASE_URL is not set, so this fills the local SQLite database, not the online one.")
    count = store.replace_examples(exchanges)
    print(f"Uploaded {count} counsellor examples to {store.describe()}")


if __name__ == "__main__":
    main()
