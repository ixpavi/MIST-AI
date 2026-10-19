"""Load the knowledge base into Postgres (table kb_chunks).

  python ingest_kb.py --init       apply database/schema.sql first (tables, seed rows), then ingest
  python ingest_kb.py              ingest backend/data/*.json + qa_seed.csv + university_info rows
  python ingest_kb.py --promote    also add thumbs-up answers from chat_logs as FAQ entries

Embeddings are created when GEMINI_API_KEY is set; without it chunks are stored without embeddings
(keyword retrieval still works) and a later run fills them in. Connection settings come from the
environment / .env (DATABASE_URL or DB_HOST, DB_PORT, DB_USER, DB_PASS, DB_NAME).
"""
import argparse
import csv
import hashlib
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend"))

import database  # noqa: E402
import engine  # noqa: E402

DATA = Path(__file__).resolve().parent / "backend" / "data"


def _id(prefix, text):
    return f"{prefix}-{hashlib.sha1(text.encode('utf-8')).hexdigest()[:10]}"


def qa_seed_chunks():
    path = DATA / "qa_seed.csv"
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return [{"id": _id("qa", r["question"]), "topic": "FAQ", "campus": "All",
                 "text": f"Q: {r['question'].strip()} A: {r['answer'].strip()}", "source": None}
                for r in csv.DictReader(f) if r.get("question") and r.get("answer")]


def university_info_chunks():
    rows = database.execute_fetch("SELECT topic, question, answer, source_url FROM university_info")
    return [{"id": _id("uni", q or a), "topic": t or "General", "campus": "All",
             "text": f"Q: {q} A: {a}" if q else a, "source": src}
            for t, q, a, src in rows if a]


def promoted_chunks():
    rows = database.execute_fetch(
        "SELECT question, response FROM chat_logs WHERE rating = 1 AND grounded ORDER BY chat_id")
    return [{"id": _id("fb", q), "topic": "FAQ", "campus": "All", "text": f"Q: {q} A: {a}", "source": None}
            for q, a in rows if q and a]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", action="store_true", help="apply database/schema.sql before ingesting")
    ap.add_argument("--promote", action="store_true", help="add thumbs-up answers as FAQ entries")
    args = ap.parse_args()

    if not database.is_configured():
        sys.exit("Database is not configured. Set DATABASE_URL or DB_HOST/DB_USER/DB_PASS (env or .env).")
    if args.init:
        print("schema:", "applied" if database.init_db() else "FAILED")

    chunks = engine.load_local_kb() + qa_seed_chunks() + university_info_chunks()
    if args.promote:
        promoted = promoted_chunks()
        print(f"promoting {len(promoted)} thumbs-up answers")
        chunks += promoted
    seen, unique = set(), []
    for c in chunks:
        if c["id"] not in seen:
            seen.add(c["id"])
            unique.append(c)

    # keep embeddings already stored so reruns only embed new or changed text
    existing = {c["id"]: c for c in database.load_chunks()}
    for c in unique:
        old = existing.get(c["id"])
        if old and old["text"] == c["text"] and old.get("embedding"):
            c["embedding"] = old["embedding"]
    engine.ensure_embeddings(unique)
    database.upsert_chunks(unique)
    embedded = sum(1 for c in unique if c.get("embedding"))
    print(f"stored {len(unique)} chunks ({embedded} with embeddings)")
    if embedded < len(unique):
        print("note: set GEMINI_API_KEY and rerun to add embeddings; keyword retrieval works meanwhile.")


if __name__ == "__main__":
    main()
