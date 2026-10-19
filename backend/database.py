"""PostgreSQL (Supabase) access for MIST AI: pooled connections, no hard-coded credentials."""
import logging
import os
import threading
from contextlib import contextmanager

import psycopg2
from psycopg2 import pool

import config

log = logging.getLogger("mist.db")
SCHEMA_PATH = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "database", "schema.sql"))

_pool = None
_lock = threading.Lock()


def is_configured():
    return bool(config.get("DATABASE_URL") or (config.get("DB_HOST") and config.get("DB_USER") and config.get("DB_PASS")))


def _connect_args():
    url = config.get("DATABASE_URL")
    if url:
        return {"dsn": url, "connect_timeout": 8}
    host = config.get("DB_HOST", "localhost")
    local = host in ("localhost", "127.0.0.1")
    return {
        "host": host, "port": config.get("DB_PORT", "5432"), "dbname": config.get("DB_NAME", "postgres"),
        "user": config.get("DB_USER"), "password": config.get("DB_PASS"),
        "sslmode": config.get("DB_SSLMODE", "prefer" if local else "require"), "connect_timeout": 8,
    }


def _get_pool():
    global _pool
    if _pool is None:
        with _lock:
            if _pool is None:
                if not is_configured():
                    raise RuntimeError("Database is not configured (set DATABASE_URL or DB_HOST/DB_USER/DB_PASS)")
                _pool = pool.ThreadedConnectionPool(1, 6, **_connect_args())
    return _pool


@contextmanager
def connection():
    p = _get_pool()
    conn = p.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        try:
            p.putconn(conn, close=bool(conn.closed))
        except Exception:
            pass


def reset_pool():
    global _pool
    with _lock:
        if _pool is not None:
            try:
                _pool.closeall()
            except Exception:
                pass
        _pool = None


def execute_query(sql, params=None, returning=False):
    """Write query. With returning=True the first row is returned."""
    try:
        with connection() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchone() if returning else None
    except psycopg2.OperationalError:
        reset_pool()   # stale pooled connection after a DB restart/pause
        raise


def execute_fetch(sql, params=None, fetch_one=False):
    """SELECT helper. Returns [] / None on failure so one bad query never breaks a chat turn."""
    try:
        with connection() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchone() if fetch_one else cur.fetchall()
    except psycopg2.OperationalError as e:
        log.warning("DB unreachable: %s", str(e)[:120])
        reset_pool()
    except Exception as e:
        log.warning("query failed: %s", str(e)[:160])
    return None if fetch_one else []


def db_ok():
    try:
        with connection() as conn, conn.cursor() as cur:
            cur.execute("select 1")
            return True
    except Exception:
        reset_pool()
        return False


def init_db():
    """Create tables / seed data. The schema is idempotent (IF NOT EXISTS / ON CONFLICT DO NOTHING)."""
    if not is_configured():
        log.warning("init_db skipped: database not configured")
        return False
    try:
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            sql = f.read()
        with connection() as conn, conn.cursor() as cur:
            cur.execute(sql)
        log.info("schema ready")
        return True
    except Exception as e:
        log.error("init_db failed: %s", str(e)[:200])
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("OK" if init_db() else "FAILED")


# ---------------------------------------------------------------- knowledge base / logging helpers
def upsert_chunks(chunks):
    import json
    with connection() as conn, conn.cursor() as cur:
        for c in chunks:
            cur.execute(
                """INSERT INTO kb_chunks (id, topic, campus, text, source, embedding, updated_at)
                   VALUES (%s,%s,%s,%s,%s,%s, now())
                   ON CONFLICT (id) DO UPDATE SET topic=EXCLUDED.topic, campus=EXCLUDED.campus,
                     text=EXCLUDED.text, source=EXCLUDED.source, embedding=EXCLUDED.embedding, updated_at=now()""",
                (c["id"], c.get("topic", "General"), c.get("campus", "All"), c["text"], c.get("source"),
                 json.dumps(c["embedding"]) if c.get("embedding") else None))


def load_chunks():
    import json
    rows = execute_fetch("SELECT id, topic, campus, text, source, embedding FROM kb_chunks ORDER BY id")
    out = []
    for id_, topic, campus, text, source, emb in rows:
        if isinstance(emb, str):
            emb = json.loads(emb)
        out.append({"id": id_, "topic": topic, "campus": campus, "text": text, "source": source, "embedding": emb})
    return out


def log_chat(session_id, question, answer, res):
    """Returns the new chat_id or None. Never raises."""
    try:
        row = execute_query(
            """INSERT INTO chat_logs (session_id, question, response, route, intent, intent_conf, grounded)
               VALUES (%s,%s,%s,%s,%s,%s,%s) RETURNING chat_id""",
            (session_id, question, answer, res.get("route"), res.get("intent"),
             res.get("intent_conf"), bool(res.get("grounded"))), returning=True)
        return row[0] if row else None
    except Exception as e:
        log.warning("log_chat failed: %s", str(e)[:120])
        return None


def save_feedback(chat_id, rating):
    try:
        row = execute_query("UPDATE chat_logs SET rating=%s WHERE chat_id=%s RETURNING chat_id",
                            (rating, chat_id), returning=True)
        return row is not None
    except Exception as e:
        log.warning("save_feedback failed: %s", str(e)[:120])
        return False


def get_stats():
    rows = execute_fetch(
        """SELECT count(*), count(*) FILTER (WHERE grounded), count(*) FILTER (WHERE rating = 1),
                  count(*) FILTER (WHERE rating = -1) FROM chat_logs""")
    if not rows:
        return {"total": 0, "grounded": 0, "up": 0, "down": 0}
    t, g, u, d = rows[0]
    return {"total": t, "grounded": g, "up": u, "down": d}
