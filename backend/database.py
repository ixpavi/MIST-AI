"""
database.py - PostgreSQL connection helper for MIST AI
"""

import psycopg2
import os
import traceback


DB_CONFIG = {
    "host":     os.getenv("DB_HOST", "localhost"),
    "port":     os.getenv("DB_PORT", "5432"),
    "dbname":   os.getenv("DB_NAME", "mist_ai"),
    "user":     os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASS", "1234"),
}


def get_connection():
    """Return a new psycopg2 connection."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except psycopg2.OperationalError as e:
        print(f"[DB ERROR] Cannot connect to database: {e}")
        raise


def execute_query(sql, params=None):
    """Execute a write query (INSERT/UPDATE/DELETE) and commit."""
    conn = None
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute(sql, params)
        conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
        print(f"[DB ERROR] execute_query failed: {e}")
        raise
    finally:
        if conn:
            conn.close()


def execute_fetch(sql, params=None, fetch_one=False):
    """Execute a SELECT query and return results."""
    conn = None
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchone() if fetch_one else cur.fetchall()
    except Exception as e:
        print(f"[DB ERROR] execute_fetch failed: {e}")
        return None if fetch_one else []
    finally:
        if conn:
            conn.close()


def init_db():
    """Run schema.sql to create tables and seed data."""
    schema_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "database", "schema.sql")
    schema_path = os.path.normpath(schema_path)

    if not os.path.exists(schema_path):
        print(f"[WARNING] schema.sql not found at {schema_path}")
        return False

    with open(schema_path, "r", encoding="utf-8") as f:
        sql = f.read()

    conn = None
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()
        print("[OK] Database schema initialized successfully.")
        return True
    except Exception as e:
        if conn:
            conn.rollback()
        print(f"[ERROR] Failed to initialize database: {e}")
        traceback.print_exc()
        return False
    finally:
        if conn:
            conn.close()


if __name__ == "__main__":
    init_db()
