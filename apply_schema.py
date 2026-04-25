import psycopg2

with open("database/schema.sql", "r", encoding="utf-8") as f:
    sql = f.read()

try:
    conn = psycopg2.connect(
        host="aws-1-ap-northeast-1.pooler.supabase.com",
        port=6543,
        dbname="postgres",
        user="postgres.xvwwfrhkoocyhncjqsnj",
        password="Pavi@mistsupabase"
    )
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(sql)
    print("SUCCESS: Schema applied.")
except Exception as e:
    print(f"ERROR: {e}")
