import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

conn = psycopg2.connect(host=os.getenv('DB_HOST'), port=os.getenv('DB_PORT'), dbname=os.getenv('DB_NAME'), user=os.getenv('DB_USER'), password=os.getenv('DB_PASS'))
cur = conn.cursor()
cur.execute("SELECT subject_name, subject_code FROM pyq_resources WHERE subject_name='Database Management Systems';")
print(cur.fetchall())
