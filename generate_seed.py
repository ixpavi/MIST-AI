import os, psycopg2; from dotenv import load_dotenv; load_dotenv()
conn=psycopg2.connect(host=os.getenv('DB_HOST'), port=os.getenv('DB_PORT'), dbname=os.getenv('DB_NAME'), user=os.getenv('DB_USER'), password=os.getenv('DB_PASS'))
cur=conn.cursor()
cur.execute('SELECT subject_name, subject_code, pyq_link FROM pyq_resources ORDER BY pyq_link')
rows = cur.fetchall()

sql = "INSERT INTO pyq_resources (subject_name, subject_code, pyq_link, source) VALUES\n"
values = []
for name, code, link in rows:
    name = name.replace("'", "''")
    values.append(f"    ('{name}', '{code}', '{link}', 'The Helpers')")

sql += ",\n".join(values) + "\nON CONFLICT (subject_name, subject_code) DO NOTHING;"

with open("seed_data.txt", "w") as f:
    f.write(sql)
