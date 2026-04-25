import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

conn = psycopg2.connect(host=os.getenv('DB_HOST'), port=os.getenv('DB_PORT'), dbname=os.getenv('DB_NAME'), user=os.getenv('DB_USER'), password=os.getenv('DB_PASS'))
cur = conn.cursor()

# Delete old mock data that has broken links
cur.execute("DELETE FROM pyq_resources WHERE pyq_link NOT LIKE '%%/subjects/%%'")

# Add acronyms to all new scraped data
cur.execute("SELECT subject_name FROM pyq_resources")
rows = cur.fetchall()
for r in rows:
    name = r[0]
    # Fix Database to Data Base to generate DBMS instead of DMS
    mod_name = name.replace("Database", "Data Base").replace("-", " ")
    words = [w for w in mod_name.split() if w.upper() not in ('AND', 'OF', 'IN', 'FOR', 'TO', 'THE', '&')]
    acronym = "".join([w[0].upper() for w in words if w])
    if len(acronym) >= 2:
        cur.execute("UPDATE pyq_resources SET subject_code = %s WHERE subject_name = %s", (acronym, name))
        
conn.commit()
print("Acronyms updated successfully.")
