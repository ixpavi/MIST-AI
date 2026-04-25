import os
import requests
import psycopg2
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import urllib.parse

load_dotenv()

def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT", "5432"),
        dbname=os.getenv("DB_NAME", "postgres"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASS")
    )

def scrape_helpers():
    base_url = "https://thehelpers.vercel.app"
    subjects = set()
    
    # Scrape semesters 1 to 8
    for sem in range(1, 9):
        sem_url = f"{base_url}/semesters/{sem}"
        print(f"Scraping {sem_url}...")
        try:
            response = requests.get(sem_url, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                # Look for all links that contain /subjects/
                for a in soup.find_all("a", href=True):
                    href = a["href"]
                    if "/subjects/" in href:
                        subject_name = a.text.strip()
                        subject_link = base_url + href if href.startswith("/") else href
                        subjects.add((subject_name, subject_link))
        except Exception as e:
            print(f"Error on sem {sem}: {e}")
            
    print(f"Found {len(subjects)} subjects. Inserting into DB...")
    
    conn = get_db_connection()
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            for name, link in subjects:
                # Insert into pyq_resources
                cur.execute("""
                    INSERT INTO pyq_resources (subject_name, subject_code, pyq_link, source)
                    VALUES (%s, %s, %s, 'The Helpers')
                    ON CONFLICT (subject_name, subject_code) 
                    DO UPDATE SET pyq_link = EXCLUDED.pyq_link;
                """, (name, "", link))
        print("Success! Database populated with subject PYQ links.")
    finally:
        conn.close()

if __name__ == "__main__":
    scrape_helpers()
