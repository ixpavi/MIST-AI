import os
import io
import re
import requests
import psycopg2
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from dotenv import load_dotenv

# Load database credentials from .env
load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "postgres")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS")

def get_db_connection():
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS
    )

def clean_text(text):
    """Remove extra spaces, newlines, and tabs."""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def scrape_pdf(url):
    """Download a PDF from a URL and extract its text."""
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        
        pdf_file = io.BytesIO(response.content)
        reader = PdfReader(pdf_file)
        
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + " "
                
        # Use filename as title if it's a PDF
        title = url.split("/")[-1].replace(".pdf", "").replace("-", " ").title()
        if not title:
            title = "SRM PDF Document"
            
        return title, clean_text(text)
    except Exception as e:
        print(f"[ERROR] Failed to read PDF {url}: {e}")
        return None, None

def scrape_webpage(url):
    """Scrape text from a standard webpage."""
    try:
        # Use a standard browser User-Agent so we don't get blocked
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Get the title
        title = soup.title.string if soup.title else "SRM Webpage"
        title = title.replace("\n", "").strip()
        
        # Remove navigation, headers, footers, and scripts so we only get pure content
        for element in soup(["script", "style", "nav", "header", "footer"]):
            element.decompose()
            
        text = soup.get_text(separator=' ')
        return title, clean_text(text)
    except Exception as e:
        print(f"[ERROR] Failed to scrape {url}: {e}")
        return None, None

def save_to_database(title, url, content):
    """Save the scraped content into the Supabase database."""
    if not content or len(content) < 50:
        print(f"[WARNING] Not enough content found for {url}. Skipping.")
        return

    try:
        conn = get_db_connection()
        conn.autocommit = True
        with conn.cursor() as cur:
            # Insert or update if the URL already exists
            cur.execute("""
                INSERT INTO website_content (page_title, url, content)
                VALUES (%s, %s, %s)
                ON CONFLICT (url) 
                DO UPDATE SET 
                    page_title = EXCLUDED.page_title,
                    content = EXCLUDED.content;
            """, (title, url, content))
        print(f"[SUCCESS] Saved '{title}' to database!")
    except Exception as e:
        print(f"[ERROR] Database error for {url}: {e}")
    finally:
        if 'conn' in locals() and conn:
            conn.close()

def run_scraper(urls):
    """Main function to iterate through URLs and scrape them."""
    print("Starting SRM Web Scraper...")
    for url in urls:
        print(f"\nScraping: {url}")
        
        if url.lower().endswith(".pdf"):
            title, content = scrape_pdf(url)
        else:
            title, content = scrape_webpage(url)
            
        if title and content:
            save_to_database(title, url, content)
    
    print("\nScraping Complete! MIST AI can now answer questions using this data.")

if __name__ == "__main__":
    # Add any official SRM URLs or PDF links here!
    TARGET_URLS = [
        "https://www.srmist.edu.in/",
        "https://www.srmist.edu.in/about-us/",
        "https://www.srmist.edu.in/placements/",
        "https://www.srmist.edu.in/admission-india/"
    ]
    
    run_scraper(TARGET_URLS)
