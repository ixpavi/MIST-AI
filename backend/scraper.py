"""
scraper.py - Web scraper for SRM university pages
Stores scraped content into the website_content table.
"""

import requests
from bs4 import BeautifulSoup
from database import get_connection


TARGET_URLS = [
    ("SRM Home",        "https://www.srmist.edu.in"),
    ("Admissions",      "https://www.srmist.edu.in/admissions"),
    ("Placements",      "https://www.srmist.edu.in/placements"),
    ("Departments",     "https://www.srmist.edu.in/departments"),
    ("Scholarships",    "https://www.srmist.edu.in/scholarships"),
    ("Hostel",          "https://www.srmist.edu.in/hostel"),
    ("Student Life",    "https://www.srmist.edu.in/student-life"),
    ("Research",        "https://www.srmist.edu.in/research"),
]


def fetch_page(url, timeout=15):
    """Fetch raw HTML from a URL."""
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print(f"[ERROR] Failed to fetch {url}: {e}")
        return None


def extract_text(html):
    """Extract and clean paragraph text from HTML."""
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    paragraphs = soup.find_all("p")
    text_parts = []
    for p in paragraphs:
        text = p.get_text(separator=" ", strip=True)
        if len(text) > 20:
            text_parts.append(text)

    return "\n".join(text_parts)


def store_content(page_title, url, content):
    """Upsert scraped content into the website_content table."""
    if not content.strip():
        print(f"[SKIP] {page_title}: no usable content.")
        return

    conn = None
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO website_content (page_title, url, content)
                VALUES (%s, %s, %s)
                ON CONFLICT (url) DO UPDATE
                    SET page_title = EXCLUDED.page_title,
                        content    = EXCLUDED.content
            """, (page_title, url, content))
        conn.commit()
        print(f"[OK] Stored: {page_title}")
    except Exception as e:
        if conn:
            conn.rollback()
        print(f"[ERROR] DB error for {page_title}: {e}")
    finally:
        if conn:
            conn.close()


def run_scraper():
    """Crawl all target URLs and store their content."""
    print("=" * 60)
    print("  MIST AI - Web Scraper")
    print("=" * 60)

    for title, url in TARGET_URLS:
        print(f"\n-> Scraping: {title} ({url})")
        html = fetch_page(url)
        if html:
            content = extract_text(html)
            store_content(title, url, content)

    print("\n" + "=" * 60)
    print("  Scraping complete!")
    print("=" * 60)


if __name__ == "__main__":
    run_scraper()
