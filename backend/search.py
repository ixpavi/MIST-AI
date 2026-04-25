"""
search.py - Database search and retrieval engine for MIST AI
Uses PostgreSQL full-text search to find relevant answers.
"""

import re
import string
import traceback
from database import execute_fetch


# ---------------------------------------------------------------------------
# Category keywords for intent detection
# ---------------------------------------------------------------------------
CATEGORY_KEYWORDS = {
    "pyq": [
        "pyq", "previous year", "previous year paper", "question paper",
        "old paper", "past paper", "past papers", "model paper",
        "sample paper", "question bank"
    ],
    "portal_feature": [
        "attendance", "timetable", "fee", "fees", "pay", "payment",
        "hall ticket", "hallticket", "scholarship", "transport", "bus",
        "hostel allocation", "internal marks", "marks", "course registration",
        "register course", "exam result", "results", "exam schedule",
        "book search", "e-resources", "download"
    ],
    "location": [
        "where is", "location", "located", "find", "direction",
        "library", "tech park", "food court", "canteen", "auditorium",
        "sports", "health center", "placement office", "hostel block",
        "main building", "university building"
    ],
    "portal": [
        "portal", "academia", "student portal", "login", "website",
        "exam portal", "library portal"
    ],
    "university_info": [
        "hostel", "transport", "scholarship", "department", "campus",
        "placement", "admit", "admission", "club", "fest", "facility",
        "facilities", "library timing", "library", "exam", "hall ticket",
        "fee structure", "minimum attendance"
    ],
}


def normalize(text):
    """Lowercase and strip punctuation from user input."""
    if not text:
        return ""
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text)
    return text


def detect_category(text):
    """Detect the question category based on keyword matching."""
    norm = normalize(text)
    if not norm:
        return "general"

    # Check in priority order
    for kw in CATEGORY_KEYWORDS["pyq"]:
        if kw in norm:
            return "pyq"

    for kw in CATEGORY_KEYWORDS["portal_feature"]:
        if kw in norm:
            return "portal_feature"

    for kw in CATEGORY_KEYWORDS["location"]:
        if kw in norm:
            return "location"

    for kw in CATEGORY_KEYWORDS["portal"]:
        if kw in norm:
            return "portal"

    for kw in CATEGORY_KEYWORDS["university_info"]:
        if kw in norm:
            return "university_info"

    return "general"


# ---------------------------------------------------------------------------
# Category-specific search functions
# ---------------------------------------------------------------------------

def search_portals(query):
    """Search the portals table."""
    try:
        rows = execute_fetch(
            """
            SELECT DISTINCT portal_name, link, managed_by
            FROM portals
            WHERE LOWER(portal_name) LIKE %s
            LIMIT 3
            """,
            (f"%{query}%",)
        )
        if rows:
            parts = []
            for name, link, managed in rows:
                parts.append(f"{name}\nLink: {link}\nManaged by: {managed}")
            return "\n\n".join(parts)
    except Exception as e:
        print(f"[SEARCH ERROR] search_portals: {e}")
    return None


def search_portal_features(query):
    """Search portal features using full-text search with ILIKE fallback."""
    try:
        # Try full-text search first
        rows = execute_fetch(
            """
            SELECT DISTINCT pf.feature_name, pf.description, p.portal_name, p.link
            FROM portal_features pf
            JOIN portals p ON p.portal_id = pf.portal_id
            WHERE to_tsvector('english', pf.feature_name || ' ' || COALESCE(pf.description, ''))
                  @@ plainto_tsquery('english', %s)
            ORDER BY ts_rank(
                to_tsvector('english', pf.feature_name || ' ' || COALESCE(pf.description, '')),
                plainto_tsquery('english', %s)
            ) DESC
            LIMIT 3
            """,
            (query, query)
        )

        # Fallback to ILIKE
        if not rows:
            rows = execute_fetch(
                """
                SELECT DISTINCT pf.feature_name, pf.description, p.portal_name, p.link
                FROM portal_features pf
                JOIN portals p ON p.portal_id = pf.portal_id
                WHERE LOWER(pf.feature_name) LIKE %s
                   OR LOWER(COALESCE(pf.description, '')) LIKE %s
                LIMIT 3
                """,
                (f"%{query}%", f"%{query}%")
            )

        if rows:
            parts = []
            for feat, desc, portal, link in rows:
                parts.append(f"{desc}\n\n(Available on {portal}: {link})")
            return "\n\n".join(parts)
    except Exception as e:
        print(f"[SEARCH ERROR] search_portal_features: {e}")
    return None


def search_pyq():
    """Return the fixed PYQ answer from the university_info table."""
    return "You can find all previous year question papers (PYQs) and study materials at The Helpers website: https://thehelpers.vercel.app/"


def search_locations(query):
    """Search campus locations."""
    try:
        rows = execute_fetch(
            """
            SELECT DISTINCT place_name, description
            FROM locations
            WHERE to_tsvector('english', place_name || ' ' || COALESCE(description, ''))
                  @@ plainto_tsquery('english', %s)
            ORDER BY ts_rank(
                to_tsvector('english', place_name || ' ' || COALESCE(description, '')),
                plainto_tsquery('english', %s)
            ) DESC
            LIMIT 3
            """,
            (query, query)
        )

        if not rows:
            rows = execute_fetch(
                """
                SELECT DISTINCT place_name, description
                FROM locations
                WHERE LOWER(place_name) LIKE %s
                   OR LOWER(COALESCE(description, '')) LIKE %s
                LIMIT 3
                """,
                (f"%{query}%", f"%{query}%")
            )

        if rows:
            parts = []
            for name, desc in rows:
                parts.append(f"{name}\n{desc}")
            return "\n\n".join(parts)
    except Exception as e:
        print(f"[SEARCH ERROR] search_locations: {e}")
    return None


def search_university_info(query):
    """Search general university information using full-text search."""
    try:
        rows = execute_fetch(
            """
            SELECT DISTINCT question, answer, source_url
            FROM university_info
            WHERE to_tsvector('english', COALESCE(question, '') || ' ' || COALESCE(answer, ''))
                  @@ plainto_tsquery('english', %s)
            ORDER BY ts_rank(
                to_tsvector('english', COALESCE(question, '') || ' ' || COALESCE(answer, '')),
                plainto_tsquery('english', %s)
            ) DESC
            LIMIT 1
            """,
            (query, query)
        )

        if not rows:
            rows = execute_fetch(
                """
                SELECT DISTINCT question, answer, source_url
                FROM university_info
                WHERE LOWER(COALESCE(question, '')) LIKE %s
                   OR LOWER(COALESCE(answer, '')) LIKE %s
                   OR LOWER(COALESCE(topic, '')) LIKE %s
                LIMIT 1
                """,
                (f"%{query}%", f"%{query}%", f"%{query}%")
            )

        if rows:
            parts = []
            for q, a, url in rows:
                part = str(a) if a else ""
                if url:
                    part += f"\n\nMore info: {url}"
                parts.append(part)
            return "\n\n".join(parts)
    except Exception as e:
        print(f"[SEARCH ERROR] search_university_info: {e}")
    return None


def search_website_content(query):
    """Full-text search on scraped website content as a fallback."""
    try:
        rows = execute_fetch(
            """
            SELECT page_title, url,
                   ts_headline('english', content,
                               plainto_tsquery('english', %s),
                               'StartSel=, StopSel=, MaxFragments=2, MaxWords=60')
                   AS snippet
            FROM website_content
            WHERE to_tsvector('english', content) @@ plainto_tsquery('english', %s)
            ORDER BY ts_rank(to_tsvector('english', content),
                             plainto_tsquery('english', %s)) DESC
            LIMIT 2
            """,
            (query, query, query)
        )

        if rows:
            parts = []
            for title, url, snippet in rows:
                parts.append(f"From {title} ({url}):\n{snippet}")
            return "\n\n".join(parts)
    except Exception as e:
        print(f"[SEARCH ERROR] search_website_content: {e}")
    return None


# ---------------------------------------------------------------------------
# Main search dispatcher
# ---------------------------------------------------------------------------

DEFAULT_REPLY = (
    "I'm sorry, I couldn't find specific information about that. "
    "You can try asking about:\n"
    "  - Hostel, fees, attendance, scholarships\n"
    "  - Previous year question papers (PYQ)\n"
    "  - Campus locations (library, tech park, food court)\n"
    "  - University portals (Academia, Student Portal)\n"
    "  - Placements, admissions, departments\n\n"
    "Or visit the SRM website: https://www.srmist.edu.in"
)

GREETINGS = {"hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening", "sup", "yo"}

def check_identity(text):
    for phrase in ["who made", "who created", "who developed", "who is your developer", "what is mist", "what does mist stand for"]:
        if phrase in text:
            return True
    return False

def search(user_message):
    """
    Main search function. Never crashes - always returns a string.
    """
    try:
        if not user_message or not user_message.strip():
            return "Please type a question so I can help you!"

        normalized = normalize(user_message)
        if not normalized:
            return "Please type a question so I can help you!"
            
        if normalized in GREETINGS:
            return "Hello! I am MIST AI, your SRM University assistant. How can I help you today?"
            
        if check_identity(normalized):
            return "I am MIST AI, the official student assistant for SRM University KTR campus. I was developed by GitHub user ixpavi (github.com/ixpavi)."

        category = detect_category(user_message)
        result = None

        # Route to the appropriate search function
        if category == "pyq":
            result = search_pyq()

        elif category == "portal_feature":
            result = search_portal_features(normalized)

        elif category == "portal":
            result = search_portals(normalized)

        elif category == "location":
            result = search_locations(normalized)

        elif category == "university_info":
            result = search_university_info(normalized)

        # Fallback chain
        if not result:
            result = search_university_info(normalized)
        if not result:
            result = search_portal_features(normalized)
        if not result:
            result = search_locations(normalized)
        if not result:
            result = search_website_content(normalized)
        if not result:
            result = DEFAULT_REPLY

        return result

    except Exception as e:
        print(f"[SEARCH ERROR] Unexpected error in search(): {e}")
        traceback.print_exc()
        return DEFAULT_REPLY
