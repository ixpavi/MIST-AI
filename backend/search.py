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
    "faculty": [
        "faculty", "professor", "prof", "teacher", "lecturer",
        "staff", "hod", "head of department", "dean",
        "who teaches", "who is dr", "who is prof",
        "sir", "maam", "madam", "ma'am"
    ],
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

    # Check in priority order  (faculty first so names don't leak to LLM)
    for kw in CATEGORY_KEYWORDS["faculty"]:
        if kw in norm:
            return "faculty"

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

# ---------------------------------------------------------------------------
# Faculty search  (returns SRM staff-finder link)
# ---------------------------------------------------------------------------

# Words to strip when extracting the faculty name from the query
_FACULTY_STRIP_WORDS = {
    "who", "is", "tell", "me", "about", "details", "of", "the",
    "faculty", "professor", "prof", "teacher", "lecturer", "staff",
    "hod", "head", "department", "dean", "sir", "maam", "madam",
    "ma'am", "give", "show", "find", "search", "info", "information",
    "teaches", "teaching", "at", "srm", "srmist", "university",
    "ktr", "kattankulathur", "campus",
    "please", "can", "you", "do", "know", "i", "want", "to",
    "what", "where", "which", "how", "contact", "email", "phone",
    "number", "cabin", "office", "room", "a", "an", "and", "in",
    "for", "from", "with", "on",
}


def _extract_faculty_name(user_text):
    """Pull a likely person-name from the user's query.

    Strategy:
    1. Strip common noise words.
    2. Whatever is left is treated as the name.
    3. Title-case it for display.
    """
    norm = normalize(user_text)
    tokens = norm.split()
    name_tokens = [t for t in tokens if t not in _FACULTY_STRIP_WORDS]
    return " ".join(name_tokens).strip().title()


def search_faculty(user_text):
    """Return the SRM staff-finder link for a faculty name.

    The SRM website provides a live search at
    https://www.srmist.edu.in/staff-finder/  which lists all matching
    faculty (including duplicates).  We also provide the direct faculty
    listing page.
    """
    name = _extract_faculty_name(user_text)

    staff_finder = "https://www.srmist.edu.in/staff-finder/"
    faculty_list = "https://www.srmist.edu.in/faculty/"

    if name:
        # Build a slug for a direct profile guess
        slug = name.lower().replace(" ", "-")
        slug_parts = slug.split("-")
        has_title = slug_parts[0] in ["dr", "mr", "mrs", "ms"]

        if has_title:
            direct_link = f'https://www.srmist.edu.in/faculty/{slug}/'
        else:
            dr_slug = f"dr-{slug}"
            direct_link = (f'https://www.srmist.edu.in/faculty/{dr_slug}/\n'
                           f'   https://www.srmist.edu.in/faculty/{slug}/')

        return (
            f"You can find the profile and details of {name} on the SRM website:\n\n"
            f"1. Search on Staff Finder (lists all matching faculty, including those with the same name):\n"
            f"   {staff_finder}\n\n"
            f"2. Direct profile link:\n"
            f"   {direct_link}\n\n"
            f"3. Browse the full faculty directory:\n"
            f"   {faculty_list}"
        )

    # No name could be extracted - return the general links
    return (
        "You can search for any faculty member on the SRM Staff Finder:\n"
        f"{staff_finder}\n\n"
        f"Or browse the complete faculty listing:\n"
        f"{faculty_list}"
    )


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


def search_pyq(query):
    """Search for specific subject PYQ links or return the generic link."""
    try:
        # Remove common words to isolate the subject name
        clean_query = query.replace("pyq", "").replace("previous year question", "").replace("paper", "").replace("give me", "").replace("for", "").strip()
        
        if len(clean_query) >= 2:
            # Create a flexible word match (e.g. "data structures" -> "%data%structures%")
            words = clean_query.lower().split()
            flex_words = "%" + "%".join(words) + "%"
            
            rows = execute_fetch(
                """
                SELECT subject_name, pyq_link
                FROM pyq_resources
                WHERE LOWER(subject_code) = %s OR LOWER(subject_name) LIKE %s
                LIMIT 1
                """,
                (clean_query.lower(), flex_words)
            )
            if rows:
                name, link = rows[0]
                return f"Here are the previous year question papers (PYQs) and study materials for {name}:\n{link}"
    except Exception as e:
        print(f"[SEARCH ERROR] search_pyq: {e}")
        
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
              AND ts_rank(to_tsvector('english', content), plainto_tsquery('english', %s)) > 0.4
            ORDER BY ts_rank(to_tsvector('english', content),
                             plainto_tsquery('english', %s)) DESC
            LIMIT 2
            """,
            (query, query, query, query)
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
    "  - Faculty / Professor details\n"
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
        if category == "faculty":
            # Faculty queries are answered with direct links, never sent to LLM
            return search_faculty(user_message)

        elif category == "pyq":
            result = search_pyq(normalized)

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
