"""DB-backed skills (faculty finder, PYQs, portals, locations, university info).

Each skill returns a plain-text answer or None. They replace the old whole-sentence LIKE matching with
keyword-based OR full-text queries, so "where can i check my attendance" finds the 'attendance' feature."""
import re

from database import execute_fetch

STOP = set("a an the is are was were of to in on for and or do does did i you me my it its what how when where "
           "which who can will be at by with about tell please give show find srm srmist ktr kattankulathur "
           "campus university want need get check".split())


def keywords(text):
    return [w for w in re.findall(r"[a-z0-9]+", (text or "").lower()) if w not in STOP and len(w) > 1]


def _tsq(text):
    kws = keywords(text)
    return " | ".join(kws) if kws else None


def _like_patterns(text):
    return [f"%{k}%" for k in keywords(text)] or None


# ---------------------------------------------------------------- faculty
_FACULTY_STRIP = {
    "who", "is", "tell", "me", "about", "details", "of", "the", "faculty", "professor", "prof", "teacher",
    "lecturer", "staff", "hod", "head", "department", "dean", "sir", "maam", "madam", "ma'am", "give", "show",
    "find", "search", "info", "information", "teaches", "teaching", "at", "srm", "srmist", "university", "ktr",
    "kattankulathur", "campus", "please", "can", "you", "do", "know", "i", "want", "to", "what", "where",
    "which", "how", "contact", "email", "phone", "number", "cabin", "office", "room", "a", "an", "and", "in",
    "for", "from", "with", "on",
}
STAFF_FINDER = "https://www.srmist.edu.in/staff-finder/"
FACULTY_LIST = "https://www.srmist.edu.in/faculty/"


def faculty(text):
    name = " ".join(t for t in re.findall(r"[a-z']+", text.lower()) if t not in _FACULTY_STRIP).title()
    if not name:
        return (f"You can search for any faculty member on the SRM Staff Finder:\n{STAFF_FINDER}\n\n"
                f"Or browse the complete faculty listing:\n{FACULTY_LIST}")
    slug = name.lower().replace(" ", "-")
    direct = (f"{FACULTY_LIST}{slug}/" if slug.split("-")[0] in ("dr", "mr", "mrs", "ms")
              else f"{FACULTY_LIST}dr-{slug}/\n   {FACULTY_LIST}{slug}/")
    return (f"You can find the profile of {name} on the SRM website:\n\n"
            f"1. Staff Finder (lists every matching faculty member):\n   {STAFF_FINDER}\n\n"
            f"2. Possible direct profile link:\n   {direct}\n\n3. Full faculty directory:\n   {FACULTY_LIST}")


# ---------------------------------------------------------------- tables
def portals(text):
    pats = _like_patterns(text)
    if not pats:
        return None
    rows = execute_fetch("""SELECT DISTINCT portal_name, link, managed_by FROM portals
                            WHERE portal_name ILIKE ANY(%s) LIMIT 3""", (pats,))
    if rows:
        return "\n\n".join(f"{n}\nLink: {l}\nManaged by: {m}" for n, l, m in rows)
    return None


def portal_features(text):
    tsq = _tsq(text)
    if not tsq:
        return None
    rows = execute_fetch("""
        SELECT pf.feature_name, pf.description, p.portal_name, p.link,
               ts_rank(to_tsvector('english', pf.feature_name || ' ' || COALESCE(pf.description, '')),
                       to_tsquery('english', %s)) AS r
        FROM portal_features pf JOIN portals p ON p.portal_id = pf.portal_id
        WHERE to_tsvector('english', pf.feature_name || ' ' || COALESCE(pf.description, '')) @@ to_tsquery('english', %s)
        ORDER BY r DESC LIMIT 2""", (tsq, tsq))
    if rows:
        return "\n\n".join(f"{d}\n\n(Available on {p}: {l})" for _, d, p, l, _ in rows)
    return None


def pyq(text):
    kws = [k for k in keywords(text) if k not in ("pyq", "pyqs", "previous", "year", "question", "paper", "papers",
                                                    "old", "past", "model", "sample", "bank", "study", "material")]
    if kws:
        rows = execute_fetch("""SELECT subject_name, pyq_link FROM pyq_resources
                                WHERE LOWER(subject_code) = ANY(%s) OR subject_name ILIKE %s LIMIT 1""",
                             (kws, "%" + "%".join(kws) + "%"))
        if rows:
            return (f"Here are the previous year question papers (PYQs) and study material for "
                    f"{rows[0][0]}:\n{rows[0][1]}")
    return "You can find previous year question papers (PYQs) and study material at The Helpers: https://thehelpers.vercel.app/"


def locations(text):
    tsq = _tsq(text)
    if not tsq:
        return None
    rows = execute_fetch("""
        SELECT place_name, description FROM locations
        WHERE to_tsvector('english', place_name || ' ' || COALESCE(description, '')) @@ to_tsquery('english', %s)
        ORDER BY ts_rank(to_tsvector('english', place_name || ' ' || COALESCE(description, '')),
                         to_tsquery('english', %s)) DESC LIMIT 2""", (tsq, tsq))
    if rows:
        return "\n\n".join(f"{n}\n{d}" for n, d in rows)
    return None


def university_info(text):
    tsq = _tsq(text)
    if not tsq:
        return None
    rows = execute_fetch("""
        SELECT answer, source_url FROM university_info
        WHERE to_tsvector('english', COALESCE(question, '') || ' ' || COALESCE(answer, '')) @@ to_tsquery('english', %s)
        ORDER BY ts_rank(to_tsvector('english', COALESCE(question, '') || ' ' || COALESCE(answer, '')),
                         to_tsquery('english', %s)) DESC LIMIT 1""", (tsq, tsq))
    if rows:
        a, url = rows[0]
        return (a or "") + (f"\n\nMore info: {url}" if url else "")
    return None


SKILL_BY_INTENT = {
    "faculty": [faculty],
    "pyq": [pyq],
    "portal": [portals, portal_features],
    "portal_feature": [portal_features, portals],
    "location": [locations],
}


def run(intent, text):
    """Run the skills for an intent; returns (answer, skill_name) or (None, None)."""
    for fn in SKILL_BY_INTENT.get(intent, []):
        try:
            out = fn(text)
        except Exception:
            out = None
        if out:
            return out, fn.__name__
    return None, None
