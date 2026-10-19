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


# ---------------------------------------------------------------- faculty & departments
import json
from pathlib import Path

STAFF_FINDER = "https://www.srmist.edu.in/staff-finder/"
FACULTY_LIST = "https://www.srmist.edu.in/faculty/"
DEPT_LIST = "https://www.srmist.edu.in/departments/"
_DEPTS = None

ROLE_WORDS = re.compile(r"\b(hod|head of (the )?department|head of dept|heads?|dean|chairperson)\b", re.I)
# words that are never part of a person's name
_NOT_NAME = {
    "who", "is", "are", "was", "tell", "me", "about", "details", "of", "the", "faculty", "professor", "prof",
    "teacher", "lecturer", "staff", "hod", "head", "department", "dept", "dean", "sir", "maam", "madam",
    "give", "show", "find", "search", "info", "information", "teaches", "teaching", "at", "srm", "srmist",
    "university", "ktr", "kattankulathur", "campus", "please", "can", "you", "do", "know", "i", "want", "to",
    "what", "where", "which", "how", "contact", "email", "phone", "number", "cabin", "office", "room", "a", "an",
    "and", "in", "for", "from", "with", "on", "so", "ok", "okay", "hey", "hi", "hello", "then", "also", "now",
    "well", "actually", "just", "my", "our", "his", "her", "this", "that", "there", "name", "mr", "mrs", "ms",
    "dr", "s", "ma", "yes", "no", "thanks", "thank", "pls", "plz", "mist", "ai",
}


def departments():
    global _DEPTS
    if _DEPTS is None:
        path = Path(__file__).with_name("data") / "departments.json"
        _DEPTS = json.loads(path.read_text(encoding="utf-8"))
    return _DEPTS


def _match_department(text):
    """Department named in the text. Exact whole-word alias match first; if several departments are mentioned the
    one mentioned LAST wins (so 'hod of ctech ... what about mechanical' means mechanical), longest alias on ties.
    Then a typo-tolerant match on single words ('ctec' -> CTECH, 'mechanicl' -> mechanical)."""
    import difflib
    t = " " + re.sub(r"[^a-z0-9 ]+", " ", text.lower()) + " "
    best, best_key = None, (-1, -1)
    for d in departments():
        for alias in d["aliases"]:
            pos = t.rfind(f" {alias} ")
            if pos >= 0 and (pos, len(alias)) > best_key:
                best, best_key = d, (pos, len(alias))
    if best:
        return best
    singles = {}
    for d in departments():
        for alias in d["aliases"]:
            if " " not in alias and len(alias) >= 4:
                singles[alias] = d
    for tok in t.split():
        if len(tok) < 4 or tok in _NOT_NAME:
            continue
        m = difflib.get_close_matches(tok, list(singles), n=1, cutoff=0.8)
        if m and m[0][0] == tok[0]:
            return singles[m[0]]
    return None


def department_chunks():
    """Knowledge-base entries so retrieval can also surface department/HOD facts."""
    out = []
    for i, d in enumerate(departments()):
        bits = [f"{d['name']} ({d['short']}), Kattankulathur. Head of the Department: {d['hod']}."]
        if d.get("email"):
            bits.append(f"HOD email: {d['email']}.")
        if d.get("phone"):
            bits.append(f"Phone: {d['phone']}.")
        bits.append(f"Location: {d['location']}. Official page: {d['url']}")
        out.append({"id": f"dept-{i:02d}", "topic": "Departments", "campus": "Kattankulathur (KTR)",
                    "text": " ".join(bits), "source": d["url"]})
    return out


def _department_answer(d):
    lines = [f"{d['name']} (SRMIST Kattankulathur)", f"Head of the Department: {d['hod']}"]
    if d.get("email"):
        lines.append(f"HOD email: {d['email']}")
    if d.get("phone"):
        lines.append(f"Phone: {d['phone']}")
    lines.append(f"Location: {d['location']}")
    lines.append(f"Official page: {d['url']}")
    lines.append("")
    lines.append("(As listed on the department's page on srmist.edu.in. HODs change, so confirm there.)")
    return "\n".join(lines)


def faculty(text):
    """Faculty / HOD questions. Never invents names or profile links."""
    dept = _match_department(text)
    low = text.lower()
    if dept and (ROLE_WORDS.search(text) or "faculty" in low or "staff" in low):
        return _department_answer(dept)
    if ROLE_WORDS.search(text):
        names = ", ".join(d["short"] for d in departments())
        return ("I can tell you the HOD for these departments: " + names + ". Try 'HOD of CTECH' or "
                "'who is the HOD of mechanical'.\n\nFor any other department, open the department list and "
                "pick yours:\n" + DEPT_LIST + "\n\nOr search a person on Staff Finder:\n" + STAFF_FINDER)
    tokens = [t for t in re.findall(r"[a-z']+", low) if t not in _NOT_NAME and len(t) > 1]
    name = " ".join(tokens).title()
    if 1 <= len(tokens) <= 4 and not dept:
        return ("I don't store individual faculty profiles, so I can't confirm details for '" + name + "'. "
                "Search the name on the official Staff Finder, which lists every matching faculty member:\n"
                + STAFF_FINDER + "\n\nFull faculty directory:\n" + FACULTY_LIST)
    return ("You can search any faculty member on the SRM Staff Finder:\n" + STAFF_FINDER + "\n\n"
            "Or browse the complete faculty directory:\n" + FACULTY_LIST)


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
