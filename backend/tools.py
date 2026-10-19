"""Deterministic SRM tools. These answer from structured data / rules with no language model involved,
so numbers and dates are exact and never hallucinated."""
import json
import re
from datetime import date
from pathlib import Path

DATA = Path(__file__).with_name("data")

# ---- SRMJEEE 2026 schedule (https://www.srmist.edu.in/srmjeee/) --------------------------------
JEEE_PHASES = [
    {"phase": 1, "apply_by": date(2026, 4, 16), "exam_start": date(2026, 4, 24), "exam_end": date(2026, 4, 29)},
    {"phase": 2, "apply_by": date(2026, 6, 4), "exam_start": date(2026, 6, 10), "exam_end": date(2026, 6, 15)},
    {"phase": 3, "apply_by": date(2026, 6, 30), "exam_start": date(2026, 7, 4), "exam_end": date(2026, 7, 5)},
]
JEEE_URL = "https://www.srmist.edu.in/srmjeee/"


def _fmt(d):
    return f"{d.day} {d.strftime('%B %Y')}"


def jeee_status(today=None):
    """Tells the applicant where each SRMJEEE 2026 phase stands relative to today."""
    today = today or date.today()
    lines, upcoming = [], None
    for p in JEEE_PHASES:
        if today <= p["apply_by"]:
            state = f"applications open until {_fmt(p['apply_by'])}"
            upcoming = upcoming or p
        elif today <= p["exam_end"]:
            state = "applications closed, exam window in progress / upcoming"
            upcoming = upcoming or p
        else:
            state = "closed (exam completed)"
        lines.append(f"- Phase {p['phase']}: apply by {_fmt(p['apply_by'])}; exam "
                     f"{p['exam_start'].day}-{_fmt(p['exam_end'])} - {state}")
    head = (f"Today is {_fmt(today)}. Here is the SRMJEEE 2026 schedule (official dates):\n\n" + "\n".join(lines))
    if upcoming is None:
        tail = ("\n\nAll three SRMJEEE 2026 phases are over. Dates for the next cycle are announced on "
                f"{JEEE_URL} and the application fee will again be Rs. 1,400 per phase.")
    else:
        tail = (f"\n\nNext step: Phase {upcoming['phase']} - apply by {_fmt(upcoming['apply_by'])} at "
                "https://applications.srmist.edu.in/btech (Rs. 1,400, non-refundable).")
    return head + tail


# ---- programme / fee lookup --------------------------------------------------------------------
_PROGS = None
CAMPUS_ALIASES = {
    "ktr": "Kattankulathur (KTR)", "kattankulathur": "Kattankulathur (KTR)",
    "ramapuram": "Ramapuram", "rmp": "Ramapuram", "vadapalani": "Vadapalani", "vdp": "Vadapalani",
    "ncr": "Delhi-NCR (Ghaziabad)", "delhi": "Delhi-NCR (Ghaziabad)", "ghaziabad": "Delhi-NCR (Ghaziabad)",
    "haryana": "Haryana (Sonepat)", "sonepat": "Haryana (Sonepat)",
    "amaravati": "Amaravati (SRM University-AP)", "andhra": "Amaravati (SRM University-AP)",
}
STOPWORDS = {"engineering", "technology", "the", "of", "in", "and", "at", "fee", "fees", "for", "what", "is",
              "how", "much", "does", "cost", "btech", "tech", "mtech", "course", "program", "programme"}


def programs():
    global _PROGS
    if _PROGS is None:
        _PROGS = json.loads((DATA / "programs.json").read_text(encoding="utf-8"))
    return _PROGS


def inr(n):
    s = str(int(n))
    if len(s) <= 3:
        return s
    head, tail = s[:-3], s[-3:]
    parts = []
    while len(head) > 2:
        parts.insert(0, head[-2:])
        head = head[:-2]
    if head:
        parts.insert(0, head)
    return ",".join(parts + [tail])


def program_text(p):
    mode = "" if p["mode"] == "Full Time" else f" ({p['mode'].lower()})"
    return (f"{p['degree'].replace(' ', '')} {p['program']}{mode} at {p['campus']}: {p['years']} years, "
            f"intake {p['intake']}, annual fee Rs. {inr(p['fee'])}")


def program_lookup(query):
    """Returns (answer_text, matches) or (None, [])."""
    q = query.lower()
    words = set(re.findall(r"[a-z]+", q))
    campus = next((CAMPUS_ALIASES[w] for w in words if w in CAMPUS_ALIASES), None)
    degree = "M.Tech" if "mtech" in words or "m" in words and "tech" in words else (
        "B.Tech" if "btech" in words or "b" in words and "tech" in words else None)
    scored = []
    for p in programs():
        pw = {w for w in re.findall(r"[a-z]+", p["program"].lower()) if w not in STOPWORDS}
        if not pw:
            continue
        hit = len(pw & words)
        if hit == 0 or hit < len(pw) * 0.5:
            continue
        score = hit / len(pw)
        if campus and p["campus"] == campus:
            score += 0.5
        if degree and p["degree"].replace(" ", "") == degree:
            score += 0.3
        scored.append((score, p))
    if not scored:
        return None, []
    scored.sort(key=lambda x: -x[0])
    best = scored[0][0]
    matches = [p for s, p in scored if s >= best - 1e-9][:6]
    if campus:
        matches = [p for p in matches if p["campus"] == campus] or matches
    body = "\n".join(f"- {program_text(p)}" for p in matches)
    return ("From the programme table on the SRMIST engineering admissions page:\n\n" + body +
            "\n\nThese are the annual fees listed there; scholarships can reduce them. Full table and "
            "latest fees: https://www.srmist.edu.in/admission-india/engineering/"), matches


def program_chunks():
    """One knowledge-base entry per programme so retrieval can also find them."""
    out = []
    for i, p in enumerate(programs()):
        out.append({"id": f"prog-{i:03d}", "topic": "Programs", "campus": p["campus"],
                    "text": program_text(p) + " (SRMIST engineering admissions programme table).",
                    "source": "https://www.srmist.edu.in/admission-india/engineering/"})
    return out
