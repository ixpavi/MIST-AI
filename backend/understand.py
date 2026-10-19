"""Query understanding: turn messy student messages into a clean standalone question *before* routing.

  1. normalise   - slang, SMS spellings and common Hinglish ("hostal fees kitna hai", "wat is d fee")
  2. spell-fix   - fuzzy-match unknown words against the SRM vocabulary (built from the knowledge base)
  3. follow-ups  - "and how many girls?" / "what about its fees" are rewritten using the conversation

Steps 1-2 are local and instant. Step 3 uses the language model only when the message looks like a follow-up,
and falls back to a simple context merge if the model is unavailable."""
import difflib
import re

# slang / SMS / Hinglish -> plain English (whole-word replacements)
SLANG = {
    "wat": "what", "wht": "what", "whts": "what is", "whats": "what is", "what's": "what is", "wats": "what is", "hw": "how", "hows": "how is", "wen": "when", "wer": "where",
    "plz": "please", "pls": "please", "u": "you", "ur": "your", "r": "are", "d": "the", "abt": "about", "bcz": "because",
    "info": "information", "hostal": "hostel", "hostle": "hostel", "hostels": "hostel", "colege": "college",
    "collage": "college", "univ": "university", "uni": "university", "sem": "semester", "exm": "exam", "exms": "exams",
    "kitna": "how much", "kitne": "how many", "kab": "when", "kahan": "where", "kaha": "where", "kaise": "how",
    "kya": "what", "kon": "who", "kaun": "who", "hai": "", "hain": "", "mein": "in",
    "batao": "tell me", "bataiye": "tell me", "chahiye": "need", "fee": "fees", "pyqs": "pyq", "papers": "paper",
    "dept": "department", "depts": "departments", "h.o.d": "hod", "hod's": "hod",
}
# never spell-correct these (short or very common words, or words where fuzzy matching hurts)
KEEP = set("a an the is are was were of to in on for and or do does did i you me my it its what how when where which "
           "who can will be at by with about tell please give show find need want get has have this that there their "
           "they them then also much many much more most any some".split())
PRONOUNS = {"it", "its", "that", "this", "those", "these", "there", "they", "them", "their", "he", "she", "his", "her"}
CONNECTORS = ("and ", "also ", "what about", "how about", "then ", "so ", "ok ", "okay ", "but ", "what if")

_vocab_cache = {"key": None, "words": set()}


def build_vocab(chunks, extra=()):
    key = (len(chunks), len(extra))
    if _vocab_cache["key"] == key:
        return _vocab_cache["words"]
    words = set()
    for c in chunks:
        for w in re.findall(r"[a-z]{4,}", (c.get("topic", "") + " " + c.get("text", "")).lower()):
            words.add(w)
    for e in extra:
        for w in re.findall(r"[a-z]{4,}", e.lower()):
            words.add(w)
    _vocab_cache.update(key=key, words=words)
    return words


def normalise(text):
    t = re.sub(r"\s+", " ", text.lower().strip())
    t = re.sub(r"[^a-z0-9\s'.\-?/]", " ", t)
    out = []
    for w in t.split():
        w = w.strip(".,!?;:'\"")
        if not w:
            continue
        rep = SLANG.get(w)
        out.append(w if rep is None else rep)
    return re.sub(r"\s+", " ", " ".join(out)).strip()


def spell_fix(text, vocab):
    """Replace unknown words by their closest SRM-vocabulary word. Strict on purpose: same first letter, and
    near-identical for short words, so real words and names are not damaged."""
    if not vocab:
        return text
    fixed = []
    for w in text.split():
        if len(w) < 4 or w in KEEP or w in vocab or not w.isalpha():
            fixed.append(w)
            continue
        cutoff = 0.9 if len(w) <= 5 else 0.84
        cands = [c for c in difflib.get_close_matches(w, vocab, n=3, cutoff=cutoff) if c[0] == w[0]]
        fixed.append(cands[0] if cands else w)
    return " ".join(fixed)


def looks_like_followup(text, has_history, names_own_subject=None):
    """True when the message only makes sense with the previous turn. A short message that names its own subject
    (for example 'dsbs hod') is standalone: merging it with the previous question used to answer the OLD subject."""
    if not has_history:
        return False
    t = " " + text.lower().strip() + " "
    toks = re.findall(r"[a-z0-9']+", t)
    if any(p in PRONOUNS for p in toks):
        return True
    if text.lower().strip().startswith(CONNECTORS):
        return True
    if names_own_subject and names_own_subject(text):
        return False
    return len(toks) <= 3


def rewrite_followup(text, history, llm_fn):
    """Make a follow-up standalone. llm_fn(prompt) -> str|None. Falls back to merging with the last user question."""
    last_user = next((m["text"] for m in reversed(history) if m.get("role") == "user"), "")
    if llm_fn and history:
        convo = "\n".join(f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['text'][:300]}" for m in history[-4:])
        prompt = ("Rewrite the user's LAST message as one complete, standalone question about SRM University, using the "
                  "conversation for context. If it is already standalone, return it unchanged. Fix spelling. "
                  "Output only the question, nothing else.\n\n" + convo + f"\nUser (last message): {text}\nStandalone question:")
        out = llm_fn(prompt)
        if out:
            out = out.strip().strip('"').splitlines()[0][:300]
            if len(out.split()) >= 5 and len(out) <= 300:
                return out, "llm"
    if last_user:
        return f"{last_user} {text}", "merge"
    return text, "none"


def prepare(text, history, vocab, llm_fn=None, names_own_subject=None):
    """Returns (standalone_query, notes) where notes lists what was changed (for logging/explainability)."""
    notes = []
    norm = normalise(text)
    if not norm:
        return text, notes
    fixed = spell_fix(norm, vocab)
    if fixed != norm:
        notes.append("spelling")
    if norm != re.sub(r"\s+", " ", text.lower().strip()):
        notes.append("slang")
    q = fixed
    if looks_like_followup(text, bool(history), names_own_subject):
        q2, how = rewrite_followup(q, history, llm_fn)
        if how != "none":
            q = q2
            notes.append("followup:" + how)
    return q, notes
