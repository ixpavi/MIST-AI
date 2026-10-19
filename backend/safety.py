"""Input safety layer. Runs before any routing, so abusive, malicious or nonsensical input never reaches the
skills, the database queries or the language model.

check(text) -> (category, reply) when the message should be handled here, otherwise None."""
import re

try:
    from better_profanity import profanity
    profanity.load_censor_words()
except Exception:      # optional dependency
    profanity = None

_BASIC_BAD = re.compile(r"\b(fuck\w*|shit\w*|bitch\w*|bastard|asshole|dick|slut|whore|cunt|motherfucker|bc|mc|chutiya\w*|madarchod|bhosdi\w*|gandu|harami)\b", re.I)

HELP = ("I can help with SRM questions: admissions and SRMJEEE, fees, hostels, placements, PYQs, portals, campus "
        "locations, departments and HODs, library, transport and exam rules.")

RE_INJECTION = re.compile(
    r"(ignore|forget|disregard|override)\b.{0,40}\b(previous|prior|above|earlier|all|your)\b.{0,30}\b(instruction|rule|prompt|direction)s?"
    r"|system\s*prompt|reveal\b.{0,25}\b(prompt|instruction|rule|secret|key)s?|show\b.{0,25}\b(your|the)\b.{0,15}\b(prompt|instruction|rule)s?"
    r"|you\s+are\s+now\b|act\s+as\s+(an?\s+)?(dan|unrestricted|jailbroken)|\bjailbreak|\bdeveloper\s+mode|\bdan\s+mode"
    r"|(api|secret)[\s_-]*key|\bgemini_api|\bdb_pass|\.env\b|environment\s+variables?|connection\s+string"
    r"|(database|db|server|admin|root)\s+password|\bpassword\s+of\s+(the\s+)?(database|db|server|admin)", re.I)
RE_SQL = re.compile(r"(\bdrop\s+table\b|\bunion\s+select\b|\bdelete\s+from\b|\binsert\s+into\b|\bselect\b.+\bfrom\b|'\s*or\s*'?1'?\s*=\s*'?1|;\s*--|--\s*$|\bxp_cmdshell\b)", re.I)
RE_MARKUP = re.compile(r"<\s*/?\s*(script|img|iframe|svg|object|embed|link|style)\b|javascript\s*:|on(error|load|click)\s*=", re.I)
RE_HACK = re.compile(
    r"\b(hack|hacking|crack|bypass|breach|exploit|ddos|phish\w*|brute\s*force|keylog\w*|steal|leak)\b.{0,50}"
    r"\b(portal|academia|server|website|site|wifi|wi-fi|network|database|account|password|exam\s*cell|marks|attendance|login|otp|id)\b"
    r"|\b(change|edit|alter|modify|increase|fake|forge|tamper)\b.{0,30}\b(my\s+)?(attendance|marks|grades?|cgpa|result|certificate|id\s*card|hall\s*ticket)\b"
    r"|\b(proxy\s+attendance|fake\s+certificate|fake\s+(marksheet|degree)|buy\s+(a\s+)?(degree|marksheet))\b", re.I)
RE_CHEAT = re.compile(
    r"\b(cheat|cheating|copy|copying|malpractice|leak(ed)?\s+paper|paper\s+leak)\b.{0,50}\b(exam|test|cycle\s*test|semester|viva|assignment|record)\b"
    r"|\b(exam|test)\b.{0,50}\b(cheat|cheating|without\s+getting\s+caught|chit|answers?\s+leak)\b", re.I)
RE_PRIVATE = re.compile(
    r"\b(phone|mobile|contact|number|address|marks|cgpa|grades?|result|aadhaar|aadhar|email|whatsapp|instagram|location)\b"
    r".{0,25}\b(of|for)\b.{0,15}\b(student|girl|boy|him|her|that\s+guy|roommate|classmate|friend|someone|my\s+crush)\b"
    r"|\b(student|girl|boy|roommate|classmate|crush)\b.{0,20}\b(named|called)?\b.{0,20}\b(phone|mobile|number|address|marks|cgpa|aadhaar|whatsapp)\b", re.I)


def is_abusive(text):
    if _BASIC_BAD.search(text):
        return True
    return bool(profanity and profanity.contains_profanity(text))


def check(text):
    t = (text or "").strip()
    letters = re.findall(r"[^\W\d_]", t, flags=re.UNICODE)
    if RE_MARKUP.search(t) or RE_SQL.search(t):
        return "attack", ("That looks like code or a technical command rather than a question. " + HELP)
    if RE_INJECTION.search(t):
        return "injection", ("I can only answer SRM-related questions, and I don't share my internal instructions, keys or "
                             "configuration. " + HELP)
    if RE_HACK.search(t):
        return "harmful", ("Sorry, I can't help with hacking, tampering with records, or bypassing any system. If you have a "
                           "genuine problem with the portal, attendance or marks, contact your department office or the "
                           "Controller of Examinations through https://www.srmist.edu.in . " + HELP)
    if RE_CHEAT.search(t):
        return "harmful", ("I can't help with cheating. SRMIST treats exam malpractice seriously: it can cancel all subjects "
                           "written in that semester and, in serious cases, lead to expulsion. I'm happy to help with exam "
                           "rules, the academic calendar or how to get revaluation.")
    if RE_PRIVATE.search(t):
        return "privacy", ("I can't share personal details such as phone numbers, addresses or marks of individual students. "
                           "For official records contact the university office. " + HELP)
    if is_abusive(t):
        return "abuse", "Let's keep it respectful, please. " + HELP
    if len(letters) < 2:
        return "nonsense", "I couldn't read a question there. Could you type it again? " + HELP
    return None
