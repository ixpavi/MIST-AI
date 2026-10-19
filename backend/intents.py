"""SRM intent classifier - a small model trained in this repo (TF-IDF + logistic regression).
Train:  python train_intent.py        Use:  intents.predict("hostel fee kitna hai") -> ("hostel", 0.8)"""
from pathlib import Path

import joblib

MODEL_FILE = Path(__file__).with_name("data") / "intent_model.joblib"
import logging
log = logging.getLogger("mist.intents")

# label -> example questions. This is the training set; extend it and re-run train_intent.py.
TRAIN = {
    "admissions": [
        "how do I get admission to SRM", "what is the admission process", "admission requirements for btech",
        "am I eligible for SRM with 60 percent", "eligibility criteria for engineering", "how to apply to SRMIST",
        "documents needed for admission", "can I get admission without jee", "last date for admission",
        "admission for mtech", "how to join SRM", "admission for architecture course", "age limit for admission",
        "can nri students apply", "direct admission at srm", "which entrance exam does srm accept", "counselling process",
        "how much is the counselling fee", "provisional allotment letter", "admission for biomedical engineering"],
    "exam": [
        "srmjeee exam pattern", "how many questions in srmjeee", "is there negative marking", "srmjeee syllabus",
        "when is srmjeee phase 2", "srmjeee 2026 dates", "what is the last date to apply for srmjeee",
        "srmjeee slot booking", "how to download admit card", "srmjeee mock test", "exam duration srmjeee",
        "srmjeee result when", "can I take srmjeee on mobile", "srmjeee application fee", "how many phases of srmjeee",
        "is srmjeee online", "srmjeee phase 3 deadline", "system requirements for the online exam", "srmjeee rank card",
        "can I change my exam slot"],
    "scholarship": [
        "scholarships at srm", "how to get fee waiver", "100 percent scholarship", "scholarship for toppers",
        "merit scholarship srm", "sports scholarship", "scholarship based on 12th marks", "fee concession for top rankers",
        "is there any scholarship for girls", "scholarship for jee rank", "how much scholarship can I get",
        "iit jee rankers scholarship", "scholarship eligibility", "financial aid options", "education loan help"],
    "fees": [
        "btech fees at srm", "what is the fee for computer science", "annual tuition fee", "how much does mtech cost",
        "fee structure of srm", "total cost of btech", "ramapuram fees", "fees for aerospace engineering",
        "is srm expensive", "fee for civil engineering at ktr", "what is the price of the course", "tuition fee per year",
        "fees for architecture", "program fee vadapalani", "how much do I have to pay per year"],
    "programs": [
        "which courses does srm offer", "list of btech branches", "does srm have ai and data science", "mtech specialisations",
        "is there an aerospace engineering course", "programs offered at srm", "does srm offer law", "mba at srm",
        "which departments are there", "can I study medicine at srm", "does srm have agriculture", "b arch duration",
        "how long is the btech course", "how many seats in civil engineering", "intake for aerospace engineering",
        "cse specialisations", "online degree at srm", "phd programs", "hotel management course", "biotechnology course"],
    "hostel": [
        "hostel facilities", "how to book hostel", "hostel fees", "is there ac hostel", "hostel rules", "mess food options",
        "girls hostel capacity", "how many hostels at ktr", "hostel for first year students", "hostel contact number",
        "is hostel compulsory", "who is eligible for hostel", "international hostel", "hostel wifi", "can I stay in hostel without admission",
        "hostel refund policy", "veg and non veg mess", "hostel booking date", "boys hostel blocks", "hostel warden contact"],
    "placements": [
        "placement record of srm", "highest package", "which companies visit campus", "average salary after btech",
        "does google recruit from srm", "placement percentage", "career centre contact", "how are placements for cse",
        "top recruiters", "internship opportunities", "core company placements", "microsoft package at srm",
        "placement training", "how many offers last year", "day one companies", "is srm good for placements",
        "packages above 20 lakhs", "placement for ece", "jobs after graduation", "campus recruitment process"],
    "rankings": [
        "nirf ranking of srm", "naac grade", "is srm a good university", "srm rank in india", "accreditation of srm",
        "is srm ugc approved", "srm world ranking", "abet accreditation", "srm among top universities", "qs ranking",
        "research ranking nirf", "srm reputation", "is srm deemed university", "what is srm ranking in engineering",
        "naac a++ meaning", "which is better srm or vit", "quality of education at srm"],
    "campuses": [
        "where is srm located", "how many campuses does srm have", "ktr campus details", "difference between ktr and ramapuram",
        "does srm have a campus in delhi", "srm vadapalani", "where is trichy campus", "how to reach srm kattankulathur",
        "srm amaravati", "srm sikkim campus", "campus size in acres", "number of students at srm", "address of srm",
        "which campus is best", "srm haryana sonepat", "distance from chennai airport", "baburayanpettai campus",
        "campus facilities", "library at srm", "campus tour"],
    "contact": [
        "srm phone number", "admission helpline number", "email id of admissions office", "how can I contact srm",
        "customer care number", "who to call for admission query", "contact details of hostel", "srm website link",
        "office timings", "complaint redressal contact", "grievance portal", "student portal link", "enquiry number",
        "toll free number", "address and contact"],
    "student_life": [
        "student clubs at srm", "sports facilities", "gym in campus", "cultural fests", "technical clubs", "nss ncc at srm",
        "events at srm", "is there a cricket ground", "anti ragging policy", "student life at srm", "food courts on campus",
        "transport facilities", "hackathons at srm", "community radio", "innovation and incubation centre",
        "extracurricular activities", "swimming pool", "clubs for coding", "student safety", "campus life"],
    "greeting": ["hi", "hello", "hey", "hii", "good morning", "good evening", "hey there", "hello mist", "namaste",
                 "yo", "sup", "hi there", "hola", "good afternoon", "hey mist ai"],
    "about_bot": [
        "who are you", "what can you do", "who created you", "who made you", "what is mist ai", "are you an ai",
        "are you chatgpt", "are you gemini", "what model are you", "how do you work", "tell me about yourself",
        "who built this", "what is your name", "are you official", "how were you trained", "what data do you use",
        "can you help me", "what are your features", "which technology powers you", "are you a bot"],
    "thanks": ["thanks", "thank you", "thx", "thanks a lot", "thank you so much", "ok thanks", "great thanks", "cool thank you",
               "appreciate it", "that helps thanks"],
    "offtopic": [
        "who is the prime minister of india", "what is the capital of france", "write a python program to sort a list",
        "tell me a joke", "who won the world cup", "weather today", "how to cook pasta", "solve this equation",
        "what is machine learning", "recommend a movie", "latest iphone price", "explain quantum computing",
        "translate this to french", "best laptop to buy", "how to lose weight", "stock market tips",
        "write an essay on pollution", "who is elon musk", "what is bitcoin", "song lyrics", "football scores",
        "how to invest money", "history of rome", "capital of japan", "tell me a story"],
}

# intent -> knowledge-base topics it should prefer during retrieval
TOPICS = {
    "admissions": {"Admissions", "SRMJEEE", "Scholarships"}, "exam": {"SRMJEEE"}, "scholarship": {"Scholarships"},
    "fees": {"Programs", "Admissions", "Hostels", "Scholarships"}, "programs": {"Programs", "Academics", "Admissions"},
    "hostel": {"Hostels"}, "placements": {"Placements"}, "rankings": {"Rankings", "Accreditation"},
    "campuses": {"Campuses", "About", "Student life"}, "contact": {"Contact"}, "student_life": {"Student life"},
}
LABELS = {"admissions": "Admissions", "exam": "SRMJEEE exam", "scholarship": "Scholarships", "fees": "Fees",
          "programs": "Programmes", "hostel": "Hostels", "placements": "Placements", "rankings": "Rankings",
          "campuses": "Campuses", "contact": "Contact", "student_life": "Student life", "greeting": "Greeting",
          "about_bot": "About MIST AI", "thanks": "Thanks", "offtopic": "Outside SRM"}

_model = None


def is_loaded():
    return _model is not None


def build_pipeline():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import FeatureUnion, Pipeline
    feats = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, lowercase=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True, lowercase=True)),
    ])
    return Pipeline([("f", feats), ("clf", LogisticRegression(C=20, max_iter=2000))])


def load():
    """Load the trained model; if the pickle is missing or incompatible with this scikit-learn, retrain in memory."""
    global _model
    if _model is None:
        try:
            _model = joblib.load(MODEL_FILE)
        except Exception as e:
            log.warning("intent model not loadable (%s); retraining in memory", str(e)[:80])
            X, y = augmented_train()
            _model = build_pipeline().fit(X, y)
    return _model


EMB_MODEL_FILE = Path(__file__).with_name("data") / "intent_model_emb.joblib"
_emb_model = None


def load_emb():
    """Logistic regression on local sentence embeddings (understands paraphrases). None if not trained."""
    global _emb_model
    if _emb_model is None and EMB_MODEL_FILE.exists():
        try:
            _emb_model = joblib.load(EMB_MODEL_FILE)
        except Exception as e:
            log.warning("embedding intent model not loadable: %s", str(e)[:80])
    return _emb_model


def predict(text, vec=None):
    """Returns (intent, confidence) or (None, 0.0). With a sentence embedding `vec`, the TF-IDF model and the
    embedding model vote together, so wording that never appeared in training still maps to the right topic."""
    m = load()
    if m is None:
        return None, 0.0
    probs = m.predict_proba([text])[0]
    classes = list(m.classes_)
    em = load_emb() if vec is not None else None
    if em is not None and list(em.classes_) == classes:
        probs = 0.5 * probs + 0.5 * em.predict_proba([vec])[0]
    i = int(probs.argmax())
    return classes[i], float(probs[i])


# ---- augmentation: short keyword queries + polite wrappers (training only) ----------------------
KEYWORDS = {
    "admissions": ["admission", "admissions process", "apply", "eligibility", "application", "counselling", "seat allotment", "nri admission"],
    "exam": ["srmjeee", "srmjeee dates", "exam pattern", "syllabus", "admit card", "slot booking", "negative marking", "exam date"],
    "scholarship": ["scholarship", "scholarships", "fee waiver", "merit scholarship", "financial aid"],
    "fees": ["fees", "fee", "tuition", "course fee", "btech fee", "annual fee", "cost of study", "fee structure"],
    "programs": ["courses", "programs", "branches", "specialisations", "departments", "seats", "intake", "duration of course"],
    "hostel": ["hostel", "hostels", "mess", "accommodation", "hostel booking", "room", "warden"],
    "placements": ["placements", "recruiters", "package", "salary", "companies visiting", "career centre", "internships"],
    "rankings": ["nirf", "naac", "ranking", "accreditation", "rank", "ugc", "reputation"],
    "campuses": ["campus", "campuses", "location", "ktr", "ramapuram", "vadapalani", "trichy", "ncr campus", "address", "library"],
    "contact": ["phone number", "contact", "email", "helpline", "website", "enquiry"],
    "student_life": ["clubs", "sports", "fest", "events", "gym", "student life", "ragging", "transport", "canteen", "hackathon"],
    "about_bot": ["who are you", "about you", "your name", "what are you", "your creator"],
    "greeting": ["hi", "hello", "hey", "hii", "good morning"],
    "thanks": ["thanks", "thank you", "thx"],
    "offtopic": ["capital of", "joke", "weather", "python code", "movie", "cricket score", "recipe", "news today", "prime minister", "math problem"],
}
WRAPPERS = ["{}", "tell me about {}", "can you tell me about {}", "i want to know about {}", "{} at srm", "srm {}",
            "{} details please", "what about {}", "help with {}", "info on {}"]


def augmented_train():
    X, y = [], []
    for label, qs in TRAIN.items():
        X += qs
        y += [label] * len(qs)
    for label, kws in KEYWORDS.items():
        for k in kws:
            for w in WRAPPERS:
                if label in ("greeting", "thanks") and w != "{}":
                    continue
                if label == "offtopic" and "srm" in w:
                    continue
                X.append(w.format(k))
                y.append(label)
    return X, y


# ---- held-out test set: never used for training ---------------------------------------------------
TEST = [
    ("what documents do I need to bring for admission", "admissions"), ("am I eligible if I got 58 percent in 12th", "admissions"),
    ("how can I join srm after 12th", "admissions"), ("is there an age limit to apply", "admissions"),
    ("how long is the srmjeee test", "exam"), ("when will srmjeee results come", "exam"), ("what subjects come in srmjeee", "exam"),
    ("can i book a different slot for the entrance exam", "exam"), ("is a laptop needed for the entrance test", "exam"),
    ("do toppers get free education here", "scholarship"), ("how can i reduce my tuition with a scholarship", "scholarship"),
    ("what rank gives a fee waiver", "scholarship"),
    ("how much is the yearly fee for civil engineering", "fees"), ("cost of mtech aerospace", "fees"), ("what will the course cost me", "fees"),
    ("is a b arch degree offered at ramapuram", "programs"), ("how many years is mtech", "programs"), ("which branches are there in engineering", "programs"),
    ("what degrees can i study here", "programs"),
    ("how do i reserve a room in the hostel", "hostel"), ("what food is served in the mess", "hostel"), ("how many girls can the hostels hold", "hostel"),
    ("who do i contact for hostel issues", "hostel"), ("are ac rooms available", "hostel"),
    ("what is the highest salary offered", "placements"), ("which big companies hire srm students", "placements"), ("does microsoft come to campus", "placements"),
    ("how is the career centre helpful", "placements"),
    ("what grade did naac give", "rankings"), ("where does srm stand in nirf", "rankings"), ("is srm recognised by ugc", "rankings"),
    ("how many campuses are there", "campuses"), ("how big is the kattankulathur campus", "campuses"), ("is there a campus in haryana", "campuses"),
    ("how do i reach the college from chennai", "campuses"),
    ("what is the number of the admissions office", "contact"), ("give me the official email for queries", "contact"), ("where can i find the student portal", "portal"),
    ("which student clubs can i join", "student_life"), ("are there sports grounds", "student_life"), ("what cultural events happen", "student_life"),
    ("is ragging allowed", "student_life"),
    ("hello there", "greeting"), ("good evening mist", "greeting"), ("heyy", "greeting"),
    ("who developed you", "about_bot"), ("what are you capable of", "about_bot"), ("are you powered by google", "about_bot"),
    ("thanks mist", "thanks"), ("thank u so much", "thanks"),
    ("who is the president of usa", "offtopic"), ("write me a poem", "offtopic"), ("how to learn guitar", "offtopic"),
    ("what is the speed of light", "offtopic"), ("best pizza recipe", "offtopic"), ("explain neural networks", "offtopic"),
]


# ---- MIST-AI specific skills (answered from the Postgres tables) ----------------------------------
TRAIN.update({
    "faculty": [
        "who is dr rajesh", "find professor kumar", "faculty details of anand", "hod of cse department", "who is the dean of engineering",
        "contact details of professor", "search faculty by name", "teacher info", "who teaches data structures", "staff finder",
        "tell me about prof sharma", "faculty list for ece", "i want to find a lecturer", "who is the head of department",
        "professor email id", "where does dr priya sit", "cabin number of professor", "find madam from it department"],
    "pyq": [
        "give me pyq", "previous year question papers", "pyq for dbms", "question paper for operating systems",
        "old papers for maths", "model question paper", "past papers for computer networks", "where can i get pyqs",
        "question bank for data structures", "previous year paper for toc", "sample papers", "pyq of cse semester 3",
        "i need last year question paper", "study material for dbms", "pyq for artificial intelligence"],
    "portal": [
        "academia link", "student portal link", "what is academia", "exam portal", "library portal", "how to login to student portal",
        "srm portal website", "which portal for results", "portal for course registration", "where is the net id login",
        "link to sp srmist", "hall ticket portal", "academia login", "official student portal", "portals available"],
    "portal_feature": [
        "where can i check attendance", "how to download hall ticket", "how do i pay fees online", "how to check internal marks",
        "course registration steps", "where to see exam results", "how to see timetable", "how to apply for bus pass",
        "how to check exam schedule", "where can i see my marks", "how to register for courses", "book search in library portal",
        "how to apply for scholarship online", "attendance percentage check", "how to get hostel allocation"],
    "location": [
        "where is the library", "where is tech park", "food court location", "where is the auditorium", "directions to health center",
        "where is the placement office", "main building location", "where is the sports complex", "find the canteen",
        "how to get to university building", "where is the hostel block", "where is the administrative office", "location of tp2",
        "where is the medical centre", "which building is the library"],
})
KEYWORDS.update({
    "faculty": ["faculty", "professor", "hod", "dean", "lecturer", "staff finder", "teacher"],
    "pyq": ["pyq", "previous year papers", "question paper", "question bank", "past papers", "old papers"],
    "portal": ["portal", "academia", "student portal", "net id login", "sp srmist"],
    "portal_feature": ["attendance", "hall ticket", "internal marks", "timetable", "course registration", "exam results", "bus pass"],
    "location": ["where is", "location", "directions", "library location", "tech park", "food court", "auditorium", "health centre"],
})
TEST.extend([
    ("who is the hod of mechanical", "faculty"), ("find dr arun from it", "faculty"), ("professor contact for ece", "faculty"),
    ("pyq for compiler design", "pyq"), ("any previous papers for physics", "pyq"), ("where do i download old question papers", "pyq"),
    ("what is the link of academia", "portal"), ("which website to log in to the student portal", "portal"),
    ("how can i see my attendance", "portal_feature"), ("how to download my hall ticket", "portal_feature"), ("where do i register for courses", "portal_feature"),
    ("where is the library building", "location"), ("how do i reach the food court", "location"), ("where is the health center", "location"),
])
TOPICS.update({
    "faculty": set(), "pyq": set(), "portal": {"Portals"}, "portal_feature": {"Portals"}, "location": {"Campuses", "About"},
})
LABELS.update({"faculty": "Faculty", "pyq": "PYQ papers", "portal": "Portals", "portal_feature": "Portal how-to",
               "location": "Campus locations"})
# intents answered directly from the MIST-AI database tables (skills.py)
SKILL_INTENTS = {"faculty", "pyq", "portal", "portal_feature", "location"}

TRAIN["offtopic"] += [
    "who is virat kohli", "who is the ceo of google", "who is the prime minister of canada", "who is shah rukh khan",
    "who is the richest person", "who is the president of france", "who is taylor swift", "who is messi",
    "who invented the telephone", "who wrote hamlet", "who is the owner of tesla", "who is albert einstein",
]


# ---- academics / exam rules / facilities / safety ------------------------------------------------
TRAIN.update({
    "academics": [
        "academic calendar 2026-27", "when do classes start", "when are the cycle tests", "when do semester exams begin",
        "what is the attendance requirement", "minimum attendance for exams", "what happens if I am detained",
        "how to apply for revaluation", "revaluation last date", "is there a model exam", "when is the model exam",
        "what is the exam policy", "what are the exam rules", "when does the even semester start",
        "what are the working days", "what is a day order", "malpractice in exams punishment", "makeup exam rules",
        "internal assessment dates", "when is the last working day", "how are internals calculated"],
    "facilities": [
        "library timings", "what time does the library close", "how many books can i borrow", "library fine for late return",
        "can i borrow a laptop from the library", "bus facility", "how to book college bus", "bus timings from chennai",
        "bus fee", "how to get bus pass", "transport office contact", "anti ragging helpline number",
        "ragging complaint", "how to report ragging", "is the central library open on sunday", "library membership",
        "lost library book what to do", "duplicate bus pass cost", "campus safety contact", "student helpline"],
})
KEYWORDS.update({
    "academics": ["academic calendar", "cycle test", "model exam", "attendance", "detained", "revaluation", "semester dates", "day order", "exam rules"],
    "facilities": ["library timings", "library fine", "borrow books", "bus", "bus pass", "transport", "anti ragging", "helpline"],
})
TEST.extend([
    ("when is the next semester exam", "academics"), ("what is the minimum attendance percentage", "academics"),
    ("can i apply for revaluation after results", "academics"), ("when is internal assessment 2", "academics"),
    ("how late is the central library open", "facilities"), ("how much is the fine for a late book", "facilities"),
    ("what are the bus timings", "facilities"), ("who do i call for ragging", "facilities"),
])
TOPICS.update({
    "academics": {"Academics", "Exams"}, "facilities": {"Library", "Transport", "Safety", "Student life"},
})
LABELS.update({"academics": "Academics & exams", "facilities": "Library, transport & safety"})
