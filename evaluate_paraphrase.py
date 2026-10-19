"""Does MIST understand questions worded differently from the stored facts?   python evaluate_paraphrase.py

Compares the old lexical retriever (TF-IDF) with the local semantic model on 60 paraphrased questions, and shows
how confident the system is when it is right vs when it should say 'not in my knowledge base'."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend"))
os.environ.setdefault("GEMINI_API_KEY", "")          # this test never needs the language model

import engine  # noqa: E402
import intents  # noqa: E402
import semantic  # noqa: E402

PARAPHRASE = [
    ("do i have to pay to take the entrance test", ["jeee-apply", "adm-fee"]),
    ("will wrong answers cut my marks in the entrance", ["jeee-pattern"]),
    ("how long is the online entrance paper", ["jeee-pattern"]),
    ("can i appear for the exam from my house", ["jeee-pattern"]),
    ("which documents are needed while applying for the exam", ["jeee-misc", "adm-policy-docs", "jeee-apply"]),
    ("what marks do i need in 12th to be eligible", ["jeee-eligibility", "adm-elig-bio"]),
    ("i want to change my test slot", ["jeee-slot-admit"]),
    ("how much money to confirm my seat after counselling", ["jeee-counselling"]),
    ("do toppers pay full fees", ["jeee-scholarship", "scholar-merit", "adm-scholar"]),
    ("my parents work at srm will i get a discount", ["scholar-other"]),
    ("my father is in the army any concession", ["scholar-other"]),
    ("how do i cancel my admission and get money back", ["adm-policy-refund"]),
    ("is there any agent i should contact for admission", ["adm-policy-docs"]),
    ("what do i need for MBA entry", ["adm-mba-law"]),
    ("tell me about studying abroad for a semester", ["ir-abroad"]),
    ("discount for NRI children", ["adm-international"]),
    ("where can girls stay on campus and how many seats", ["hostel-capacity"]),
    ("what rules do hostel residents have to follow", ["hostel-rules-1"]),
    ("can my mom and dad sleep over in my hostel room", ["hostel-rules-2"]),
    ("what is served to eat in hostel", ["hostel-capacity"]),
    ("who do i contact if my hostel room has a problem", ["hostel-contact", "hostel-rules-2"]),
    ("how do i reserve a room in the dorm", ["hostel-apply"]),
    ("whats the highest salary any student got", ["placements-stats"]),
    ("which top firms come to hire students", ["placements-tiers", "placements-recruiters"]),
    ("how good are the job opportunities after the course", ["placements-career", "placements-tiers"]),
    ("is srm recognized by the government", ["rank-naac", "about-overview"]),
    ("where does the university stand in national rankings", ["rank-nirf"]),
    ("how many students study there", ["about-ktr", "about-overview"]),
    ("how many campuses and where", ["about-campuses"]),
    ("what time does the book place open and close", ["lib-hours"]),
    ("how many days can i keep a book", ["lib-borrow"]),
    ("what do i pay if i return a book late", ["lib-fines"]),
    ("how can i travel to campus every day from the city", ["transport-info"]),
    ("when does the college bus leave in the evening", ["transport-timings"]),
    ("which societies can a student join for dance and music", ["clubs-central"]),
    ("any coding communities for CSE students", ["clubs-ctech", "clubs-cintel-nwc"]),
    ("someone is harassing freshers what number to call", ["antiragging"]),
    ("when will my even semester classes begin", ["cal-sh-even"]),
    ("when are the first tests of the odd semester", ["cal-sh-odd"]),
    ("what if i dont have enough attendance", ["exam-rules"]),
    ("can i get my paper rechecked", ["exam-revaluation"]),
    ("what is the punishment for copying in exams", ["exam-malpractice"]),
    ("how far is the college from the airport", ["reach-ktr"]),
    ("can i visit the campus on weekends", ["reach-visit"]),
    ("what sports can i play and is there a pool", ["sports-facilities"]),
    ("how do i join the university cricket team", ["sports-quota"]),
    ("is there wifi food court and bank in the campus", ["campus-life"]),
    ("what exams does srm accept for B arch", ["adm-pg"]),
    ("who runs the cse department", ["dept-"]),
    ("contact email for the head of mechanical engineering", ["dept-"]),
    ("which faculties does the university have", ["colleges", "about-overview"]),
    ("where to log in for hostel booking and bus", ["portals-links", "hostel-apply", "transport-info"]),
    ("what are the official website and phone number", ["contact-general", "adm-contacts"]),
    ("phone number for engineering admissions enquiry", ["adm-contacts"]),
    ("how is the srmjeee test conducted", ["jeee-pattern"]),
    ("what subjects are asked in the entrance exam", ["jeee-pattern", "jeee-eligibility"]),
    ("how many phases does the entrance have and what dates", ["adm-phases", "jeee-"]),
    ("what are the sections of the online entrance test", ["jeee-pattern"]),
]

HELDOUT = [
    ("kya library sunday ko khula rehta hai", ["lib-hours"]),
    ("how many text books can first year students take home", ["lib-borrow"]),
    ("laptop rental from the book centre", ["lib-borrow"]),
    ("price of the college bus ticket for the year", ["transport-info"]),
    ("is there a gym", ["sports-facilities", "campus-life"]),
    ("any place to get cash on campus", ["campus-life"]),
    ("can i get a refund if i leave the dorm midway", ["hostel-rules-2", "hostel-fees"]),
    ("who is the in charge of dorms", ["hostel-contact"]),
    ("what are the dates for the second cycle test in sept", ["cal-sh-odd"]),
    ("what is the last day of classes in april", ["cal-sh-even"]),
    ("does srm teach law", ["colleges", "adm-mba-law"]),
    ("how to get a seat in pharmacy", ["adm-health"]),
    ("scholarship for students whose parents earn little", ["scholar-other"]),
    ("what is the minimum percentage needed in physics chemistry maths", ["jeee-eligibility"]),
    ("is it compulsory to attend all three phases", ["jeee-misc", "adm-phases"]),
    ("how much is the counselling deposit", ["jeee-counselling"]),
    ("can i get the application fee back", ["adm-fee", "jeee-apply"]),
    ("do they provide any coaching for athletes", ["sports-quota"]),
    ("contact number to report ragging incidents", ["antiragging"]),
    ("what time does the evening bus leave for city", ["transport-timings"]),
    ("which student organisations are for ai and machine learning", ["clubs-cintel-nwc"]),
    ("how do i find out the hod email of aerospace", ["dept-"]),
    ("what is the best way to reach campus by train", ["reach-ktr"]),
    ("recognition and grade given by naac", ["rank-naac"]),
    ("how many offers did students receive in 2018", ["placements-stats"]),
    ("where can overseas students apply", ["adm-international"]),
    ("is there transport for medical students", ["transport-timings"]),
    ("what is the penalty for plagiarism or cheating in semester exams", ["exam-malpractice"]),
    ("what happens if i cannot attend an exam because of fever", ["exam-revaluation"]),
    ("distance education options", ["colleges"]),
]

NEGATIVE = [
    "tell me a joke", "who won the cricket world cup", "what is the capital of france", "best pizza recipe",
    "write python code for sorting", "weather in london today", "how to lose weight fast", "who is the prime minister",
    "explain quantum computing", "latest iphone price", "how do i invest in stocks", "recommend a movie to watch",
    "what is the meaning of life", "translate hello to french", "how to cook biryani", "who is elon musk",
    "what is the speed of light", "how do vaccines work", "bitcoin price", "football match score",
]


def run(chunks, mode, data=None):
    """Returns (hits@1, hits@3, rows) where rows = [(correct@1, top_score)] for paraphrases."""
    orig = semantic.ready
    if mode == "tfidf":
        semantic.ready = lambda: False
    try:
        h1 = h3 = 0
        rows = []
        for q, want in (data or PARAPHRASE):
            intent, _ = intents.predict(q)
            top, score, _ = engine.retrieve(q, chunks, intent=intent, k=3)
            ids = [c["id"] for c in top]
            ok1 = bool(ids) and any(ids[0].startswith(w) for w in want)
            ok3 = any(i.startswith(w) for i in ids for w in want)
            h1 += ok1
            h3 += ok3
            rows.append((ok1, score))
        neg = []
        for q in NEGATIVE:
            top, score, _ = engine.retrieve(q, chunks, intent=None, k=3)
            neg.append((bool(top), score))
        return h1, h3, rows, neg
    finally:
        semantic.ready = orig


def main():
    chunks = engine.load_local_kb()
    n = len(PARAPHRASE)
    h1, h3, _, neg = run(chunks, "tfidf")
    print(f"KB entries: {len(chunks)}   paraphrased questions: {n}   off-topic questions: {len(NEGATIVE)}\n")
    print(f"TF-IDF only    : hit@1 {h1}/{n} = {h1 / n:.0%}   hit@3 {h3}/{n} = {h3 / n:.0%}   off-topic leaks {sum(x for x, _ in neg)}/{len(neg)}")
    if not semantic.load_blocking():
        print("local semantic model unavailable:", semantic.status())
        return
    h1, h3, rows, neg = run(chunks, "semantic")
    print(f"Local semantic : hit@1 {h1}/{n} = {h1 / n:.0%}   hit@3 {h3}/{n} = {h3 / n:.0%}   off-topic leaks {sum(x for x, _ in neg)}/{len(neg)}")
    print("\nConfidence calibration (semantic): answering directly (no LLM) when score >= T")
    print("   T     answered   correct   precision | off-topic above T")
    for t in np.arange(0.60, 0.96, 0.05):
        sel = [ok for ok, sc in rows if sc >= t]
        leak = sum(1 for _, sc in neg if sc >= t)
        prec = (sum(sel) / len(sel)) if sel else float("nan")
        print(f"  {t:.2f}   {len(sel):3d}/{n}     {sum(sel):3d}      {prec:6.1%}   | {leak}/{len(neg)}")
    hn = len(HELDOUT)
    t1, t3, _, _ = run(chunks, "tfidf", HELDOUT)
    s1, s3, hrows, _ = run(chunks, "semantic", HELDOUT)
    print(f"\nFRESH held-out set ({hn} questions never used for tuning):")
    print(f"  TF-IDF only   : hit@1 {t1}/{hn} = {t1 / hn:.0%}   hit@3 {t3}/{hn} = {t3 / hn:.0%}")
    print(f"  Local semantic: hit@1 {s1}/{hn} = {s1 / hn:.0%}   hit@3 {s3}/{hn} = {s3 / hn:.0%}")
    sel = [ok for ok, sc in hrows if sc >= engine.EXTRACT_MIN]
    print(f"  Answered directly (score >= {engine.EXTRACT_MIN}): {len(sel)}/{hn}, correct {sum(sel)}"
          + (f" = {sum(sel) / len(sel):.0%}" if sel else ""))
    print("\nTop scores of off-topic questions:", sorted(round(sc, 2) for _, sc in neg)[-5:])


if __name__ == "__main__":
    main()
