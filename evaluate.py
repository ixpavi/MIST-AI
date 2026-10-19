"""Measure MIST AI without the language model:  python evaluate.py
 - intent accuracy on the held-out question set
 - retrieval hit@1 / hit@3 on questions whose answer chunk is known
 - routing checks (deadline / fee questions must use exact tools, off-topic must be declined)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend"))

import engine  # noqa: E402
import intents  # noqa: E402
import tools  # noqa: E402

# question -> id (or id prefix) of the knowledge-base entry that should be retrieved
RETRIEVAL = [
    ("how many questions are in the srmjeee exam", "jeee-pattern"),
    ("is there negative marking in srmjeee", "jeee-pattern"),
    ("how much do I pay at counselling", "jeee-counselling"),
    ("what is the srmjeee application fee", "jeee-apply"),
    ("can I change my exam slot", "jeee-slot-admit"),
    ("do I get a scholarship for a good rank", "jeee-scholarship"),
    ("what are the eligibility marks for btech", "jeee-eligibility"),
    ("how do I book a hostel room", "hostel-apply"),
    ("how many hostel blocks are there for girls", "hostel-capacity"),
    ("who can stay in the hostel", "hostel-eligibility"),
    ("hostel helpdesk email", "hostel-contact"),
    ("which companies pay more than 20 lakh", "placements-tiers"),
    ("highest package at srm", "placements-stats"),
    ("what is srm naac grade", "rank-naac"),
    ("nirf rank 2025", "rank-nirf"),
    ("how many campuses does srm have", "about-campuses"),
    ("how large is the ktr campus", "about-ktr"),
    ("admission helpline number", "adm-contacts"),
    ("which entrance exam for b arch", "adm-pg"),
    ("is srm a deemed university founded when", "about-overview"),
    ("what sports facilities are there", "sports"),
    ("who made mist ai", "mist-about"),
    # --- added with the campus knowledge base: also phrased the way students really type ---
    ("library timings", "lib-hours"),
    ("what time does the central library close", "lib-hours"),
    ("how many books can a ug student borrow", "lib-borrow"),
    ("fine for returning a library book late", "lib-fines"),
    ("how do i get a bus pass", "transport-info"),
    ("college bus timings morning", "transport-timings"),
    ("anti ragging helpline number", "antiragging"),
    ("what scholarships are there for rank 300", "scholar-merit"),
    ("scholarship for children of army personnel", "scholar-other"),
    ("is there a refund if i withdraw admission", "adm-policy-refund"),
    ("how to apply for mba at srm", "adm-mba-law"),
    ("fee waiver for nri students", "adm-international"),
    ("when do classes begin for first year", "cal-sh-odd"),
    ("when is the even semester model exam", "cal-sh-even"),
    ("what happens if my attendance is low", "exam-rules"),
    ("how long do i have to apply for revaluation", "exam-revaluation"),
    ("how do i get to college from the airport", "reach-ktr"),
    ("is there a swimming pool", "sports-facilities"),
    ("can parents stay in hostel as guests", "hostel-rules-2"),
    ("are there nss and ncc units", "campus-life"),
    ("clubs for coding students", "clubs-ctech"),
    ("what are the student clubs", "clubs-central"),
    ("phone number of the transport office", "transport-timings"),
    ("who is the hod of data science", "dept-"),
]

NEGATIVE = ["tell me a joke", "who won ipl", "what is the capital of france", "best pizza recipe", "write python code for sorting",
            "weather in london", "how to lose weight"]

ROUTING = [
    ("when is the last date for srmjeee phase 3", "tool"),
    ("what is the fee for civil engineering at ktr", "tool"),
    ("what is the capital of france", "decline"),
    ("hello", "rule"),
    ("who are you", "rule"),
]


def rag_retrieve(q, chunks):
    return engine.retrieve(q, chunks, intent=None, k=3)


def main():
    X, y = intents.augmented_train()          # fresh model that has never seen the test questions
    pipe = intents.build_pipeline().fit(X, y)
    tx, ty = zip(*intents.TEST)
    ok = sum(p == t for p, t in zip(pipe.predict(list(tx)), ty))
    print(f"Intent model (held-out, unseen questions): {ok}/{len(tx)} = {ok / len(tx):.1%}")

    chunks = engine.load_local_kb()
    engine.ensure_embeddings(chunks)
    mode = "hybrid (embeddings + keywords)" if any(c.get("embedding") for c in chunks) else "keyword only"
    h1 = h3 = 0
    misses = []
    for q, want in RETRIEVAL:
        intent, _ = intents.predict(q)
        top, _, _ = engine.retrieve(q, chunks, intent=intent, k=3)
        ids = [c["id"] for c in top]
        h1 += bool(ids) and ids[0].startswith(want)
        hit = any(i.startswith(want) for i in ids)
        h3 += hit
        if not hit:
            misses.append((q, want, ids))
    n = len(RETRIEVAL)
    print(f"Retrieval [{mode}] over {len(chunks)} entries: hit@1 {h1}/{n} = {h1 / n:.0%}, hit@3 {h3}/{n} = {h3 / n:.0%}")
    for q, want, ids in misses:
        print(f"  miss: {q!r} wanted {want}, got {ids}")

    leaks = []
    for q in NEGATIVE:
        top, _, _ = rag_retrieve(q, chunks)
        if top:
            leaks.append((q, [c["id"] for c in top][:2]))
    print(f"Off-topic questions that wrongly retrieve SRM entries: {len(leaks)}/{len(NEGATIVE)}")
    for q, ids in leaks:
        print(f"  leak: {q!r} -> {ids}")

    r_ok = 0
    for q, route in ROUTING:
        got = engine.answer(q)["route"]
        r_ok += got == route
        if got != route:
            print(f"  routing miss: {q!r} wanted {route}, got {got}")
    print(f"Routing: {r_ok}/{len(ROUTING)}")
    print(f"Tools self-check: SRMJEEE tool says -> {tools.jeee_status().splitlines()[0]}")


if __name__ == "__main__":
    main()
