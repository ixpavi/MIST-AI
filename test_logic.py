import os; import sys; sys.path.insert(0, os.path.abspath('backend')); from search import execute_fetch

query = "give me dbms pyq"
clean_query = query.replace("pyq", "").replace("previous year question", "").replace("paper", "").replace("give me", "").replace("for", "").strip()
print("clean_query:", repr(clean_query))

words = clean_query.lower().split()
flex_words = "%" + "%".join(words) + "%"
print("flex_words:", repr(flex_words))

rows = execute_fetch(
    "SELECT subject_name, pyq_link FROM pyq_resources WHERE LOWER(subject_code) = %s OR LOWER(subject_name) LIKE %s LIMIT 1",
    (clean_query.lower(), flex_words)
)
print("ROWS:", rows)
