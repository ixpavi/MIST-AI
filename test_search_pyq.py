import os, sys
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.abspath('backend'))
from search import execute_fetch

def test():
    query = "give me chemistry pyq"
    clean_query = query.replace("pyq", "").replace("previous year question", "").replace("paper", "").replace("give me", "").replace("for", "").strip()
    words = clean_query.lower().split()
    flex_words = "%" + "%".join(words) + "%"
    
    print(repr(clean_query.lower()), repr(flex_words))
    
    try:
        rows = execute_fetch(
            """
            SELECT subject_name, pyq_link
            FROM pyq_resources
            WHERE LOWER(subject_code) = %s OR LOWER(subject_name) LIKE %s
            LIMIT 1
            """,
            (clean_query.lower(), flex_words)
        )
        print("ROWS:", rows)
    except Exception as e:
        print("ERROR:", e)

test()
