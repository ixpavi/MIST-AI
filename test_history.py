import os, sys
sys.path.insert(0, os.path.abspath('backend'))
from llm import ask_llm

history = [
    {"role": "user", "parts": [{"text": "Who is the HOD of CSE?"}]},
    {"role": "model", "parts": [{"text": "The HOD of CSE is Dr. Revathi."}]}
]

response = ask_llm("What is her email address?", history)
print("Response:", response)
