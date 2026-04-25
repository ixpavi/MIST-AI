import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv(".env")
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

try:
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content("What is the latest news on srmist.edu.in?", tools="google_search_retrieval")
    print("SUCCESS")
    print(response.text)
except Exception as e:
    print(f"ERROR: {e}")
