import os
import warnings

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

if load_dotenv:
    ENV_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".env"))
    load_dotenv(ENV_PATH)

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        import google.generativeai as genai
except Exception as import_error:
    genai = None
    print(f"[LLM WARNING] google-generativeai is not available: {import_error}")


MODEL_NAMES = ("gemini-1.5-flash", "gemini-2.0-flash", "gemini-flash-latest")


def ask_llm(prompt: str) -> str:
    if not prompt or not prompt.strip():
        return "Please type a question so I can help you!"

    if genai is None:
        return "AI service temporarily unavailable."

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "AI service is not configured."

    try:
        genai.configure(api_key=api_key)
        last_error = None
        for model_name in MODEL_NAMES:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(
                    prompt.strip(),
                    request_options={"timeout": 20}
                )
                text = getattr(response, "text", "") or ""
                return text.strip() or "AI service temporarily unavailable."
            except Exception as model_error:
                last_error = model_error
                continue

        if last_error:
            print(f"[LLM ERROR] Gemini fallback failed: {type(last_error).__name__}")
        return "AI service temporarily unavailable."
    except Exception as error:
        print(f"[LLM ERROR] Gemini fallback failed: {type(error).__name__}")
        return "AI service temporarily unavailable."
