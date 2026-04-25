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


def ask_llm(prompt: str) -> str:
    if not prompt or not prompt.strip():
        return "Please type a question so I can help you!"

    if genai is None:
        return "AI service temporarily unavailable."

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "AI service is not configured."

    # Strict system instructions for SRM context and formatting
    system_instruction = (
        "You are MIST AI, the official student assistant for SRM University KTR (Kattankulathur) campus. "
        "CRITICAL RULES: "
        "1. Strictly answer ONLY in the context of SRM University. If the user asks something outside of the SRM context or generic trivia, politely decline to answer. "
        "2. Do NOT use any Markdown formatting! No asterisks (*), no hashtags (#), no backticks (`), and no bold text. "
        "3. Provide answers in plain text only. Keep it concise, helpful, and fast. "
        f"\n\nUser Question: {prompt.strip()}"
    )

    try:
        genai.configure(api_key=api_key)
        
        # Use fastest models to ensure quick responses
        fast_models = ("gemini-1.5-flash-8b", "gemini-1.5-flash", "gemini-flash-latest")
        
        # Strict safety settings to block profanity/bad language
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_LOW_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_LOW_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_LOW_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_LOW_AND_ABOVE"}
        ]

        last_error = None
        for model_name in fast_models:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(
                    system_instruction,
                    safety_settings=safety_settings,
                    request_options={"timeout": 15}
                )
                
                try:
                    text = response.text
                except ValueError:
                    # Raised when response is blocked by safety settings
                    return "I cannot answer that. Please keep the conversation respectful and appropriate."
                
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
