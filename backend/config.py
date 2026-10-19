"""Central configuration. Everything comes from environment variables (Render dashboard) or a local .env file."""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    load_dotenv(Path(__file__).resolve().parent / ".env")
except Exception:
    pass


def get(name, default=None):
    value = os.getenv(name)
    return value if value else default


GEMINI_API_KEY = get("GEMINI_API_KEY")
CHAT_MODEL = get("GEMINI_CHAT_MODEL", "gemini-2.5-flash-lite")
# Tried in order; a busy or retired model falls through to the next one.
CHAT_MODELS = [CHAT_MODEL] + [m for m in ("gemini-3.5-flash", "gemini-3.8-flash", "gemini-flash-lite-latest")
                              if m != CHAT_MODEL]
EMBED_MODEL = get("GEMINI_EMBED_MODEL", "gemini-embedding-001")
EMBED_DIM = int(get("GEMINI_EMBED_DIM", "768"))
MAX_MESSAGE_CHARS = 500
