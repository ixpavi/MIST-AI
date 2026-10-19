"""Flask API for MIST AI.  POST /chat, POST /feedback, GET /health, GET /stats."""
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

import config
import database
import engine
import semantic
import intents

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("mist.app")

FRONTEND_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend"))
MAX_HISTORY = 10
ACTIONS = {"faculty"}

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
CORS(app)

# Non-fatal: without a database the app still answers from the bundled knowledge base.
try:
    database.init_db()
except Exception as e:
    log.warning("init_db skipped: %s", str(e)[:120])


RATE_LIMIT = 30          # requests
RATE_WINDOW = 60         # per seconds, per client
_hits = {}


def client_ip():
    fwd = request.headers.get("X-Forwarded-For", "")
    return (fwd.split(",")[0].strip() if fwd else request.remote_addr) or "unknown"


def rate_limited():
    """Sliding-window limiter: protects the free-tier model quota from hammering."""
    import time
    from collections import deque
    now = time.time()
    q = _hits.setdefault(client_ip(), deque())
    while q and now - q[0] > RATE_WINDOW:
        q.popleft()
    if len(q) >= RATE_LIMIT:
        return True
    q.append(now)
    if len(_hits) > 5000:
        _hits.clear()
    return False


def clean_history(raw):
    """Keep only well-formed user/model turns from the client, trimmed to a sane size."""
    out = []
    if not isinstance(raw, list):
        return out
    for m in raw[-2 * MAX_HISTORY:]:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        text = m.get("text")
        if text is None and isinstance(m.get("parts"), list) and m["parts"] and isinstance(m["parts"][0], dict):
            text = m["parts"][0].get("text")
        if role not in ("user", "model") or not isinstance(text, str) or not text.strip():
            continue
        out.append({"role": role, "text": text.strip()[:config.MAX_MESSAGE_CHARS]})
    return out


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/chat", methods=["POST"])
def chat():
    if rate_limited():
        return jsonify({"reply": "You are sending messages very quickly. Please wait a few seconds and try again."}), 429
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or not isinstance(data.get("message"), str):
        return jsonify({"reply": "Please send a message."}), 400
    message = data["message"].strip()
    if not message:
        return jsonify({"reply": "Please type a question so I can help you!"}), 400
    if len(message) > config.MAX_MESSAGE_CHARS:
        return jsonify({"reply": f"Please keep your question under {config.MAX_MESSAGE_CHARS} characters."}), 400
    action = data.get("action") if data.get("action") in ACTIONS else None
    session_id = str(data.get("session_id") or "")[:64] or None

    try:
        res = engine.answer(message, clean_history(data.get("history")), action=action)
    except Exception:
        log.exception("engine failed")
        return jsonify({"reply": "Sorry, something went wrong on my side. Please try again."}), 500

    chat_id = database.log_chat(session_id, message, res["reply"], res) if database.is_configured() else None
    return jsonify({
        "reply": res["reply"], "route": res["route"], "intent": res["intent"],
        "intent_conf": res["intent_conf"], "sources": res.get("sources", []), "chat_id": chat_id,
    })


@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.get_json(silent=True) or {}
    chat_id, rating = data.get("chat_id"), data.get("rating")
    if not isinstance(chat_id, int) or isinstance(chat_id, bool) or rating not in (1, -1):
        return jsonify({"ok": False, "error": "chat_id (int) and rating (1 or -1) are required"}), 400
    return jsonify({"ok": bool(database.save_feedback(chat_id, rating))})


@app.route("/health")
def health():
    _, kb_source = engine.get_kb()
    return jsonify({
        "semantic_model": semantic.status()["status"], "status": "ok",
        "db_configured": database.is_configured(),
        "db_ok": database.db_ok() if database.is_configured() else False,
        "kb_source": kb_source,
        "intent_model_loaded": intents.is_loaded(),
        "llm_configured": bool(config.GEMINI_API_KEY),
    })


@app.route("/stats")
def stats():
    if not database.is_configured():
        return jsonify({"total": 0, "grounded": 0, "up": 0, "down": 0, "db": False})
    return jsonify({**database.get_stats(), "db": True})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
