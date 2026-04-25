"""
app.py - Flask backend for MIST AI chatbot
"""

import os
import sys
import traceback

LOCAL_DEPS = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".deps")
if os.path.isdir(LOCAL_DEPS) and LOCAL_DEPS not in sys.path:
    sys.path.insert(0, LOCAL_DEPS)

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Add backend directory to path so imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from search import DEFAULT_REPLY, search
from llm import ask_llm
from database import execute_query, execute_fetch, init_db


# ---------------------------------------------------------------------------
# Flask app setup
# ---------------------------------------------------------------------------
FRONTEND_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend")
)

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
CORS(app)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Serve the chat frontend."""
    try:
        return send_from_directory(app.static_folder, "index.html")
    except Exception as e:
        print(f"[ERROR] Failed to serve index.html: {e}")
        return jsonify({"error": "Frontend not found"}), 500


@app.route("/chat", methods=["POST"])
def chat():
    """
    POST /chat
    Body: { "message": "user question" }
    Returns: { "reply": "chatbot answer" }
    """
    try:
        data = request.get_json(silent=True)
        if not data or "message" not in data:
            return jsonify({"reply": "Please send a message."}), 400

        user_message = str(data["message"]).strip()
        if not user_message:
            return jsonify({"reply": "Please type a question so I can help you!"}), 400

        # Search the database first; Gemini is only a fallback for no-match cases.
        reply = search(user_message)
        if not reply or reply.strip() == DEFAULT_REPLY:
            reply = ask_llm(user_message)

        # Log the conversation (non-blocking - errors here should not break the response)
        try:
            execute_query(
                "INSERT INTO chat_logs (question, response) VALUES (%s, %s)",
                (user_message, reply)
            )
        except Exception as log_err:
            print(f"[WARNING] Failed to log chat: {log_err}")

        return jsonify({"reply": reply})

    except Exception as e:
        print(f"[ERROR] /chat route failed: {e}")
        traceback.print_exc()
        return jsonify({"reply": "Sorry, something went wrong on the server. Please try again."}), 500


@app.route("/history", methods=["GET"])
def history():
    """Return the last 50 chat log entries."""
    try:
        rows = execute_fetch(
            """
            SELECT chat_id, question, response, timestamp
            FROM chat_logs
            ORDER BY timestamp DESC
            LIMIT 50
            """
        )
        logs = []
        for r in rows:
            logs.append({
                "id": r[0],
                "question": r[1],
                "response": r[2],
                "timestamp": r[3].isoformat() if r[3] else None
            })
        return jsonify(logs)
    except Exception as e:
        print(f"[ERROR] /history route failed: {e}")
        return jsonify([]), 500


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("[*] Initializing database...")
    init_db()
    print("[*] Starting MIST AI server on http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
