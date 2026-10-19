"""Tests the free-LLM provider chain against a local fake OpenAI-compatible server (no real keys needed).
    python test_llm_providers.py"""
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# fully Gemini-free configuration
os.environ["GEMINI_API_KEY"] = ""
os.environ["GROQ_API_KEY"] = "test-key"
os.environ["LLM_CHAIN"] = "groq"
os.environ["RENDER"] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend"))

BEHAVIOUR = {"mode": "ok", "calls": []}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        BEHAVIOUR["calls"].append(body["model"])
        mode = BEHAVIOUR["mode"]
        if mode == "429_first_model" and body["model"] == "llama-3.3-70b-versatile":
            self.send_response(429)
            self.end_headers()
            self.wfile.write(b'{"error":"rate limited"}')
            return
        if mode == "500":
            self.send_response(500)
            self.end_headers()
            return
        text = "Wrong number: you need 99% attendance." if mode == "bad_number" else \
            f"[{'big' if '70b' in body['model'] else 'small'}] Parents can request short-term guest accommodation with prior notice."
        out = {"choices": [{"message": {"content": text}}]}
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(out).encode())


srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
threading.Thread(target=srv.serve_forever, daemon=True).start()
os.environ["GROQ_BASE_URL"] = f"http://127.0.0.1:{srv.server_port}/v1"

import engine  # noqa: E402
import llm_providers  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


Q = "can my parents stay with me in the hostel"
check("configured providers = groq only (Gemini-free)", llm_providers.configured() == ["groq"], str(llm_providers.configured()))

BEHAVIOUR.update(mode="ok", calls=[])
r = engine.answer(Q, [])
check("answer comes from the free provider", r["route"] == "kb+llm" and str(r.get("model", "")).startswith("groq:"), f"{r['route']} {r.get('model')}")
check("answer text is the provider's", "[big]" in r["reply"], r["reply"][:70])

llm_providers._cooldown.clear()
engine._answer_cache.clear()
BEHAVIOUR.update(mode="429_first_model", calls=[])
r = engine.answer(Q, [])
check("429 on the big model falls through to the small model", "[small]" in r["reply"], str(BEHAVIOUR["calls"]))
BEHAVIOUR["calls"].clear()
engine._answer_cache.clear()
engine.answer(Q + " please", [])
check("rate-limited model is skipped afterwards (cooldown)", "llama-3.3-70b-versatile" not in BEHAVIOUR["calls"], str(BEHAVIOUR["calls"]))

llm_providers._cooldown.clear()
engine._answer_cache.clear()
BEHAVIOUR.update(mode="500", calls=[])
r = engine.answer(Q, [])
check("provider outage -> still answers from the knowledge base", r["route"] == "kb" and r["reply"].strip() != "", f"{r['route']}: {r['reply'][:60]!r}")

llm_providers._cooldown.clear()
engine._answer_cache.clear()
BEHAVIOUR.update(mode="bad_number", calls=[])
r = engine.answer("what is the attendance rule", [])
check("numeric guard drops an invented '99%'", "99%" not in r["reply"], r["reply"][:80])

llm_providers._cooldown.clear()
BEHAVIOUR.update(mode="ok", calls=[])
text = engine._rewrite_llm("Rewrite: and how many girls?")
check("follow-up rewrite uses the fast model", text and "[small]" in text, str(BEHAVIOUR["calls"]))

os.environ["GROQ_API_KEY"] = ""
check("no key -> provider silently disabled", llm_providers.configured() == [] and engine.llm_status() == [])

print("\nFAILED: " + ", ".join(FAILS) if FAILS else "\nAll provider-chain checks passed.")
sys.exit(1 if FAILS else 0)
