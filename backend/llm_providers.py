"""Free / low-cost LLM providers behind one tiny interface.

Groq, OpenRouter and Cerebras all expose an OpenAI-compatible /chat/completions API, so one small client covers
them. No SDK, no training: these are ready-made hosted models used as an optional 'wording layer'.

Configure with environment variables (only providers that have a key are used):
    LLM_CHAIN            order to try, default "groq,openrouter,cerebras,gemini"  (drop "gemini" to be Gemini-free)
    GROQ_API_KEY         https://console.groq.com/keys          (free tier, no card)
    OPENROUTER_API_KEY   https://openrouter.ai/keys             (free ':free' models)
    CEREBRAS_API_KEY     https://cloud.cerebras.ai
    <NAME>_MODEL / <NAME>_FAST_MODEL / <NAME>_BASE_URL   optional overrides
"""
import json
import logging
import time
import urllib.error
import urllib.request

import config

log = logging.getLogger("mist.llm")

PROVIDERS = {
    "groq": {"base": "https://api.groq.com/openai/v1", "key": "GROQ_API_KEY",
             "models": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"], "fast": ["llama-3.1-8b-instant"]},
    "openrouter": {"base": "https://openrouter.ai/api/v1", "key": "OPENROUTER_API_KEY",
                   "models": ["openrouter/free"], "fast": ["openrouter/free"]},
    "cerebras": {"base": "https://api.cerebras.ai/v1", "key": "CEREBRAS_API_KEY",
                 "models": ["gpt-oss-120b"], "fast": ["gpt-oss-120b"]},
}
DEFAULT_CHAIN = "groq,openrouter,cerebras,gemini"
_cooldown = {}
COOLDOWN = {429: 600, 401: 3600, 403: 3600, 404: 3600, "5xx": 60, "other": 30}


def chain():
    raw = config.get("LLM_CHAIN", DEFAULT_CHAIN)
    return [x.strip().lower() for x in raw.split(",") if x.strip()]


def configured():
    """Names of providers that can be called right now (key present)."""
    out = []
    for name in chain():
        if name == "gemini":
            if config.get("GEMINI_API_KEY"):
                out.append(name)
        elif name in PROVIDERS and config.get(PROVIDERS[name]["key"]):
            out.append(name)
    return out


def _models(name, fast):
    p = PROVIDERS[name]
    custom = config.get(name.upper() + ("_FAST_MODEL" if fast else "_MODEL"))
    return [custom] if custom else list(p["fast" if fast else "models"])


def _blocked(key):
    return _cooldown.get(key, 0) > time.time()


def _trip(key, code, retry_after=None):
    if code == 429:
        try:        # providers say how long to wait; per-minute limits reset in seconds, so do not freeze for 10 minutes
            secs = min(max(int(float(retry_after)), 5), 600) if retry_after else 20
        except (TypeError, ValueError):
            secs = 20
    else:
        secs = COOLDOWN.get(code) or (COOLDOWN["5xx"] if isinstance(code, int) and code >= 500 else COOLDOWN["other"])
    _cooldown[key] = time.time() + secs
    log.warning("%s paused for %ss (%s)", key, secs, code)


def _call(name, model, system, prompt, max_tokens, temperature, timeout):
    p = PROVIDERS[name]
    base = config.get(name.upper() + "_BASE_URL", p["base"]).rstrip("/")
    body = {"model": model, "temperature": temperature, "max_tokens": max_tokens,
            "messages": ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(
        base + "/chat/completions", data=json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + config.get(p["key"]), "Content-Type": "application/json",
                 "User-Agent": "mist-ai/1.0", "HTTP-Referer": "https://mist-ai-r6t1.onrender.com", "X-Title": "MIST AI"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read())
    msg = (data.get("choices") or [{}])[0].get("message") or {}
    return (msg.get("content") or "").strip()


def chat(name, system, prompt, max_tokens=900, temperature=0.3, fast=False, timeout=12):
    """Try one OpenAI-compatible provider. Returns (text, 'provider:model') or (None, None)."""
    if name not in PROVIDERS or not config.get(PROVIDERS[name]["key"]):
        return None, None
    for model in _models(name, fast):
        key = f"{name}:{model}"
        if _blocked(key):
            continue
        try:
            text = _call(name, model, system, prompt, max_tokens, temperature, timeout)
            if text:
                return text, key
        except urllib.error.HTTPError as e:
            _trip(key, e.code, e.headers.get("retry-after") if e.headers else None)
        except Exception as e:                      # timeouts, DNS, bad JSON ...
            _trip(key, "other")
            log.warning("%s failed: %s", key, str(e)[:100])
    return None, None
