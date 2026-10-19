"""Local semantic encoder (no API, no quota): BAAI/bge-small-en-v1.5 as a quantised ONNX model via fastembed.

It lets MIST match a question to the right fact by *meaning*, so paraphrases that share no words with the stored
text still work ("is there a place to swim" -> swimming pool entry). The model (~65 MB) is downloaded once on first
use in a background thread, so the app never blocks; until it is ready MIST uses its TF-IDF retriever.

Disable with LOCAL_EMBEDDINGS=0 (e.g. on a very small server)."""
import logging
import os
import threading
from pathlib import Path

import numpy as np

import config

log = logging.getLogger("mist.semantic")
MODEL_NAME = config.get("LOCAL_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
CACHE_DIR = str(Path(__file__).with_name("models"))
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_model = None
_state = {"status": "idle", "error": None}       # idle | loading | ready | failed | disabled
_lock = threading.Lock()


def enabled():
    """On by default on a normal machine. Off by default on Render's free tier (RENDER is set there): the model
    plus runtime needs well over 1 GB of RAM, which would crash a 512 MB instance. Opt in with LOCAL_EMBEDDINGS=1
    on a host that has the memory (and install requirements-semantic.txt)."""
    default = "0" if os.getenv("RENDER") else "1"
    return str(config.get("LOCAL_EMBEDDINGS", default)).lower() not in ("0", "false", "no", "off")


def _load():
    global _model
    with _lock:
        if _model is not None or _state["status"] in ("loading", "disabled"):
            return
        if not enabled():
            _state["status"] = "disabled"
            return
        _state["status"] = "loading"
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        from fastembed import TextEmbedding
        _model = TextEmbedding(MODEL_NAME, cache_dir=CACHE_DIR)
        list(_model.embed(["warm up"]))
        _state["status"] = "ready"
        log.info("local semantic model ready")
    except Exception as e:           # no internet, no memory, package missing ... -> TF-IDF fallback
        _state.update(status="failed", error=str(e)[:160])
        log.warning("local semantic model unavailable: %s", str(e)[:160])


def start_warmup():
    """Load the model in the background (idempotent)."""
    if _state["status"] == "idle":
        threading.Thread(target=_load, name="semantic-warmup", daemon=True).start()


def load_blocking():
    """For scripts and tests that need the model right now."""
    _load()
    return ready()


def ready():
    return _state["status"] == "ready" and _model is not None


def status():
    return dict(_state)


def _norm(m):
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return m / n


def encode_docs(texts):
    if not ready() or not texts:
        return None
    return _norm(np.array(list(_model.embed(list(texts))), dtype="float32"))


def encode_queries(texts):
    if not ready() or not texts:
        return None
    return _norm(np.array(list(_model.embed([QUERY_PREFIX + t for t in texts])), dtype="float32"))


def encode_plain(texts):
    """Symmetric (sentence-to-sentence) encoding, used for intent training and sentence selection."""
    if not ready() or not texts:
        return None
    return _norm(np.array(list(_model.embed(list(texts))), dtype="float32"))
