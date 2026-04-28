"""
verify_search.py
────────────────────────────────────────────────────────────────────────────────
Standalone Flask verification UI for the CLIP search API.

Each query mode (text / image / multimodal) shows two result grids
side-by-side:

    ┌─────────────────────┬─────────────────────┐
    │  Image-space hits   │  Text-space hits     │
    │  (searched img idx) │  (searched txt idx)  │
    └─────────────────────┴─────────────────────┘

Run:
    pip install flask requests Pillow
    python verify_search.py

    SEARCH_API_URL=http://localhost:8000 python verify_search.py

Then open: http://localhost:5050
────────────────────────────────────────────────────────────────────────────────
"""

import base64
import io
import json
import os
import time

import requests
# from flask import Flask, jsonify, render_template_string, request
from flask import Flask, jsonify, render_template, request
app = Flask(__name__)

SEARCH_API_URL  = os.environ.get("SEARCH_API_URL", "http://localhost:8000").rstrip("/")
REQUEST_TIMEOUT = 30


# ── Internal helpers ───────────────────────────────────────────────────────────

def _post(endpoint: str, payload: dict) -> tuple[dict, float]:
    url = f"{SEARCH_API_URL}{endpoint}"
    t0  = time.perf_counter()
    resp = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    resp.raise_for_status()
    return resp.json(), round(elapsed_ms, 1)


def _health() -> tuple[dict, float]:
    url = f"{SEARCH_API_URL}/health"
    t0  = time.perf_counter()
    resp = requests.get(url, timeout=10)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    resp.raise_for_status()
    return resp.json(), round(elapsed_ms, 1)


def _image_to_b64(file_storage) -> str:
    return base64.b64encode(file_storage.read()).decode("utf-8")


# ── API proxy routes ───────────────────────────────────────────────────────────

@app.route("/api/health")
def api_health():
    try:
        data, ms = _health()
        return jsonify({"ok": True, "data": data, "elapsed_ms": ms})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 502


@app.route("/api/search/text", methods=["POST"])
def api_text_search():
    body  = request.get_json(force=True)
    query = body.get("query", "").strip()
    top_k = int(body.get("top_k", 5))
    if not query:
        return jsonify({"ok": False, "error": "query is required"}), 400
    try:
        data, ms = _post("/search/text", {"query": query, "top_k": top_k})
        return jsonify({"ok": True, "data": data, "elapsed_ms": ms})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 502


@app.route("/api/search/image", methods=["POST"])
def api_image_search():
    top_k = int(request.form.get("top_k", 5))
    if "image" not in request.files or request.files["image"].filename == "":
        return jsonify({"ok": False, "error": "No image file provided"}), 400
    b64 = _image_to_b64(request.files["image"])
    try:
        data, ms = _post("/search/image", {"image_b64": b64, "top_k": top_k})
        return jsonify({"ok": True, "data": data, "elapsed_ms": ms})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 502


@app.route("/api/search/multimodal", methods=["POST"])
def api_multimodal_search():
    top_k = int(request.form.get("top_k", 5))
    query = request.form.get("query", "").strip()
    b64   = ""
    if "image" in request.files and request.files["image"].filename != "":
        b64 = _image_to_b64(request.files["image"])
    if not query and not b64:
        return jsonify({"ok": False, "error": "Provide text and/or image"}), 400
    try:
        data, ms = _post("/search/multimodal",
                         {"query": query, "image_b64": b64, "top_k": top_k})
        return jsonify({"ok": True, "data": data, "elapsed_ms": ms})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 502


@app.route("/api/reload", methods=["POST"])
def api_reload():
    try:
        resp = requests.post(f"{SEARCH_API_URL}/reload", timeout=60)
        resp.raise_for_status()
        return jsonify({"ok": True, "data": resp.json()})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 502


@app.route("/")
def index():
    return render_template("index.html", api_url=SEARCH_API_URL)

if __name__ == "__main__":
    port = int(os.environ.get("VERIFY_PORT", 5050))
    print(f"\n  CLIP Search Verifier")
    print(f"  ─────────────────────")
    print(f"  Search API : {SEARCH_API_URL}")
    print(f"  Open       : http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=True)