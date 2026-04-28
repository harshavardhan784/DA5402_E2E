"""
src/search_api.py  (OPTIMISED)
──────────────────────
FastAPI search service backed by two FAISS indexes:

    index_img.bin   — one image embedding per unique product
    index_txt.bin   — one averaged-text embedding per unique product
    meta.json       — one metadata record per unique product (same order)

Key optimisations vs original:
  1. CLIP model + FAISS indexes loaded EAGERLY at startup (lifespan event),
     not lazily on the first request — eliminates cold-start latency.
  2. Both FAISS indexes searched in one Python call (_search_both) which
     runs two index.search() calls back-to-back in the same thread;
     no extra HTTP round-trips.
  3. torch.inference_mode() instead of torch.no_grad() (slightly faster).
  4. Embeddings cast to float32 in-place, not via a copy.
  5. Workers: keep at 1 when GPU is shared; use --workers 2+ only on CPU.

Search endpoints — all return BOTH image-space and text-space results:

    POST /search/text        { "query": "...", "top_k": 10 }
    POST /search/image       { "image_b64": "<base64>", "top_k": 10 }
    POST /search/multimodal  { "query": "...", "image_b64": "...", "top_k": 10 }

Response shape:
    {
      "image_results": [ { "original_index", "image_url", "title",
                           "category_2", "category_3", "score" }, ... ],
      "text_results":  [ same shape, ranked by text-embedding similarity ]
    }

Management endpoints:
    GET  /health
    POST /reload   — hot-swap both FAISS indexes after DAG update

Start (CPU, assignment):
    uvicorn src.search_api:app --host 0.0.0.0 --port 8000 --workers 1
"""

from __future__ import annotations

import base64
import io
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator

log = logging.getLogger(__name__)


# ── Globals: model + index (populated at startup) ──────────────────────────────

_model      = None
_preprocess = None
_tokenizer  = None
_device     = None


def _load_clip():
    """Load CLIP once into globals. Called at startup."""
    global _model, _preprocess, _tokenizer, _device
    import torch
    import open_clip
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = os.environ.get("CLIP_MODEL",      "ViT-B-32")
    pretrained  = os.environ.get("CLIP_PRETRAINED", "openai")
    log.info("Loading CLIP model=%s pretrained=%s device=%s", model_name, pretrained, _device)
    _model, _, _preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained)
    _model = _model.to(_device).eval()
    _tokenizer = open_clip.get_tokenizer(model_name)
    log.info("CLIP model ready.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: load CLIP + FAISS before the first request arrives.
    This eliminates the cold-start penalty on the very first query.
    """
    log.info("[startup] Loading CLIP model …")
    _load_clip()

    log.info("[startup] Loading FAISS indexes …")
    from faiss_loader import get_img_index, get_txt_index, get_meta
    img = get_img_index()
    txt = get_txt_index()
    meta = get_meta()
    log.info("[startup] Ready — %d vectors (dim=%d)", img.ntotal, img.d)

    yield  # server runs here

    log.info("[shutdown] Bye.")


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(title="CLIP Product Search API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
Instrumentator().instrument(app).expose(app)


# ── Embedding helpers ──────────────────────────────────────────────────────────

def _embed_text(query: str) -> np.ndarray:
    """Encode a text query → unit-norm float32 (1, D)."""
    import torch
    with torch.inference_mode():
        tok = _tokenizer([query]).to(_device)
        emb = _model.encode_text(tok, normalize=True)
    return emb.cpu().numpy().astype("float32", copy=False)


def _embed_image(image_b64: str) -> np.ndarray:
    """Decode base64 image → unit-norm float32 (1, D)."""
    import torch
    from PIL import Image
    img_bytes = base64.b64decode(image_b64)
    img       = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    tensor    = _preprocess(img).unsqueeze(0).to(_device)
    with torch.inference_mode():
        emb = _model.encode_image(tensor, normalize=True)
    return emb.cpu().numpy().astype("float32", copy=False)


def _embed_multimodal(query: str, image_b64: str) -> np.ndarray:
    """Average text + image embeddings (whichever are present), renormalise."""
    vectors = []
    if query.strip():
        vectors.append(_embed_text(query))
    if image_b64.strip():
        vectors.append(_embed_image(image_b64))
    if not vectors:
        raise ValueError("At least one of query or image_b64 must be non-empty.")
    avg  = np.mean(np.concatenate(vectors, axis=0), axis=0, keepdims=True)
    norm = np.linalg.norm(avg)
    if norm > 1e-9:
        avg /= norm          # in-place — no extra allocation
    return avg.astype("float32", copy=False)


# ── FAISS search helpers ───────────────────────────────────────────────────────

def _faiss_search(index, meta: list, query_vec: np.ndarray, top_k: int) -> list:
    k = min(top_k, index.ntotal)
    scores, indices = index.search(query_vec, k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(meta):
            continue
        entry = meta[idx]
        results.append({
            "original_index": entry["original_index"],
            "image_url":      entry.get("image_url", ""),
            "title":          entry.get("title", ""),
            "category_2":     entry.get("category_2", ""),
            "category_3":     entry.get("category_3", ""),
            "score":          float(score),
        })
    return results


def _search_both(query_vec: np.ndarray, top_k: int) -> dict:
    """
    Search image-index AND text-index with the same vector.
    Both calls happen in a single Python function — no extra HTTP hops.
    """
    from faiss_loader import get_img_index, get_txt_index, get_meta
    meta      = get_meta()
    img_index = get_img_index()
    txt_index = get_txt_index()
    return {
        "image_results": _faiss_search(img_index, meta, query_vec, top_k),
        "text_results":  _faiss_search(txt_index, meta, query_vec, top_k),
    }


# ── Request / response models ──────────────────────────────────────────────────

class TextSearchRequest(BaseModel):
    query: str
    top_k: int = 10

class ImageSearchRequest(BaseModel):
    image_b64: str
    top_k: int = 10

class MultimodalSearchRequest(BaseModel):
    query:     Optional[str] = ""
    image_b64: Optional[str] = ""
    top_k:     int = 10


# ── Search endpoints ───────────────────────────────────────────────────────────

@app.post("/search/text")
def search_text(req: TextSearchRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")
    try:
        return _search_both(_embed_text(req.query), req.top_k)
    except Exception as e:
        log.exception("search/text failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/image")
def search_image(req: ImageSearchRequest):
    if not req.image_b64.strip():
        raise HTTPException(status_code=400, detail="image_b64 must not be empty")
    try:
        return _search_both(_embed_image(req.image_b64), req.top_k)
    except Exception as e:
        log.exception("search/image failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/multimodal")
def search_multimodal(req: MultimodalSearchRequest):
    if not (req.query or "").strip() and not (req.image_b64 or "").strip():
        raise HTTPException(status_code=400, detail="Provide query and/or image_b64")
    try:
        return _search_both(_embed_multimodal(req.query or "", req.image_b64 or ""), req.top_k)
    except Exception as e:
        log.exception("search/multimodal failed")
        raise HTTPException(status_code=500, detail=str(e))


# ── Management endpoints ───────────────────────────────────────────────────────

@app.get("/health")
def health():
    try:
        from faiss_loader import get_img_index, get_txt_index, get_meta
        img_index = get_img_index()
        txt_index = get_txt_index()
        meta      = get_meta()
        return {
            "status":       "ok",
            "img_vectors":  img_index.ntotal,
            "txt_vectors":  txt_index.ntotal,
            "meta_entries": len(meta),
            "dim":          img_index.d,
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@app.post("/reload")
def reload():
    """Hot-swap both FAISS indexes from disk without restarting the server."""
    try:
        from faiss_loader import reload_index
        n = reload_index()
        return {"status": "ok", "vector_count": n}
    except Exception as e:
        log.exception("reload failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ready")
def ready():
    # If this returns, the lifespan (startup) has completed
    return {"status": "ready"}