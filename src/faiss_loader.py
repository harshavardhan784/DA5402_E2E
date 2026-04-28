"""
src/faiss_loader.py
────────────────────
Thread-safe FAISS index (image + text) + metadata singleton.

Two separate FAISS indexes, one shared meta list, all aligned to the same
N_unique products in the same order:

    index_img.bin   — one image embedding per unique original_index
    index_txt.bin   — one averaged text embedding per unique original_index
    meta.json       — one metadata dict per unique original_index

Loaded once at search-api startup. Hot-swapped atomically via reload_index()
which is called from POST /reload after each DAG run updates the files on disk.

Usage:
    from faiss_loader import get_img_index, get_txt_index, get_meta, reload_index

    img_index = get_img_index()
    txt_index = get_txt_index()
    meta      = get_meta()

    n = reload_index()   # hot-swap after DAG writes new index files
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# ── Paths (overridable via env) ────────────────────────────────────────────────

def _faiss_dir() -> Path:
    return Path(os.environ.get("DATA_DIR", "/opt/data")) / "faiss"

def _img_index_path() -> Path:
    return _faiss_dir() / "index_img.bin"

def _txt_index_path() -> Path:
    return _faiss_dir() / "index_txt.bin"

def _meta_path() -> Path:
    return _faiss_dir() / "meta.json"


# ── Internal singletons ────────────────────────────────────────────────────────

_lock:      threading.Lock   = threading.Lock()
_img_index: Optional[object] = None   # faiss.Index over image embeddings
_txt_index: Optional[object] = None   # faiss.Index over averaged text embeddings
_meta:      Optional[list]   = None   # list of dicts, one per unique product


# ── Public API ─────────────────────────────────────────────────────────────────

def get_img_index():
    """Return the image FAISS index, loading on first call."""
    global _img_index, _txt_index, _meta
    if _img_index is None:
        with _lock:
            if _img_index is None:
                _img_index, _txt_index, _meta = _load()
    return _img_index


def get_txt_index():
    """Return the text FAISS index, loading on first call."""
    global _img_index, _txt_index, _meta
    if _txt_index is None:
        with _lock:
            if _txt_index is None:
                _img_index, _txt_index, _meta = _load()
    return _txt_index


def get_meta() -> list:
    """Return the metadata list, loading on first call."""
    global _img_index, _txt_index, _meta
    if _meta is None:
        with _lock:
            if _meta is None:
                _img_index, _txt_index, _meta = _load()
    return _meta


def reload_index() -> int:
    """
    Hot-swap both FAISS indexes and metadata from disk.
    Called by POST /reload after the DAG writes new index files.
    Returns the new vector count (same for both indexes).
    """
    global _img_index, _txt_index, _meta
    new_img, new_txt, new_meta = _load()
    with _lock:
        _img_index = new_img
        _txt_index = new_txt
        _meta      = new_meta
    log.info(
        "[faiss_loader] Reloaded — img_index: %d vectors  txt_index: %d vectors  meta: %d entries",
        new_img.ntotal, new_txt.ntotal, len(new_meta),
    )
    return new_img.ntotal


# ── Internal loader ────────────────────────────────────────────────────────────

def _load():
    """
    Load index_img.bin + index_txt.bin + meta.json from disk.
    Raises FileNotFoundError if any file is missing.
    Raises RuntimeError if counts are inconsistent.
    """
    import faiss  # lazy import — module usable before faiss is installed

    img_path  = _img_index_path()
    txt_path  = _txt_index_path()
    meta_path = _meta_path()

    for p in (img_path, txt_path, meta_path):
        if not p.exists():
            raise FileNotFoundError(
                f"FAISS file not found: {p}. "
                "Run the Airflow pipeline at least once to build the indexes."
            )

    log.info("[faiss_loader] Loading image index from %s", img_path)
    img_index = faiss.read_index(str(img_path))

    log.info("[faiss_loader] Loading text index from %s", txt_path)
    txt_index = faiss.read_index(str(txt_path))

    log.info("[faiss_loader] Loading meta from %s", meta_path)
    with open(meta_path) as f:
        meta = json.load(f)

    # All three must be perfectly aligned
    if img_index.ntotal != txt_index.ntotal:
        raise RuntimeError(
            f"[faiss_loader] img_index.ntotal={img_index.ntotal} != "
            f"txt_index.ntotal={txt_index.ntotal}. Re-run the pipeline."
        )
    if img_index.ntotal != len(meta):
        raise RuntimeError(
            f"[faiss_loader] index.ntotal={img_index.ntotal} != len(meta)={len(meta)}. "
            "The indexes and meta.json are out of sync — re-run the pipeline."
        )

    log.info(
        "[faiss_loader] Loaded: %d unique products  dim=%d",
        img_index.ntotal, img_index.d,
    )
    return img_index, txt_index, meta