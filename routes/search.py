"""
routes/search.py
─────────────────
All /search/* and /reload endpoints.

Every endpoint hits the MLflow model server (mlflow models serve) via
_call_mlflow(), which sends a dataframe_records payload to /invocations.

Query modes supported:
  POST /search/text        — text query  → image results + text results
  POST /search/image       — image query → image results + text results
  POST /search/multimodal  — both        → image results + text results

Each response carries TWO ranked lists:
  image_results : top_k items ranked by image-embedding cosine similarity
  text_results  : top_k items ranked by text-embedding cosine similarity

POST /reload — hot-swap FAISS index after DAG run (calls faiss_loader.reload_index)
"""

from __future__ import annotations

import os
from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(tags=["search"])

MLFLOW_SERVE_URL = os.environ.get("MLFLOW_SERVE_URL", "http://localhost:5001")
INVOCATIONS_URL  = f"{MLFLOW_SERVE_URL}/invocations"


# ── Request / Response schemas ─────────────────────────────────────────────────

class TextSearchRequest(BaseModel):
    query: str
    top_k: int = 10

class ImageSearchRequest(BaseModel):
    image_b64: str
    top_k: int = 10

class MultimodalSearchRequest(BaseModel):
    query:     str = ""
    image_b64: str = ""
    top_k:     int = 10

class SearchResult(BaseModel):
    original_index: int
    image_url:      str
    augmented_text: str
    category_2:     str
    category_3:     str
    score:          float

class DualSearchResponse(BaseModel):
    """Both image-similarity and text-similarity result lists."""
    image_results: list[SearchResult]
    text_results:  list[SearchResult]


# ── Helper: call MLflow /invocations ──────────────────────────────────────────

async def _call_mlflow(
    query_text:  str,
    query_image: str,
    top_k:       int,
    search_mode: str,   # "image" | "text" | "both"
) -> list[dict]:
    """
    Send one row to the MLflow pyfunc /invocations endpoint.
    search_mode is passed so the wrapper can rank results differently
    for image-similarity vs text-similarity.
    """
    payload = {
        "dataframe_records": [
            {
                "query_text":  query_text,
                "query_image": query_image,
                "top_k":       top_k,
                "search_mode": search_mode,
            }
        ]
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(INVOCATIONS_URL, json=payload)

    if resp.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"MLflow server error {resp.status_code}: {resp.text[:400]}",
        )

    data        = resp.json()
    predictions = data.get("predictions", data)

    # Handle both MLflow 2.x and 3.x response shapes
    if isinstance(predictions, list):
        return predictions
    if isinstance(predictions, dict):
        if "dataframe_records" in predictions:
            return predictions["dataframe_records"]
        if "data" in predictions and "columns" in predictions:
            cols = predictions["columns"]
            return [dict(zip(cols, row)) for row in predictions["data"]]
    return []


async def _dual_search(query_text: str, query_image: str, top_k: int) -> DualSearchResponse:
    """
    Issue two parallel calls to MLflow:
      - search_mode="image" → ranks by image-embedding similarity
      - search_mode="text"  → ranks by text-embedding similarity
    """
    import asyncio

    img_task  = asyncio.create_task(
        _call_mlflow(query_text, query_image, top_k, search_mode="image"))
    text_task = asyncio.create_task(
        _call_mlflow(query_text, query_image, top_k, search_mode="text"))

    image_results_raw, text_results_raw = await asyncio.gather(img_task, text_task)
    return DualSearchResponse(
        image_results=[SearchResult(**r) for r in image_results_raw],
        text_results =[SearchResult(**r) for r in text_results_raw],
    )


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post("/reload")
async def reload_index():
    """
    Trigger a FAISS index hot-swap after the DAG writes a new index on disk.
    The Airflow DAG calls this endpoint via the reload_faiss_index task.
    """
    try:
        from faiss_loader import reload_index as _reload  # noqa: PLC0415
        n = _reload()
        return {"status": "reloaded", "vector_count": n}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search/text", response_model=DualSearchResponse)
async def search_text(req: TextSearchRequest):
    """
    Text query → image results + text results.
    Both ranked lists are returned; the client can display either or both.
    """
    return await _dual_search(
        query_text  = req.query,
        query_image = "",
        top_k       = req.top_k,
    )


@router.post("/search/image", response_model=DualSearchResponse)
async def search_image(req: ImageSearchRequest):
    """
    Image query (base64) → image results + text results.
    Both ranked lists are returned.
    """
    return await _dual_search(
        query_text  = "",
        query_image = req.image_b64,
        top_k       = req.top_k,
    )


@router.post("/search/multimodal", response_model=DualSearchResponse)
async def search_multimodal(req: MultimodalSearchRequest):
    """
    Text + image query → image results + text results.
    Query vectors are averaged before FAISS search (handled in clip_mlflow_wrapper).
    Both ranked lists are returned.
    """
    if not req.query and not req.image_b64:
        raise HTTPException(status_code=400, detail="Provide query text or image or both.")
    return await _dual_search(
        query_text  = req.query,
        query_image = req.image_b64,
        top_k       = req.top_k,
    )