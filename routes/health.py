"""
routes/health.py
─────────────────
GET /health  — liveness probe for the search-api container.
Also reports whether the MLflow serve endpoint is reachable.
"""

import os

import httpx
from fastapi import APIRouter

router = APIRouter(tags=["health"])

MLFLOW_SERVE_URL = os.environ.get("MLFLOW_SERVE_URL", "http://localhost:5001")


@router.get("/health")
async def health():
    """
    Returns:
        status         : "ok"
        mlflow_url     : configured MLflow serve URL
        mlflow_healthy : True if /ping responds 200, else False
        faiss_loaded   : True if the FAISS index singleton is initialised
    """
    # Check MLflow serve
    mlflow_healthy = False
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(f"{MLFLOW_SERVE_URL}/ping")
            mlflow_healthy = r.status_code == 200
    except Exception:
        pass

    # Check FAISS singleton (non-blocking — don't force a load here)
    try:
        from faiss_loader import _img_index, _txt_index  # noqa: PLC0415
        faiss_loaded = _img_index is not None and _txt_index is not None
    except Exception:
        faiss_loaded = False

    return {
        "status":         "ok",
        "mlflow_url":     MLFLOW_SERVE_URL,
        "mlflow_healthy": mlflow_healthy,
        "faiss_loaded":   faiss_loaded,
    }

@router.get("/ready")
async def ready():
    """
    Readiness probe — returns 503 until FAISS index is loaded.
    Kubernetes/Docker will not route traffic until this returns 200.
    """
    try:
        from faiss_loader import get_img_index, get_txt_index, get_meta
        idx = get_img_index()
        _ = get_txt_index()  # ensure txt index is also loaded
        meta = get_meta()
        if idx.ntotal == 0 or not meta:
            from fastapi import Response
            return Response(
                content='{"status":"not_ready","reason":"empty index"}',
                status_code=503,
                media_type="application/json"
            )
        return {"status": "ready", "vectors": idx.ntotal}
    except Exception as e:
        from fastapi import Response
        return Response(
            content=f'{{"status":"not_ready","reason":"{str(e)}"}}',
            status_code=503,
            media_type="application/json"
        )