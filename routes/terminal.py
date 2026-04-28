"""
routes/terminal.py
───────────────────
Lightweight operational endpoints for inspecting live system state.

GET  /terminal/faiss-info      — index size, dimension, path
GET  /terminal/env             — non-secret env vars for debugging
GET  /terminal/logs?tail=N     — last N lines of uvicorn log (if log file exists)
POST /terminal/dvc-status      — run `dvc status` in ROOT_DIR, return stdout
"""

import logging
import os
import subprocess
from pathlib import Path

from fastapi import APIRouter, Query

router = APIRouter(prefix="/terminal", tags=["terminal"])
log    = logging.getLogger(__name__)

ROOT_DIR = Path(os.environ.get("ROOT_DIR", "/opt"))
DATA_DIR = Path(os.environ.get("DATA_DIR", "/opt/data"))
LOG_FILE = Path(os.environ.get("UVICORN_LOG", "/tmp/uvicorn.log"))


@router.get("/faiss-info")
async def faiss_info():
    """Return current FAISS index metadata without triggering a reload."""
    try:
        from faiss_loader import get_img_index, get_txt_index, get_meta  # noqa: PLC0415
        img_index = get_img_index()
        txt_index = get_txt_index()
        meta      = get_meta()
        return {
            "img_ntotal":     img_index.ntotal,
            "txt_ntotal":     txt_index.ntotal,
            "dimension":      img_index.d,
            "img_index_path": str(DATA_DIR / "faiss" / "index_img.bin"),
            "txt_index_path": str(DATA_DIR / "faiss" / "index_txt.bin"),
            "meta_count":     len(meta),
            "sample":         meta[:3] if meta else [],
        }
    except FileNotFoundError as e:
        return {"error": str(e), "img_ntotal": 0, "txt_ntotal": 0}
    except Exception as e:
        return {"error": str(e)}


@router.get("/env")
async def env_info():
    """Expose non-secret environment variables for debugging."""
    safe_keys = {
        "DATA_DIR", "SRC_DIR", "ROOT_DIR", "META_DIR",
        "MLFLOW_SERVE_URL", "MLFLOW_TRACKING_URI", "MLFLOW_EXPERIMENT",
        "AIRFLOW_API_URL", "SEARCH_API_URL",
        "PYTHONPATH", "HOSTNAME",
    }
    return {k: os.environ.get(k, "") for k in sorted(safe_keys)}


@router.get("/logs")
async def tail_logs(tail: int = Query(default=50, le=500)):
    """Return the last `tail` lines of the uvicorn log file."""
    if not LOG_FILE.exists():
        return {"detail": f"Log file not found at {LOG_FILE}"}
    lines = LOG_FILE.read_text(errors="replace").splitlines()
    return {"lines": lines[-tail:], "total_lines": len(lines)}


@router.post("/dvc-status")
async def dvc_status():
    """Run `dvc status` in ROOT_DIR and return the output."""
    result = subprocess.run(
        ["dvc", "status"],
        capture_output=True, text=True, cwd=str(ROOT_DIR),
    )
    return {
        "stdout":      result.stdout[-3000:],
        "stderr":      result.stderr[-1000:],
        "returncode":  result.returncode,
    }


@router.post("/dvc-push")
async def dvc_push():
    """Run `dvc push` to sync DVC-tracked data (including image_cache) to remote."""
    result = subprocess.run(
        ["dvc", "push"],
        capture_output=True, text=True, cwd=str(ROOT_DIR),
    )
    return {
        "stdout":     result.stdout[-3000:],
        "stderr":     result.stderr[-1000:],
        "returncode": result.returncode,
    }