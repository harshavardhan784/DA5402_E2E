"""
routes/pipeline.py
───────────────────
POST /pipeline/trigger  — manually trigger the Airflow DAG via its REST API.
GET  /pipeline/status   — get the latest DAG run status.
"""

import os

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/pipeline", tags=["pipeline"])

AIRFLOW_API_URL = os.environ.get("AIRFLOW_API_URL",    "http://airflow-webserver:8080/api/v2")
AIRFLOW_USER    = os.environ.get("AIRFLOW_API_USER",   "admin")
AIRFLOW_PASS    = os.environ.get("AIRFLOW_API_PASS",   "admin")
DAG_ID          = "clip_product_retrieval_pipeline"


def _auth() -> tuple[str, str]:
    return (AIRFLOW_USER, AIRFLOW_PASS)


class TriggerRequest(BaseModel):
    conf: dict = {}


@router.post("/trigger")
async def trigger_pipeline(req: TriggerRequest = TriggerRequest()):
    """Manually trigger the clip_product_retrieval_pipeline DAG."""
    url = f"{AIRFLOW_API_URL}/dags/{DAG_ID}/dagRuns"
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.post(url, json={"conf": req.conf}, auth=_auth())
    if r.status_code not in (200, 201):
        raise HTTPException(
            status_code=502,
            detail=f"Airflow error {r.status_code}: {r.text[:300]}",
        )
    return r.json()


@router.get("/status")
async def pipeline_status():
    """Return the most recent DAG run record."""
    url = f"{AIRFLOW_API_URL}/dags/{DAG_ID}/dagRuns?limit=1&order_by=-start_date"
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(url, auth=_auth())
    if r.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Airflow error {r.status_code}: {r.text[:300]}",
        )
    runs = r.json().get("dag_runs", [])
    return runs[0] if runs else {"detail": "No DAG runs found"}