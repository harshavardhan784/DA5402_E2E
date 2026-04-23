"""
dags/clip_pipeline_dag.py
──────────────────────────
Weekly CLIP product-retrieval retraining pipeline.

Pipeline stages
───────────────
 1. detect_new_data          Check that incoming CSV exists and has new rows.
 2. dvc_pull_latest          Pull latest data version tracked by DVC.
 3. augment_texts            Run augment_texts.py → data/processed/week{N}.csv
 4. build_replay_buffer      Combine new + old data with hard negatives.
 5. branch_on_baseline       BranchPythonOperator: Day-0 (no baseline) OR normal path.

 ── Day-0 path (no last_run_id.txt) ──────────────────────────────────────────
 6a. initial_train           Fine-tune on augmented new data directly (no replay).
                             Saves new run_id to data/meta/last_run_id.txt.

 ── Normal path (baseline exists) ────────────────────────────────────────────
 6b. evaluate_current_model  Zero-shot / last run baseline recall on new data.
 6c. drift_check             compute_drift_score() → XCom {drift_detected, score}
 6d. branch_on_drift         BranchPythonOperator: retrain OR skip.
 6d-i.  retrain_model        clip_experiments.py --mode linear_probe on replay buffer.
 6d-ii. skip_training        No-op log.

 ── Shared tail ──────────────────────────────────────────────────────────────
 7. update_faiss_index       Regenerate embeddings → rebuild FAISS index.
 8. dvc_commit_new_data      DVC add/push + git commit.
 9. pipeline_complete        Terminal no-op.

XCom keys used
──────────────
  baseline_run_id   : MLflow run ID of the last training run (used by drift + replay)
  new_csv_path      : path to augmented new-week CSV
  replay_csv_path   : path to replay-buffer CSV (Week N only; initial_train uses new_csv_path)
  drift_score       : float score from drift_check
  drift_detected    : bool from drift_check
  new_run_id        : MLflow run ID produced by initial_train OR retrain_model

Environment variables (set in docker-compose)
──────────────────────────────────────────────
  MLFLOW_TRACKING_URI   http://mlflow:5000
  MLFLOW_EXPERIMENT     clip_product_retrieval
  DATA_DIR              /opt/airflow/data
  SRC_DIR               /opt/airflow/src
"""

from __future__ import annotations

import glob
import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.providers.standard.operators.python import (
    PythonOperator,
    BranchPythonOperator,
)
from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.task.trigger_rule import TriggerRule

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Environment helpers (resolved at task-runtime, not import time)
# ─────────────────────────────────────────────────────────────────────────────

def _data_dir() -> Path:
    return Path(os.environ.get("DATA_DIR", "/opt/airflow/data"))

def _src_dir() -> Path:
    return Path(os.environ.get("SRC_DIR", "/opt/airflow/src"))

def _tracking_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")

def _experiment() -> str:
    return os.environ.get("MLFLOW_EXPERIMENT", "clip_product_retrieval")


# ─────────────────────────────────────────────────────────────────────────────
# DAG default args
# ─────────────────────────────────────────────────────────────────────────────

default_args = {
    "owner":            "mlops",
    "depends_on_past":  False,
    "retries":          1,                      # testing: 1 | production: 3
    "retry_delay":      timedelta(seconds=30),
    "email_on_failure": False,
}

# ─────────────────────────────────────────────────────────────────────────────
# TASK FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

# ── 1. Detect new data ────────────────────────────────────────────────────────

def _state_file() -> Path:
    return _data_dir() / "meta" / ".processed_csvs.json"

def _load_state() -> dict:
    sf = _state_file()
    if sf.exists():
        try:
            return json.loads(sf.read_text())
        except Exception:
            return {}
    return {}

def _save_state(state: dict):
    sf = _state_file()
    sf.parent.mkdir(parents=True, exist_ok=True)
    sf.write_text(json.dumps(state, indent=2))

def _file_sig(path: str) -> dict:
    st = os.stat(path)
    return {"mtime": st.st_mtime, "size": st.st_size}

def _find_unprocessed_csv() -> str | None:
    incoming_dir = str(_data_dir() / "incoming")
    state = _load_state()
    log.info("[detect] Scanning: %s", incoming_dir)
    all_csvs = sorted(glob.glob(os.path.join(incoming_dir, "*.csv")))
    log.info("[detect] All CSVs found: %s", all_csvs)
    for csv_path in all_csvs:
        sig = _file_sig(csv_path)
        prev = state.get(csv_path)
        log.info("[detect] %s  sig=%s  prev=%s", csv_path, sig, prev)
        if prev is None or prev["mtime"] != sig["mtime"] or prev["size"] != sig["size"]:
            log.info("[detect] → UNPROCESSED: %s", csv_path)
            return csv_path
    log.info("[detect] No unprocessed CSVs found.")
    return None

def _mark_csv_processed(csv_path: str):
    state = _load_state()
    state[csv_path] = _file_sig(csv_path)
    _save_state(state)
    log.info("[detect] Marked as processed: %s", csv_path)


def detect_new_data(**ctx):
    log.info("[detect] DATA_DIR env   = %s", os.environ.get("DATA_DIR", "NOT SET"))
    log.info("[detect] _data_dir()    = %s", _data_dir())
    log.info("[detect] cwd            = %s", os.getcwd())

    incoming_dir = _data_dir() / "incoming"
    incoming_dir.mkdir(parents=True, exist_ok=True)

    data_root = _data_dir()
    log.info("[detect] Contents of %s:", data_root)
    for p in sorted(data_root.rglob("*")):
        log.info("[detect]   %s", p)

    csv_path_str = _find_unprocessed_csv()

    if csv_path_str is None:
        existing = list(incoming_dir.glob("*.csv"))
        if existing:
            raise ValueError(
                f"CSVs exist in {incoming_dir} but all are already processed: "
                f"{[str(p) for p in existing]}. "
                "To reprocess, delete data/meta/.processed_csvs.json and re-trigger."
            )
        raise FileNotFoundError(
            f"No CSV found in {incoming_dir}. "
            "Copy your CSV there, then trigger the DAG manually."
        )

    csv_path = Path(csv_path_str)
    import pandas as pd
    df = pd.read_csv(csv_path, nrows=1000)
    n = len(df)
    log.info("[detect] CSV columns: %s", list(df.columns))
    log.info("[detect] CSV shape (sample): %d rows", n)

    if n == 0:
        raise ValueError(f"{csv_path} is empty.")

    week_label = csv_path.stem
    log.info("[detect] week_label=%s  incoming_csv=%s  rows=%d", week_label, csv_path, n)

    ctx["ti"].xcom_push(key="incoming_csv",         value=str(csv_path))
    ctx["ti"].xcom_push(key="week_label",            value=week_label)
    ctx["ti"].xcom_push(key="n_new_rows",            value=n)
    ctx["ti"].xcom_push(key="incoming_csv_to_mark",  value=str(csv_path))


# ── 2. DVC pull ───────────────────────────────────────────────────────────────

def dvc_pull(**ctx):
    import subprocess
    result = subprocess.run(
        ["dvc", "pull", "--force"],
        capture_output=True, text=True,
        cwd="/opt/airflow",
    )
    log.info("dvc pull stdout: %s", result.stdout[-500:])
    if result.returncode != 0:
        log.warning("dvc pull failed (may be first run): %s", result.stderr[-300:])


# ── 3. Augment texts ──────────────────────────────────────────────────────────

def augment_texts(**ctx):
    import sys
    sys.path.insert(0, str(_src_dir()))
    from augment_texts import build_augmented_dataset

    ti         = ctx["ti"]
    incoming   = ti.xcom_pull(key="incoming_csv", task_ids="detect_new_data")
    week_label = ti.xcom_pull(key="week_label",   task_ids="detect_new_data")

    out_path = _data_dir() / "processed" / f"{week_label}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("Augmenting %s → %s", incoming, out_path)
    df_aug = build_augmented_dataset(
        csv_path  = incoming,
        n_samples = 4000,
        seed      = 42,
    )
    df_aug.to_csv(out_path, index=False)
    log.info("Augmented dataset: %d rows saved to %s", len(df_aug), out_path)

    ti.xcom_push(key="new_csv_path", value=str(out_path))


# ── 4. Build replay buffer ────────────────────────────────────────────────────

def build_replay_buffer(**ctx):
    import sys
    import pandas as pd
    sys.path.insert(0, str(_src_dir()))
    from replay_buffer import (
        build_replay_buffer as _build,
        load_baseline_from_mlflow,
    )

    ti         = ctx["ti"]
    new_csv    = ti.xcom_pull(key="new_csv_path", task_ids="augment_texts")
    week_label = ti.xcom_pull(key="week_label",   task_ids="detect_new_data")

    meta_dir    = _data_dir() / "meta"
    run_id_file = meta_dir / "last_run_id.txt"

    replay_path = _data_dir() / "processed" / f"{week_label}_replay.csv"
    replay_path.parent.mkdir(parents=True, exist_ok=True)

    new_df = pd.read_csv(new_csv)

    if not run_id_file.exists():
        # Day 0 — no previous run; downstream branch will route to initial_train.
        # Still write replay_path so XCom key exists and update_faiss_index is happy,
        # but initial_train intentionally reads new_csv_path instead.
        log.info("No previous run_id found — using new data only (Day 0).")
        new_df.to_csv(replay_path, index=False)
        ti.xcom_push(key="replay_csv_path", value=str(replay_path))
        ti.xcom_push(key="baseline_run_id", value=None)
        return

    baseline_run_id = run_id_file.read_text().strip()
    log.info("Building replay buffer against baseline run_id=%s", baseline_run_id)

    baseline_data = load_baseline_from_mlflow(baseline_run_id, _tracking_uri())

    old_csvs = [p for p in (_data_dir() / "processed").glob("*.csv")
                if "_replay" not in p.name and p.name != f"{week_label}.csv"]
    old_csv = str(old_csvs[-1]) if old_csvs else None

    if old_csv:
        combined = _build(
            old_csv_path  = old_csv,
            new_df        = new_df,
            baseline_data = baseline_data,
            replay_ratio  = 0.20,
        )
    else:
        log.warning("No old CSV found — skipping replay.")
        combined = new_df

    combined.to_csv(replay_path, index=False)
    log.info("Replay buffer: %d rows saved to %s", len(combined), replay_path)

    ti.xcom_push(key="replay_csv_path", value=str(replay_path))
    ti.xcom_push(key="baseline_run_id", value=baseline_run_id)


# ── 5. Branch on baseline ─────────────────────────────────────────────────────
# This is the new gate. Runs immediately after build_replay_buffer.
# No baseline → initial_train (skip eval + drift entirely).
# Baseline exists → normal evaluate → drift → branch_on_drift path.

def branch_on_baseline(**ctx):
    run_id_file = _data_dir() / "meta" / "last_run_id.txt"
    if run_id_file.exists():
        log.info("Baseline found — taking normal eval/drift path.")
        return "evaluate_current_model"
    log.info("No baseline — routing to initial_train.")
    return "initial_train"


# ── 6a. Initial train (Day 0) ─────────────────────────────────────────────────
# Trains directly on the augmented new-data CSV (new_csv_path from augment_texts).
# Does NOT use the replay buffer — there is nothing old to replay.
# Identical hyperparameters to retrain_model; only the MLflow tag differs.

def initial_train(**ctx):
    import sys
    sys.path.insert(0, str(_src_dir()))
    from clip_experiments import Config, run_linear_probe
    import mlflow

    ti         = ctx["ti"]
    new_csv    = ti.xcom_pull(key="new_csv_path", task_ids="augment_texts")
    week_label = ti.xcom_pull(key="week_label",   task_ids="detect_new_data")

    mlflow.set_tracking_uri(_tracking_uri())
    mlflow.set_experiment(_experiment())

    cfg = Config(
        csv_path            = new_csv,
        mode                = "linear_probe",
        epochs              = 2,
        batch_size          = 32,
        lr                  = 5e-6,
        week_label          = week_label,
        mlflow_tracking_uri = _tracking_uri(),
        mlflow_experiment   = _experiment(),
        mlflow_run_name     = f"initial_train_{week_label}",
    )

    with mlflow.start_run(run_name=cfg.mlflow_run_name) as run:
        new_run_id = run.info.run_id
        mlflow.set_tag("stage",      "initial_train")
        mlflow.set_tag("week_label", week_label)
        final_metrics = run_linear_probe(cfg, device="cpu", run_id=new_run_id)

    log.info("Initial train complete. run_id=%s  metrics=%s", new_run_id, final_metrics)

    # Persist so next week knows its baseline
    meta_dir = _data_dir() / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "last_run_id.txt").write_text(new_run_id)

    ti.xcom_push(key="new_run_id", value=new_run_id)


# ── 6b. Evaluate current model ────────────────────────────────────────────────
# Only reached when a baseline exists (normal weekly path).

def evaluate_current_model(**ctx):
    import sys
    sys.path.insert(0, str(_src_dir()))
    from clip_experiments import Config, run_zero_shot
    import mlflow

    ti              = ctx["ti"]
    new_csv         = ti.xcom_pull(key="new_csv_path",    task_ids="augment_texts")
    week_label      = ti.xcom_pull(key="week_label",      task_ids="detect_new_data")
    baseline_run_id = ti.xcom_pull(key="baseline_run_id", task_ids="build_replay_buffer")

    mlflow.set_tracking_uri(_tracking_uri())
    mlflow.set_experiment(_experiment())

    cfg = Config(
        csv_path            = new_csv,
        mode                = "zero_shot",
        week_label          = week_label,
        mlflow_tracking_uri = _tracking_uri(),
        mlflow_experiment   = _experiment(),
        mlflow_run_name     = f"eval_pre_retrain_{week_label}",
    )

    with mlflow.start_run(run_name=cfg.mlflow_run_name) as run:
        eval_run_id = run.info.run_id
        mlflow.set_tag("stage",           "pre_retrain_eval")
        mlflow.set_tag("baseline_run_id", str(baseline_run_id))
        final_metrics = run_zero_shot(cfg, device="cpu", run_id=eval_run_id)

    log.info("Pre-retrain eval run_id=%s  metrics=%s", eval_run_id, final_metrics)
    ti.xcom_push(key="eval_run_id", value=eval_run_id)


# ── 6c. Drift check ───────────────────────────────────────────────────────────

def drift_check(**ctx):
    import sys
    sys.path.insert(0, str(_src_dir()))
    from drift_detection import (
        load_baseline_from_mlflow,
        compute_drift_score,
        DRIFT_SCORE_THRESHOLD,
    )

    ti              = ctx["ti"]
    new_csv         = ti.xcom_pull(key="new_csv_path",    task_ids="augment_texts")
    baseline_run_id = ti.xcom_pull(key="baseline_run_id", task_ids="build_replay_buffer")

    # baseline_run_id is guaranteed non-None here because branch_on_baseline
    # only routes to evaluate_current_model when last_run_id.txt exists.
    baseline = load_baseline_from_mlflow(baseline_run_id, _tracking_uri())
    result   = compute_drift_score(
        new_csv  = new_csv,
        baseline = baseline,
        device   = "cpu",
    )

    log.info(result.report())
    log.info("Drift score=%.4f  detected=%s  threshold=%.2f",
             result.drift_score, result.drift_detected, DRIFT_SCORE_THRESHOLD)

    ti.xcom_push(key="drift_detected", value=result.drift_detected)
    ti.xcom_push(key="drift_score",    value=result.drift_score)


# ── 6d. Branch on drift ───────────────────────────────────────────────────────

def branch_on_drift(**ctx):
    ti             = ctx["ti"]
    drift_detected = ti.xcom_pull(key="drift_detected", task_ids="drift_check")
    score          = ti.xcom_pull(key="drift_score",    task_ids="drift_check")
    log.info("Branch: drift_detected=%s  score=%s", drift_detected, score)
    return "retrain_model" if drift_detected else "skip_training"


# ── 6d-i. Retrain model ───────────────────────────────────────────────────────

def retrain_model(**ctx):
    import sys
    sys.path.insert(0, str(_src_dir()))
    from clip_experiments import Config, run_linear_probe
    import mlflow

    ti         = ctx["ti"]
    replay_csv = ti.xcom_pull(key="replay_csv_path", task_ids="build_replay_buffer")
    week_label = ti.xcom_pull(key="week_label",      task_ids="detect_new_data")

    mlflow.set_tracking_uri(_tracking_uri())
    mlflow.set_experiment(_experiment())

    cfg = Config(
        csv_path            = replay_csv,
        mode                = "linear_probe",
        epochs              = 10,
        batch_size          = 32,
        lr                  = 5e-6,
        week_label          = week_label,
        mlflow_tracking_uri = _tracking_uri(),
        mlflow_experiment   = _experiment(),
        mlflow_run_name     = f"linear_probe_{week_label}",
    )

    with mlflow.start_run(run_name=cfg.mlflow_run_name) as run:
        new_run_id = run.info.run_id
        mlflow.set_tag("stage",      "retrain")
        mlflow.set_tag("week_label", week_label)
        final_metrics = run_linear_probe(cfg, device="cpu", run_id=new_run_id)

    log.info("Retrain complete. run_id=%s  metrics=%s", new_run_id, final_metrics)

    meta_dir = _data_dir() / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "last_run_id.txt").write_text(new_run_id)

    ti.xcom_push(key="new_run_id", value=new_run_id)


# ── 6d-ii. Skip training ──────────────────────────────────────────────────────

def skip_training(**ctx):
    ti    = ctx["ti"]
    score = ti.xcom_pull(key="drift_score", task_ids="drift_check")
    log.info("No significant drift (score=%.4f). Skipping retraining.", score or 0.0)


# ── 7. Update FAISS index ─────────────────────────────────────────────────────
# Reads new_run_id from whichever training task actually ran:
#   initial_train  → task_ids="initial_train"
#   retrain_model  → task_ids="retrain_model"
# Both push the same XCom key "new_run_id", so we pull from both and take whichever
# is non-None. Falls back to eval_run_id if neither ran (skip_training path).

def update_faiss_index(**ctx):
    import sys
    import json
    import numpy as np
    sys.path.insert(0, str(_src_dir()))

    try:
        import faiss
    except ImportError:
        log.warning("faiss-cpu not installed — skipping FAISS update.")
        return

    from clip_experiments import (
        Config, load_linear_probe_from_mlflow, load_zero_shot_model,
        CLIPProductDataset, collate_skip_none, compute_embeddings,
    )
    import mlflow
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader

    ti         = ctx["ti"]
    week_label = ti.xcom_pull(key="week_label",     task_ids="detect_new_data")
    eval_run_id = ti.xcom_pull(key="eval_run_id",   task_ids="evaluate_current_model")

    # Pull new_run_id from both possible training tasks; one will be None/skipped.
    run_id_from_initial = ti.xcom_pull(key="new_run_id", task_ids="initial_train")
    run_id_from_retrain = ti.xcom_pull(key="new_run_id", task_ids="retrain_model")
    new_run_id = run_id_from_initial or run_id_from_retrain

    active_run_id = new_run_id or eval_run_id

    mlflow.set_tracking_uri(_tracking_uri())

    cfg = Config(
        csv_path            = str(_data_dir() / "processed" / f"{week_label}.csv"),
        mlflow_tracking_uri = _tracking_uri(),
        week_label          = week_label,
    )
    device = "cpu"

    if active_run_id:
        try:
            model, preprocess, tokenizer = load_linear_probe_from_mlflow(
                active_run_id, cfg, device)
            log.info("Loaded fine-tuned model from run %s", active_run_id)
        except Exception as e:
            log.warning("Could not load fine-tuned model (%s). Falling back to zero-shot.", e)
            model, preprocess, tokenizer = load_zero_shot_model(cfg, device)
    else:
        model, preprocess, tokenizer = load_zero_shot_model(cfg, device)

    all_csv = sorted((_data_dir() / "processed").glob("week*.csv"))
    if not all_csv:
        log.warning("No processed CSVs found — cannot build FAISS index.")
        return

    dfs = [pd.read_csv(p) for p in all_csv]
    df_all = pd.concat(dfs, ignore_index=True)
    df_title = df_all[df_all["method"] == "title_clean"].drop_duplicates("original_index")

    dataset = CLIPProductDataset(df_title, tokenizer, preprocess,
                                 cache_dir=str(_data_dir() / "image_cache"),
                                 mode="val")
    loader  = DataLoader(dataset, batch_size=64, collate_fn=collate_skip_none, num_workers=0)

    img_embs, txt_embs = compute_embeddings(model, loader, device)

    img_np = img_embs.numpy().astype("float32")
    txt_np = txt_embs.numpy().astype("float32")
    dim    = img_np.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(img_np)

    faiss_dir = _data_dir() / "faiss"
    faiss_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(faiss_dir / "index.bin"))

    meta = df_title[["original_index", "image_url",
                      "augmented_text"]].to_dict(orient="records")
    (faiss_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    log.info("FAISS index updated: %d vectors  dim=%d  saved to %s",
             index.ntotal, dim, faiss_dir)

    if active_run_id:
        with mlflow.start_run(run_id=active_run_id):
            mlflow.log_artifact(str(faiss_dir / "index.bin"), artifact_path="faiss")
            mlflow.log_artifact(str(faiss_dir / "meta.json"),  artifact_path="faiss")


# ── 8. DVC commit ─────────────────────────────────────────────────────────────

def dvc_commit_new_data(**ctx):
    import subprocess

    ti         = ctx["ti"]
    week_label = ti.xcom_pull(key="week_label", task_ids="detect_new_data")

    # Pull new_run_id from both training tasks; whichever ran will be non-None.
    run_id_from_initial = ti.xcom_pull(key="new_run_id", task_ids="initial_train")
    run_id_from_retrain = ti.xcom_pull(key="new_run_id", task_ids="retrain_model")
    new_run_id = run_id_from_initial or run_id_from_retrain

    if not new_run_id:
        log.info("No new model run — skipping DVC commit.")
        return

    # On Day 0 there is no replay CSV, so only track what actually exists.
    files_to_track = []
    replay_csv = _data_dir() / "processed" / f"{week_label}_replay.csv"
    if replay_csv.exists():
        files_to_track.append(str(replay_csv))
    faiss_index = _data_dir() / "faiss" / "index.bin"
    if faiss_index.exists():
        files_to_track.append(str(faiss_index))

    cmds = [["dvc", "add", f] for f in files_to_track]
    cmds += [
        ["dvc", "push"],
        ["git", "add", "--all", "*.dvc", ".dvc/"],
        ["git", "commit", "-m",
         f"data: {week_label} batch + FAISS index  mlflow_run={new_run_id}"],
    ]
    for cmd in cmds:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd="/opt/airflow")
        if result.returncode != 0:
            log.warning("CMD %s failed: %s", " ".join(cmd), result.stderr[-200:])
        else:
            log.info("CMD %s OK", " ".join(cmd))

    incoming_csv = ti.xcom_pull(key="incoming_csv_to_mark", task_ids="detect_new_data")
    if incoming_csv:
        _mark_csv_processed(incoming_csv)
        log.info("CSV marked as processed.")


# ─────────────────────────────────────────────────────────────────────────────
# DAG DEFINITION
# ─────────────────────────────────────────────────────────────────────────────

with DAG(
    dag_id                  = "clip_product_retrieval_pipeline",
    default_args            = default_args,
    description             = "CLIP continual-learning pipeline with drift detection",
    schedule                = timedelta(minutes=5),   # testing: every 5 min | production: timedelta(weeks=1)
    start_date              = datetime(2025, 1, 1),
    catchup                 = False,
    max_active_runs         = 1,
    is_paused_upon_creation = False,
    tags                    = ["clip", "mlops", "retrieval"],
) as dag:

    t_detect = PythonOperator(
        task_id         = "detect_new_data",
        python_callable = detect_new_data,
    )

    t_dvc_pull = PythonOperator(
        task_id         = "dvc_pull_latest",
        python_callable = dvc_pull,
    )

    t_augment = PythonOperator(
        task_id           = "augment_texts",
        python_callable   = augment_texts,
        execution_timeout = timedelta(minutes=30),
    )

    t_replay = PythonOperator(
        task_id           = "build_replay_buffer",
        python_callable   = build_replay_buffer,
        execution_timeout = timedelta(minutes=20),
    )

    # ── NEW: gate on whether a baseline model exists ──────────────────────────
    t_branch_baseline = BranchPythonOperator(
        task_id         = "branch_on_baseline",
        python_callable = branch_on_baseline,
    )

    # ── Day-0 path ────────────────────────────────────────────────────────────
    t_initial_train = PythonOperator(
        task_id           = "initial_train",
        python_callable   = initial_train,
        execution_timeout = timedelta(hours=4),
    )

    # ── Normal path ───────────────────────────────────────────────────────────
    t_eval = PythonOperator(
        task_id           = "evaluate_current_model",
        python_callable   = evaluate_current_model,
        execution_timeout = timedelta(hours=1),
    )

    t_drift = PythonOperator(
        task_id           = "drift_check",
        python_callable   = drift_check,
        execution_timeout = timedelta(minutes=30),
    )

    t_branch_drift = BranchPythonOperator(
        task_id         = "branch_on_drift",
        python_callable = branch_on_drift,
    )

    t_retrain = PythonOperator(
        task_id           = "retrain_model",
        python_callable   = retrain_model,
        execution_timeout = timedelta(hours=4),
    )

    t_skip = PythonOperator(
        task_id         = "skip_training",
        python_callable = skip_training,
    )

    # ── Shared tail ───────────────────────────────────────────────────────────
    t_faiss = PythonOperator(
        task_id           = "update_faiss_index",
        python_callable   = update_faiss_index,
        trigger_rule      = TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
        execution_timeout = timedelta(hours=1),
    )

    t_dvc_commit = PythonOperator(
        task_id         = "dvc_commit_new_data",
        python_callable = dvc_commit_new_data,
        trigger_rule    = TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )

    t_done = EmptyOperator(
        task_id      = "pipeline_complete",
        trigger_rule = TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # TASK DEPENDENCIES
    # ─────────────────────────────────────────────────────────────────────────
    #
    #   detect_new_data
    #        │
    #   dvc_pull_latest
    #        │
    #   augment_texts
    #        │
    #   build_replay_buffer
    #        │
    #   branch_on_baseline
    #       │                      │
    #  [no baseline]          [baseline exists]
    #       │                      │
    #  initial_train     evaluate_current_model
    #       │                      │
    #       │               drift_check
    #       │                      │
    #       │             branch_on_drift
    #       │                ├── retrain_model
    #       │                └── skip_training
    #       │                      │
    #       └──────────────────────┘
    #                              │
    #                    update_faiss_index
    #                              │
    #                    dvc_commit_new_data
    #                              │
    #                    pipeline_complete
    #
    (
        t_detect
        >> t_dvc_pull
        >> t_augment
        >> t_replay
        >> t_branch_baseline
        >> [t_initial_train, t_eval]
    )

    t_eval >> t_drift >> t_branch_drift >> [t_retrain, t_skip]

    # All terminal training tasks feed into the shared tail
    [t_initial_train, t_retrain, t_skip] >> t_faiss >> t_dvc_commit >> t_done