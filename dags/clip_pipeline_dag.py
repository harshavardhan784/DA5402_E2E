"""
dags/clip_pipeline_dag.py
──────────────────────────
Weekly CLIP product-retrieval retraining pipeline.

Pipeline stages
───────────────
 1. detect_new_data          Check incoming/ for a CSV whose replay file doesn't exist yet.
 2. dvc_pull_latest          Pull latest data version tracked by DVC.
 3. augment_texts            Run augment_texts.py → data/processed/{week}.csv
 4. build_replay_buffer      Combine new augmented data + previous week's replay.
 5. branch_on_baseline       Day-0 (no last_run_id.txt) OR normal path.

 ── Day-0 path ───────────────────────────────────────────────────────────────
 6a. initial_train           Train on week1_replay.csv (written by build_replay_buffer).
                             Saves run_id to data/meta/last_run_id.txt.

 ── Normal path (baseline exists) ────────────────────────────────────────────
 6b. evaluate_current_model  Zero-shot baseline recall on new augmented data.
 6c. drift_check             compute_drift_score() → XCom {drift_detected, score}
 6d. branch_on_drift         BranchPythonOperator: retrain OR skip.
 6d-i.  retrain_model        linear_probe on replay buffer CSV.
 6d-ii. skip_training        No-op log.

 ── Shared tail ──────────────────────────────────────────────────────────────
 7. update_faiss_index       Rebuild FAISS index from all processed week CSVs.
 8. log_provenance           Log reproducibility record to MLflow (replaces DVC commit).
 9. pipeline_complete        Terminal no-op.

Processed folder conventions
─────────────────────────────
  data/processed/week1.csv          ← augmented data for week1 only
  data/processed/week1_replay.csv   ← week1 augmented (Day-0, no old data to mix)
  data/processed/week2_replay.csv   ← week2 augmented + samples from week1_replay
  data/processed/week5_replay.csv   ← week5 augmented + samples from week4_replay
                                       (week4_replay already contains week1-4 mix)

  Existence of {week}_replay.csv    ← means this week was fully processed.
  detect_new_data uses this as its "already done" gate — no hidden state files.

XCom keys
──────────
  incoming_csv    : absolute path to incoming raw CSV
  week_label      : stem of filename (e.g. 'week2')
  new_csv_path    : path to augmented CSV  (data/processed/week2.csv)
  replay_csv_path : path to replay CSV     (data/processed/week2_replay.csv)
  baseline_run_id : MLflow run_id from last_run_id.txt (None on Day-0)
  eval_run_id     : MLflow run_id of pre-retrain evaluation
  new_run_id      : MLflow run_id from initial_train OR retrain_model
  drift_detected  : bool
  drift_score     : float

Environment variables (set in docker-compose)
──────────────────────────────────────────────
  MLFLOW_TRACKING_URI   http://mlflow:5000
  MLFLOW_EXPERIMENT     clip_product_retrieval
  DATA_DIR              /opt/airflow/data
  SRC_DIR               /opt/airflow/src
"""

from __future__ import annotations

import glob
import logging
import os
import re
import subprocess
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
# Environment helpers  (resolved at task-runtime, not import time)
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
# Week-label helpers
# ─────────────────────────────────────────────────────────────────────────────

def _week_number(stem: str) -> int:
    """
    Extract the trailing integer from a week stem.
    'week1'  → 1
    'week12' → 12
    Returns 0 if no number found (sorts to front, treated as oldest).
    """
    m = re.search(r"(\d+)$", stem)
    return int(m.group(1)) if m else 0


def _replay_exists(week_label: str) -> bool:
    """Return True if this week has already been fully processed."""
    return (_data_dir() / "processed" / f"{week_label}_replay.csv").exists()


def _find_previous_replay(current_week_label: str) -> str | None:
    """
    Find the replay CSV from the immediately preceding week.

    Convention: week5_replay.csv already contains the cumulative mix of
    weeks 1-5, so week6 only needs to sample from week5_replay.csv.

    Returns the absolute path string or None if no previous replay exists
    (Day-0 case — current week IS week1 or no replay was ever written).
    """
    processed_dir = _data_dir() / "processed"
    current_num   = _week_number(current_week_label)

    # Collect all *_replay.csv files excluding the current week's own replay
    replays = [
        p for p in processed_dir.glob("*_replay.csv")
        if _week_number(p.stem.replace("_replay", "")) < current_num
    ]

    if not replays:
        return None

    # Pick the one with the highest week number (immediately previous)
    replays.sort(key=lambda p: _week_number(p.stem.replace("_replay", "")))
    chosen = replays[-1]
    log.info("[replay_lookup] current=%s  previous_replay=%s", current_week_label, chosen)
    return str(chosen)


# ─────────────────────────────────────────────────────────────────────────────
# DAG default args
# ─────────────────────────────────────────────────────────────────────────────

default_args = {
    "owner":            "mlops",
    "depends_on_past":  False,
    "retries":          1,
    "retry_delay":      timedelta(seconds=30),
    "email_on_failure": False,
}

# ─────────────────────────────────────────────────────────────────────────────
# TASK FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

# ── 1. Detect new data ────────────────────────────────────────────────────────
#
# Gate: a CSV in incoming/ is considered "new" if and only if its corresponding
# {stem}_replay.csv does NOT yet exist in data/processed/.
#
# This replaces the mtime/size state-file approach.  It is stateless — the
# processed/ folder is the ground truth.  If week1_replay.csv exists, week1.csv
# will never be picked up again regardless of mtime or file changes.
#
# To reprocess a week: delete data/processed/{week}_replay.csv and re-trigger.

def detect_new_data(**ctx):
    """
    Scan data/incoming/ for the first CSV that has not yet been processed.
    Raises FileNotFoundError if no CSVs exist.
    Raises ValueError if all CSVs are already processed (replay files exist).
    """
    import pandas as pd

    week_label = ctx["ti"].xcom_pull(key="week_label", task_ids="detect_new_data")
    tag = f"[detect]"

    log.info("%s DATA_DIR=%s", tag, os.environ.get("DATA_DIR", "NOT SET"))
    log.info("%s _data_dir()=%s", tag, _data_dir())

    incoming_dir = _data_dir() / "incoming"
    incoming_dir.mkdir(parents=True, exist_ok=True)

    all_csvs = sorted(
        glob.glob(str(incoming_dir / "*.csv")),
        key=lambda p: _week_number(Path(p).stem),
    )
    log.info("%s CSVs found in incoming/: %s", tag, all_csvs)

    if not all_csvs:
        raise FileNotFoundError(
            f"No CSV found in {incoming_dir}. "
            "Copy your weekly CSV there and re-trigger the DAG."
        )

    chosen = None
    for csv_path in all_csvs:
        stem = Path(csv_path).stem
        if _replay_exists(stem):
            log.info("%s %s → ALREADY PROCESSED (replay exists), skipping.", tag, stem)
        else:
            log.info("%s %s → UNPROCESSED (no replay found), selecting.", tag, stem)
            chosen = csv_path
            break   # process one week per DAG run

    if chosen is None:
        processed = [Path(p).stem for p in all_csvs]
        raise ValueError(
            f"All CSVs in incoming/ are already processed: {processed}. "
            "To reprocess a week, delete data/processed/{week}_replay.csv "
            "and re-trigger the DAG."
        )

    csv_path   = Path(chosen)
    week_label = csv_path.stem

    # Quick sanity check on the CSV — log column names so mismatches are visible
    df_peek = pd.read_csv(csv_path, nrows=5)
    log.info("%s [%s] columns=%s  peek_rows=%d",
             tag, week_label, list(df_peek.columns), len(df_peek))

    if len(df_peek) == 0:
        raise ValueError(f"{csv_path} appears empty (0 rows in first 5).")

    # Log full row count (capped for speed)
    n_rows = sum(1 for _ in open(csv_path)) - 1   # fast line count
    log.info("%s [%s] incoming_csv=%s  total_rows~=%d",
             tag, week_label, csv_path, n_rows)

    ctx["ti"].xcom_push(key="incoming_csv",  value=str(csv_path))
    ctx["ti"].xcom_push(key="week_label",    value=week_label)
    ctx["ti"].xcom_push(key="n_rows",        value=n_rows)

    log.info("%s [%s] ✓ Detection complete.", tag, week_label)


# ── 2. DVC pull ───────────────────────────────────────────────────────────────

def dvc_pull(**ctx):
    """
    Pull DVC-tracked files.  Logs stdout/stderr in full so you can see exactly
    which files were fetched or skipped.  Failure is non-fatal on first run
    (no remote configured yet).
    """
    ti         = ctx["ti"]
    week_label = ti.xcom_pull(key="week_label", task_ids="detect_new_data")
    tag        = f"[dvc_pull][{week_label}]"

    result = subprocess.run(
        ["dvc", "pull", "--force"],
        capture_output=True, text=True,
        cwd="/opt/airflow",
    )
    if result.stdout:
        log.info("%s stdout:\n%s", tag, result.stdout[-1000:])
    if result.returncode != 0:
        log.warning(
            "%s dvc pull exited with code %d — may be first run or no remote. "
            "stderr:\n%s",
            tag, result.returncode, result.stderr[-500:],
        )
    else:
        log.info("%s dvc pull succeeded.", tag)


# ── 3. Augment texts ──────────────────────────────────────────────────────────

def augment_texts(**ctx):
    """
    Produce data/processed/{week_label}.csv from the raw incoming CSV.

    Input  : incoming raw CSV  (columns: title, image_links, selling_price, …)
    Output : augmented CSV     (columns: original_index, image_url, method,
                                          augmented_text, category_2, category_3,
                                          selling_price)

    No category_filter — processes all categories in the incoming file.
    """
    import sys
    sys.path.insert(0, str(_src_dir()))
    from augment_texts import build_augmented_dataset

    ti         = ctx["ti"]
    incoming   = ti.xcom_pull(key="incoming_csv", task_ids="detect_new_data")
    week_label = ti.xcom_pull(key="week_label",   task_ids="detect_new_data")
    tag        = f"[augment][{week_label}]"

    out_path = _data_dir() / "processed" / f"{week_label}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("%s Augmenting  %s  →  %s", tag, incoming, out_path)

    df_aug = build_augmented_dataset(
        csv_path  = incoming,
        n_samples = 4000,
        seed      = 42,
        # no category_filter — use all categories
    )
    df_aug.to_csv(out_path, index=False)

    log.info(
        "%s ✓ Augmented: %d rows, %d unique images  saved to %s",
        tag, len(df_aug), df_aug["original_index"].nunique(), out_path,
    )
    ti.xcom_push(key="new_csv_path", value=str(out_path))


# ── 4. Build replay buffer ────────────────────────────────────────────────────

def build_replay_buffer(**ctx):
    """
    Combine current week's augmented data with a sample from the previous
    week's replay CSV.

    Why previous replay (not augmented)?
        week5_replay.csv already contains the cumulative hard-negative mix of
        weeks 1-5.  Sampling from it preserves that breadth automatically.

    Day-0 (no previous replay exists):
        Writes new augmented data directly to {week}_replay.csv.
        Sets baseline_run_id=None so branch_on_baseline routes to initial_train.

    Week N (previous replay found):
        Calls build_replay_buffer() from replay_buffer.py which does:
          - random sample from old replay
          - hard-negative sample (cosine-far from baseline centroid)
          - deduplication
        Result written to {week}_replay.csv.
    """
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
    tag        = f"[replay][{week_label}]"

    run_id_file = _data_dir() / "meta" / "last_run_id.txt"
    replay_path = _data_dir() / "processed" / f"{week_label}_replay.csv"
    replay_path.parent.mkdir(parents=True, exist_ok=True)

    new_df = pd.read_csv(new_csv)
    log.info("%s new augmented data: %d rows, %d images",
             tag, len(new_df), new_df["original_index"].nunique())

    # ── Day-0: no baseline run, no previous replay ────────────────────────────
    if not run_id_file.exists():
        log.info(
            "%s No last_run_id.txt found — Day-0 path. "
            "Writing augmented data directly to replay path (no mixing).",
            tag,
        )
        new_df.to_csv(replay_path, index=False)
        log.info("%s ✓ Day-0 replay written: %d rows → %s", tag, len(new_df), replay_path)
        ti.xcom_push(key="replay_csv_path", value=str(replay_path))
        ti.xcom_push(key="baseline_run_id", value=None)
        return

    # ── Week N: baseline exists ────────────────────────────────────────────────
    baseline_run_id = run_id_file.read_text().strip()
    log.info("%s baseline_run_id=%s", tag, baseline_run_id)

    prev_replay = _find_previous_replay(week_label)

    if prev_replay is None:
        log.warning(
            "%s baseline run_id exists but no previous replay CSV found. "
            "This is unexpected after Day-0. Falling back to new data only.",
            tag,
        )
        combined = new_df
    else:
        log.info("%s Previous replay: %s", tag, prev_replay)
        prev_df = pd.read_csv(prev_replay)
        log.info(
            "%s Previous replay stats: %d rows, %d images",
            tag, len(prev_df), prev_df["original_index"].nunique(),
        )

        baseline_data = load_baseline_from_mlflow(baseline_run_id, _tracking_uri())

        combined = _build(
            old_csv_path  = prev_replay,
            new_df        = new_df,
            baseline_data = baseline_data,
            replay_ratio  = 0.20,
        )

    combined.to_csv(replay_path, index=False)
    log.info(
        "%s ✓ Replay buffer written: %d rows, %d corresponding unique images → %s",
        tag, len(combined), combined["original_index"].nunique(), replay_path,
    )
    ti.xcom_push(key="replay_csv_path", value=str(replay_path))
    ti.xcom_push(key="baseline_run_id", value=baseline_run_id)


# ── 5. Branch on baseline ─────────────────────────────────────────────────────

def branch_on_baseline(**ctx):
    """
    Route based on whether a trained baseline model exists.
      No last_run_id.txt  →  initial_train   (Day-0)
      Has last_run_id.txt →  evaluate_current_model (normal weekly path)
    """
    ti         = ctx["ti"]
    week_label = ti.xcom_pull(key="week_label", task_ids="detect_new_data")
    tag        = f"[branch_baseline][{week_label}]"

    run_id_file = _data_dir() / "meta" / "last_run_id.txt"
    if run_id_file.exists():
        run_id = run_id_file.read_text().strip()
        log.info("%s Baseline found (run_id=%s) → drift_check", tag, run_id)
        return "drift_check"

    log.info("%s No baseline → initial_train (Day-0)", tag)
    return "initial_train"


# ── 6a. Initial train (Day-0) ─────────────────────────────────────────────────

def initial_train(**ctx):
    """
    Day-0 training: linear probe on week1_replay.csv.

    Uses the replay CSV (not just the augmented CSV) so the training path is
    identical to subsequent weeks — only the absence of old data differs.

    Writes run_id to data/meta/last_run_id.txt so next week has a baseline.
    """
    import sys
    sys.path.insert(0, str(_src_dir()))
    from clip_experiments import Config, run_linear_probe
    import mlflow

    ti         = ctx["ti"]
    replay_csv = ti.xcom_pull(key="replay_csv_path", task_ids="build_replay_buffer")
    week_label = ti.xcom_pull(key="week_label",      task_ids="detect_new_data")
    tag        = f"[initial_train][{week_label}]"

    log.info("%s Training on replay CSV: %s", tag, replay_csv)

    mlflow.set_tracking_uri(_tracking_uri())
    mlflow.set_experiment(_experiment())

    cfg = Config(
        csv_path            = replay_csv,
        image_cache_dir     = str(_data_dir() / "image_cache"),
        mode                = "linear_probe",
        epochs              = 2,          # quick first run; increase for production
        batch_size          = 32,
        lr                  = 1e-3,
        week_label          = week_label,
        mlflow_tracking_uri = _tracking_uri(),
        mlflow_experiment   = _experiment(),
        mlflow_run_name     = f"initial_train_{week_label}",
    )

    with mlflow.start_run(run_name=cfg.mlflow_run_name) as run:
        new_run_id = run.info.run_id
        mlflow.set_tag("stage",      "initial_train")
        mlflow.set_tag("week_label", week_label)
        mlflow.set_tag("pipeline",   "clip_product_retrieval")
        log.info("%s MLflow run_id=%s", tag, new_run_id)
        final_metrics = run_linear_probe(cfg, device="cpu", run_id=new_run_id)

    # Persist baseline for next week
    meta_dir = _data_dir() / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "last_run_id.txt").write_text(new_run_id)

    log.info(
        "%s ✓ Initial train complete. run_id=%s  Recall@1=%.4f  saved baseline.",
        tag, new_run_id, final_metrics.get("Recall@1", 0.0),
    )
    ti.xcom_push(key="new_run_id", value=new_run_id)


# ── 6b. Evaluate current model ────────────────────────────────────────────────

# def evaluate_current_model(**ctx):
#     """
#     Run linear_probe evaluation of the current baseline model on new augmented data.
#     Records pre-retrain metrics in MLflow so you can track per-week degradation.
#     """
#     import sys
#     sys.path.insert(0, str(_src_dir()))
#     from clip_experiments import Config, run_linear_probe
#     import mlflow

#     ti              = ctx["ti"]
#     new_csv         = ti.xcom_pull(key="new_csv_path",    task_ids="augment_texts")
#     week_label      = ti.xcom_pull(key="week_label",      task_ids="detect_new_data")
#     baseline_run_id = ti.xcom_pull(key="baseline_run_id", task_ids="build_replay_buffer")
#     tag             = f"[evaluate][{week_label}]"

#     log.info("%s Evaluating baseline run_id=%s on new data: %s",
#              tag, baseline_run_id, new_csv)

#     mlflow.set_tracking_uri(_tracking_uri())
#     mlflow.set_experiment(_experiment())

#     cfg = Config(
#         csv_path            = new_csv,
#         image_cache_dir     = str(_data_dir() / "image_cache"),
#         mode                = "linear_probe",
#         week_label          = week_label,
#         mlflow_tracking_uri = _tracking_uri(),
#         mlflow_experiment   = _experiment(),
#         mlflow_run_name     = f"eval_pre_retrain_{week_label}",
#     )

#     with mlflow.start_run(run_name=cfg.mlflow_run_name) as run:
#         eval_run_id = run.info.run_id
#         mlflow.set_tag("stage",           "pre_retrain_eval")
#         mlflow.set_tag("week_label",      week_label)
#         mlflow.set_tag("baseline_run_id", str(baseline_run_id))
#         mlflow.set_tag("pipeline",        "clip_product_retrieval")
#         log.info("%s MLflow eval run_id=%s", tag, eval_run_id)
#         final_metrics = run_linear_probe(cfg, device="cpu", run_id=eval_run_id)

#     log.info(
#         "%s ✓ Eval complete. run_id=%s  Recall@1=%.4f  Consistency@1=%.4f",
#         tag, eval_run_id,
#         final_metrics.get("Recall@1",      0.0),
#         final_metrics.get("Consistency@1", 0.0),
#     )
#     ti.xcom_push(key="eval_run_id", value=eval_run_id)


# ── 6c. Drift check ───────────────────────────────────────────────────────────

def drift_check(**ctx):
    """
    Compare new-data embeddings against the baseline distribution.
    baseline_run_id is guaranteed non-None here (branch_on_baseline
    only routes here when last_run_id.txt exists).
    """
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
    week_label      = ti.xcom_pull(key="week_label",      task_ids="detect_new_data")
    tag             = f"[drift][{week_label}]"

    log.info("%s Comparing new data against baseline run_id=%s", tag, baseline_run_id)

    baseline = load_baseline_from_mlflow(baseline_run_id, _tracking_uri())
    result   = compute_drift_score(new_csv=new_csv, baseline=baseline, device="cpu")

    log.info("%s %s", tag, result.report())
    log.info(
        "%s drift_score=%.4f  detected=%s  threshold=%.2f",
        tag, result.drift_score, result.drift_detected, DRIFT_SCORE_THRESHOLD,
    )

    ti.xcom_push(key="drift_detected", value=result.drift_detected)
    ti.xcom_push(key="drift_score",    value=result.drift_score)


# ── 6d. Branch on drift ───────────────────────────────────────────────────────

def branch_on_drift(**ctx):
    ti         = ctx["ti"]
    week_label = ti.xcom_pull(key="week_label",    task_ids="detect_new_data")
    detected   = ti.xcom_pull(key="drift_detected", task_ids="drift_check")
    score      = ti.xcom_pull(key="drift_score",    task_ids="drift_check")
    tag        = f"[branch_drift][{week_label}]"

    decision = "retrain_model" if detected else "skip_training"
    log.info("%s drift_detected=%s  score=%.4f  → %s", tag, detected, score or 0.0, decision)
    return decision


# ── 6d-i. Retrain model ───────────────────────────────────────────────────────

def retrain_model(**ctx):
    """
    Linear probe retrain on the replay buffer CSV.
    Uses {week}_replay.csv which contains current week + previous weeks' mix.
    Updates last_run_id.txt so next week uses this run as its baseline.
    """
    import sys
    sys.path.insert(0, str(_src_dir()))
    from clip_experiments import Config, run_linear_probe
    import mlflow

    ti         = ctx["ti"]
    replay_csv = ti.xcom_pull(key="replay_csv_path", task_ids="build_replay_buffer")
    week_label = ti.xcom_pull(key="week_label",      task_ids="detect_new_data")
    tag        = f"[retrain][{week_label}]"

    log.info("%s Retraining on replay CSV: %s", tag, replay_csv)

    mlflow.set_tracking_uri(_tracking_uri())
    mlflow.set_experiment(_experiment())

    cfg = Config(
        csv_path            = replay_csv,
        image_cache_dir     = str(_data_dir() / "image_cache"),
        mode                = "linear_probe",
        epochs              = 4,
        batch_size          = 32,
        lr                  = 1e-3,
        week_label          = week_label,
        mlflow_tracking_uri = _tracking_uri(),
        mlflow_experiment   = _experiment(),
        mlflow_run_name     = f"retrain_linear_probe_{week_label}",
    )

    with mlflow.start_run(run_name=cfg.mlflow_run_name) as run:
        new_run_id = run.info.run_id
        mlflow.set_tag("stage",      "retrain")
        mlflow.set_tag("week_label", week_label)
        mlflow.set_tag("pipeline",   "clip_product_retrieval")
        log.info("%s MLflow run_id=%s", tag, new_run_id)
        final_metrics = run_linear_probe(cfg, device="cpu", run_id=new_run_id)

    # Update baseline for next week
    meta_dir = _data_dir() / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "last_run_id.txt").write_text(new_run_id)

    log.info(
        "%s ✓ Retrain complete. run_id=%s  Recall@1=%.4f  baseline updated.",
        tag, new_run_id, final_metrics.get("Recall@1", 0.0),
    )
    ti.xcom_push(key="new_run_id", value=new_run_id)


# ── 6d-ii. Skip training ──────────────────────────────────────────────────────

def skip_training(**ctx):
    ti         = ctx["ti"]
    week_label = ti.xcom_pull(key="week_label",   task_ids="detect_new_data")
    score      = ti.xcom_pull(key="drift_score",  task_ids="drift_check")
    tag        = f"[skip_training][{week_label}]"
    log.info("%s No significant drift (score=%.4f). Retraining skipped.", tag, score or 0.0)


# ── 7. Update FAISS index ─────────────────────────────────────────────────────

def update_faiss_index(**ctx):
    """
    Rebuild the FAISS index from ALL processed week CSVs (not replay — those
    have duplicates across weeks).  Uses only method='title_clean' rows so
    each product appears exactly once.

    Loads the best available model:
      1. new_run_id from this run's training task  (initial_train or retrain_model)
      2. eval_run_id from evaluate_current_model
      3. zero-shot CLIP (fallback)
    """
    import sys
    import json
    import numpy as np
    sys.path.insert(0, str(_src_dir()))

    ti         = ctx["ti"]
    week_label = ti.xcom_pull(key="week_label",   task_ids="detect_new_data")
    tag        = f"[faiss][{week_label}]"

    try:
        import faiss
    except ImportError:
        log.warning("%s faiss-cpu not installed — skipping FAISS update.", tag)
        return

    from clip_experiments import (
        Config,
        load_linear_probe_from_mlflow,
        load_zero_shot_model,
        CLIPProductDataset,
        collate_skip_none,
        compute_embeddings,
    )
    import mlflow
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader

    # eval_run_id         = ti.xcom_pull(key="eval_run_id",  task_ids="evaluate_current_model")
    run_id_from_initial = ti.xcom_pull(key="new_run_id",   task_ids="initial_train")
    run_id_from_retrain = ti.xcom_pull(key="new_run_id",   task_ids="retrain_model")
    new_run_id          = run_id_from_initial or run_id_from_retrain
    active_run_id       = new_run_id

    log.info(
        "%s run_ids — initial=%s  retrain=%s  active=%s",
        tag, run_id_from_initial, run_id_from_retrain, active_run_id,
    )

    mlflow.set_tracking_uri(_tracking_uri())

    cfg = Config(
        csv_path            = str(_data_dir() / "processed" / f"{week_label}.csv"),
        image_cache_dir     = str(_data_dir() / "image_cache"),
        mlflow_tracking_uri = _tracking_uri(),
        week_label          = week_label,
    )
    device = "cpu"

    # ── Load model ────────────────────────────────────────────────────────────
    if active_run_id:
        try:
            # load_linear_probe_from_mlflow returns (model, probe, preprocess, tokenizer)
            model, probe, preprocess, tokenizer = load_linear_probe_from_mlflow(
                active_run_id, cfg, device)
            log.info("%s Loaded linear-probe model from run %s", tag, active_run_id)
        except Exception as e:
            log.warning(
                "%s Could not load linear-probe model (%s) — falling back to zero-shot.", tag, e)
            model, preprocess, tokenizer = load_zero_shot_model(cfg, device)
            probe = None
    else:
        log.info("%s No active run_id — using zero-shot model.", tag)
        model, preprocess, tokenizer = load_zero_shot_model(cfg, device)
        probe = None

    # ── Load all processed week CSVs (augmented, not replay) ──────────────────
    # These have the format: original_index | image_url | method | augmented_text …
    # We pick method='title_clean' so each original product appears once.
    all_week_csvs = sorted(
        (_data_dir() / "processed").glob("week*.csv"),
        key=lambda p: _week_number(p.stem),
    )
    # Exclude replay files — they contain duplicates across weeks
    all_week_csvs = [p for p in all_week_csvs if "_replay" not in p.name]
    log.info("%s Building index from %d week CSV(s): %s",
             tag, len(all_week_csvs), [p.name for p in all_week_csvs])

    if not all_week_csvs:
        log.warning("%s No processed week CSVs found — cannot build FAISS index.", tag)
        return

    dfs      = [pd.read_csv(p) for p in all_week_csvs]
    df_all   = pd.concat(dfs, ignore_index=True)
    df_title = (
        df_all[df_all["method"] == "title_clean"]
        .drop_duplicates("original_index")
        .reset_index(drop=True)
    )
    log.info("%s Index corpus: %d unique products across %d weeks",
             tag, len(df_title), len(all_week_csvs))

    dataset = CLIPProductDataset(
        df_title, tokenizer, preprocess,
        cache_dir=str(_data_dir() / "image_cache"),
        mode="val",
    )
    loader = DataLoader(
        dataset, batch_size=64,
        collate_fn=collate_skip_none, num_workers=0,
    )

    # compute_embeddings returns (img_embs, txt_embs, orig_idx_list)
    img_embs, txt_embs, orig_idx = compute_embeddings(model, loader, device,
                                                       probe_head=probe)

    img_np = img_embs.numpy().astype("float32")
    dim    = img_np.shape[1]
    log.info("%s Embedding shape: img=%s  txt=%s  dim=%d",
             tag, img_embs.shape, txt_embs.shape, dim)

    # ── Build FAISS index on image embeddings ─────────────────────────────────
    index = faiss.IndexFlatIP(dim)   # cosine similarity on unit-normed vectors
    index.add(img_np)

    faiss_dir = _data_dir() / "faiss"
    faiss_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(faiss_dir / "index.bin"))

    # ── Write metadata: orig_idx → image_url + title for lookup ──────────────
    meta = (
        df_title[["original_index", "image_url", "augmented_text"]]
        .to_dict(orient="records")
    )
    (faiss_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    log.info(
        "%s ✓ FAISS index saved: %d vectors  dim=%d  path=%s",
        tag, index.ntotal, dim, faiss_dir,
    )

    # ── Log index as MLflow artifact so it's versioned with the run ───────────
    if active_run_id:
        with mlflow.start_run(run_id=active_run_id):
            mlflow.log_artifact(str(faiss_dir / "index.bin"), artifact_path="faiss")
            mlflow.log_artifact(str(faiss_dir / "meta.json"),  artifact_path="faiss")
        log.info("%s FAISS artifacts logged to MLflow run %s", tag, active_run_id)


# ── 8. Log provenance ─────────────────────────────────────────────────────────

def log_provenance(**ctx):
    """
    Replace the old DVC commit step with a provenance record logged to MLflow.

    What gets recorded:
      - week_label
      - incoming CSV path
      - git commit hash of the codebase at run time
      - MLflow run_id of the model used/produced this week
      - paths of all output files (augmented CSV, replay CSV, FAISS index)

    Why MLflow instead of DVC commit?
      DVC local-only mode has no remote to push to, so `dvc push` always fails.
      The provenance record in MLflow gives you full reproducibility:
        1. Find the week's run in MLflow UI by tag week_label=weekN
        2. Read the provenance artifact to find git_commit + input CSV path
        3. Check out that commit, place the CSV, re-run — identical results.

    DVC is still used for `dvc pull` (step 2) to hydrate data pointers.
    Committing/pushing is skipped until you configure a DVC remote.
    """
    import mlflow
    import json

    ti         = ctx["ti"]
    week_label = ti.xcom_pull(key="week_label",   task_ids="detect_new_data")
    incoming   = ti.xcom_pull(key="incoming_csv", task_ids="detect_new_data")
    tag        = f"[provenance][{week_label}]"

    run_id_from_initial = ti.xcom_pull(key="new_run_id", task_ids="initial_train")
    run_id_from_retrain = ti.xcom_pull(key="new_run_id", task_ids="retrain_model")
    new_run_id          = run_id_from_initial or run_id_from_retrain

    if not new_run_id:
        log.info("%s No new model run this week — skipping provenance log.", tag)
        return

    # Git commit hash of the code that ran this pipeline
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True, cwd="/opt/airflow",
        ).strip()
    except Exception:
        git_commit = "unknown"

    provenance = {
        "week_label":       week_label,
        "incoming_csv":     incoming,
        "git_commit":       git_commit,
        "model_run_id":     new_run_id,
        "augmented_csv":    str(_data_dir() / "processed" / f"{week_label}.csv"),
        "replay_csv":       str(_data_dir() / "processed" / f"{week_label}_replay.csv"),
        "faiss_index":      str(_data_dir() / "faiss" / "index.bin"),
        "faiss_meta":       str(_data_dir() / "faiss" / "meta.json"),
    }

    log.info("%s Provenance record:\n%s", tag, json.dumps(provenance, indent=2))

    mlflow.set_tracking_uri(_tracking_uri())
    with mlflow.start_run(run_id=new_run_id):
        mlflow.log_dict(provenance, f"provenance/{week_label}_provenance.json")
        mlflow.set_tag("week_label",  week_label)
        mlflow.set_tag("git_commit",  git_commit)
        mlflow.set_tag("pipeline",    "clip_product_retrieval")

    log.info(
        "%s ✓ Provenance logged to MLflow run %s (git=%s).",
        tag, new_run_id, git_commit[:8],
    )


# ─────────────────────────────────────────────────────────────────────────────
# DAG DEFINITION
# ─────────────────────────────────────────────────────────────────────────────

with DAG(
    dag_id                  = "clip_product_retrieval_pipeline",
    default_args            = default_args,
    description             = "CLIP continual-learning pipeline with drift detection",
    schedule                = timedelta(minutes=5),   # dev: 5 min | prod: timedelta(weeks=1)
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
    # t_eval = PythonOperator(
    #     task_id           = "evaluate_current_model",
    #     python_callable   = evaluate_current_model,
    #     execution_timeout = timedelta(hours=1),
    # )

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

    t_provenance = PythonOperator(
        task_id         = "log_provenance",
        python_callable = log_provenance,
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
    #       │                        │
    #  [no baseline]            [baseline exists]
    #       │                        │
    #  initial_train       evaluate_current_model
    #       │                        │
    #       │                  drift_check
    #       │                        │
    #       │               branch_on_drift
    #       │                  ├── retrain_model
    #       │                  └── skip_training
    #       │                        │
    #       └────────────────────────┘
    #                                │
    #                     update_faiss_index
    #                                │
    #                       log_provenance
    #                                │
    #                      pipeline_complete
    #
    (
        t_detect
        >> t_dvc_pull
        >> t_augment
        >> t_replay
        >> t_branch_baseline
        >> [t_initial_train, t_drift]
    )

    t_drift >> t_branch_drift >> [t_retrain, t_skip]

    [t_initial_train, t_retrain, t_skip] >> t_faiss >> t_provenance >> t_done