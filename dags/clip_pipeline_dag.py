from __future__ import annotations

import glob
import logging
import os
import yaml
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
    return Path(os.environ.get("DATA_DIR", "/opt/project/data"))

def _src_dir() -> Path:
    return Path(os.environ.get("SRC_DIR", "/opt/project/src"))

def _root_dir() -> Path:
    return Path(os.environ.get("ROOT_DIR", "/opt/project"))

def _meta_dir() -> Path:
    return Path(os.environ.get("META_DIR", "/opt/project/data/meta"))
    

def _tracking_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")

def _experiment() -> str:
    return os.environ.get("MLFLOW_EXPERIMENT", "clip_product_retrieval")


# ─────────────────────────────────────────────────────────────────────────────
# Week-label helpers
# ─────────────────────────────────────────────────────────────────────────────

def _week_number(stem: str) -> int:
    m = re.search(r"(\d+)$", stem)
    return int(m.group(1)) if m else 0


def _replay_exists(week_label: str) -> bool:
    return (_data_dir() / "processed" / f"{week_label}_replay.csv").exists()


def _find_previous_replay(current_week_label: str) -> str | None:
    processed_dir = _data_dir() / "processed"
    current_num   = _week_number(current_week_label)

    replays = [
        p for p in processed_dir.glob("*_replay.csv")
        if _week_number(p.stem.replace("_replay", "")) < current_num
    ]

    if not replays:
        return None

    replays.sort(key=lambda p: _week_number(p.stem.replace("_replay", "")))
    chosen = replays[-1]
    log.info("[replay_lookup] current=%s  previous_replay=%s", current_week_label, chosen)
    return str(chosen)

def _load_yaml_safe(path: Path):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")   # creates empty file
        return {}
    try:
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}
    
def _save_yaml_atomic(path: Path, data: dict):
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        yaml.safe_dump(data, f)
    tmp_path.replace(path)

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

def detect_new_data(**ctx):
    import pandas as pd

    tag = "[detect]"

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
            break

    if chosen is None:
        from airflow.exceptions import AirflowSkipException
        processed = [Path(p).stem for p in all_csvs]
        raise AirflowSkipException(
            f"All CSVs in incoming/ are already processed: {processed}. "
            "Drop a new weekly CSV into incoming/ to trigger processing."
        )

    csv_path   = Path(chosen)
    week_label = csv_path.stem

    df_peek = pd.read_csv(csv_path, nrows=5)
    log.info("%s [%s] columns=%s  peek_rows=%d",
             tag, week_label, list(df_peek.columns), len(df_peek))

    if len(df_peek) == 0:
        raise ValueError(f"{csv_path} appears empty (0 rows in first 5).")

    with open(csv_path, "rb") as _f:
        n_rows = sum(1 for _ in _f) - 1
    log.info("%s [%s] incoming_csv=%s  rows=%d", tag, week_label, csv_path, n_rows)

    ctx["ti"].xcom_push(key="incoming_csv", value=str(csv_path))
    ctx["ti"].xcom_push(key="week_label",   value=week_label)
    ctx["ti"].xcom_push(key="n_rows",       value=n_rows)

    log.info("%s [%s] ✓ Detection complete.", tag, week_label)


# ── 2. DVC pull ───────────────────────────────────────────────────────────────

def dvc_pull(**ctx):
    ti         = ctx["ti"]
    week_label = ti.xcom_pull(key="week_label", task_ids="detect_new_data")
    tag        = f"[dvc_pull][{week_label}]"
    cwd        = str(_root_dir())

    subprocess.run(
        ["git", "config", "--global", "--add", "safe.directory", cwd],
        check=False,
    )

    # ── Step 1: Git init if not initialized ──────────────────────────────────
    git_dir = Path(cwd) / ".git"
    if not git_dir.exists():
        log.info("%s .git not found — initializing git repo.", tag)
        subprocess.run(["git", "init"], cwd=cwd, check=True)
        subprocess.run(["git", "config", "user.email", "airflow@mlops.local"],
                       cwd=cwd, check=True)
        subprocess.run(["git", "config", "user.name", "Airflow"],
                       cwd=cwd, check=True)
        # initial commit so DVC has something to attach to
        subprocess.run(["git", "add", ".gitignore"], cwd=cwd, check=False)
        subprocess.run(["git", "commit", "-m", "init: git repo"],
                       cwd=cwd, check=False)
        log.info("%s ✓ Git initialized.", tag)
    else:
        log.info("%s .git already exists — skipping git init.", tag)

    # ── Step 2: DVC init if not initialized ──────────────────────────────────
    dvc_dir = Path(cwd) / ".dvc"
    if not dvc_dir.exists():
        # Only delete full_corpus.csv — it gets rebuilt by augment_texts.
        # DO NOT delete params.yaml or logs_for_revert.yaml; those are
        # written by the pipeline itself and must survive a DVC re-init.
        corpus_path = _data_dir() / "full_corpus.csv"

        deleted_any = False
        for path in [corpus_path]:
            if path.exists():
                path.unlink()
                log.warning("%s Deleted (will be rebuilt): %s", tag, path)
                deleted_any = True

        if not deleted_any:
            log.info("%s No files found to delete.", tag)
            
                    
        log.info("%s .dvc not found — initializing DVC.", tag)
        subprocess.run(["dvc", "init"], cwd=cwd, check=True)
        subprocess.run(["git", "add", ".dvc", ".dvcignore"],
                       cwd=cwd, check=False)
        subprocess.run(["git", "commit", "-m", "init: dvc"],
                       cwd=cwd, check=False)
        log.info("%s ✓ DVC initialized.", tag)
    else:
        log.info("%s .dvc already exists — skipping dvc init.", tag)

    # ── Step 3: DVC remote if not configured ─────────────────────────────────
    remote_check = subprocess.run(
        ["dvc", "remote", "list"],
        capture_output=True, text=True, cwd=cwd,
    )
    if "localremote" not in remote_check.stdout:
        log.info("%s No DVC remote found — adding localremote.", tag)
        subprocess.run(
            ["dvc", "remote", "add", "-d", "localremote", "/opt/airflow/dvc-remote"],
            cwd=cwd, check=True,
        )
        subprocess.run(["git", "add", ".dvc/config"], cwd=cwd, check=False)
        subprocess.run(["git", "commit", "-m", "init: dvc remote"],
                       cwd=cwd, check=False)
        log.info("%s ✓ DVC remote configured.", tag)
    else:
        log.info("%s DVC remote already configured.", tag)
    
    # ── Step 4: DVC-track data (idempotent — skipped if pointer exists) ──
    data_dvc = Path(cwd) / "data.dvc"
    if not data_dvc.exists():
        data_dir = _data_dir()
        data_dir.mkdir(parents=True, exist_ok=True)   # ensure dir exists before dvc add
        log.info("%s data.dvc not found — running dvc add data.", tag)
        subprocess.run(["dvc", "add", "data"], cwd=cwd, check=False)
        subprocess.run(["git", "add", "data.dvc", ".gitignore"],
                       cwd=cwd, check=False)
        subprocess.run(["git", "commit", "-m", "init: track data via DVC"],
                       cwd=cwd, check=False)
        log.info("%s \u2713 data tracked by DVC.", tag)
    else:
        log.info("%s data.dvc already exists — skipping dvc add.", tag)

    # ── Step 5: DVC pull (existing logic) ────────────────────────────────────
    result = subprocess.run(
        ["dvc", "pull", "--force"],
        capture_output=True, text=True,
        cwd=cwd,
    )
    if result.stdout:
        log.info("%s stdout:\n%s", tag, result.stdout[-1000:])
    if result.stderr:
        log.info("%s stderr:\n%s", tag, result.stderr[-500:])
    if result.returncode != 0:
        log.warning(
            "%s dvc pull exited with code %d — may be first run or no remote configured.",
            tag, result.returncode,
        )
    else:
        log.info("%s dvc pull succeeded.", tag)
        

# ── 3. Augment texts ──────────────────────────────────────────────────────────

def augment_texts(**ctx):
    """
    Produce data/processed/{week_label}_augmented.csv from the raw incoming CSV,
    then merge it into data/full_corpus.csv.
    """
    import sys
    import pandas as pd
    sys.path.insert(0, str(_src_dir()))
    from augment_texts import build_augmented_dataset

    ti         = ctx["ti"]
    incoming   = ti.xcom_pull(key="incoming_csv", task_ids="detect_new_data")
    week_label = ti.xcom_pull(key="week_label",   task_ids="detect_new_data")
    tag        = f"[augment][{week_label}]"

    out_path = _data_dir() / "processed" / f"{week_label}_augmented.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("%s Augmenting  %s  →  %s", tag, incoming, out_path)

    df_aug = build_augmented_dataset(
        csv_path  = incoming,
        n_samples = 4000,
        seed      = 42,
    )
    df_aug.to_csv(out_path, index=False)

    log.info(
        "%s ✓ Augmented: %d rows, %d unique images  saved to %s",
        tag, len(df_aug), df_aug["original_index"].nunique(), out_path,
    )

    corpus_path = _data_dir() / "full_corpus.csv"

    # ── Week-1 guard: if this is week 1, always start the corpus fresh —
    #    ignore whatever full_corpus.csv contains (stale data from a previous
    #    week-1 run).  We overwrite it below via the atomic tmp rename, so no
    #    delete/unlink is needed (avoids PermissionError on root-owned files).
    if _week_number(week_label) == 1:
        if corpus_path.exists():
            log.warning(
                "%s Week-1 detected and full_corpus.csv already existed — "
                "ignoring stale corpus to avoid duplicate week-1 data.", tag
            )
        log.info("%s Week-1: initialising corpus from current week only.", tag)
        df_corpus = df_aug.copy()
    elif corpus_path.exists():
        df_prev = pd.read_csv(corpus_path)
        log.info("%s Existing corpus: %d rows", tag, len(df_prev))
        df_corpus = pd.concat([df_prev, df_aug], ignore_index=True)
    else:
        log.info("%s No existing corpus — initialising with current week.", tag)
        df_corpus = df_aug.copy()

    # Write directly — avoids PermissionError from rename/unlink on
    # root-owned files. to_csv opens and truncates in place which only
    # requires directory write permission, not file ownership.
    df_corpus.to_csv(corpus_path, index=False)

    log.info(
        "%s ✓ Corpus updated: %d rows, %d unique images → %s",
        tag, len(df_corpus), df_corpus["original_index"].nunique(), corpus_path,
    )

    ti.xcom_push(key="new_csv_path",    value=str(out_path))
    ti.xcom_push(key="corpus_csv_path", value=str(corpus_path))


# ── 4. Build replay buffer ────────────────────────────────────────────────────

def build_replay_buffer(**ctx):
    import sys
    import os
    import yaml
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

    params_path = _root_dir() / "params.yaml"

    replay_path = _data_dir() / "processed" / f"{week_label}_replay.csv"
    replay_path.parent.mkdir(parents=True, exist_ok=True)

    new_df = pd.read_csv(new_csv)
    log.info("%s new augmented data: %d rows, %d images",
             tag, len(new_df), new_df["original_index"].nunique())

    # 🔹 Read baseline run_id from params.yaml
    if not os.path.exists(params_path):
        log.info("%s No params.yaml found — Day-0 path.", tag)
        baseline_run_id = None
    else:
        with open(params_path) as f:
            params = yaml.safe_load(f) or {}

        baseline_run_id = params.get("run_id")

    # Day-0 condition
    if baseline_run_id is None:
        log.info(
            "%s No baseline run_id found — Day-0 path. "
            "Writing augmented data directly (no replay mixing).",
            tag,
        )
        new_df.to_csv(replay_path, index=False)

        log.info("%s ✓ Day-0 replay written: %d rows → %s",
                 tag, len(new_df), replay_path)

        ti.xcom_push(key="replay_csv_path", value=str(replay_path))
        ti.xcom_push(key="baseline_run_id", value=None)
        return

    log.info("%s baseline_run_id=%s", tag, baseline_run_id)

    prev_replay = _find_previous_replay(week_label)

    if prev_replay is None:
        log.warning(
            "%s baseline run_id exists but no previous replay CSV found. "
            "Falling back to new data only.",
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

        baseline_data = load_baseline_from_mlflow(
            baseline_run_id, _tracking_uri()
        )

        combined = _build(
            old_csv_path  = prev_replay,
            new_df        = new_df,
            baseline_data = baseline_data,
            replay_ratio  = 0.20,
            week_number   = _week_number(week_label),
        )

    combined.to_csv(replay_path, index=False)

    log.info(
        "%s ✓ Replay buffer written: %d rows, %d unique images → %s",
        tag, len(combined), combined["original_index"].nunique(), replay_path,
    )

    ti.xcom_push(key="replay_csv_path", value=str(replay_path))
    ti.xcom_push(key="baseline_run_id", value=baseline_run_id)
    

# ── 5. Branch on baseline ─────────────────────────────────────────────────────

def branch_on_baseline(**ctx):
    ti = ctx["ti"]

    week_label = ti.xcom_pull(key="week_label", task_ids="detect_new_data")
    baseline_run_id = ti.xcom_pull(
        key="baseline_run_id",
        task_ids="build_replay_buffer"
    )

    tag = f"[branch][{week_label}]"

    if baseline_run_id is not None:
        log.info("%s Baseline found (run_id=%s) → drift_check",
                 tag, baseline_run_id)
        return "drift_check"

    log.info("%s No baseline → initial_train (Day-0)", tag)
    return "initial_train"

# ── 6a. Initial train (Day-0) ─────────────────────────────────────────────────


def initial_train(**ctx):
    import sys
    sys.path.insert(0, str(_src_dir()))
    from clip_experiments import Config, run_linear_probe
    import mlflow

    ti         = ctx["ti"]
    replay_csv = ti.xcom_pull(key="replay_csv_path", task_ids="build_replay_buffer")
    week_label = ti.xcom_pull(key="week_label",      task_ids="detect_new_data")
    tag        = f"[initial_train][{week_label}]"

    log.info("%s Training on replay CSV: %s", tag, replay_csv)

    # ── Week-1 / Day-0: clear params.yaml and logs_for_revert.yaml so
    #    log_provenance starts from a clean slate.  The corpus deletion
    #    (to avoid double-stacking week-1 data) is handled in augment_texts.
    #    On Week 2+ initial_train is never called, so this is safe.
    params_path = _root_dir() / "params.yaml"
    revert_path = _meta_dir() / "logs_for_revert.yaml"

    params_path.parent.mkdir(parents=True, exist_ok=True)
    revert_path.parent.mkdir(parents=True, exist_ok=True)
    params_path.write_text("")
    revert_path.write_text("")
    log.info("%s Week-1: cleared params.yaml and logs_for_revert.yaml.", tag)

    mlflow.set_tracking_uri(_tracking_uri())
    mlflow.set_experiment(_experiment())

    cfg = Config(
        csv_path            = replay_csv,
        image_cache_dir     = str(_data_dir() / "image_cache"),
        mode                = "linear_probe",
        epochs              = 2,
        batch_size          = 32,
        lr                  = 1e-3,
        week_label          = week_label,
        mlflow_tracking_uri = _tracking_uri(),
        mlflow_experiment   = _experiment(),
        mlflow_run_name     = f"initial_train_{week_label}",
    )

    with mlflow.start_run(run_name=cfg.mlflow_run_name) as run:
        new_run_id = run.info.run_id
        mlflow.set_tag("mode",       cfg.mode)
        mlflow.set_tag("stage",      "initial_train")
        mlflow.set_tag("week_label", week_label)
        mlflow.set_tag("pipeline",   "clip_product_retrieval")
        log.info("%s MLflow run_id=%s", tag, new_run_id)
        final_metrics = run_linear_probe(cfg, device="cpu", run_id=new_run_id)

    # ── Write the initial entry into logs_for_revert.yaml and SAVE it.
    #    (Bug fix: the original code built the dict but never called
    #     _save_yaml_atomic, so it was thrown away every time.)
    revert_data = _load_yaml_safe(revert_path)
    if week_label not in revert_data:
        revert_data[week_label] = {}
    revert_data[week_label].update({
        "mlflow_run_id": new_run_id,
        "stage":         "initial_train",
    })
    _save_yaml_atomic(revert_path, revert_data)
    log.info("%s ✓ logs_for_revert.yaml written for %s.", tag, week_label)

    log.info(
        "%s ✓ Initial train complete. run_id=%s  Recall@1=%.4f",
        tag, new_run_id, final_metrics.get("Recall@1", 0.0),
    )
    ti.xcom_push(key="new_run_id", value=new_run_id)

# ── 6b. Drift check ───────────────────────────────────────────────────────────

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


# ── 6c. Branch on drift ───────────────────────────────────────────────────────

def branch_on_drift(**ctx):
    ti         = ctx["ti"]
    week_label = ti.xcom_pull(key="week_label",     task_ids="detect_new_data")
    detected   = ti.xcom_pull(key="drift_detected", task_ids="drift_check")
    score      = ti.xcom_pull(key="drift_score",    task_ids="drift_check")
    tag        = f"[branch_drift][{week_label}]"

    decision = "retrain_model" if detected else "skip_training"
    log.info("%s drift_detected=%s  score=%.4f  → %s", tag, detected, score or 0.0, decision)
    return decision


# ── 6c-i. Retrain model ───────────────────────────────────────────────────────


def retrain_model(**ctx):
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
        epochs              = 2,
        batch_size          = 32,
        lr                  = 1e-3,
        week_label          = week_label,
        mlflow_tracking_uri = _tracking_uri(),
        mlflow_experiment   = _experiment(),
        mlflow_run_name     = f"retrain_linear_probe_{week_label}",
    )

    with mlflow.start_run(run_name=cfg.mlflow_run_name) as run:
        new_run_id = run.info.run_id
        mlflow.set_tag("mode", cfg.mode)
        mlflow.set_tag("stage",      "retrain")
        mlflow.set_tag("week_label", week_label)
        mlflow.set_tag("pipeline",   "clip_product_retrieval")
        log.info("%s MLflow run_id=%s", tag, new_run_id)
        final_metrics = run_linear_probe(cfg, device="cpu", run_id=new_run_id)

    log.info(
        "%s ✓ Retrain complete. run_id=%s  Recall@1=%.4f.",
        tag, new_run_id, final_metrics.get("Recall@1", 0.0),
    )
    ti.xcom_push(key="new_run_id", value=new_run_id)


# ── 6c-ii. Skip training ──────────────────────────────────────────────────────

def skip_training(**ctx):
    ti         = ctx["ti"]
    week_label = ti.xcom_pull(key="week_label",      task_ids="detect_new_data")
    score      = ti.xcom_pull(key="drift_score",     task_ids="drift_check")
    baseline_run_id = ti.xcom_pull(key="baseline_run_id", task_ids="build_replay_buffer")
    tag = f"[skip_training][{week_label}]"
    log.info("%s No significant drift (score=%.4f). Retraining skipped.", tag, score or 0.0)
    if baseline_run_id:
        ti.xcom_push(key="new_run_id", value=baseline_run_id)
        log.info("%s Forwarding baseline run_id=%s to log_provenance.", tag, baseline_run_id)
    else:
        log.warning("%s No baseline_run_id available — provenance will be skipped.", tag)


# ── 7. Update FAISS index ─────────────────────────────────────────────────────
def update_faiss_index(**ctx):
    import sys
    import json
    import numpy as np
    sys.path.insert(0, str(_src_dir()))

    ti         = ctx["ti"]
    week_label = ti.xcom_pull(key="week_label",      task_ids="detect_new_data")
    corpus_csv = ti.xcom_pull(key="corpus_csv_path", task_ids="augment_texts")
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
        embed_full_corpus,
    )
    import mlflow
    import pandas as pd

    run_id_from_initial = ti.xcom_pull(key="new_run_id", task_ids="initial_train")
    run_id_from_retrain = ti.xcom_pull(key="new_run_id", task_ids="retrain_model")
    run_id_from_skip    = ti.xcom_pull(key="new_run_id", task_ids="skip_training")
    active_run_id       = run_id_from_initial or run_id_from_retrain or run_id_from_skip

    log.info(
        "%s run_ids — initial=%s  retrain=%s  skip=%s  active=%s",
        tag, run_id_from_initial, run_id_from_retrain, run_id_from_skip, active_run_id,
    )

    mlflow.set_tracking_uri(_tracking_uri())

    cfg = Config(
        image_cache_dir     = str(_data_dir() / "image_cache"),
        mlflow_tracking_uri = _tracking_uri(),
        week_label          = week_label,
        combined_csv        = corpus_csv or str(_data_dir() / "full_corpus.csv"),
        corpus_emb_dir      = str(_data_dir() / "corpus_embeddings"),
    )
    device = "cpu"

    if active_run_id:
        try:
            model, probe, preprocess, tokenizer = load_linear_probe_from_mlflow(
                active_run_id, cfg, device)
            log.info("%s Loaded linear-probe model from run %s", tag, active_run_id)
        except Exception as e:
            log.warning(
                "%s Could not load linear-probe (%s) — falling back to zero-shot.", tag, e)
            model, preprocess, tokenizer = load_zero_shot_model(cfg, device)
            probe = None
    else:
        log.info("%s No active run_id — using zero-shot model.", tag)
        model, preprocess, tokenizer = load_zero_shot_model(cfg, device)
        probe = None

    if not Path(cfg.combined_csv).exists():
        log.warning(
            "%s full_corpus.csv not found at %s — skipping embed + FAISS.",
            tag, cfg.combined_csv,
        )
        return

    active = mlflow.active_run()
    if active:
        log.info(
            "%s Found leaked MLflow run %s — closing before re-opening.",
            tag, active.info.run_id,
        )
        mlflow.end_run()

    _run_ctx = (
        mlflow.start_run(run_id=active_run_id)
        if active_run_id
        else mlflow.start_run(run_name=f"zero_shot_faiss_{week_label}")
    )
    with _run_ctx:
        
        embed_full_corpus(
            model          = model,
            probe_head     = probe,
            cfg            = cfg,
            device         = device,
            run_id         = active_run_id or "zero_shot",
            combined_csv   = cfg.combined_csv,
            local_save_dir = cfg.corpus_emb_dir,
        )

        emb_dir = Path(cfg.corpus_emb_dir)
        npy_img = emb_dir / "corpus_image_embeddings.npy"
        npy_txt = emb_dir / "corpus_text_embeddings.npy"
        npy_idx = emb_dir / "corpus_orig_idx.npy"

        img_np  = np.load(str(npy_img))
        txt_np  = np.load(str(npy_txt))
        idx_np  = np.load(str(npy_idx))

        orig_idx_keys = [str(k) for k in idx_np.tolist()]

        log.info(
            "%s Corpus embeddings loaded: N_unique=%d N_rows=%d D=%d",
            tag, len(idx_np), len(txt_np), img_np.shape[1],
        )

        # ── Sanity: both npy arrays must be N_unique × D (avg done in embed_full_corpus)
        assert img_np.shape == txt_np.shape, (
            f"{tag} img/txt shape mismatch: {img_np.shape} vs {txt_np.shape}. "
            "Re-run embed_full_corpus."
        )
        assert len(idx_np) == img_np.shape[0], (
            f"{tag} idx_np length {len(idx_np)} != img rows {img_np.shape[0]}"
        )

        dim     = img_np.shape[1]
        faiss_dir = _data_dir() / "faiss"
        faiss_dir.mkdir(parents=True, exist_ok=True)

        # ── Image FAISS index ─────────────────────────────────────────────────
        idx_img = faiss.IndexFlatIP(dim)
        idx_img.add(img_np)
        faiss.write_index(idx_img, str(faiss_dir / f"{week_label}_index_img.bin"))
        log.info("%s ✓ image FAISS: %d vectors  dim=%d", tag, idx_img.ntotal, dim)

        # ── Text FAISS index ──────────────────────────────────────────────────
        idx_txt = faiss.IndexFlatIP(dim)
        idx_txt.add(txt_np)
        faiss.write_index(idx_txt, str(faiss_dir / f"{week_label}_index_txt.bin"))
        log.info("%s ✓ text  FAISS: %d vectors  dim=%d", tag, idx_txt.ntotal, dim)

        # ── Shared meta (one entry per unique product, aligned to idx_np) ─────
        df_corpus = pd.read_csv(cfg.combined_csv)
        # collect ALL augmented texts per product so the UI can show them
        all_texts = (
            df_corpus.groupby("original_index")["augmented_text"]
            .apply(list)
            .to_dict()
        )
        df_lookup = (
            df_corpus
            .sort_values("original_index")
            .drop_duplicates("original_index", keep="first")
            .set_index("original_index")
        )

        meta = []
        n_missing = 0
        for k in orig_idx_keys:
            oid = int(k)
            if oid in df_lookup.index:
                row = df_lookup.loc[oid]
                meta.append({
                    "original_index":  oid,
                    "image_url":       str(row.get("image_url", "")),
                    "title":           str(row.get("augmented_text", "")),
                    "augmented_texts": all_texts.get(oid, []),
                    "category_2":      str(row.get("category_2", "")),
                    "category_3":      str(row.get("category_3", "")),
                })
            else:
                n_missing += 1
                meta.append({
                    "original_index":  oid,
                    "image_url":       "",
                    "title":           "",
                    "augmented_texts": [],
                    "category_2":      "",
                    "category_3":      "",
                })

        if n_missing:
            log.warning("%s %d FAISS vectors have no matching corpus row.", tag, n_missing)

        if len(meta) != idx_img.ntotal:
            raise RuntimeError(
                f"{tag} meta length {len(meta)} != FAISS ntotal {idx_img.ntotal} — "
                "ordering is broken. Do not serve this index."
            )

        (faiss_dir / "meta.json").write_text(json.dumps(meta, indent=2))

        log.info(
            "%s ✓ FAISS done: %d unique products  dim=%d  path=%s",
            tag, len(meta), dim, faiss_dir,
        )

        mlflow.log_artifact(str(faiss_dir / "index_img.bin"), artifact_path="faiss")
        mlflow.log_artifact(str(faiss_dir / "index_txt.bin"), artifact_path="faiss")
        mlflow.log_artifact(str(faiss_dir / "meta.json"),     artifact_path="faiss")
        mlflow.log_artifact(str(npy_img),                     artifact_path="embeddings/corpus")
        mlflow.log_artifact(str(npy_txt),                     artifact_path="embeddings/corpus")
        mlflow.log_artifact(str(npy_idx),                     artifact_path="embeddings/corpus")
        mlflow.log_metric("corpus_size", len(meta))

    log.info("%s FAISS + corpus embeddings logged to MLflow run %s",
             tag, active_run_id or "zero_shot")



# ── 7b. DVC-track image_cache ────────────────────────────────────────────────

def dvc_add_data(**ctx):
    """
    Track data/image_cache/ as a plain DVC-managed directory (outside the
    pipeline stages).  This runs after every FAISS update so newly cached
    images are committed to DVC storage.

    Steps:
      1. dvc add data          → writes data.dvc
      2. git add data.dvc .gitignore
      3. git commit  (non-fatal if nothing changed)
      4. dvc push    (non-fatal — may have no remote configured)
    """
    ti         = ctx["ti"]
    week_label = ti.xcom_pull(key="week_label", task_ids="detect_new_data")
    tag        = f"[dvc_data][{week_label}]"
    cwd        = str(_root_dir())

    data_dir = _data_dir()
    if not data_dir.exists():
        log.info("%s data dir does not exist yet — skipping DVC add.", tag)
        return

    # Step 1: dvc add
    result = subprocess.run(
        ["dvc", "add", "data"],
        capture_output=True, text=True, cwd=cwd,
    )
    log.info("%s dvc add stdout:\n%s", tag, result.stdout[-500:])
    if result.returncode != 0:
        log.warning("%s dvc add exited %d:\n%s", tag, result.returncode, result.stderr[-300:])

    # Step 2: git add the .dvc pointer + .gitignore update
    subprocess.run(
        ["git", "add", "data.dvc", ".gitignore"],
        cwd=cwd, check=False,
    )

    # Step 3: git commit (non-fatal if nothing changed)
    commit_result = subprocess.run(
        ["git", "commit", "-m", f"dvc: track data after {week_label}"],
        capture_output=True, text=True, cwd=cwd,
    )
    if commit_result.returncode == 0:
        log.info("%s ✓ git commit: data.dvc", tag)
    else:
        log.info("%s git commit — nothing new to commit (ok).", tag)

    # Step 4: dvc push (non-fatal)
    push_result = subprocess.run(
        ["dvc", "push"],
        capture_output=True, text=True, cwd=cwd,
    )
    if push_result.returncode == 0:
        log.info("%s ✓ dvc push data complete.", tag)
    else:
        log.warning("%s dvc push exited %d (non-fatal):\n%s",
                    tag, push_result.returncode, push_result.stderr[-300:])


# ── 8. Log provenance ─────────────────────────────────────────────────────────
def log_provenance(**ctx):
    import mlflow
    import json

    ti         = ctx["ti"]
    week_label = ti.xcom_pull(key="week_label",   task_ids="detect_new_data")
    incoming   = ti.xcom_pull(key="incoming_csv", task_ids="detect_new_data")
    tag        = f"[provenance][{week_label}]"

    run_id_from_initial = ti.xcom_pull(key="new_run_id", task_ids="initial_train")
    run_id_from_retrain = ti.xcom_pull(key="new_run_id", task_ids="retrain_model")
    run_id_from_skip    = ti.xcom_pull(key="new_run_id", task_ids="skip_training")
    new_run_id          = run_id_from_initial or run_id_from_retrain or run_id_from_skip

    params_path = _root_dir() / "params.yaml"
    revert_path = _meta_dir() / "logs_for_revert.yaml"

    if not new_run_id:
        log.info("%s No new model run this week — skipping provenance log.", tag)
        return

    import hashlib
    dvc_lock_path = _root_dir() / "dvc.lock"
    if os.path.exists(dvc_lock_path):
        with open(dvc_lock_path, "rb") as f:
            dvc_lock_hash = hashlib.md5(f.read()).hexdigest()
    else:
        dvc_lock_hash = "no-lock-file"

    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True, cwd=_root_dir(),
        ).strip()
    except Exception:
        git_commit = "unknown"

    try:
        dvc_status = subprocess.check_output(
            ["dvc", "status", "--cloud"],
            text=True, cwd=_root_dir(),
        ).strip()
    except Exception:
        dvc_status = "unknown"

    # ------------------------------------------------------------------
    # Step 1: snapshot the CURRENT params.yaml into logs_for_revert.yaml
    # BEFORE overwriting params.yaml — so we have a "what was true last
    # week" record.  On Day-0 params.yaml is empty so this is a no-op.
    # ------------------------------------------------------------------
    try:
        revert_path.parent.mkdir(parents=True, exist_ok=True)
        revert_data  = _load_yaml_safe(revert_path)
        old_params   = _load_yaml_safe(params_path)   # snapshot of previous week

        if week_label not in revert_data:
            revert_data[week_label] = {}

        # Merge the old params snapshot + new git/dvc info into this week's entry.
        # If initial_train already wrote a partial entry (mlflow_run_id, stage),
        # .update() will extend it rather than overwrite it.
        revert_data[week_label].update({
            "mlflow_run_id":    new_run_id,
            "git_commit":       git_commit,
            "dvc_lock_md5":     dvc_lock_hash,
            # snapshot of what params.yaml contained at the START of this week's run
            "params_snapshot":  old_params,
        })

        _save_yaml_atomic(revert_path, revert_data)
        log.info("%s ✓ logs_for_revert.yaml updated for %s.", tag, week_label)

    except Exception as e:
        log.warning("%s logs_for_revert update failed (non-fatal): %s", tag, e)

    # ------------------------------------------------------------------
    # Step 2: overwrite params.yaml with THIS week's values.
    # Bug fix: was `run_id` (NameError) — corrected to `new_run_id`.
    # ------------------------------------------------------------------
    try:
        params = _load_yaml_safe(params_path)

        params.update({
            "week":                week_label,
            "prev_week":           params.get("week", ""),
            "run_id":              new_run_id,          # ← fixed (was bare `run_id`)
            "mlflow_tracking_uri": _tracking_uri(),
            "week_number":         _week_number(week_label),
        })

        _save_yaml_atomic(params_path, params)
        log.info("%s ✓ params.yaml updated for %s  run_id=%s", tag, week_label, new_run_id)

    except Exception as e:
        log.warning("%s params.yaml update failed (non-fatal): %s", tag, e)

    # ------------------------------------------------------------------
    # Step 3: git commit both files + tag + dvc push
    # ------------------------------------------------------------------
    try:
        subprocess.run([
            "git", "add",
            "dvc.lock",
            "params.yaml",
            "data/meta/logs_for_revert.yaml",
            "dags/clip_pipeline_dag.py",
            "docker/",
            "monitoring/",
            "src/",
            "scripts/",
            "routes/",
            "reproduce.sh",
            "verify_search.py",
            "docker-compose.yaml",
        ], cwd=_root_dir(), check=False)

        subprocess.run(["git", "commit", "-m",
                        f"chore: {week_label} pipeline run {new_run_id[:8]}"],
                       cwd=_root_dir(), check=False)

        tag_name = f"{week_label}/run-{new_run_id[:8]}"
        subprocess.run(["git", "tag", "-a", tag_name, "-m",
                        f"MLflow run_id={new_run_id}"],
                       cwd=_root_dir(), check=False)

        subprocess.run(["dvc", "push"], cwd=_root_dir(), check=False)

        log.info("%s ✓ Git tag=%s, dvc push done.", tag, tag_name)
    except Exception as e:
        log.warning("%s Git/DVC commit step failed (non-fatal): %s", tag, e)

    # ------------------------------------------------------------------
    # Step 4: log provenance artifact to MLflow
    # ------------------------------------------------------------------
    provenance = {
        "week_label":      week_label,
        "incoming_csv":    incoming,
        "git_commit":      git_commit,
        "model_run_id":    new_run_id,
        "augmented_csv":   str(_data_dir() / "processed" / f"{week_label}_augmented.csv"),
        "replay_csv":      str(_data_dir() / "processed" / f"{week_label}_replay.csv"),
        "faiss_img_index": str(_data_dir() / "faiss" / f"{week_label}_index_img.bin"),
        "faiss_txt_index": str(_data_dir() / "faiss" / f"{week_label}_index_txt.bin"),
        "faiss_meta":      str(_data_dir() / "faiss" / "meta.json"),
        "dvc_status":      dvc_status,
    }

    log.info("%s Provenance record:\n%s", tag, json.dumps(provenance, indent=2))

    mlflow.set_tracking_uri(_tracking_uri())
    active = mlflow.active_run()
    if active:
        log.info("[provenance] Found leaked run %s — closing.", active.info.run_id)
        mlflow.end_run()

    with mlflow.start_run(run_id=new_run_id):
        mlflow.log_dict(provenance, f"provenance/{week_label}_provenance.json")
        mlflow.set_tag("week_label",   week_label)
        mlflow.set_tag("git_commit",   git_commit)
        mlflow.set_tag("pipeline",     "clip_product_retrieval")
        mlflow.set_tag("dvc_lock_md5", dvc_lock_hash)

    log.info(
        "%s ✓ Provenance logged to MLflow run %s (git=%s).",
        tag, new_run_id, git_commit[:8],
    )



# ── 9. Register model ─────────────────────────────────────────────────────────

def register_model(**ctx):
    import sys
    sys.path.insert(0, str(_src_dir()))

    ti         = ctx["ti"]
    week_label = ti.xcom_pull(key="week_label",   task_ids="detect_new_data")
    tag        = f"[register][{week_label}]"

    run_id_from_initial = ti.xcom_pull(key="new_run_id", task_ids="initial_train")
    run_id_from_retrain = ti.xcom_pull(key="new_run_id", task_ids="retrain_model")
    run_id_from_skip    = ti.xcom_pull(key="new_run_id", task_ids="skip_training")
    active_run_id       = run_id_from_initial or run_id_from_retrain or run_id_from_skip

    if not active_run_id:
        log.warning("%s No run_id — skipping model registration.", tag)
        return

    import mlflow
    mlflow.set_tracking_uri(_tracking_uri())
    client = mlflow.MlflowClient()
    run    = client.get_run(active_run_id)
    mode   = run.data.tags.get("mode", "zero_shot")

    cmd = [
        "python", str(_src_dir() / "register_clip_model.py"),
        "--run_id",       active_run_id,
        "--mode",         mode,
        "--faiss_index_img",  str(_data_dir() / "faiss" / f"{week_label}_index_img.bin"),
        "--faiss_index_txt",  str(_data_dir() / "faiss" / f"{week_label}_index_txt.bin"),
        "--faiss_meta",   str(_data_dir() / "faiss" / "meta.json"),
        "--tracking_uri", _tracking_uri(),
    ]

    log.info("%s Running: %s", tag, " ".join(cmd))
    subprocess.run(cmd, check=True)
    log.info("%s Model registered to MLflow Model Registry.", tag)


# ── 10. Start serving containers

def start_serving(**ctx):
    import docker
    import subprocess

    ti         = ctx["ti"]
    week_label = ti.xcom_pull(key="week_label", task_ids="detect_new_data")
    tag        = f"[serving][{week_label}]"

    client = docker.from_env()

    for container_name in ["mlflow_serve", "search_api"]:
        try:
            container = client.containers.get(container_name)

            if container.status == "running":
                log.info("%s %s running → restarting", tag, container_name)
                container.restart(timeout=30)

            else:
                log.info("%s %s exists but stopped → starting", tag, container_name)
                container.start()

        except docker.errors.NotFound:
            log.warning("%s %s not found → creating via docker compose", tag, container_name)

            subprocess.run(
                ["docker-compose", "up", "-d", "mlflow-serve", "search-api"],
                check=True
            )

        except Exception as e:
            log.error("%s Failed for %s: %s", tag, container_name, e)
            raise
# ── 11. Reload FAISS in search-api ────────────────────────────────────────────

def reload_faiss_index(**ctx):
    import time
    import requests as req

    ti         = ctx["ti"]
    week_label = ti.xcom_pull(key="week_label", task_ids="detect_new_data")
    tag        = f"[reload_faiss][{week_label}]"

    search_api_url = os.environ.get("SEARCH_API_URL", "http://search_api:8000")

    # Wait for search-api to be healthy after restart (max 3 min)
    for attempt in range(18):
        try:
            r = req.get(f"{search_api_url}/health", timeout=5)
            if r.status_code == 200:
                log.info("%s search-api is healthy after %d attempts.", tag, attempt + 1)
                break
        except Exception:
            pass
        log.info("%s Waiting for search-api... attempt %d/18", tag, attempt + 1)
        time.sleep(10)
    else:
        raise RuntimeError(f"{tag} search-api did not become healthy in time.")

    # Trigger reload
    r = req.post(f"{search_api_url}/reload", timeout=60)
    r.raise_for_status()
    data = r.json()
    log.info("%s ✓ FAISS reloaded — vector_count=%s", tag, data.get("vector_count"))



# ─────────────────────────────────────────────────────────────────────────────
# DAG DEFINITION
# ─────────────────────────────────────────────────────────────────────────────

with DAG(
    dag_id                  = "clip_product_retrieval_pipeline",
    default_args            = default_args,
    description             = "CLIP continual-learning pipeline with drift detection",
    schedule                = timedelta(weeks=1),
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

    t_initial_train = PythonOperator(
        task_id           = "initial_train",
        python_callable   = initial_train,
        execution_timeout = timedelta(hours=4),
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

    t_faiss = PythonOperator(
        task_id           = "update_faiss_index",
        python_callable   = update_faiss_index,
        trigger_rule      = TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
        execution_timeout = timedelta(hours=1),
    )

    t_dvc_cache = PythonOperator(
        task_id         = "dvc_add_data",
        python_callable = dvc_add_data,
        trigger_rule    = TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )

    t_provenance = PythonOperator(
        task_id         = "log_provenance",
        python_callable = log_provenance,
        trigger_rule    = TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )

    t_register = PythonOperator(
        task_id         = "register_model",
        python_callable = register_model,
        trigger_rule    = TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )

    t_done = EmptyOperator(
        task_id      = "pipeline_complete",
        trigger_rule = TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )

    t_start_serving = PythonOperator(
        task_id         = "start_serving",
        python_callable = start_serving,
        trigger_rule    = TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )

    t_reload_faiss = PythonOperator(
        task_id           = "reload_faiss_index",
        python_callable   = reload_faiss_index,
        trigger_rule      = TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
        execution_timeout = timedelta(minutes=5),
    )

    (
        t_detect
        >> t_dvc_pull
        >> t_augment
        >> t_replay
        >> t_branch_baseline
        >> [t_initial_train, t_drift]
    )

    t_drift >> t_branch_drift >> [t_retrain, t_skip]

    [t_initial_train, t_retrain, t_skip] >> t_faiss >> [t_dvc_cache, t_provenance]
    t_dvc_cache >> t_provenance
    t_provenance >> t_register >> t_start_serving >> t_reload_faiss >> t_done