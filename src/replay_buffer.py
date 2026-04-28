"""
src/replay_buffer.py
------------------------------
Builds a replay buffer for continual learning by fetching baselines from MLflow.
"""

import json
import logging
import argparse
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import mlflow

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# MLFLOW BASELINE LOADER

def load_baseline_from_mlflow(run_id: str, tracking_uri: str) -> dict:
    """
    Downloads the drift_reference.json from MLflow to use as a baseline for 
    hard negative sampling.
    """
    mlflow.set_tracking_uri(tracking_uri)
    log.info(f"Fetching baseline from MLflow run: {run_id}")
    
    # Download the json artifact directly
    ref_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, 
        artifact_path="drift/drift_reference.json"
    )
    
    with open(ref_path) as f:
        return json.load(f)

# RANDOM SAMPLE FROM OLD DATA

def sample_random_old(old_csv_path: str, n_images: int, seed: int = 42) -> pd.DataFrame:
    df = pd.read_csv(old_csv_path)
    unique_ids = df["original_index"].unique()
    rng = np.random.default_rng(seed)
    n_pick = min(n_images, len(unique_ids))
    chosen = rng.choice(unique_ids, size=n_pick, replace=False)
    result = df[df["original_index"].isin(chosen)].copy()
    log.info("Random replay: %d images → %d rows", n_pick, len(result))
    return result

# HARD NEGATIVES FROM OLD DATA (Sync'd with MLflow format)

def get_hard_negatives(
    old_csv_path: str,
    baseline_data: dict,  # Now accepts the dict directly
    n_images: int,
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
    device: str = "cpu",
) -> pd.DataFrame:
    try:
        import torch
        import open_clip
    except ImportError:
        log.warning("open_clip not available — falling back to random")
        return sample_random_old(old_csv_path, n_images, seed=99)

    df = pd.read_csv(old_csv_path)
    canon = df[df["method"] == "title_clean"].drop_duplicates("original_index")
    if len(canon) == 0:
        canon = df.drop_duplicates("original_index")

    # Terminology Sync: clip_experiments.py stores centroid in ['text_embedding_stats']['dim_mean']
    if "text_embedding_stats" in baseline_data:
        centroid = np.array(baseline_data["text_embedding_stats"]["dim_mean"])
    else:
        centroid = np.array(baseline_data["dim_mean"])

    tokenizer = open_clip.get_tokenizer(model_name)
    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = model.to(device).eval()

    texts = canon["augmented_text"].tolist()
    all_embs = []
    with torch.no_grad():
        for i in range(0, len(texts), 64):
            batch = texts[i: i + 64]
            tokens = tokenizer(batch).to(device)
            emb = model.encode_text(tokens, normalize=True)
            all_embs.append(emb.cpu().numpy())

    embs = np.vstack(all_embs)
    cos_sims = embs @ centroid
    n_pick = min(n_images, len(cos_sims))
    hard_positions = np.argsort(cos_sims)[:n_pick]
    hard_orig_ids = canon.iloc[hard_positions]["original_index"].values

    result = df[df["original_index"].isin(hard_orig_ids)].copy()
    log.info("Hard negatives: %d images (avg cosine=%.4f) → %d rows",
             n_pick, float(cos_sims[hard_positions].mean()), len(result))
    return result

# BUILD COMBINED TRAINING SET

def build_replay_buffer(
    old_csv_path: str,
    new_df: pd.DataFrame,
    baseline_data: dict,
    replay_ratio: float = 0.40,
    seed: int = 42,
    device: str = "cpu",
    week_number: int = 0,       # BUG 7 FIX: caller passes the numeric week so
) -> pd.DataFrame:              # offset is fixed per-week, not additive over runs.
    
    
    # week_number is used to namespace replay original_index values so the same
    # real product never collides with new-data indices regardless of how many
    # times the buffer is rebuilt.

    # Offset scheme (deterministic, non-cumulative):
    #   random replay : 1_000_000 * week_number  + original_index
    #   hard negatives: 2_000_000 * week_number  + original_index

    # Any week_number > 0 guarantees no overlap with real h_index values
    # (Flipkart h_index fits comfortably below 1M).  If week_number=0 (Day-0
    # guard) no old data is mixed in anyway so the offset is never applied.
    

    if not Path(old_csv_path).exists():
        log.warning("Old CSV not found — using new data only")
        return new_df.copy()

    n_new_images = new_df["original_index"].nunique()
    n_replay = max(2, int(n_new_images * replay_ratio))
    n_random = n_replay // 2
    n_hard = n_replay - n_random

    random_df = sample_random_old(old_csv_path, n_random, seed=seed)
    hard_df = get_hard_negatives(old_csv_path, baseline_data, n_hard, device=device)

    random_base = 1_000_000 * week_number
    hard_base   = 2_000_000 * week_number

    random_df = random_df.copy()
    random_df["original_index"] = random_df["original_index"] + random_base
    hard_df = hard_df.copy()
    hard_df["original_index"] = hard_df["original_index"] + hard_base

    combined = pd.concat([new_df, random_df, hard_df], ignore_index=True)
    return combined.drop_duplicates(subset=["original_index", "augmented_text"]).reset_index(drop=True)

# MAIN

def main():
    # CLI 
    parser = argparse.ArgumentParser(
        description=(
            "Drift-gated replay buffer builder.\n\n"
            "Behaviour\n"
            "---------\n"
            "  drift NOT detected  →  write new_csv as-is to --output\n"
            "  drift detected      →  build replay buffer (old + hard negatives)\n"
            "                         and write combined CSV to --output\n\n"
            "The --output path is always data/incoming/<weekN>.csv so the\n"
            "Airflow DAG picks it up for training regardless of which branch ran."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # data paths
    parser.add_argument("--old_csv",       required=True,
                        help="Augmented CSV from the previous week (for replay sampling).")
    parser.add_argument("--new_csv",       required=True,
                        help="Augmented CSV for the current week (already processed by augment_texts.py).")
    parser.add_argument("--output",        required=True,
                        help="Destination path, e.g. data/incoming/week2.csv")
    # MLflow
    parser.add_argument("--run_id",        required=True,
                        help="MLflow run_id of the previous training run (used as drift baseline).")
    parser.add_argument("--tracking_uri",  default="sqlite:///mlflow.db")
    # replay
    parser.add_argument("--ratio",         type=float, default=0.40,
                        help="Fraction of new-data images to sample from old data (default 0.40).")
    parser.add_argument("--week_number",   type=int, default=0,
                        help="Numeric week index (e.g. 2 for week2). Used to compute a "
                             "fixed, non-cumulative index offset for replay rows so the "
                             "same product never gets two different original_index values "
                             "across pipeline runs. (BUG 7 FIX)")
    parser.add_argument("--device",        default="cpu")
    # drift weights (forwarded to compute_drift_score)
    parser.add_argument("--weight_embedding",   type=float, default=0.50,
                        help="Weight for embedding-shift signal (default 0.50).")
    parser.add_argument("--weight_recall",      type=float, default=0.25,
                        help="Weight for recall-drop signal (default 0.25).")
    parser.add_argument("--weight_consistency", type=float, default=0.25,
                        help="Weight for consistency-drop signal (default 0.25).")
    parser.add_argument("--batch_size",    type=int, default=64,
                        help="Batch size used when encoding embeddings for drift check.")
    args = parser.parse_args()
    

    # ── Imports from drift_detection (local to main to avoid circular deps) ──
    # drift_detection.load_baseline_from_mlflow returns a DriftBaseline dataclass
    # which is what compute_drift_score() expects.
    # replay_buffer.load_baseline_from_mlflow (this file) returns a plain dict
    # which is what get_hard_negatives() expects — both are kept.
    from drift_detection import (           # type: ignore
        load_baseline_from_mlflow as _load_drift_baseline,
        compute_drift_score,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # AFTER — handle Day-0 where prev_week="" in params.yaml
    if not args.old_csv or not Path(args.old_csv).exists():
        log.info(
            "No old_csv provided or file does not exist ('%s') — "
            "Day-0 path: writing new data as-is to output.",
            args.old_csv,
        )
        pd.read_csv(args.new_csv).to_csv(args.output, index=False)
        return   # or sys.exit(0) if this is at module level


    # 1. Load new data
    new_df = pd.read_csv(args.new_csv)
    log.info("New data loaded: %d rows from %s", len(new_df), args.new_csv)

    # 2. Drift check
    # Load DriftBaseline (contains centroid arrays + cfg) for compute_drift_score
    drift_baseline = _load_drift_baseline(args.run_id, args.tracking_uri)

    result = compute_drift_score(
        new_csv            = args.new_csv,
        baseline           = drift_baseline,
        device             = args.device,
        batch_size         = args.batch_size,
        weight_embedding   = args.weight_embedding,
        weight_recall      = args.weight_recall,
        weight_consistency = args.weight_consistency,
    )
    print(result.report())

    # 3. Gate on drift
    if not result.drift_detected:
        # No significant drift — pass new data through unchanged.
        # Airflow will still run augment + evaluate but will skip retraining.
        new_df.to_csv(output_path, index=False)
        log.info(
            "No drift detected (score=%.4f < threshold). "
            "Saved new data as-is to %s",
            result.drift_score, output_path,
        )
        return

    # 4. Drift detected — build replay buffer
    log.info(
        "Drift detected (score=%.4f). Building replay buffer from %s",
        result.drift_score, args.old_csv,
    )

    # Load baseline as plain dict for get_hard_negatives (needs dim_mean directly)
    baseline_dict = load_baseline_from_mlflow(args.run_id, args.tracking_uri)

    combined_df = build_replay_buffer(
        old_csv_path  = args.old_csv,
        new_df        = new_df,
        baseline_data = baseline_dict,
        replay_ratio  = args.ratio,
        device        = args.device,
        week_number   = args.week_number,   # BUG 7 FIX
    )

    combined_df.to_csv(output_path, index=False)
    log.info(
        "Replay buffer saved: %d rows (new=%d) to %s",
        len(combined_df), len(new_df), output_path,
    )


if __name__ == "__main__":
    main()