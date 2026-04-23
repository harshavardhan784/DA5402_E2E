"""
src/models/drift_detection.py
-------------------------------
Detect data drift on new product batches relative to a past training run.

Drift score formula (0.0 = no drift, 1.0 = maximum drift):

  drift_score = 0.50 * embedding_signal
              + 0.25 * recall_signal
              + 0.25 * consistency_signal

  embedding_signal  (50 %)
  ─────────────────────────
  Average of two sub-signals, each in [0, 1]:
    • centroid_shift  = 1 − cosine_similarity(new_centroid, baseline_centroid)
    • pairwise_delta  = clamp(|new_pairwise_mean − baseline_pairwise_mean| / 0.20, 0, 1)
      (normalised so a 0.20 absolute change maps to a full signal of 1.0)

  recall_signal  (25 %)
  ──────────────────────
  Average relative drop across Recall@1 and Recall@5, each clamped to [0, 1]:
    drop_k = clamp((baseline_k − new_k) / baseline_k, 0, 1)
  A drop of exactly 15 % (DRIFT_METRIC_THRESHOLD) produces a partial signal.

  consistency_signal  (25 %)
  ───────────────────────────
  Same as recall_signal but for Consistency@1 and Consistency@5.

  drift_detected = drift_score >= DRIFT_SCORE_THRESHOLD  (default 0.30)

Per-mode embedding spaces
──────────────────────────
  zero_shot    → encode new data with plain CLIP (no adaptation)
  linear_probe → encode with frozen CLIP → LinearProbeHead (from probe_best.pt)
  finetune     → encode with fine-tuned CLIP (from finetune_best.pt)

Usage
─────
  from drift_detection import compute_drift_score, load_baseline_from_mlflow

  baseline = load_baseline_from_mlflow(past_run_id, tracking_uri)
  result   = compute_drift_score(new_csv, baseline, device="cpu")

  if result.drift_detected:
      print(f"DRIFT  score={result.drift_score:.3f}")
      print(result.report())

CLI:
  python drift_detection.py \\
      --run_id <past_mlflow_run_id> \\
      --new_csv data/processed/week2.csv \\
      --tracking_uri sqlite:///mlflow.db
"""

import json
import logging
import argparse
import tempfile
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import mlflow

# Imported from clip_experiments — the three loader helpers + dataset utilities
from clip_experiments import (
    Config,
    LinearProbeHead,
    CLIPProductDataset,
    collate_skip_none,
    compute_embeddings,
    recall_at_k,
    consistency_at_k,
    load_zero_shot_model,
    load_linear_probe_from_mlflow,
    load_finetune_from_mlflow,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")


# ─────────────────────────────────────────────────────────────────────────────
# THRESHOLDS
# ─────────────────────────────────────────────────────────────────────────────

DRIFT_SCORE_THRESHOLD   = 0.30   # combined score above this → drift_detected=True
DRIFT_METRIC_THRESHOLD  = 0.15   # 15 % relative metric drop → non-zero metric signal
PAIRWISE_NORM_RANGE     = 0.20   # absolute pairwise-cosine delta mapped to [0, 1]


# ─────────────────────────────────────────────────────────────────────────────
# RESULT DATACLASS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DriftResult:
    """
    All numbers are in [0, 1].  Higher = more drift.

    Weights
    ───────
      embedding_signal   → 50 %
      recall_signal      → 25 %
      consistency_signal → 25 %
    """
    run_id:              str
    mode:                str
    week_label:          str
    category_label:      str
    n_baseline:          int
    n_new:               int

    # ── Signal 1: embedding shift (50 %) ──────────────────────────────────
    centroid_shift_text:   float = 0.0   # 1 − cos(new_centroid, baseline_centroid) for text
    centroid_shift_image:  float = 0.0   # same for image
    pairwise_delta_text:   float = 0.0   # normalised |pairwise_mean_new − baseline|
    pairwise_delta_image:  float = 0.0
    embedding_signal:      float = 0.0   # mean of four above sub-signals

    # ── Signal 2a: recall drop (25 %) ─────────────────────────────────────
    recall_at_1_baseline:  float = 0.0
    recall_at_5_baseline:  float = 0.0
    recall_at_1_new:       float = 0.0
    recall_at_5_new:       float = 0.0
    recall_signal:         float = 0.0   # mean relative drop, clamped [0,1]

    # ── Signal 2b: consistency drop (25 %) ────────────────────────────────
    consistency_at_1_baseline: float = 0.0
    consistency_at_5_baseline: float = 0.0
    consistency_at_1_new:      float = 0.0
    consistency_at_5_new:      float = 0.0
    consistency_signal:        float = 0.0

    # ── Combined ──────────────────────────────────────────────────────────
    drift_score:    float = 0.0
    drift_detected: bool  = False

    def report(self) -> str:
        lines = [
            f"{'─'*60}",
            f"  Drift Report  run_id={self.run_id}  mode={self.mode}",
            f"  week={self.week_label}  category={self.category_label}",
            f"  baseline N={self.n_baseline}  new N={self.n_new}",
            f"{'─'*60}",
            f"  [Signal 1 — Embedding shift  50%]",
            f"    text  centroid_shift  : {self.centroid_shift_text:.4f}",
            f"    image centroid_shift  : {self.centroid_shift_image:.4f}",
            f"    text  pairwise_delta  : {self.pairwise_delta_text:.4f}",
            f"    image pairwise_delta  : {self.pairwise_delta_image:.4f}",
            f"    → embedding_signal    : {self.embedding_signal:.4f}",
            f"  [Signal 2a — Recall drop     25%]",
            f"    Recall@1  baseline={self.recall_at_1_baseline:.4f}  new={self.recall_at_1_new:.4f}",
            f"    Recall@5  baseline={self.recall_at_5_baseline:.4f}  new={self.recall_at_5_new:.4f}",
            f"    → recall_signal       : {self.recall_signal:.4f}",
            f"  [Signal 2b — Consistency drop 25%]",
            f"    Cons@1  baseline={self.consistency_at_1_baseline:.4f}  new={self.consistency_at_1_new:.4f}",
            f"    Cons@5  baseline={self.consistency_at_5_baseline:.4f}  new={self.consistency_at_5_new:.4f}",
            f"    → consistency_signal  : {self.consistency_signal:.4f}",
            f"{'─'*60}",
            f"  DRIFT SCORE : {self.drift_score:.4f}  "
            f"(threshold={DRIFT_SCORE_THRESHOLD})",
            f"  DRIFT DETECTED : {'YES ⚠' if self.drift_detected else 'no'}",
            f"{'─'*60}",
        ]
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# BASELINE LOADER
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DriftBaseline:
    """
    Everything needed to score new data against a past run.

    Loaded from MLflow artifacts/drift/ of the reference run.
    """
    run_id:        str
    mode:          str                      # zero_shot | linear_probe | finetune
    reference:     dict                     # full drift_reference.json
    txt_centroid:  np.ndarray               # (D,) unit-normed text centroid
    img_centroid:  np.ndarray               # (D,) unit-normed image centroid
    cfg:           Config                   # reconstructed Config for model loading


def load_baseline_from_mlflow(run_id: str,
                               tracking_uri: str = "sqlite:///mlflow.db") -> DriftBaseline:
    """
    Download drift_reference.json and centroid .npy files from a past MLflow run.

    Parameters
    ----------
    run_id       : MLflow run ID of the training run to use as baseline
    tracking_uri : MLflow tracking URI (must match the one used during training)

    Returns
    -------
    DriftBaseline — ready to pass into compute_drift_score()
    """
    mlflow.set_tracking_uri(tracking_uri)

    ref_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="drift/drift_reference.json")
    with open(ref_path) as f:
        reference = json.load(f)

    txt_cent_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="drift/baseline_text_centroid.npy")
    img_cent_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="drift/baseline_image_centroid.npy")

    txt_centroid = np.load(txt_cent_path)
    img_centroid = np.load(img_cent_path)

    ri = reference["run_info"]
    cfg = Config(
        model_name     = ri["model_name"],
        pretrained     = ri["pretrained"],
        embed_dim      = ri["embed_dim"],
        probe_hidden   = ri.get("probe_hidden"),
        probe_dropout  = ri.get("probe_dropout", 0.10),
        mode           = ri["mode"],
        week_label     = ri["week_label"],
        category_label = ri["category_label"],
        mlflow_tracking_uri = tracking_uri,
    )

    log.info("Baseline loaded  run_id=%s  mode=%s  week=%s",
             run_id, ri["mode"], ri["week_label"])
    return DriftBaseline(
        run_id       = run_id,
        mode         = ri["mode"],
        reference    = reference,
        txt_centroid = txt_centroid,
        img_centroid = img_centroid,
        cfg          = cfg,
    )


# ─────────────────────────────────────────────────────────────────────────────
# ENCODE NEW DATA — mode-aware
# ─────────────────────────────────────────────────────────────────────────────

def _encode_new_data(
        new_csv: str,
        baseline: DriftBaseline,
        device: str,
        batch_size: int = 64,
        num_workers: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Encode new_csv through the *same model space* used during training.

    zero_shot    → plain CLIP, no weights loaded from MLflow
    linear_probe → frozen CLIP + LinearProbeHead from probe_best.pt
    finetune     → fine-tuned CLIP from finetune_best.pt

    Returns (img_embs, txt_embs) tensors on CPU.
    """
    import pandas as pd
    df = pd.read_csv(new_csv)
    cfg  = baseline.cfg
    mode = baseline.mode

    if mode == "zero_shot":
        model, preprocess, tokenizer = load_zero_shot_model(cfg, device)
        probe_head = None

    elif mode == "linear_probe":
        model, probe_head, preprocess, tokenizer = load_linear_probe_from_mlflow(
            baseline.run_id, cfg, device)
        probe_head.eval()

    elif mode == "finetune":
        model, preprocess, tokenizer = load_finetune_from_mlflow(
            baseline.run_id, cfg, device)
        probe_head = None

    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Use eval mode — one canonical text per image (title_clean)
    ds = CLIPProductDataset(df, tokenizer, preprocess,
                            cache_dir=cfg.image_cache_dir, mode="eval")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, collate_fn=collate_skip_none)

    img_embs, txt_embs, _ = compute_embeddings(model, loader, device, probe_head)
    log.info("New data encoded  mode=%s  n=%d", mode, len(img_embs))
    return img_embs, txt_embs


# ─────────────────────────────────────────────────────────────────────────────
# EMBEDDING SIGNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _centroid_shift(new_embs: np.ndarray, baseline_centroid: np.ndarray) -> float:
    """
    1 − cosine_similarity(new_centroid, baseline_centroid).
    Range [0, 2]; we clamp to [0, 1] for interpretability.
    A value of 0 means the distribution centroid hasn't moved.
    """
    new_centroid = new_embs.mean(axis=0)
    new_centroid = new_centroid / (np.linalg.norm(new_centroid) + 1e-9)
    cos_sim = float(np.dot(new_centroid, baseline_centroid))
    return float(np.clip(1.0 - cos_sim, 0.0, 1.0))


def _pairwise_delta(new_embs: np.ndarray, baseline_pairwise_mean: float) -> float:
    """
    Normalised absolute change in mean pairwise cosine similarity.
    PAIRWISE_NORM_RANGE (default 0.20) maps to a full signal of 1.0.
    """
    sub = new_embs[:500] if len(new_embs) > 500 else new_embs
    sim_mat = sub @ sub.T
    np.fill_diagonal(sim_mat, np.nan)
    new_pairwise = float(np.nanmean(sim_mat))
    delta = abs(new_pairwise - baseline_pairwise_mean)
    return float(np.clip(delta / PAIRWISE_NORM_RANGE, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# METRIC SIGNAL HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _relative_drop(baseline_val: float, new_val: float) -> float:
    """
    Relative drop from baseline, clamped to [0, 1].
    A 15 % drop (DRIFT_METRIC_THRESHOLD) produces a partial signal;
    a 100 % drop produces 1.0.
    Zero or negative baseline → return 0 to avoid division by zero.
    """
    if baseline_val <= 0:
        return 0.0
    return float(np.clip((baseline_val - new_val) / baseline_val, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def compute_drift_score(
        new_csv: str,
        baseline: DriftBaseline,
        device: str = "cpu",
        batch_size: int = 64,
        num_workers: int = 0,
        eval_ks: tuple = (1, 5, 10),
        weight_embedding: float = 0.20,
        weight_recall: float = 0.4,
        weight_consistency: float = 0.4,
) -> DriftResult:
    """
    Weights must sum to 1.0.  Defaults match the original 50/25/25 split.
    Override from CLI with --weight_embedding, --weight_recall, --weight_consistency.
    """
    total = weight_embedding + weight_recall + weight_consistency
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"Weights must sum to 1.0, got {total:.4f} "
            f"(embedding={weight_embedding}, recall={weight_recall}, "
            f"consistency={weight_consistency})"
        )

    # ── 1. Encode new data in the correct embedding space ─────────────────
    img_embs, txt_embs = _encode_new_data(
        new_csv, baseline, device, batch_size, num_workers)

    txt_np = txt_embs.numpy()
    img_np = img_embs.numpy()

    ref     = baseline.reference
    txt_bl  = ref["text_embedding_stats"]
    img_bl  = ref["image_embedding_stats"]
    met_bl  = ref["baseline_metrics"]

    # ── 2. Embedding signal (50 %) ─────────────────────────────────────────
    cs_txt = _centroid_shift(txt_np, baseline.txt_centroid)
    cs_img = _centroid_shift(img_np, baseline.img_centroid)
    pd_txt = _pairwise_delta(txt_np, txt_bl["pairwise_cosine_mean"])
    pd_img = _pairwise_delta(img_np, img_bl["pairwise_cosine_mean"])
    emb_signal = float(np.mean([cs_txt, cs_img, pd_txt, pd_img]))

    # ── 3. Retrieval metrics on new data ───────────────────────────────────
    new_recall  = recall_at_k(img_embs, txt_embs, ks=eval_ks)
    new_consist = consistency_at_k(img_embs, txt_embs, ks=eval_ks)

    r1_new = new_recall.get("Recall@1",      0.0)
    r5_new = new_recall.get("Recall@5",      0.0)
    c1_new = new_consist.get("Consistency@1", 0.0)
    c5_new = new_consist.get("Consistency@5", 0.0)

    r1_bl  = met_bl.get("Recall@1",       0.0)
    r5_bl  = met_bl.get("Recall@5",       0.0)
    c1_bl  = met_bl.get("Consistency@1",  0.0)
    c5_bl  = met_bl.get("Consistency@5",  0.0)

    # ── 4. Recall signal (25 %) ────────────────────────────────────────────
    recall_signal = float(np.mean([
        _relative_drop(r1_bl, r1_new),
        _relative_drop(r5_bl, r5_new),
    ]))

    # ── 5. Consistency signal (25 %) ───────────────────────────────────────
    consistency_signal = float(np.mean([
        _relative_drop(c1_bl, c1_new),
        _relative_drop(c5_bl, c5_new),
    ]))

    # ── 6. Combined score ──────────────────────────────────────────────────
    drift_score = (
        weight_embedding    * emb_signal
      + weight_recall       * recall_signal
      + weight_consistency  * consistency_signal
    )

    result = DriftResult(
        run_id         = baseline.run_id,
        mode           = baseline.mode,
        week_label     = baseline.cfg.week_label,
        category_label = baseline.cfg.category_label,
        n_baseline     = txt_bl["n_samples"],
        n_new          = len(txt_np),

        centroid_shift_text   = cs_txt,
        centroid_shift_image  = cs_img,
        pairwise_delta_text   = pd_txt,
        pairwise_delta_image  = pd_img,
        embedding_signal      = emb_signal,

        recall_at_1_baseline  = r1_bl,
        recall_at_5_baseline  = r5_bl,
        recall_at_1_new       = r1_new,
        recall_at_5_new       = r5_new,
        recall_signal         = recall_signal,

        consistency_at_1_baseline = c1_bl,
        consistency_at_5_baseline = c5_bl,
        consistency_at_1_new      = c1_new,
        consistency_at_5_new      = c5_new,
        consistency_signal        = consistency_signal,

        drift_score    = drift_score,
        drift_detected = drift_score >= DRIFT_SCORE_THRESHOLD,
    )

    log.info("Drift score=%.4f  detected=%s", drift_score, result.drift_detected)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Compute drift score between a past training run and new data")
    p.add_argument("--run_id",       required=True,
                   help="MLflow run_id of the baseline training run")
    p.add_argument("--new_csv",      required=True,
                   help="Path to new week's augmented CSV")
    p.add_argument("--tracking_uri", default="sqlite:///mlflow.db")
    p.add_argument("--device",       default="cpu")
    p.add_argument("--batch_size",   type=int, default=64)
    p.add_argument("--threshold",          type=float, default=DRIFT_SCORE_THRESHOLD,
                   help="drift_score >= threshold → drift_detected=True")
    p.add_argument("--weight_embedding",   type=float, default=0.50,
                   help="Weight for embedding shift signal (default 0.50)")
    p.add_argument("--weight_recall",      type=float, default=0.25,
                   help="Weight for recall drop signal (default 0.25)")
    p.add_argument("--weight_consistency", type=float, default=0.25,
                   help="Weight for consistency drop signal (default 0.25). "
                        "All three weights must sum to 1.0.")
    return p.parse_args()


def main():
    import sys
    global DRIFT_SCORE_THRESHOLD

    args = parse_args()
    DRIFT_SCORE_THRESHOLD = args.threshold

    baseline = load_baseline_from_mlflow(args.run_id, args.tracking_uri)
    result   = compute_drift_score(
        new_csv            = args.new_csv,
        baseline           = baseline,
        device             = args.device,
        batch_size         = args.batch_size,
        weight_embedding   = args.weight_embedding,
        weight_recall      = args.weight_recall,
        weight_consistency = args.weight_consistency,
    )
    print(result.report())
    sys.exit(0 if not result.drift_detected else 1)


if __name__ == "__main__":
    main()