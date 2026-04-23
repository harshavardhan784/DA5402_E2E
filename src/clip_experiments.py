"""
src/models/clip_experiments.py
-------------------------------
Three CLIP training modes with full MLflow artifact logging.

All models, embeddings, metrics, and checkpoints go to MLflow — nothing
is kept only on local disk.

MLflow artifact layout per run:
  artifacts/
    model/                    ← mlflow.pytorch.log_model
    checkpoints/              ← state_dicts at exponential epochs + best
    embeddings/               ← val image + text embedding .npy files
    drift/
      drift_reference.json    ← baseline stats + final metrics for drift detection
      baseline_text_centroid.npy  ← averaged text embedding vector (mode-specific space)
    confusion/                ← retrieval failure analysis CSV
    eval_metrics_<label>.json
    requirements.txt
"""

import os
os.environ["MLFLOW_CONFIGURE_LOGGING"] = "0"

import io
import json
import math
import tempfile
import time
import hashlib
import logging
import random
import argparse
import subprocess
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

import open_clip
import mlflow
import mlflow.pytorch

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw): return it

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    csv_path:               str   = "data/processed/week1.csv"
    image_cache_dir:        str   = "data/image_cache"
    val_fraction:           float = 0.15
    seed:                   int   = 42

    model_name:             str   = "ViT-B-32"
    pretrained:             str   = "openai"

    mode:                   str   = "zero_shot"
    epochs:                 int   = 10
    batch_size:             int   = 32
    lr:                     float = 1e-3
    weight_decay:           float = 1e-4
    warmup_epochs:          int   = 1
    grad_clip:              float = 1.0
    num_workers:            int   = 0
    embed_dim:              int   = 512

    probe_hidden:           Optional[int] = None
    probe_dropout:          float = 0.10
    unfreeze_last_n_blocks: int   = 2
    replay_ratio:           float = 0.20

    eval_ks:                List[int] = field(default_factory=lambda: [1, 5, 10])

    mlflow_tracking_uri:    str   = "sqlite:///mlflow.db"
    mlflow_experiment:      str   = "clip_product_retrieval"
    mlflow_run_name:        Optional[str] = None

    checkpoint_dir:         str   = "checkpoints"
    top_failures:           int   = 20
    week_label:             str   = "week1"
    category_label:         str   = "mens_wear"


# ─────────────────────────────────────────────────────────────────────────────
# EXPONENTIAL EPOCH SET
# ─────────────────────────────────────────────────────────────────────────────

def exponential_checkpoint_epochs(total_epochs: int) -> set:
    """
    Returns checkpoint epochs as powers of 2 strictly less than total_epochs,
    plus total_epochs itself.

      total_epochs=10  → {2, 4, 8, 10}
      total_epochs=32  → {2, 4, 8, 16, 32}
      total_epochs=5   → {2, 4, 5}
    """
    epochs, k = set(), 1
    while True:
        e = 2 ** k
        if e >= total_epochs:
            break
        epochs.add(e)
        k += 1
    epochs.add(total_epochs)
    return epochs


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE LOADING
# ─────────────────────────────────────────────────────────────────────────────

def _cache_path(url: str, cache_dir: str) -> Path:
    return Path(cache_dir) / f"{hashlib.md5(url.encode()).hexdigest()}.jpg"


def load_image(url: str, cache_dir: str = "data/image_cache",
               timeout: int = 10) -> Optional[Image.Image]:
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    cp = _cache_path(url, cache_dir)
    if cp.exists():
        try:
            return Image.open(cp).convert("RGB")
        except Exception:
            cp.unlink(missing_ok=True)
    if not HAS_REQUESTS:
        return None
    try:
        resp = requests.get(url, timeout=timeout, stream=True)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        img.save(cp, format="JPEG", quality=90)
        return img
    except Exception as e:
        log.debug("Image load failed %s: %s", url[:60], e)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────

def split_by_index(df: pd.DataFrame, val_fraction: float = 0.15,
                   seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ids = df["original_index"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n_val = max(1, int(len(ids) * val_fraction))
    val_ids   = set(ids[:n_val])
    train_ids = set(ids[n_val:])
    return (df[df["original_index"].isin(train_ids)].copy(),
            df[df["original_index"].isin(val_ids)].copy())


class CLIPProductDataset(Dataset):
    def __init__(self, df, tokenizer, preprocess,
                 cache_dir="data/image_cache", mode="train"):
        self.preprocess = preprocess
        self.tokenizer  = tokenizer
        self.cache_dir  = cache_dir
        self.is_train   = (mode == "train")
        self.groups = {}
        for idx, grp in df.groupby("original_index"):
            texts   = grp["augmented_text"].tolist()
            methods = grp["method"].tolist()
            url     = grp["image_url"].iloc[0]
            eval_text = (texts[methods.index("title_clean")]
                         if "title_clean" in methods else texts[0])
            self.groups[idx] = {"url": url, "texts": texts, "eval_text": eval_text}
        self.indices = sorted(self.groups.keys())

    def __len__(self): return len(self.indices)

    def __getitem__(self, i):
        g = self.groups[self.indices[i]]
        img = load_image(g["url"], self.cache_dir)
        if img is None:
            return None
        text = random.choice(g["texts"]) if self.is_train else g["eval_text"]
        return {
            "image":    self.preprocess(img),
            "text":     self.tokenizer([text])[0],
            "orig_idx": self.indices[i],
        }


def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return {
        "image":    torch.stack([b["image"]    for b in batch]),
        "text":     torch.stack([b["text"]     for b in batch]),
        "orig_idx": torch.tensor([b["orig_idx"] for b in batch], dtype=torch.long),
    }


# ─────────────────────────────────────────────────────────────────────────────
# LOSS
# ─────────────────────────────────────────────────────────────────────────────

class InfoNCELoss(nn.Module):
    def __init__(self, logit_scale=None, init_temperature: float = 0.07):
        super().__init__()
        if logit_scale is not None:
            self.logit_scale = logit_scale
        else:
            self.logit_scale = nn.Parameter(
                torch.tensor(math.log(1.0 / init_temperature)))

    def forward(self, img_f, txt_f):
        scale  = self.logit_scale.exp().clamp(max=100)
        logits = scale * img_f @ txt_f.T
        labels = torch.arange(len(logits), device=logits.device)
        return 0.5 * (F.cross_entropy(logits, labels) +
                      F.cross_entropy(logits.T, labels))


# ─────────────────────────────────────────────────────────────────────────────
# LINEAR PROBE HEAD
# ─────────────────────────────────────────────────────────────────────────────

class LinearProbeHead(nn.Module):
    def __init__(self, in_dim=512, hidden_dim=None, dropout=0.1):
        super().__init__()
        def _head():
            if hidden_dim is None:
                return nn.Sequential(nn.Linear(in_dim, in_dim, bias=False))
            return nn.Sequential(
                nn.Linear(in_dim, hidden_dim, bias=False), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(hidden_dim, in_dim, bias=False))
        self.image_head = _head()
        self.text_head  = _head()
        for mod in [self.image_head, self.text_head]:
            for layer in mod:
                if isinstance(layer, nn.Linear):
                    nn.init.eye_(layer.weight)

    def forward(self, img_f, txt_f):
        return (F.normalize(self.image_head(img_f), dim=-1),
                F.normalize(self.text_head(txt_f),  dim=-1))


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_embeddings(model, dataloader, device, probe_head=None):
    model.eval()
    if probe_head is not None:
        probe_head.eval()
    all_img, all_txt, all_idx = [], [], []
    for batch in tqdm(dataloader, desc="Embedding", leave=False):
        if batch is None:
            continue
        img_f = model.encode_image(batch["image"].to(device), normalize=True)
        txt_f = model.encode_text(batch["text"].to(device),   normalize=True)
        if probe_head is not None:
            img_f, txt_f = probe_head(img_f, txt_f)
        all_img.append(img_f.cpu())
        all_txt.append(txt_f.cpu())
        all_idx.extend(batch["orig_idx"].tolist())
    return torch.cat(all_img), torch.cat(all_txt), all_idx


def recall_at_k(img_embs, txt_embs, ks=(1, 5, 10)):
    sim    = img_embs @ txt_embs.T
    labels = torch.arange(len(sim)).unsqueeze(1)
    return {f"Recall@{k}": (sim.topk(min(k, sim.shape[1]), dim=1).indices == labels)
                             .any(dim=1).float().mean().item() for k in ks}


def consistency_at_k(img_embs, txt_embs, ks=(1, 5, 10)):
    N    = img_embs.shape[0]
    mask = torch.eye(N, dtype=torch.bool)
    i_sim = img_embs @ img_embs.T;  i_sim[mask] = -1e9
    t_sim = txt_embs @ img_embs.T;  t_sim[mask] = -1e9
    out = {}
    for k in ks:
        k_eff = min(k, N - 1)
        top_i = i_sim.topk(k_eff, dim=1).indices
        top_t = t_sim.topk(k_eff, dim=1).indices
        scores = [len(set(top_i[i].tolist()) & set(top_t[i].tolist())) / k_eff
                  for i in range(N)]
        out[f"Consistency@{k}"] = float(np.mean(scores))
    return out


def evaluate(model, loader, device, ks=(1, 5, 10), probe_head=None, label="eval"):
    img_embs, txt_embs, _ = compute_embeddings(model, loader, device, probe_head)
    metrics = {**recall_at_k(img_embs, txt_embs, ks),
               **consistency_at_k(img_embs, txt_embs, ks)}
    log.info("── Eval [%s] N=%d ──", label, len(img_embs))
    for k, v in metrics.items():
        log.info("  %-22s : %.4f", k, v)
    return metrics, img_embs, txt_embs


# ─────────────────────────────────────────────────────────────────────────────
# DRIFT BASELINE — computed from already-materialised embedding tensors
# ─────────────────────────────────────────────────────────────────────────────

def _compute_embedding_stats(embs: np.ndarray) -> dict:
    """
    Compute distribution statistics from an (N, D) embedding matrix.
    These are the numbers that drift_detection.py will compare against.

    Returns
    -------
    dict with keys:
        n_samples               int
        embed_dim               int
        cosine_to_centroid_mean float  — mean cos-sim of each vector to centroid
        cosine_to_centroid_std  float
        pairwise_cosine_mean    float  — mean off-diagonal pairwise cos-sim (≤500 subsample)
        dim_mean                list   — D-dim centroid (unit-normed), used as reference vector
        dim_std                 list   — D-dim per-dimension std
    """
    centroid      = embs.mean(axis=0)
    centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-9)
    cos_to_centroid = embs @ centroid_norm               # (N,)

    sub = embs[:500] if len(embs) > 500 else embs
    sim_mat = sub @ sub.T
    np.fill_diagonal(sim_mat, np.nan)
    pairwise_mean = float(np.nanmean(sim_mat))

    return {
        "n_samples":                len(embs),
        "embed_dim":                int(embs.shape[1]),
        "cosine_to_centroid_mean":  float(cos_to_centroid.mean()),
        "cosine_to_centroid_std":   float(cos_to_centroid.std()),
        "pairwise_cosine_mean":     pairwise_mean,
        "dim_mean":                 centroid_norm.tolist(),   # D-dim — logged as .npy too
        "dim_std":                  embs.std(axis=0).tolist(),
    }


def build_drift_reference(
        txt_embs: torch.Tensor,
        img_embs: torch.Tensor,
        metrics: dict,
        cfg: Config,
        run_id: str,
) -> dict:
    """
    Build the drift reference dict that will be stored as
    artifacts/drift/drift_reference.json in the current MLflow run.

    What each section is for
    ─────────────────────────
    run_info      — lets drift_detection.py reload the exact model + probe from
                    the same run_id, so new data is encoded in the *same space*.

    embedding_stats (text + image) — the baseline distribution.
                    drift_detection.py encodes new data through the same model,
                    computes the same stats, and compares (Signal 1, 50 % weight).

    baseline_metrics — the Recall@K / Consistency@K values at training time.
                    A drop > 15 % on any metric on new data fires Signal 2
                    (50 % weight: Recall@1 25 %, Recall@5 25 %,
                     Consistency@1 25 %, Consistency@5 25 %).

    Per-mode embedding space
    ─────────────────────────
    zero_shot     → raw CLIP text + image embeddings (no adaptation layer)
    linear_probe  → text + image embeddings *after* LinearProbeHead
                    (the probe transforms the CLIP space; we must compare in
                    the same transformed space, not raw CLIP)
    finetune      → text + image embeddings from the fine-tuned encoder
    """
    txt_np = txt_embs.numpy()
    img_np = img_embs.numpy()

    reference = {
        "run_info": {
            "run_id":        run_id,
            "mode":          cfg.mode,
            "model_name":    cfg.model_name,
            "pretrained":    cfg.pretrained,
            "embed_dim":     cfg.embed_dim,
            "probe_hidden":  cfg.probe_hidden,
            "probe_dropout": cfg.probe_dropout,
            "week_label":    cfg.week_label,
            "category_label": cfg.category_label,
            # Enough to reconstruct LinearProbeHead or reload finetune checkpoint
        },
        "text_embedding_stats":  _compute_embedding_stats(txt_np),
        "image_embedding_stats": _compute_embedding_stats(img_np),
        "baseline_metrics": {
            # Keep only the four metrics that feed into Signal 2
            "Recall@1":       metrics.get("Recall@1",       0.0),
            "Recall@5":       metrics.get("Recall@5",       0.0),
            "Consistency@1":  metrics.get("Consistency@1",  0.0),
            "Consistency@5":  metrics.get("Consistency@5",  0.0),
        },
    }
    return reference


def _log_drift_reference(
        txt_embs: torch.Tensor,
        img_embs: torch.Tensor,
        metrics: dict,
        cfg: Config,
        run_id: str,
):
    """
    Compute and log the drift reference to MLflow artifacts/drift/.

    Logs:
      drift/drift_reference.json          — full reference dict (stats + metrics + run_info)
      drift/baseline_text_centroid.npy    — text centroid vector (D,) for fast cosine compare
      drift/baseline_image_centroid.npy   — image centroid vector (D,)
    """
    reference = build_drift_reference(txt_embs, img_embs, metrics, cfg, run_id)

    # 1. JSON reference
    mlflow.log_dict(reference, "drift/drift_reference.json")

    # 2. Centroid .npy files — stored separately so drift_detection.py can do
    #    a single dot-product without parsing the full JSON
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        txt_centroid = np.array(reference["text_embedding_stats"]["dim_mean"],
                                dtype=np.float32)
        img_centroid = np.array(reference["image_embedding_stats"]["dim_mean"],
                                dtype=np.float32)
        np.save(tmp_dir / "baseline_text_centroid.npy",  txt_centroid)
        np.save(tmp_dir / "baseline_image_centroid.npy", img_centroid)
        mlflow.log_artifacts(str(tmp_dir), artifact_path="drift")

    log.info("Drift reference logged (mode=%s, n_text=%d, n_img=%d)",
             cfg.mode, len(txt_embs), len(img_embs))


# ─────────────────────────────────────────────────────────────────────────────
# MLFLOW ARTIFACT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _log_config(cfg: Config):
    mlflow.log_params({k: str(v) for k, v in asdict(cfg).items()})


def _log_requirements():
    try:
        reqs = subprocess.check_output(["pip", "freeze"], text=True)
    except Exception:
        reqs = "# pip freeze failed"
    mlflow.log_text(reqs, "requirements.txt")


def _log_embeddings(img_embs: torch.Tensor, txt_embs: torch.Tensor,
                    label: str = "val"):
    # Each helper gets its own isolated temp dir so files from other helpers
    # never bleed into the wrong artifact_path folder.
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        np.save(tmp_dir / f"{label}_image_embeddings.npy", img_embs.numpy())
        np.save(tmp_dir / f"{label}_text_embeddings.npy",  txt_embs.numpy())
        mlflow.log_artifacts(str(tmp_dir), artifact_path="embeddings")


def _log_checkpoint(state_dict: dict, filename: str):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / filename
        torch.save(state_dict, str(path))
        mlflow.log_artifact(str(path), artifact_path="checkpoints")


def _log_eval_metrics_artifact(metrics: dict, label: str = "final"):
    mlflow.log_dict(metrics, f"eval_metrics_{label}.json")


def _log_model_pytorch(model: nn.Module, artifact_path: str,
                       cfg: Config, extra_pip: list = None):
    """
    Log the model ONCE per run using mlflow.pytorch.log_model.
    Never call this inside the epoch loop.
    """
    pip_reqs = [
        "open-clip-torch==2.26.1",
        "torch>=2.0.0",
        "Pillow>=10.0.0",
        "numpy>=1.24.0",
    ]
    if extra_pip:
        pip_reqs.extend(extra_pip)
    mlflow.pytorch.log_model(
        pytorch_model=model,
        artifact_path=artifact_path,
        pip_requirements=pip_reqs,
    )
    log.info("Model logged → %s", artifact_path)


def _log_retrieval_failures(img_embs: torch.Tensor, txt_embs: torch.Tensor,
                             df_val: pd.DataFrame, top_failures: int = 20):
    """
    Log the `top_failures` worst image↔text paired cosine-sim cases to
    artifacts/confusion/retrieval_failures.csv.
    """
    sims      = (img_embs * txt_embs).sum(dim=1).numpy()
    worst_idx = np.argsort(sims)[:top_failures]
    unique_ids = sorted(df_val["original_index"].unique())
    rows = []
    for rank, i in enumerate(worst_idx):
        if i < len(unique_ids):
            oid = unique_ids[i]
            row = df_val[df_val["original_index"] == oid].iloc[0]
            rows.append({
                "rank":           rank + 1,
                "cosine_sim":     round(float(sims[i]), 4),
                "title":          row.get("augmented_text", "")[:80],
                "category_2":     row.get("category_2", ""),
                "image_url":      row.get("image_url", ""),
                "original_index": oid,
            })
    if rows:
        with tempfile.TemporaryDirectory() as tmp:
            pd.DataFrame(rows).to_csv(Path(tmp) / "retrieval_failures.csv", index=False)
            mlflow.log_artifacts(tmp, artifact_path="confusion")


# ─────────────────────────────────────────────────────────────────────────────
# FREEZE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _freeze_clip_except_last_n(model, n_blocks=2):
    for p in model.parameters():
        p.requires_grad = False
    targets = [
        model.visual.transformer.resblocks[-n_blocks:],
        model.transformer.resblocks[-n_blocks:],
    ]
    for group in targets:
        for block in group:
            for p in block.parameters():
                p.requires_grad = True
    for attr in ["visual.ln_post", "ln_final"]:
        parts = attr.split(".")
        obj = model
        for p in parts:
            obj = getattr(obj, p)
        for param in obj.parameters():
            param.requires_grad = True
    for attr in ["visual.proj", "text_projection"]:
        parts = attr.split(".")
        try:
            obj = model
            for p in parts:
                obj = getattr(obj, p)
            if isinstance(obj, nn.Parameter):
                obj.requires_grad = True
            else:
                for param in obj.parameters():
                    param.requires_grad = True
        except AttributeError:
            pass
    if hasattr(model, "logit_scale"):
        model.logit_scale.requires_grad = True
    frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("  Frozen=%d  Trainable=%d  (last %d blocks)", frozen, trainable, n_blocks)


# ─────────────────────────────────────────────────────────────────────────────
# SCHEDULER
# ─────────────────────────────────────────────────────────────────────────────

def _make_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN ONE EPOCH
# ─────────────────────────────────────────────────────────────────────────────

def _train_one_epoch(model, probe_head, loader, optimizer, criterion,
                     scheduler, grad_clip, device, epoch, total_epochs, mode):
    if mode == "linear_probe":
        model.eval(); probe_head.train()
    else:
        model.train()

    total_loss, n_batches = 0.0, 0
    for batch in tqdm(loader, desc=f"Epoch {epoch}/{total_epochs}", leave=False):
        if batch is None or batch["image"].shape[0] < 2:
            continue
        images = batch["image"].to(device)
        texts  = batch["text"].to(device)
        if mode == "linear_probe":
            with torch.no_grad():
                img_f = model.encode_image(images, normalize=True)
                txt_f = model.encode_text(texts,   normalize=True)
            img_f, txt_f = probe_head(img_f, txt_f)
        else:
            img_f = model.encode_image(images, normalize=True)
            txt_f = model.encode_text(texts,   normalize=True)

        loss = criterion(img_f, txt_f)
        optimizer.zero_grad()
        loss.backward()
        params = (probe_head.parameters() if mode == "linear_probe"
                  else [p for p in model.parameters() if p.requires_grad])
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(params, grad_clip)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item(); n_batches += 1

    avg = total_loss / max(n_batches, 1)
    log.info("  Epoch %d/%d  loss=%.4f", epoch, total_epochs, avg)
    return avg


# ─────────────────────────────────────────────────────────────────────────────
# COMMON DATA SETUP
# ─────────────────────────────────────────────────────────────────────────────

def _build_loaders(cfg, tokenizer, preprocess):
    df = pd.read_csv(cfg.csv_path)
    train_df, val_df = split_by_index(df, cfg.val_fraction, cfg.seed)
    log.info("Train images=%d  Val images=%d",
             train_df["original_index"].nunique(),
             val_df["original_index"].nunique())
    train_ds = CLIPProductDataset(train_df, tokenizer, preprocess,
                                  cache_dir=cfg.image_cache_dir, mode="train")
    val_ds   = CLIPProductDataset(val_df,   tokenizer, preprocess,
                                  cache_dir=cfg.image_cache_dir, mode="eval")
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers,
                              collate_fn=collate_skip_none, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers,
                              collate_fn=collate_skip_none)
    return train_loader, val_loader, train_df, val_df


# ─────────────────────────────────────────────────────────────────────────────
# MODE 1 — ZERO SHOT
# ─────────────────────────────────────────────────────────────────────────────

def run_zero_shot(cfg: Config, device: str, run_id: str) -> dict:
    """
    Drift baseline for zero_shot:
        Raw CLIP text + image embeddings (no adaptation).
        That is the space drift_detection.py must encode new data into.
    """
    log.info("═══ ZERO SHOT ═══")
    model, _, preprocess = open_clip.create_model_and_transforms(
        cfg.model_name, pretrained=cfg.pretrained)
    tokenizer = open_clip.get_tokenizer(cfg.model_name)
    model = model.to(device).eval()

    df = pd.read_csv(cfg.csv_path)
    _, val_df = split_by_index(df, cfg.val_fraction, cfg.seed)
    val_ds = CLIPProductDataset(val_df, tokenizer, preprocess,
                                cache_dir=cfg.image_cache_dir, mode="eval")
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers,
                            collate_fn=collate_skip_none)

    metrics, img_embs, txt_embs = evaluate(model, val_loader, device,
                                            cfg.eval_ks, label="zero_shot")
    _log_config(cfg)
    _log_requirements()
    _log_embeddings(img_embs, txt_embs, label="zero_shot_val")
    _log_eval_metrics_artifact(metrics, label="zero_shot")
    _log_retrieval_failures(img_embs, txt_embs, val_df, cfg.top_failures)
    _log_model_pytorch(model, "model", cfg)

    # Drift reference — raw CLIP space, no probe
    _log_drift_reference(txt_embs, img_embs, metrics, cfg, run_id)

    safe = {k.replace("@", "_"): v for k, v in metrics.items()}
    mlflow.log_metrics(safe)
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# MODE 2 — LINEAR PROBE
# ─────────────────────────────────────────────────────────────────────────────

def run_linear_probe(cfg: Config, device: str, run_id: str) -> dict:
    """
    Drift baseline for linear_probe:
        Text + image embeddings *after* LinearProbeHead.
        New data must be encoded through the same frozen CLIP + same probe
        weights (loaded from this run's checkpoints/probe_best.pt) to compare
        in the same adapted space.
    """
    log.info("═══ LINEAR PROBE ═══")
    model, _, preprocess = open_clip.create_model_and_transforms(
        cfg.model_name, pretrained=cfg.pretrained)
    tokenizer = open_clip.get_tokenizer(cfg.model_name)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False

    probe = LinearProbeHead(cfg.embed_dim, cfg.probe_hidden,
                            cfg.probe_dropout).to(device)
    log.info("  Probe trainable params: %d",
             sum(p.numel() for p in probe.parameters() if p.requires_grad))

    _log_config(cfg)
    _log_requirements()

    train_loader, val_loader, train_df, val_df = _build_loaders(
        cfg, tokenizer, preprocess)
    optimizer    = torch.optim.AdamW(probe.parameters(),
                                     lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion    = InfoNCELoss(init_temperature=0.07)
    total_steps  = cfg.epochs * len(train_loader)
    warmup_steps = cfg.warmup_epochs * len(train_loader)
    scheduler    = _make_scheduler(optimizer, warmup_steps, total_steps)

    ckpt_epochs = exponential_checkpoint_epochs(cfg.epochs)
    log.info("  Checkpoint epochs: %s", sorted(ckpt_epochs))

    best_recall, best_metrics = 0.0, {}
    img_embs = txt_embs = None
    for epoch in range(1, cfg.epochs + 1):
        epoch_loss = _train_one_epoch(
            model, probe, train_loader, optimizer, criterion, scheduler,
            cfg.grad_clip, device, epoch, cfg.epochs, "linear_probe")
        metrics, img_embs, txt_embs = evaluate(
            model, val_loader, device, cfg.eval_ks,
            probe_head=probe, label=f"ep{epoch}")
        metrics["train_loss"] = epoch_loss

        safe = {k.replace("@", "_"): v for k, v in metrics.items()}
        mlflow.log_metrics(safe, step=epoch)

        if epoch in ckpt_epochs:
            _log_checkpoint({"epoch": epoch, "probe_state": probe.state_dict(),
                              "metrics": metrics}, f"probe_epoch{epoch}.pt")

        recall1 = metrics.get("Recall@1", 0.0)
        if recall1 > best_recall:
            best_recall = recall1
            best_metrics = metrics
            _log_checkpoint({"epoch": epoch, "probe_state": probe.state_dict(),
                              "metrics": metrics}, "probe_best.pt")
            _log_embeddings(img_embs, txt_embs, label=f"best_ep{epoch}")
            log.info("  ✓ New best Recall@1=%.4f", best_recall)

    _log_eval_metrics_artifact(best_metrics, label="linear_probe_best")
    _log_retrieval_failures(img_embs, txt_embs, val_df, cfg.top_failures)

    # Drift reference — post-probe space (img_embs / txt_embs are already
    # the probe-transformed vectors from the best-recall epoch)
    _log_drift_reference(txt_embs, img_embs, best_metrics, cfg, run_id)

    # Log probe only — CLIP backbone is frozen/unchanged so no need to re-log it
    _log_model_pytorch(probe, "model", cfg)

    safe_best = {f"best_{k.replace('@','_')}": v for k, v in best_metrics.items()}
    mlflow.log_metrics(safe_best)
    return best_metrics


# ─────────────────────────────────────────────────────────────────────────────
# MODE 3 — FINETUNE
# ─────────────────────────────────────────────────────────────────────────────

def run_finetune(cfg: Config, device: str, run_id: str) -> dict:
    """
    Drift baseline for finetune:
        Text + image embeddings from the fine-tuned encoder directly.
        New data must be encoded through the fine-tuned CLIP weights
        (loaded from this run's checkpoints/finetune_best.pt).
    """
    log.info("═══ FINETUNE (last %d blocks) ═══", cfg.unfreeze_last_n_blocks)
    model, _, preprocess = open_clip.create_model_and_transforms(
        cfg.model_name, pretrained=cfg.pretrained)
    tokenizer = open_clip.get_tokenizer(cfg.model_name)
    model = model.to(device)
    _freeze_clip_except_last_n(model, cfg.unfreeze_last_n_blocks)

    _log_config(cfg)
    _log_requirements()

    train_loader, val_loader, train_df, val_df = _build_loaders(
        cfg, tokenizer, preprocess)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion    = InfoNCELoss(logit_scale=model.logit_scale)
    total_steps  = cfg.epochs * len(train_loader)
    warmup_steps = cfg.warmup_epochs * len(train_loader)
    scheduler    = _make_scheduler(optimizer, warmup_steps, total_steps)

    ckpt_epochs = exponential_checkpoint_epochs(cfg.epochs)
    log.info("  Checkpoint epochs: %s", sorted(ckpt_epochs))

    best_recall, best_metrics = 0.0, {}
    img_embs = txt_embs = None
    for epoch in range(1, cfg.epochs + 1):
        epoch_loss = _train_one_epoch(
            model, None, train_loader, optimizer, criterion, scheduler,
            cfg.grad_clip, device, epoch, cfg.epochs, "finetune")
        metrics, img_embs, txt_embs = evaluate(
            model, val_loader, device, cfg.eval_ks, label=f"ep{epoch}")
        metrics["train_loss"] = epoch_loss

        safe = {k.replace("@", "_"): v for k, v in metrics.items()}
        mlflow.log_metrics(safe, step=epoch)

        if epoch in ckpt_epochs:
            _log_checkpoint({"epoch": epoch, "model_state": model.state_dict(),
                              "metrics": metrics}, f"finetune_epoch{epoch}.pt")

        recall1 = metrics.get("Recall@1", 0.0)
        if recall1 > best_recall:
            best_recall = recall1
            best_metrics = metrics
            _log_checkpoint({"epoch": epoch, "model_state": model.state_dict(),
                              "metrics": metrics}, "finetune_best.pt")
            _log_embeddings(img_embs, txt_embs, label=f"best_ep{epoch}")
            log.info("  ✓ New best Recall@1=%.4f", best_recall)

    _log_eval_metrics_artifact(best_metrics, label="finetune_best")
    if img_embs is not None:
        _log_retrieval_failures(img_embs, txt_embs, val_df, cfg.top_failures)

    # Drift reference — fine-tuned encoder space
    if img_embs is not None:
        _log_drift_reference(txt_embs, img_embs, best_metrics, cfg, run_id)

    _log_model_pytorch(model, "model", cfg)

    safe_best = {f"best_{k.replace('@','_')}": v for k, v in best_metrics.items()}
    mlflow.log_metrics(safe_best)
    return best_metrics


# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINT LOADERS  (used by drift_detection.py)
# ─────────────────────────────────────────────────────────────────────────────

def load_zero_shot_model(cfg: Config, device: str):
    """Return a plain CLIP model (no adaptation). Used by drift_detection."""
    model, _, preprocess = open_clip.create_model_and_transforms(
        cfg.model_name, pretrained=cfg.pretrained)
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(cfg.model_name)
    return model, preprocess, tokenizer


def load_linear_probe_from_mlflow(run_id: str, cfg: Config, device: str):
    """
    Reconstruct frozen CLIP + trained LinearProbeHead from an MLflow run.
    Returns (model, probe, preprocess, tokenizer).
    """
    local_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="checkpoints/probe_best.pt")
    model, _, preprocess = open_clip.create_model_and_transforms(
        cfg.model_name, pretrained=cfg.pretrained)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    probe = LinearProbeHead(cfg.embed_dim, cfg.probe_hidden,
                            cfg.probe_dropout).to(device)
    ckpt = torch.load(local_path, map_location=device)
    probe.load_state_dict(ckpt["probe_state"])
    probe.eval()
    tokenizer = open_clip.get_tokenizer(cfg.model_name)
    log.info("Loaded probe from run %s epoch %s", run_id, ckpt.get("epoch"))
    return model, probe, preprocess, tokenizer


def load_finetune_from_mlflow(run_id: str, cfg: Config, device: str):
    """
    Reconstruct fine-tuned CLIP from an MLflow run.
    Returns (model, preprocess, tokenizer).
    """
    local_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="checkpoints/finetune_best.pt")
    model, _, preprocess = open_clip.create_model_and_transforms(
        cfg.model_name, pretrained=cfg.pretrained)
    model = model.to(device).eval()
    ckpt = torch.load(local_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    tokenizer = open_clip.get_tokenizer(cfg.model_name)
    log.info("Loaded finetune from run %s epoch %s", run_id, ckpt.get("epoch"))
    return model, preprocess, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode",            required=True,
                   choices=["zero_shot", "linear_probe", "finetune"])
    p.add_argument("--csv",             default="data/processed/week1.csv")
    p.add_argument("--epochs",          type=int,   default=10)
    p.add_argument("--batch_size",      type=int,   default=32)
    p.add_argument("--lr",              type=float, default=None)
    p.add_argument("--model_name",      default="ViT-B-32")
    p.add_argument("--pretrained",      default="openai")
    p.add_argument("--unfreeze_blocks", type=int,   default=2)
    p.add_argument("--cache_dir",       default="data/image_cache")
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--tracking_uri",    default="sqlite:///mlflow.db")
    p.add_argument("--experiment",      default="clip_product_retrieval")
    p.add_argument("--run_name",        default=None)
    p.add_argument("--week_label",      default="week1")
    p.add_argument("--category_label",  default="mens_wear")
    p.add_argument("--top_failures",    type=int,   default=20,
                   help="Number of worst retrieval pairs to log in confusion/")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s", device)

    default_lrs = {"zero_shot": 0.0, "linear_probe": 1e-3, "finetune": 5e-6}
    lr = args.lr if args.lr is not None else default_lrs[args.mode]

    cfg = Config(
        csv_path               = args.csv,
        image_cache_dir        = args.cache_dir,
        model_name             = args.model_name,
        pretrained             = args.pretrained,
        mode                   = args.mode,
        epochs                 = args.epochs,
        batch_size             = args.batch_size,
        lr                     = lr,
        unfreeze_last_n_blocks = args.unfreeze_blocks,
        mlflow_tracking_uri    = args.tracking_uri,
        mlflow_experiment      = args.experiment,
        mlflow_run_name        = args.run_name,
        seed                   = args.seed,
        week_label             = args.week_label,
        category_label         = args.category_label,
        top_failures           = args.top_failures,
    )

    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.mlflow_experiment)

    run_name = (cfg.mlflow_run_name
                or f"{cfg.mode}_{cfg.week_label}_{cfg.category_label}")

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        log.info("MLflow run_id: %s", run_id)

        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True).strip()
            mlflow.set_tag("git_commit", commit)
        except Exception:
            pass
        mlflow.set_tag("week",     cfg.week_label)
        mlflow.set_tag("category", cfg.category_label)
        mlflow.set_tag("mode",     cfg.mode)

        t0 = time.time()
        if cfg.mode == "zero_shot":
            final = run_zero_shot(cfg, device, run_id)
        elif cfg.mode == "linear_probe":
            final = run_linear_probe(cfg, device, run_id)
        elif cfg.mode == "finetune":
            final = run_finetune(cfg, device, run_id)

        mlflow.log_metric("elapsed_seconds", time.time() - t0)
        log.info("Run ID: %s", run_id)
        log.info("Final metrics: %s", final)


if __name__ == "__main__":
    main()