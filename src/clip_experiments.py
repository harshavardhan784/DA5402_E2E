"""
src/models/clip_experiments.py
-------------------------------
Three CLIP training modes with full MLflow artifact logging.

MLflow artifact layout per run
────────────────────────────────
  artifacts/
    model/                        ← mlflow.pytorch.log_model
    checkpoints/                  ← state_dicts at exponential epochs + best
    embeddings/                   ← val image + text embedding .npy files
    drift/
      drift_reference.json        ← baseline stats + final metrics
      baseline_text_centroid.npy
      baseline_image_centroid.npy
    confusion/                    ← retrieval failure analysis CSV
    provenance/                   ← week provenance JSON (written by DAG)
    eval_metrics_<label>.json
    requirements.txt

MLflow tags set on every run (searchable, used by DAG for model selection)
───────────────────────────────────────────────────────────────────────────
  pipeline     = "clip_product_retrieval"
  mode         = "zero_shot" | "linear_probe" | "finetune"
  stage        = "initial_train" | "retrain" | "pre_retrain_eval"
  week_label   = e.g. "week1"
  trained_on   = stem of the CSV used for training, e.g. "week1_replay"
                 (empty string for zero_shot — no training involved)

MLflow params set on every training run
────────────────────────────────────────
  n_train_images, n_val_images, n_train_rows, n_val_rows
  (logged inside _build_loaders so they appear on every mode that trains)
"""

import os
os.environ["MLFLOW_CONFIGURE_LOGGING"] = "0"

import importlib.metadata
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
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)


# ─────────────────────────────────────────────────────────────────────────────
# DYNAMIC PACKAGE VERSION HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _pkg_version(name: str, fallback: str = "unknown") -> str:
    """
    Return the installed version of a package at runtime.
    Falls back to `fallback` if the package is not found.
    Used by _log_model_pytorch so pip_requirements is always accurate.
    """
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return fallback


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
    category_label:         str   = "all"      # "all" = no filter applied


# ─────────────────────────────────────────────────────────────────────────────
# EXPONENTIAL CHECKPOINT EPOCHS
# ─────────────────────────────────────────────────────────────────────────────

def exponential_checkpoint_epochs(total_epochs: int) -> set:
    """
    Powers of 2 strictly less than total_epochs, plus total_epochs itself.
      10 → {2, 4, 8, 10}
      32 → {2, 4, 8, 16, 32}
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
    n_val     = max(1, int(len(ids) * val_fraction))
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
        self.groups     = {}
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
        g   = self.groups[self.indices[i]]
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
    """
    Returns (img_embs, txt_embs, orig_idx_list).
    Callers must unpack all three values.
    """
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
    out   = {}
    for k in ks:
        k_eff  = min(k, N - 1)
        top_i  = i_sim.topk(k_eff, dim=1).indices
        top_t  = t_sim.topk(k_eff, dim=1).indices
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
# DRIFT BASELINE
# ─────────────────────────────────────────────────────────────────────────────

def _compute_embedding_stats(embs: np.ndarray) -> dict:
    centroid      = embs.mean(axis=0)
    centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-9)
    cos_to_centroid = embs @ centroid_norm

    sub = embs[:500] if len(embs) > 500 else embs
    sim_mat = sub @ sub.T
    np.fill_diagonal(sim_mat, np.nan)
    pairwise_mean = float(np.nanmean(sim_mat))

    return {
        "n_samples":               len(embs),
        "embed_dim":               int(embs.shape[1]),
        "cosine_to_centroid_mean": float(cos_to_centroid.mean()),
        "cosine_to_centroid_std":  float(cos_to_centroid.std()),
        "pairwise_cosine_mean":    pairwise_mean,
        "dim_mean":                centroid_norm.tolist(),
        "dim_std":                 embs.std(axis=0).tolist(),
    }


def build_drift_reference(txt_embs, img_embs, metrics, cfg, run_id):
    txt_np = txt_embs.numpy()
    img_np = img_embs.numpy()
    return {
        "run_info": {
            "run_id":         run_id,
            "mode":           cfg.mode,
            "model_name":     cfg.model_name,
            "pretrained":     cfg.pretrained,
            "embed_dim":      cfg.embed_dim,
            "probe_hidden":   cfg.probe_hidden,
            "probe_dropout":  cfg.probe_dropout,
            "week_label":     cfg.week_label,
            "category_label": cfg.category_label,
            "trained_on":     Path(cfg.csv_path).stem,
        },
        "text_embedding_stats":  _compute_embedding_stats(txt_np),
        "image_embedding_stats": _compute_embedding_stats(img_np),
        "baseline_metrics": {
            "Recall@1":      metrics.get("Recall@1",      0.0),
            "Recall@5":      metrics.get("Recall@5",      0.0),
            "Consistency@1": metrics.get("Consistency@1", 0.0),
            "Consistency@5": metrics.get("Consistency@5", 0.0),
        },
    }


def _log_drift_reference(txt_embs, img_embs, metrics, cfg, run_id):
    reference = build_drift_reference(txt_embs, img_embs, metrics, cfg, run_id)
    mlflow.log_dict(reference, "drift/drift_reference.json")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        txt_centroid = np.array(
            reference["text_embedding_stats"]["dim_mean"], dtype=np.float32)
        img_centroid = np.array(
            reference["image_embedding_stats"]["dim_mean"], dtype=np.float32)
        np.save(tmp_dir / "baseline_text_centroid.npy",  txt_centroid)
        np.save(tmp_dir / "baseline_image_centroid.npy", img_centroid)
        mlflow.log_artifacts(str(tmp_dir), artifact_path="drift")
    log.info("Drift reference logged (mode=%s  week=%s  n_text=%d  n_img=%d)",
             cfg.mode, cfg.week_label, len(txt_embs), len(img_embs))


# ─────────────────────────────────────────────────────────────────────────────
# MLFLOW HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _set_run_tags(cfg: Config, stage: str):
    """
    Set the standard searchable tags on the active MLflow run.
    Called once at the start of every mode function.

    Tags (searchable in MLflow UI and via MlflowClient.search_runs):
      pipeline   — always "clip_product_retrieval"
      mode       — "zero_shot" | "linear_probe" | "finetune"
      stage      — "initial_train" | "retrain" | "pre_retrain_eval"
      week_label — e.g. "week2"
      trained_on — stem of cfg.csv_path, e.g. "week2_replay"
                   empty string for zero_shot (no training data)
    """
    trained_on = Path(cfg.csv_path).stem if cfg.mode != "zero_shot" else ""
    mlflow.set_tags({
        "pipeline":   "clip_product_retrieval",
        "mode":       cfg.mode,
        "stage":      stage,
        "week_label": cfg.week_label,
        "trained_on": trained_on,
    })
    log.info(
        "MLflow tags set — mode=%s  stage=%s  week=%s  trained_on=%s",
        cfg.mode, stage, cfg.week_label, trained_on or "(none)",
    )


def _log_config(cfg: Config):
    mlflow.log_params({k: str(v) for k, v in asdict(cfg).items()})


def _log_requirements():
    try:
        reqs = subprocess.check_output(["pip", "freeze"], text=True)
    except Exception:
        reqs = "# pip freeze failed"
    mlflow.log_text(reqs, "requirements.txt")


def _log_embeddings(img_embs, txt_embs, label: str = "val"):
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
    Log the model using mlflow.pytorch.log_model.
    pip_requirements uses the ACTUAL installed versions at runtime
    via importlib.metadata so they never go stale.
    """
    open_clip_ver = _pkg_version("open-clip-torch")
    torch_ver     = _pkg_version("torch")
    pillow_ver    = _pkg_version("Pillow")
    numpy_ver     = _pkg_version("numpy")

    pip_reqs = [
        f"open-clip-torch=={open_clip_ver}",
        f"torch=={torch_ver}",
        f"Pillow=={pillow_ver}",
        f"numpy=={numpy_ver}",
    ]
    if extra_pip:
        pip_reqs.extend(extra_pip)

    log.info("Logging model with pip_requirements: %s", pip_reqs)
    mlflow.pytorch.log_model(
        pytorch_model  = model,
        artifact_path  = artifact_path,
        pip_requirements = pip_reqs,
    )
    log.info("Model logged → %s", artifact_path)


def _log_retrieval_failures(img_embs, txt_embs, df_val, top_failures=20):
    sims       = (img_embs * txt_embs).sum(dim=1).numpy()
    worst_idx  = np.argsort(sims)[:top_failures]
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
            pd.DataFrame(rows).to_csv(
                Path(tmp) / "retrieval_failures.csv", index=False)
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
    log.info("  Frozen=%d  Trainable=%d  (last %d blocks unfrozen)",
             frozen, trainable, n_blocks)


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
        model.eval()
        probe_head.train()
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
        total_loss += loss.item()
        n_batches  += 1

    avg = total_loss / max(n_batches, 1)
    log.info("  Epoch %d/%d  avg_loss=%.4f  batches=%d", epoch, total_epochs, avg, n_batches)
    return avg


# ─────────────────────────────────────────────────────────────────────────────
# COMMON DATA SETUP
# ─────────────────────────────────────────────────────────────────────────────

def _build_loaders(cfg, tokenizer, preprocess):
    """
    Build train and val DataLoaders from cfg.csv_path.

    Also logs dataset split stats to the active MLflow run as params:
      n_train_images, n_val_images, n_train_rows, n_val_rows, trained_on

    These params appear on every run that calls _build_loaders so you can
    always trace which data a model was trained on.
    """
    df = pd.read_csv(cfg.csv_path)
    train_df, val_df = split_by_index(df, cfg.val_fraction, cfg.seed)

    n_train_images = train_df["original_index"].nunique()
    n_val_images   = val_df["original_index"].nunique()
    log.info(
        "Dataset split — train: %d rows / %d images   val: %d rows / %d images   "
        "source: %s",
        len(train_df), n_train_images,
        len(val_df),   n_val_images,
        Path(cfg.csv_path).name,
    )

    # Log to MLflow so every training run has data provenance as params
    mlflow.log_params({
        "n_train_rows":   len(train_df),
        "n_val_rows":     len(val_df),
        "n_train_images": n_train_images,
        "n_val_images":   n_val_images,
        "trained_on":     Path(cfg.csv_path).stem,
    })

    train_ds = CLIPProductDataset(
        train_df, tokenizer, preprocess,
        cache_dir=cfg.image_cache_dir, mode="train")
    val_ds = CLIPProductDataset(
        val_df, tokenizer, preprocess,
        cache_dir=cfg.image_cache_dir, mode="eval")

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, collate_fn=collate_skip_none, drop_last=True)
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, collate_fn=collate_skip_none)

    return train_loader, val_loader, train_df, val_df


# ─────────────────────────────────────────────────────────────────────────────
# MODE 1 — ZERO SHOT
# ─────────────────────────────────────────────────────────────────────────────

def run_zero_shot(cfg: Config, device: str, run_id: str) -> dict:
    """
    Evaluate raw CLIP embeddings with no adaptation.
    Used as pre-retrain baseline evaluation.
    No training data involved — trained_on tag is empty string.
    """
    log.info("═══ ZERO SHOT  [week=%s] ═══", cfg.week_label)

    model, _, preprocess = open_clip.create_model_and_transforms(
        cfg.model_name, pretrained=cfg.pretrained)
    tokenizer = open_clip.get_tokenizer(cfg.model_name)
    model = model.to(device).eval()

    # Tags — must be called inside an active run (caller opens the run)
    _set_run_tags(cfg, stage="pre_retrain_eval")
    _log_config(cfg)
    _log_requirements()

    df = pd.read_csv(cfg.csv_path)
    _, val_df = split_by_index(df, cfg.val_fraction, cfg.seed)
    log.info("Zero-shot eval corpus — val images: %d", val_df["original_index"].nunique())

    val_ds = CLIPProductDataset(
        val_df, tokenizer, preprocess,
        cache_dir=cfg.image_cache_dir, mode="eval")
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, collate_fn=collate_skip_none)

    metrics, img_embs, txt_embs = evaluate(
        model, val_loader, device, cfg.eval_ks, label="zero_shot")

    _log_embeddings(img_embs, txt_embs, label="zero_shot_val")
    _log_eval_metrics_artifact(metrics, label="zero_shot")
    _log_retrieval_failures(img_embs, txt_embs, val_df, cfg.top_failures)
    _log_model_pytorch(model, "model", cfg)
    _log_drift_reference(txt_embs, img_embs, metrics, cfg, run_id)

    safe = {k.replace("@", "_"): v for k, v in metrics.items()}
    mlflow.log_metrics(safe)

    log.info(
        "Zero-shot complete [week=%s] — Recall@1=%.4f  Consistency@1=%.4f",
        cfg.week_label,
        metrics.get("Recall@1",      0.0),
        metrics.get("Consistency@1", 0.0),
    )
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# MODE 2 — LINEAR PROBE
# ─────────────────────────────────────────────────────────────────────────────

def run_linear_probe(cfg: Config, device: str, run_id: str,
                     stage: str = "retrain") -> dict:
    """
    Train a LinearProbeHead on top of frozen CLIP.

    stage parameter controls the MLflow 'stage' tag:
      "initial_train" — first ever training run (Day-0)
      "retrain"       — drift-triggered retraining

    trained_on tag = stem of cfg.csv_path, e.g. "week2_replay"
    This is the key used by the DAG to verify model/data alignment.
    """
    log.info(
        "═══ LINEAR PROBE  [week=%s  stage=%s  data=%s] ═══",
        cfg.week_label, stage, Path(cfg.csv_path).name,
    )

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

    # Tags and config — called inside the active run opened by caller
    _set_run_tags(cfg, stage=stage)
    _log_config(cfg)
    _log_requirements()

    # _build_loaders also logs n_train_images, n_val_images, trained_on as params
    train_loader, val_loader, train_df, val_df = _build_loaders(
        cfg, tokenizer, preprocess)

    optimizer    = torch.optim.AdamW(
        probe.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion    = InfoNCELoss(init_temperature=0.07)
    total_steps  = cfg.epochs * len(train_loader)
    warmup_steps = cfg.warmup_epochs * len(train_loader)
    scheduler    = _make_scheduler(optimizer, warmup_steps, total_steps)

    ckpt_epochs = exponential_checkpoint_epochs(cfg.epochs)
    log.info("  Checkpoint epochs: %s  total_steps=%d  warmup_steps=%d",
             sorted(ckpt_epochs), total_steps, warmup_steps)

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
            _log_checkpoint(
                {"epoch": epoch, "probe_state": probe.state_dict(),
                 "metrics": metrics},
                f"probe_epoch{epoch:03d}.pt",
            )

        recall1 = metrics.get("Recall@1", 0.0)
        if recall1 > best_recall:
            best_recall  = recall1
            best_metrics = metrics
            _log_checkpoint(
                {"epoch": epoch, "probe_state": probe.state_dict(),
                 "metrics": metrics},
                "probe_best.pt",
            )
            _log_embeddings(img_embs, txt_embs, label=f"best_ep{epoch:03d}")
            log.info("  ✓ New best Recall@1=%.4f  epoch=%d", best_recall, epoch)

    _log_eval_metrics_artifact(best_metrics, label="linear_probe_best")
    _log_retrieval_failures(img_embs, txt_embs, val_df, cfg.top_failures)

    # Drift reference uses post-probe embeddings from the best epoch
    _log_drift_reference(txt_embs, img_embs, best_metrics, cfg, run_id)

    # Log probe head only — backbone is frozen and unchanged
    _log_model_pytorch(probe, "model", cfg)

    safe_best = {f"best_{k.replace('@', '_')}": v for k, v in best_metrics.items()}
    mlflow.log_metrics(safe_best)

    log.info(
        "Linear probe complete [week=%s  stage=%s] — "
        "best Recall@1=%.4f  Consistency@1=%.4f  run_id=%s",
        cfg.week_label, stage,
        best_metrics.get("Recall@1",      0.0),
        best_metrics.get("Consistency@1", 0.0),
        run_id,
    )
    return best_metrics


# ─────────────────────────────────────────────────────────────────────────────
# MODE 3 — FINETUNE
# ─────────────────────────────────────────────────────────────────────────────

def run_finetune(cfg: Config, device: str, run_id: str,
                 stage: str = "retrain") -> dict:
    """
    Fine-tune the last N transformer blocks of CLIP directly.
    trained_on tag = stem of cfg.csv_path.
    """
    log.info(
        "═══ FINETUNE  [week=%s  stage=%s  last %d blocks  data=%s] ═══",
        cfg.week_label, stage, cfg.unfreeze_last_n_blocks, Path(cfg.csv_path).name,
    )

    model, _, preprocess = open_clip.create_model_and_transforms(
        cfg.model_name, pretrained=cfg.pretrained)
    tokenizer = open_clip.get_tokenizer(cfg.model_name)
    model = model.to(device)
    _freeze_clip_except_last_n(model, cfg.unfreeze_last_n_blocks)

    _set_run_tags(cfg, stage=stage)
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
    log.info("  Checkpoint epochs: %s  total_steps=%d", sorted(ckpt_epochs), total_steps)

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
            _log_checkpoint(
                {"epoch": epoch, "model_state": model.state_dict(),
                 "metrics": metrics},
                f"finetune_epoch{epoch:03d}.pt",
            )

        recall1 = metrics.get("Recall@1", 0.0)
        if recall1 > best_recall:
            best_recall  = recall1
            best_metrics = metrics
            _log_checkpoint(
                {"epoch": epoch, "model_state": model.state_dict(),
                 "metrics": metrics},
                "finetune_best.pt",
            )
            _log_embeddings(img_embs, txt_embs, label=f"best_ep{epoch:03d}")
            log.info("  ✓ New best Recall@1=%.4f  epoch=%d", best_recall, epoch)

    _log_eval_metrics_artifact(best_metrics, label="finetune_best")
    if img_embs is not None:
        _log_retrieval_failures(img_embs, txt_embs, val_df, cfg.top_failures)
        _log_drift_reference(txt_embs, img_embs, best_metrics, cfg, run_id)

    _log_model_pytorch(model, "model", cfg)

    safe_best = {f"best_{k.replace('@', '_')}": v for k, v in best_metrics.items()}
    mlflow.log_metrics(safe_best)

    log.info(
        "Finetune complete [week=%s  stage=%s] — "
        "best Recall@1=%.4f  Consistency@1=%.4f  run_id=%s",
        cfg.week_label, stage,
        best_metrics.get("Recall@1",      0.0),
        best_metrics.get("Consistency@1", 0.0),
        run_id,
    )
    return best_metrics


# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINT LOADERS  (used by DAG and drift_detection.py)
# ─────────────────────────────────────────────────────────────────────────────

def load_zero_shot_model(cfg: Config, device: str):
    """Return a plain CLIP model (no adaptation)."""
    model, _, preprocess = open_clip.create_model_and_transforms(
        cfg.model_name, pretrained=cfg.pretrained)
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(cfg.model_name)
    log.info("Loaded zero-shot CLIP  model=%s  pretrained=%s",
             cfg.model_name, cfg.pretrained)
    return model, preprocess, tokenizer


def load_linear_probe_from_mlflow(run_id: str, cfg: Config, device: str):
    """
    Reconstruct frozen CLIP + trained LinearProbeHead from an MLflow run.
    Returns (model, probe, preprocess, tokenizer).

    The DAG calls _verify_run_mode() before calling this to ensure
    the run is actually a linear_probe run.
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
    log.info(
        "Loaded linear-probe from MLflow — run_id=%s  epoch=%s  "
        "Recall@1=%.4f",
        run_id,
        ckpt.get("epoch", "?"),
        ckpt.get("metrics", {}).get("Recall@1", 0.0),
    )
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
    log.info(
        "Loaded finetune from MLflow — run_id=%s  epoch=%s  Recall@1=%.4f",
        run_id,
        ckpt.get("epoch", "?"),
        ckpt.get("metrics", {}).get("Recall@1", 0.0),
    )
    return model, preprocess, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# RUN VERIFICATION HELPER  (used by DAG before loading a model)
# ─────────────────────────────────────────────────────────────────────────────

def get_run_tags(run_id: str, tracking_uri: str) -> dict:
    """
    Fetch the tags dict for a given MLflow run_id.
    Returns {} if the run doesn't exist or tags can't be fetched.

    Tags the DAG cares about:
      mode        — "linear_probe" | "zero_shot" | "finetune"
      trained_on  — e.g. "week2_replay"
      week_label  — e.g. "week2"
      stage       — "initial_train" | "retrain" | "pre_retrain_eval"

    Usage in DAG:
        tags = get_run_tags(run_id, tracking_uri)
        if tags.get("mode") != "linear_probe":
            # fall back to zero-shot
    """
    try:
        client = mlflow.MlflowClient(tracking_uri=tracking_uri)
        run    = client.get_run(run_id)
        return dict(run.data.tags)
    except Exception as e:
        log.warning("Could not fetch tags for run_id=%s: %s", run_id, e)
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN  (CLI entry point)
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
    p.add_argument("--category_label",  default="all")
    p.add_argument("--stage",           default="retrain",
                   choices=["initial_train", "retrain", "pre_retrain_eval"])
    p.add_argument("--top_failures",    type=int,   default=20)
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

    run_name = cfg.mlflow_run_name or f"{cfg.mode}_{cfg.week_label}"

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        log.info("MLflow run started — run_id=%s  name=%s", run_id, run_name)

        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True).strip()
            mlflow.set_tag("git_commit", commit)
        except Exception:
            pass

        t0 = time.time()
        if cfg.mode == "zero_shot":
            final = run_zero_shot(cfg, device, run_id)
        elif cfg.mode == "linear_probe":
            final = run_linear_probe(cfg, device, run_id, stage=args.stage)
        elif cfg.mode == "finetune":
            final = run_finetune(cfg, device, run_id, stage=args.stage)

        elapsed = time.time() - t0
        mlflow.log_metric("elapsed_seconds", elapsed)
        log.info(
            "Run complete — run_id=%s  elapsed=%.1fs  final_metrics=%s",
            run_id, elapsed, final,
        )


if __name__ == "__main__":
    main()