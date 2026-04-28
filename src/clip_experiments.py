"""
src/clip_experiments.py
-------------------------------
Three CLIP training modes with full MLflow artifact logging.

MLflow artifact layout per run
-------------------------------
  artifacts/
    model/                        - mlflow.pytorch.log_model
    checkpoints/                  - state_dicts at exponential epochs + best
    embeddings/                   - val image + text embedding .npy files
    drift/
      drift_reference.json        - baseline stats + final metrics
      baseline_text_centroid.npy
      baseline_image_centroid.npy
    confusion/                    - retrieval failure analysis CSV
    provenance/                   - week provenance JSON (written by DAG)
    eval_metrics_<label>.json
    requirements.txt

MLflow tags set on every run (searchable, used by DAG for model selection)
------------------------------------------------------------------------------
  pipeline     = "clip_product_retrieval"
  mode         = "zero_shot" | "linear_probe" | "finetune"
  stage        = "initial_train" | "retrain" | "pre_retrain_eval"
  week_label   = e.g. "week1"
  trained_on   = stem of the CSV used for training, e.g. "week1_replay"
                 (empty string for zero_shot — no training involved)

MLflow params set on every training run
------------------------------------------------------------------------------
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


# DYNAMIC PACKAGE VERSION HELPER

def _pkg_version(name: str, fallback: str = "unknown") -> str:
    """Return the installed version of a package at runtime."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return fallback

# CONFIG

@dataclass
class Config:
    csv_path:               str   = "data/processed/week1.csv"
    image_cache_dir:        str   = "data/image_cache"
    val_fraction:           float = 0.15
    seed:                   int   = 42

    model_name:             str   = "ViT-B-32"
    pretrained:             str   = "openai"

    mode:                   str   = "linear_probe"  # "zero_shot" | "linear_probe" | "finetune"
    epochs:                 int   = 10
    batch_size:             int   = 32
    lr:                     float = 1e-3
    weight_decay:           float = 1e-4
    warmup_epochs:          int   = 1
    grad_clip:              float = 1.0
    num_workers:            int   = 0
    embed_dim:              int   = 512
    max_aug_per_item:       int   = 3

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
    category_label:         str   = "all"
    corpus_emb_dir:         str   = "data/corpus_embeddings"
    combined_csv:           str   = "data/full_corpus.csv"


# EXPONENTIAL CHECKPOINT EPOCHS

def exponential_checkpoint_epochs(total_epochs: int) -> set:
    """Powers of 2 strictly less than total_epochs, plus total_epochs itself."""
    epochs, k = set(), 1
    while True:
        e = 2 ** k
        if e >= total_epochs:
            break
        epochs.add(e)
        k += 1
    epochs.add(total_epochs)
    return epochs

# IMAGE LOADING
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

# DATASET


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

# LOSS

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


# LINEAR PROBE HEAD
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

# EVALUATION
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
    img_embs, txt_embs, orig_idx = compute_embeddings(model, loader, device, probe_head)
    metrics = {**recall_at_k(img_embs, txt_embs, ks),
               **consistency_at_k(img_embs, txt_embs, ks)}
    log.info("── Eval [%s] N=%d ──", label, len(img_embs))
    for k, v in metrics.items():
        log.info("  %-22s : %.4f", k, v)
    return metrics, img_embs, txt_embs, orig_idx

# DRIFT BASELINE
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


def _log_checkpoint(state_dict: dict, filename: str):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / filename
        torch.save(state_dict, str(path))
        mlflow.log_artifact(str(path), artifact_path="checkpoints")


def _log_eval_metrics_artifact(metrics: dict, label: str = "final"):
    mlflow.log_dict(metrics, f"eval_metrics_{label}.json")


def _log_model_pytorch(model: nn.Module, artifact_path: str,
                       cfg: Config, extra_pip: list = None):
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
        pytorch_model    = model,
        artifact_path    = artifact_path,
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


# SCHEDULER

def _make_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# TRAIN ONE EPOCH
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

# COMMON DATA SETUP
def _build_loaders(cfg, tokenizer, preprocess):
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
# FULL CORPUS EMBEDDING  (for website / FAISS serving)
# ─────────────────────────────────────────────────────────────────────────────

# def embed_full_corpus(
#     model,
#     probe_head,
#     cfg: Config,
#     device: str,
#     run_id: str,
#     combined_csv: str,
#     local_save_dir: str,
# ):
#     combined_path = Path(combined_csv)
#     if not combined_path.exists():
#         log.warning("embed_full_corpus: combined CSV not found at %s — skipping.", combined_csv)
#         return

#     log.info("embed_full_corpus: loading ALL rows from %s", combined_csv)
#     df = pd.read_csv(combined_path).reset_index(drop=True)
#     log.info("embed_full_corpus: %d total rows to embed (no filtering)", len(df))

#     _, _, preprocess = open_clip.create_model_and_transforms(
#         cfg.model_name, pretrained=cfg.pretrained)
#     tokenizer = open_clip.get_tokenizer(cfg.model_name)

#     model.eval()
#     if probe_head is not None:
#         probe_head.eval()

#     all_img_embs = []
#     all_txt_embs = []
#     img_cache: dict = {}

#     batch_size = 64
#     for start in tqdm(range(0, len(df), batch_size), desc="Embedding corpus"):
#         batch_df = df.iloc[start : start + batch_size]

#         texts = batch_df["augmented_text"].fillna("").tolist()
#         tok   = tokenizer(texts)
#         with torch.no_grad():
#             txt_f = model.encode_text(tok.to(device), normalize=True)
#             if probe_head is not None:
#                 dummy_img = torch.zeros(len(batch_df), cfg.embed_dim, device=device)
#                 _, txt_f  = probe_head(dummy_img, txt_f)
#         all_txt_embs.append(txt_f.cpu().numpy())

#         # Replace the per-row image loop (lines 667–682) with this:
#         batch_img_embs = []
#         imgs_to_encode = []
#         oids_in_batch  = []

#         for _, row in batch_df.iterrows():
#             oid = int(row["original_index"])
#             if oid in img_cache:
#                 batch_img_embs.append(img_cache[oid])
#             else:
#                 img = load_image(row["image_url"], cfg.image_cache_dir)
#                 if img is None:
#                     img_cache[oid] = np.zeros(cfg.embed_dim, dtype="float32")
#                     batch_img_embs.append(img_cache[oid])
#                 else:
#                     imgs_to_encode.append((oid, preprocess(img)))
#                     batch_img_embs.append(None)  # placeholder
#             oids_in_batch.append(oid)

#         # Encode all uncached images in one forward pass
#         if imgs_to_encode:
#             oids_new, tensors = zip(*imgs_to_encode)
#             img_batch = torch.stack(tensors).to(device)
#             with torch.no_grad():
#                 f = model.encode_image(img_batch, normalize=True)
#                 if probe_head is not None:
#                     dummy = torch.zeros(len(tensors), cfg.embed_dim, device=device)
#                     f, _  = probe_head(f, dummy)
#             for oid, emb in zip(oids_new, f.cpu().numpy()):
#                 img_cache[oid] = emb

#         # Fill in placeholders
#         batch_img_embs = [img_cache[int(row["original_index"])]
#                         for _, row in batch_df.iterrows()]
#         all_img_embs.append(np.stack(batch_img_embs))

#     img_embs_all = np.concatenate(all_img_embs, axis=0).astype("float32")
#     txt_embs_all = np.concatenate(all_txt_embs, axis=0).astype("float32")

#     assert len(img_embs_all) == len(df), "Row count mismatch"

#     local_dir = Path(local_save_dir)
#     local_dir.mkdir(parents=True, exist_ok=True)

#     # ── Save full per-row parquet (keeps all augmented rows for reference) ──────
#     df_out = df.copy()
#     df_out["img_emb"] = [img_embs_all[i].tolist() for i in range(len(df))]
#     df_out["txt_emb"] = [txt_embs_all[i].tolist() for i in range(len(df))]

#     parquet_path = local_dir / "corpus_embeddings.parquet"
#     df_out.to_parquet(str(parquet_path), index=False)
#     log.info("embed_full_corpus: saved parquet → %s  (%d rows)", parquet_path, len(df_out))

#     # ── Build one embedding per unique original_index ─────────────────────────
#     #
#     # Image: already one image per product (img_cache deduplicates by oid).
#     #        We just pick the first-seen row position.
#     #
#     # Text:  AVERAGE all augmented text embeddings for the same product, then
#     #        L2-renormalise so the averaged vector is still unit-norm for IP search.
#     #
#     orig_ids_col = df["original_index"].astype(int).tolist()

#     # Group row indices by original_index (preserves insertion order → stable sort)
#     from collections import defaultdict
#     groups: dict = defaultdict(list)
#     first_seen_img: dict = {}
#     for i, oid in enumerate(orig_ids_col):
#         groups[oid].append(i)
#         if oid not in first_seen_img:
#             first_seen_img[oid] = i

#     orig_idx_keys = sorted(groups.keys())

#     img_rows = [first_seen_img[k] for k in orig_idx_keys]
#     img_np   = img_embs_all[img_rows]                            # (N_unique, D)

#     # Average + renormalise text embeddings per product
#     txt_list = []
#     max_aug_per_item = cfg.max_aug_per_item
#     import random  # optional if you want random sampling

#     for k in orig_idx_keys:
#         row_idxs = groups[k]

#         # limit number of augmentations
#         if len(row_idxs) > max_aug_per_item:
#             # Option 1: take first K (deterministic)
#             selected_idxs = row_idxs[:max_aug_per_item]

#             # Option 2 (better): random sample
#             # selected_idxs = random.sample(row_idxs, max_aug_per_item)
#         else:
#             selected_idxs = row_idxs

#         avg = txt_embs_all[selected_idxs].mean(axis=0)

#         # renormalise
#         norm = np.linalg.norm(avg)
#         if norm > 1e-9:
#             avg = avg / norm

#         txt_list.append(avg)
#     txt_np = np.stack(txt_list).astype("float32")               # (N_unique, D)

#     idx_np = np.array(orig_idx_keys, dtype="int64")             # (N_unique,)

#     assert img_np.shape == txt_np.shape, (
#         f"img/txt shape mismatch after averaging: {img_np.shape} vs {txt_np.shape}"
#     )

#     npy_img = local_dir / "corpus_image_embeddings.npy"
#     npy_txt = local_dir / "corpus_text_embeddings.npy"
#     npy_idx = local_dir / "corpus_orig_idx.npy"

#     np.save(str(npy_img), img_np)
#     np.save(str(npy_txt), txt_np)
#     np.save(str(npy_idx), idx_np)

#     log.info(
#         "embed_full_corpus: npy saved — img=%s  txt=%s  idx=%s  "
#         "N_unique=%d  N_rows(augmented)=%d  D=%d",
#         npy_img.name, npy_txt.name, npy_idx.name,
#         len(orig_idx_keys), len(df), img_np.shape[1],
#     )

#     with tempfile.TemporaryDirectory() as tmp:
#         tmp_dir = Path(tmp)
#         import shutil
#         shutil.copy(str(parquet_path), tmp_dir / "corpus_embeddings.parquet")
#         shutil.copy(str(npy_img),      tmp_dir / "corpus_image_embeddings.npy")
#         shutil.copy(str(npy_txt),      tmp_dir / "corpus_text_embeddings.npy")
#         shutil.copy(str(npy_idx),      tmp_dir / "corpus_orig_idx.npy")
#         mlflow.log_artifacts(str(tmp_dir), artifact_path="embeddings/corpus")

#     log.info("embed_full_corpus: done — %d rows  run_id=%s", len(df), run_id)

def embed_full_corpus(
    model,
    probe_head,
    cfg: Config,
    device: str,
    run_id: str,
    combined_csv: str,
    local_save_dir: str,
):
    """
    Embed the full corpus in memory-safe chunks.

    Memory budget (why the old code OOM-d):
      - Old code pre-fetched ALL ~2000 PIL images before encoding anything.
        Decoded RGB PIL images at typical product resolution (~500×500) are
        ~750 KB each → ~1.5 GB just for raw pixels, stacked on top of model
        weights (~600 MB for ViT-B-32) and growing embedding arrays → SIGKILL.

    Fix: process images in chunks of `img_chunk_size` unique products.
      1. Download / load PIL images for one chunk (ThreadPoolExecutor).
      2. Encode that chunk immediately → np.ndarray, discard PIL objects.
      3. Move to next chunk.
    Peak extra memory ≈ img_chunk_size × ~750 KB ≈ 150 MB for chunk_size=200.
    """
    combined_path = Path(combined_csv)
    if not combined_path.exists():
        log.warning("embed_full_corpus: combined CSV not found at %s — skipping.", combined_csv)
        return

    log.info("embed_full_corpus: loading ALL rows from %s", combined_csv)
    df = pd.read_csv(combined_path).reset_index(drop=True)
    log.info("embed_full_corpus: %d total rows to embed (no filtering)", len(df))

    _, _, preprocess = open_clip.create_model_and_transforms(
        cfg.model_name, pretrained=cfg.pretrained)
    tokenizer = open_clip.get_tokenizer(cfg.model_name)

    model.eval()
    if probe_head is not None:
        probe_head.eval()

    # ── Build unique-product index up front ───────────────────────────────────
    orig_ids_col = df["original_index"].astype(int).tolist()

    from collections import defaultdict
    groups: dict[int, list[int]] = defaultdict(list)
    first_seen_img: dict[int, int] = {}
    for row_i, oid in enumerate(orig_ids_col):
        groups[oid].append(row_i)
        if oid not in first_seen_img:
            first_seen_img[oid] = row_i

    # Map each original_index → its canonical image_url (first occurrence)
    oid_to_url: dict[int, str] = {
        oid: df.iloc[first_seen_img[oid]]["image_url"]
        for oid in first_seen_img
    }
    orig_idx_keys = sorted(groups.keys())

    # ── Chunked image encoding ─────────────────────────────────────────────────
    #
    # We only keep `img_chunk_size` PIL images alive at once.  After encoding
    # each chunk we store only the compact float32 embedding (512 floats ≈ 2 KB)
    # rather than the raw pixel data.
    #
    img_chunk_size = 200          # tune down if container memory is very tight
    fetch_workers  = cfg.num_workers if cfg.num_workers and cfg.num_workers > 0 else 8
    zero_emb       = np.zeros(cfg.embed_dim, dtype="float32")

    encoded_img_cache: dict[int, np.ndarray] = {}  # oid → (embed_dim,) float32

    from concurrent.futures import ThreadPoolExecutor, as_completed

    for chunk_start in range(0, len(orig_idx_keys), img_chunk_size):
        chunk_oids = orig_idx_keys[chunk_start: chunk_start + img_chunk_size]

        # ── 1. Parallel download / cache-read for this chunk ──────────────────
        def _fetch(oid: int):
            return oid, load_image(oid_to_url[oid], cfg.image_cache_dir)

        pil_cache: dict[int, Image.Image] = {}
        with ThreadPoolExecutor(max_workers=fetch_workers) as pool:
            futs = {pool.submit(_fetch, oid): oid for oid in chunk_oids}
            for fut in as_completed(futs):
                oid, img = fut.result()
                if img is not None:
                    pil_cache[oid] = img

        log.info(
            "embed_full_corpus: chunk [%d–%d] fetched %d/%d images",
            chunk_start, chunk_start + len(chunk_oids) - 1,
            len(pil_cache), len(chunk_oids),
        )

        # ── 2. Encode in sub-batches, then release PIL images immediately ─────
        encode_batch = 128
        chunk_oids_with_img = [oid for oid in chunk_oids if oid in pil_cache]
        chunk_oids_missing  = [oid for oid in chunk_oids if oid not in pil_cache]

        for b_start in range(0, len(chunk_oids_with_img), encode_batch):
            batch_oids = chunk_oids_with_img[b_start: b_start + encode_batch]
            tensors = [preprocess(pil_cache[oid]) for oid in batch_oids]
            img_batch = torch.stack(tensors).to(device)

            with torch.no_grad():
                f = model.encode_image(img_batch, normalize=True)
                if probe_head is not None:
                    dummy = torch.zeros(len(tensors), cfg.embed_dim, device=device)
                    f, _ = probe_head(f, dummy)

            for oid, emb in zip(batch_oids, f.cpu().numpy()):
                encoded_img_cache[oid] = emb

            del img_batch, tensors  # release GPU + CPU memory immediately

        # Fill zeros for images that failed to download
        for oid in chunk_oids_missing:
            encoded_img_cache[oid] = zero_emb

        # ── 3. Discard PIL images for this chunk ──────────────────────────────
        pil_cache.clear()

    log.info(
        "embed_full_corpus: image encoding done — %d/%d products embedded",
        sum(1 for v in encoded_img_cache.values() if v is not zero_emb),
        len(orig_idx_keys),
    )

    # ── Text embeddings — vectorised, no iterrows ─────────────────────────────
    all_img_embs: list[np.ndarray] = []
    all_txt_embs: list[np.ndarray] = []
    txt_batch_size = 128

    for start in tqdm(range(0, len(df), txt_batch_size), desc="Embedding corpus (text)"):
        batch_df = df.iloc[start: start + txt_batch_size]

        texts = batch_df["augmented_text"].fillna("").tolist()
        tok   = tokenizer(texts)
        with torch.no_grad():
            txt_f = model.encode_text(tok.to(device), normalize=True)
            if probe_head is not None:
                dummy_img = torch.zeros(len(batch_df), cfg.embed_dim, device=device)
                _, txt_f  = probe_head(dummy_img, txt_f)
        all_txt_embs.append(txt_f.cpu().numpy())

        # Vectorised image lookup — O(1) per row, no I/O
        batch_oids = batch_df["original_index"].astype(int).to_numpy()
        batch_img  = np.stack([encoded_img_cache[oid] for oid in batch_oids])
        all_img_embs.append(batch_img)

    img_embs_all = np.concatenate(all_img_embs, axis=0).astype("float32")
    txt_embs_all = np.concatenate(all_txt_embs, axis=0).astype("float32")

    assert len(img_embs_all) == len(df), (
        f"Row count mismatch: img_embs={len(img_embs_all)} df={len(df)}"
    )

    # ── Persist full per-row parquet (all augmented rows) ─────────────────────
    local_dir = Path(local_save_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    df_out = df.copy()
    df_out["img_emb"] = [img_embs_all[i].tolist() for i in range(len(df))]
    df_out["txt_emb"] = [txt_embs_all[i].tolist() for i in range(len(df))]
    parquet_path = local_dir / "corpus_embeddings.parquet"
    df_out.to_parquet(str(parquet_path), index=False)
    log.info("embed_full_corpus: saved parquet → %s  (%d rows)", parquet_path, len(df_out))

    # ── One embedding per unique original_index ───────────────────────────────
    # Image: first-seen row (already deduplicated by img_cache above).
    # Text:  average of up to max_aug_per_item augmented texts, then L2-renorm.

    img_rows = [first_seen_img[k] for k in orig_idx_keys]
    img_np   = img_embs_all[img_rows]                            # (N_unique, D)

    txt_list: list[np.ndarray] = []
    for k in orig_idx_keys:
        row_idxs = groups[k]
        if len(row_idxs) > cfg.max_aug_per_item:
            row_idxs = row_idxs[: cfg.max_aug_per_item]         # deterministic
        avg  = txt_embs_all[row_idxs].mean(axis=0)
        norm = np.linalg.norm(avg)
        if norm > 1e-9:
            avg = avg / norm
        txt_list.append(avg)

    txt_np = np.stack(txt_list).astype("float32")               # (N_unique, D)
    idx_np = np.array(orig_idx_keys, dtype="int64")             # (N_unique,)

    assert img_np.shape == txt_np.shape, (
        f"img/txt shape mismatch after averaging: {img_np.shape} vs {txt_np.shape}"
    )

    npy_img = local_dir / "corpus_image_embeddings.npy"
    npy_txt = local_dir / "corpus_text_embeddings.npy"
    npy_idx = local_dir / "corpus_orig_idx.npy"

    np.save(str(npy_img), img_np)
    np.save(str(npy_txt), txt_np)
    np.save(str(npy_idx), idx_np)

    log.info(
        "embed_full_corpus: npy saved — img=%s  txt=%s  idx=%s  "
        "N_unique=%d  N_rows(augmented)=%d  D=%d",
        npy_img.name, npy_txt.name, npy_idx.name,
        len(orig_idx_keys), len(df), img_np.shape[1],
    )

    with tempfile.TemporaryDirectory() as tmp:
        import shutil
        tmp_dir = Path(tmp)
        shutil.copy(str(parquet_path), tmp_dir / "corpus_embeddings.parquet")
        shutil.copy(str(npy_img),      tmp_dir / "corpus_image_embeddings.npy")
        shutil.copy(str(npy_txt),      tmp_dir / "corpus_text_embeddings.npy")
        shutil.copy(str(npy_idx),      tmp_dir / "corpus_orig_idx.npy")
        mlflow.log_artifacts(str(tmp_dir), artifact_path="embeddings/corpus")

    log.info("embed_full_corpus: done — %d rows  run_id=%s", len(df), run_id)
    
# MODE 1 — ZERO SHOT

def run_zero_shot(cfg: Config, device: str, run_id: str) -> dict:
    log.info("═══ ZERO SHOT  [week=%s] ═══", cfg.week_label)

    model, _, preprocess = open_clip.create_model_and_transforms(
        cfg.model_name, pretrained=cfg.pretrained)
    tokenizer = open_clip.get_tokenizer(cfg.model_name)
    model = model.to(device).eval()

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

    metrics, img_embs, txt_embs, orig_idx = evaluate(
        model, val_loader, device, cfg.eval_ks, label="zero_shot")

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


# MODE 2 — LINEAR PROBE

def run_linear_probe(cfg: Config, device: str, run_id: str,
                     stage: str = "retrain") -> dict:
    log.info("═══ LINEAR PROBE  [week=%s  stage=%s] ═══", cfg.week_label, stage)

    model, _, preprocess = open_clip.create_model_and_transforms(
        cfg.model_name, pretrained=cfg.pretrained)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False

    tokenizer = open_clip.get_tokenizer(cfg.model_name)

    probe = LinearProbeHead(cfg.embed_dim, cfg.probe_hidden,
                            cfg.probe_dropout).to(device)

    _set_run_tags(cfg, stage=stage)
    _log_config(cfg)
    _log_requirements()

    train_loader, val_loader, train_df, val_df = _build_loaders(cfg, tokenizer, preprocess)

    criterion = InfoNCELoss(logit_scale=model.logit_scale).to(device)
    optimizer = torch.optim.AdamW(
        probe.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    steps_per_epoch = max(len(train_loader), 1)
    total_steps     = cfg.epochs * steps_per_epoch
    warmup_steps    = cfg.warmup_epochs * steps_per_epoch
    scheduler = _make_scheduler(optimizer, warmup_steps, total_steps)

    ckpt_epochs  = exponential_checkpoint_epochs(cfg.epochs)
    best_recall1 = -1.0
    best_metrics = {}

    for epoch in range(1, cfg.epochs + 1):
        avg_loss = _train_one_epoch(
            model, probe, train_loader, optimizer, criterion,
            scheduler, cfg.grad_clip, device, epoch, cfg.epochs, "linear_probe")
        mlflow.log_metric("train_loss", avg_loss, step=epoch)

        metrics, img_embs, txt_embs, orig_idx = evaluate(
            model, val_loader, device, cfg.eval_ks, probe_head=probe,
            label=f"probe_ep{epoch}")

        safe = {f"val_{k.replace('@','_')}": v for k, v in metrics.items()}
        mlflow.log_metrics(safe, step=epoch)

        if epoch in ckpt_epochs:
            _log_checkpoint({
                "epoch":       epoch,
                "probe_state": probe.state_dict(),
                "metrics":     metrics,
            }, f"probe_ep{epoch}.pt")

        if metrics.get("Recall@1", 0.0) > best_recall1:
            best_recall1 = metrics["Recall@1"]
            best_metrics = metrics
            _log_checkpoint({
                "epoch":       epoch,
                "probe_state": probe.state_dict(),
                "metrics":     metrics,
            }, "probe_best.pt")
            log.info("  ✓ New best Recall@1=%.4f  epoch=%d", best_recall1, epoch)

    _log_eval_metrics_artifact(best_metrics, label="probe_best")
    if img_embs is not None:
        _log_retrieval_failures(img_embs, txt_embs, val_df, cfg.top_failures)
        _log_drift_reference(txt_embs, img_embs, best_metrics, cfg, run_id)

    _log_model_pytorch(model, "model", cfg)

    safe_best = {f"best_{k.replace('@', '_')}": v for k, v in best_metrics.items()}
    mlflow.log_metrics(safe_best)

    log.info(
        "Linear-probe complete [week=%s  stage=%s] — "
        "best Recall@1=%.4f  Consistency@1=%.4f  run_id=%s",
        cfg.week_label, stage,
        best_metrics.get("Recall@1",      0.0),
        best_metrics.get("Consistency@1", 0.0),
        run_id,
    )
    return best_metrics


# MODE 3 — FINETUNE

def run_finetune(cfg: Config, device: str, run_id: str,
                 stage: str = "retrain") -> dict:
    log.info("═══ FINETUNE  [week=%s  stage=%s] ═══", cfg.week_label, stage)

    model, _, preprocess = open_clip.create_model_and_transforms(
        cfg.model_name, pretrained=cfg.pretrained)
    model = model.to(device)
    tokenizer = open_clip.get_tokenizer(cfg.model_name)

    _freeze_clip_except_last_n(model, cfg.unfreeze_last_n_blocks)

    _set_run_tags(cfg, stage=stage)
    _log_config(cfg)
    _log_requirements()

    train_loader, val_loader, train_df, val_df = _build_loaders(cfg, tokenizer, preprocess)

    criterion = InfoNCELoss(logit_scale=model.logit_scale).to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr, weight_decay=cfg.weight_decay)

    steps_per_epoch = max(len(train_loader), 1)
    total_steps     = cfg.epochs * steps_per_epoch
    warmup_steps    = cfg.warmup_epochs * steps_per_epoch
    scheduler = _make_scheduler(optimizer, warmup_steps, total_steps)

    ckpt_epochs  = exponential_checkpoint_epochs(cfg.epochs)
    best_recall1 = -1.0
    best_metrics = {}
    img_embs = txt_embs = None

    for epoch in range(1, cfg.epochs + 1):
        avg_loss = _train_one_epoch(
            model, None, train_loader, optimizer, criterion,
            scheduler, cfg.grad_clip, device, epoch, cfg.epochs, "finetune")
        mlflow.log_metric("train_loss", avg_loss, step=epoch)

        metrics, img_embs, txt_embs, orig_idx = evaluate(
            model, val_loader, device, cfg.eval_ks, label=f"finetune_ep{epoch}")

        safe = {f"val_{k.replace('@','_')}": v for k, v in metrics.items()}
        mlflow.log_metrics(safe, step=epoch)

        if epoch in ckpt_epochs:
            _log_checkpoint({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "metrics":     metrics,
            }, f"finetune_ep{epoch}.pt")

        if metrics.get("Recall@1", 0.0) > best_recall1:
            best_recall1 = metrics["Recall@1"]
            best_metrics = metrics
            _log_checkpoint({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "metrics":     metrics,
            }, "finetune_best.pt")
            log.info("  ✓ New best Recall@1=%.4f  epoch=%d", best_recall1, epoch)

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


# CHECKPOINT LOADERS  (used by DAG and drift_detection.py)

def load_zero_shot_model(cfg: Config, device: str):
    model, _, preprocess = open_clip.create_model_and_transforms(
        cfg.model_name, pretrained=cfg.pretrained)
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(cfg.model_name)
    log.info("Loaded zero-shot CLIP  model=%s  pretrained=%s",
             cfg.model_name, cfg.pretrained)
    return model, preprocess, tokenizer


def load_linear_probe_from_mlflow(run_id: str, cfg: Config, device: str):
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
        "Loaded linear-probe from MLflow — run_id=%s  epoch=%s  Recall@1=%.4f",
        run_id,
        ckpt.get("epoch", "?"),
        ckpt.get("metrics", {}).get("Recall@1", 0.0),
    )
    return model, probe, preprocess, tokenizer


def load_finetune_from_mlflow(run_id: str, cfg: Config, device: str):
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


# RUN VERIFICATION HELPER  (used by DAG before loading a model)
def get_run_tags(run_id: str, tracking_uri: str) -> dict:
    try:
        client = mlflow.MlflowClient(tracking_uri=tracking_uri)
        run    = client.get_run(run_id)
        return dict(run.data.tags)
    except Exception as e:
        log.warning("Could not fetch tags for run_id=%s: %s", run_id, e)
        return {}


# MAIN  (CLI entry point)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode",            required=True,
                   choices=["zero_shot", "linear_probe", "finetune"])
    p.add_argument("--csv_path",             default="data/processed/week1.csv")
    p.add_argument("--epochs",          type=int,   default=10)
    p.add_argument("--batch_size",      type=int,   default=32)
    p.add_argument("--lr",              type=float, default=None)
    p.add_argument("--model_name",      default="ViT-B-32")
    p.add_argument("--pretrained",      default="openai")
    p.add_argument("--unfreeze_blocks", type=int,   default=2)
    p.add_argument("--cache_dir",       default="data/image_cache")
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--mlflow_tracking_uri",    default="sqlite:///mlflow.db")
    p.add_argument("--experiment",      default="clip_product_retrieval")
    p.add_argument("--run_name",        default=None)
    p.add_argument("--week_label",      default="week1")
    p.add_argument("--category_label",  default="all")
    p.add_argument("--stage",           default="retrain",
                   choices=["initial_train", "retrain", "pre_retrain_eval"])
    p.add_argument("--top_failures",    type=int,   default=20)
    # dvc.yaml train stage passes --run_id_out so the run_id is written
    # to disk for downstream stages. Without this arg parse_args() would error
    # with "unrecognised arguments: --run_id_out ..." when called via DVC.
    p.add_argument("--run_id_out",      default=None,
                   help="Optional path to write the MLflow run_id after training.")
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
        csv_path               = args.csv_path,
        image_cache_dir        = args.cache_dir,
        model_name             = args.model_name,
        pretrained             = args.pretrained,
        mode                   = args.mode,
        epochs                 = args.epochs,
        batch_size             = args.batch_size,
        lr                     = lr,
        unfreeze_last_n_blocks = args.unfreeze_blocks,
        mlflow_tracking_uri    = args.mlflow_tracking_uri,
        mlflow_experiment      = args.experiment,
        mlflow_run_name        = args.run_name,
        seed                   = args.seed,
        week_label             = args.week_label,
        category_label         = args.category_label,
        top_failures           = args.top_failures,
        max_aug_per_item       = 3
    )

    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.mlflow_experiment)
    mlflow.pytorch.autolog(log_every_n_epoch=1, log_models=False)

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

    # FIX: write run_id to --run_id_out if supplied (required by dvc.yaml
    # train stage so downstream stages can resolve the MLflow run_id).
    if args.run_id_out:
        out = Path(args.run_id_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(run_id)
        log.info("run_id written to %s", out)


if __name__ == "__main__":
    main()