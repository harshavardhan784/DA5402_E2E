"""
src/clip_mlflow_wrapper.py
──────────────────────────
Custom MLflow pyfunc wrapper that bundles:
  - CLIP backbone (open_clip ViT-B-32)
  - LinearProbeHead (optional — None for zero_shot / finetune)
  - FAISS image index
  - corpus metadata (meta.json)

Registered as a single model in MLflow Model Registry.
Served via `mlflow models serve` and called from FastAPI (search_api → routes/search.py).

Input  (pandas DataFrame, one row per query):
  Columns:
    query_text  : str   (optional — "" to skip)
    query_image : str   (optional — base64-encoded JPEG/PNG, "" to skip)
    top_k       : int   (default 10)
    search_mode : str   ("image" | "text" | "both")
                         "image" → rank by image-embedding cosine sim
                         "text"  → rank by text-embedding cosine sim
                         "both"  → average image+text vectors, then rank

Output (pandas DataFrame):
  Columns: original_index, image_url, augmented_text, category_2, category_3,
           score, search_mode
  The FastAPI layer calls /invocations twice (mode=image, mode=text) and
  zips the two lists into a DualSearchResponse.
"""

import base64
import io
import json
import os
from pathlib import Path
from typing import Optional

import mlflow.pyfunc
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image


class CLIPRetrieverModel(mlflow.pyfunc.PythonModel):
    """
    Bundles CLIP + probe + FAISS into one MLflow pyfunc model.

    Artifacts expected at load time (passed via artifacts dict):
      clip_checkpoint  : path to probe_best.pt or finetune_best.pt (or "zero_shot")
      faiss_index      : path to data/faiss/index.bin
      faiss_meta       : path to data/faiss/meta.json
      model_config     : path to model_config.json
                         (model_name, pretrained, mode, embed_dim)
    """

    def load_context(self, context):
        import open_clip
        import faiss
        from clip_experiments import LinearProbeHead

        cfg_path = context.artifacts["model_config"]
        with open(cfg_path) as f:
            cfg = json.load(f)

        self.model_name = cfg["model_name"]
        self.pretrained = cfg["pretrained"]
        self.mode       = cfg["mode"]           # "zero_shot" | "linear_probe" | "finetune"
        self.embed_dim  = cfg.get("embed_dim", 512)
        self.device     = "cuda" if torch.cuda.is_available() else "cpu"

        # ── CLIP backbone ────────────────────────────────────────────────────
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained=self.pretrained)
        self.model     = self.model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer(self.model_name)

        # ── Linear probe head (optional) ──────────────────────────────────
        self.probe = None
        ckpt_path  = context.artifacts.get("clip_checkpoint", "")
        if ckpt_path and ckpt_path != "zero_shot" and Path(ckpt_path).exists():
            probe_hidden  = cfg.get("probe_hidden", None)
            probe_dropout = cfg.get("probe_dropout", 0.1)
            self.probe    = LinearProbeHead(self.embed_dim, probe_hidden, probe_dropout)
            ckpt = torch.load(ckpt_path, map_location=self.device)
            if "probe_state" in ckpt:
                self.probe.load_state_dict(ckpt["probe_state"])
            elif "model_state" in ckpt:
                self.probe.load_state_dict(ckpt["model_state"])
            self.probe = self.probe.to(self.device).eval()

        # ── FAISS indexes ──────────────────────────────────────────────────────────
        import faiss
        self.img_index = faiss.read_index(context.artifacts["faiss_index_img"])
        self.txt_index = faiss.read_index(context.artifacts["faiss_index_txt"])

        # ── Corpus text embeddings (optional, for brute-force text search fallback) 
        txt_emb_key = "corpus_text_embeddings"
        if txt_emb_key in context.artifacts:
            self._text_embs = np.load(context.artifacts[txt_emb_key]).astype("float32")
        else:
            self._text_embs = None
            
    # ── Encoders ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _encode_text(self, text: str) -> np.ndarray:
        tok = self.tokenizer([text]).to(self.device)
        f   = self.model.encode_text(tok, normalize=True)
        if self.probe is not None:
            dummy = torch.zeros(1, self.embed_dim, device=self.device)
            _, f  = self.probe(dummy, f)
        return F.normalize(f, dim=-1).squeeze(0).cpu().numpy()

    @torch.no_grad()
    def _encode_image(self, b64_str: str) -> np.ndarray:
        img_bytes = base64.b64decode(b64_str)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        t   = self.preprocess(img).unsqueeze(0).to(self.device)
        f   = self.model.encode_image(t, normalize=True)
        if self.probe is not None:
            dummy = torch.zeros(1, self.embed_dim, device=self.device)
            f, _  = self.probe(f, dummy)
        return F.normalize(f, dim=-1).squeeze(0).cpu().numpy()

    # ── Search helpers ────────────────────────────────────────────────────────

    def _search_by_image_emb(self, query_vec: np.ndarray, top_k: int) -> list[dict]:
        """Search the image-embedding FAISS index."""
        scores, indices = self.img_index.search(
            query_vec.astype("float32").reshape(1, -1), top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.meta):
                continue
            entry = dict(self.meta[idx])
            entry["score"]       = float(score)
            entry["search_mode"] = "image"
            results.append(entry)
        return results

    def _search_by_text_emb(self, query_vec: np.ndarray, top_k: int) -> list[dict]:
        """Search the text FAISS index (or brute-force against stored text embeddings)."""
        if self._text_embs is None:
            # true text FAISS index built by the DAG
            scores, indices = self.txt_index.search(
                query_vec.astype("float32").reshape(1, -1), top_k)
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or idx >= len(self.meta):
                    continue
                entry = dict(self.meta[idx])
                entry["score"]       = float(score)
                entry["search_mode"] = "text"
                results.append(entry)
            return results

        # brute-force cosine over precomputed corpus embeddings (optional path)
        sims    = (self._text_embs @ query_vec.astype("float32")).ravel()
        top_idx = np.argsort(-sims)[:top_k]
        results = []
        for idx in top_idx:
            if idx >= len(self.meta):
                continue
            entry = dict(self.meta[idx])
            entry["score"]       = float(sims[idx])
            entry["search_mode"] = "text"
            results.append(entry)
        return results

    # ── predict ───────────────────────────────────────────────────────────────

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        """
        model_input columns:
          query_text  : str
          query_image : str  (base64)
          top_k       : int
          search_mode : "image" | "text"   (FastAPI issues one call per mode)

        Returns a DataFrame with result rows for that search_mode.
        """
        all_results = []

        for _, row in model_input.iterrows():
            query_text  = str(row.get("query_text",  "") or "")
            query_image = str(row.get("query_image", "") or "")
            top_k       = int(row.get("top_k", 10))
            mode        = str(row.get("search_mode", "image")).lower()

            # ── Build query vector ────────────────────────────────────────
            if query_text and query_image:
                t_vec = self._encode_text(query_text)
                i_vec = self._encode_image(query_image)
                q_vec = (t_vec + i_vec) / 2.0
                q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-9)
            elif query_text:
                q_vec = self._encode_text(query_text)
            elif query_image:
                q_vec = self._encode_image(query_image)
            else:
                continue  # empty query — skip

            # ── Search ────────────────────────────────────────────────────
            if mode == "text":
                results = self._search_by_text_emb(q_vec, top_k)
            elif mode == "both":
                img_res = self._search_by_image_emb(q_vec, top_k)
                txt_res = self._search_by_text_emb(q_vec, top_k)
                results = img_res + txt_res   # caller gets both lists merged; they can split by search_mode field

            else:  # "image" or fallback
                results = self._search_by_image_emb(q_vec, top_k)

            all_results.extend(results)

        return pd.DataFrame(all_results) if all_results else pd.DataFrame()