"""
src/augment_texts.py
--------------------------
Rule-based text augmentation for product image-text pairs.

Seven agumentation strategies + deduplication.

CLI usage:
    python src/data/augment_texts.py \
        --input  data/incoming/week1.csv \
        --output data/processed/week1_augmented.csv \
        [--n_samples 4000] [--seed 42]
"""

import re
import random
import logging
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# NOISE / PATTERN HELPERS

_NOISE_PAT = re.compile(
    r"\xa0+"
    r"|\s{2,}"
    r"|\(Pack of \d+\)"
    r"|\bCombo\b"
    r"|\s*\|\s*",
    re.IGNORECASE,
)
_BRACKET_PAT = re.compile(r"\([^)]{0,40}\)$")
_SQUARE_PAT  = re.compile(r"\[[^\]]{0,60}\]")
_COLOUR_PAT = re.compile(
    r"\b(red|blue|green|black|white|yellow|orange|pink|purple|grey|gray|"
    r"brown|navy|beige|gold|silver|multicolor|multicolour)\b",
    re.IGNORECASE,
)
_SIZE_PAT = re.compile(
    r"\b(\d+\s*(?:cm|mm|m|inch|inches|kg|g|gm|gms|ml|l|ltr|ft|oz|"
    r"xl|xxl|xs|s|m|l)\b)",
    re.IGNORECASE,
)
_STOP_WORDS = {
    "the", "a", "an", "and", "or", "for", "of", "in", "to", "by",
    "with", "is", "it", "at", "on", "as", "be", "its", "this",
    "that", "are", "was", "were", "from", "has", "have",
}


def _clean_title(title: str) -> str:
    t = _NOISE_PAT.sub(" ", str(title))
    t = _SQUARE_PAT.sub("", t)     # strip [128GB], [Pack of 2], etc.
    t = _BRACKET_PAT.sub("", t)
    return re.sub(r"\s+", " ", t).strip()


def _extract_keywords(text: str, min_len: int = 3) -> list:
    words = re.findall(r"[a-zA-Z0-9]+", text)
    return [w for w in words if w.lower() not in _STOP_WORDS and len(w) >= min_len]


def _price_bucket(price) -> str:
    if pd.isna(price):
        return ""
    if price < 200:   return "budget"
    if price < 500:   return "affordable"
    if price < 2000:  return "mid-range"
    return "premium"


# DATA LOADING

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8", encoding_errors="replace")
    df = df[df["title"].notna() & df["image_links"].notna()].copy()
    df = df.reset_index(drop=True)
    for col in ["selling_price", "mrp"]:
        df[col] = df[col].astype(str).str.replace(r"[₹,\s]", "", regex=True)
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# AUGMENTATION STRATEGIES

def aug_title_clean(row: pd.Series):
    return _clean_title(row["title"])


def aug_category_prefix(row: pd.Series):
    cats = []
    for col in ["category_2", "category_3"]:
        v = str(row.get(col, "")).strip()
        if v and v.lower() not in ("nan", ""):
            cats.append(v.strip())
    cat_str = " > ".join(cats) if cats else ""
    title = _clean_title(row["title"])
    return f"{cat_str}: {title}" if cat_str else None


def aug_price_context(row: pd.Series):
    price = row.get("selling_price")
    bucket = _price_bucket(price)
    if not bucket:
        return None
    kw = " ".join(_extract_keywords(_clean_title(row["title"]))[:6])
    if pd.notna(price) and price > 0:
        return f"{bucket} {kw} under ₹{int(price)}"
    return f"{bucket} {kw}"


def aug_highlights_fusion(row: pd.Series):
    highlights = str(row.get("highlights", ""))
    if highlights.lower() in ("nan", ""):
        return None
    parts = re.split(r"[,\|]", highlights)
    attrs = [p.split(":", 1)[-1].strip() for p in parts]
    attrs = [a for a in attrs if len(a) > 1]
    if not attrs:
        return None
    title_kw = _extract_keywords(_clean_title(row["title"]))[:5]
    return " ".join(title_kw) + " " + " ".join(attrs)


def aug_description_snippet(row: pd.Series):
    desc = str(row.get("description", ""))
    if desc.lower() in ("nan", "") or len(desc) < 10:
        return None
    snippet = " ".join(desc.split()[:25])
    title_words = set(_clean_title(row["title"]).lower().split())
    snippet_words = set(snippet.lower().split())
    if title_words and len(snippet_words & title_words) / len(title_words) > 0.70:
        return None
    return snippet


def aug_keyword_drop(row: pd.Series, drop_frac: float = 0.30, seed=None):
    rng = random.Random(seed if seed is not None else random.randint(0, 999999))
    words = _clean_title(row["title"]).split()
    kept = [w for w in words if w.lower() not in _STOP_WORDS and rng.random() > drop_frac]
    return " ".join(kept) if len(kept) >= 3 else None


def aug_attribute_reorder(row: pd.Series, seed=None):
    rng = random.Random(seed if seed is not None else random.randint(0, 999999))
    words = _clean_title(row["title"]).split()
    attr_indices = [i for i, w in enumerate(words) if _COLOUR_PAT.match(w) or _SIZE_PAT.match(w)]
    if len(attr_indices) < 2:
        return None
    core = [w for i, w in enumerate(words) if i not in set(attr_indices)]
    attrs = [words[i] for i in attr_indices]
    rng.shuffle(attrs)
    result = core[:]
    for pos, attr in zip(sorted(attr_indices), attrs):
        result.insert(min(pos, len(result)), attr)
    return " ".join(result)


AUGMENTERS = {
    "title_clean":         aug_title_clean,
    "category_prefix":     aug_category_prefix,
    "price_context":       aug_price_context,
    "highlights_fusion":   aug_highlights_fusion,
    "description_snippet": aug_description_snippet,
    "keyword_drop":        aug_keyword_drop,
    "attribute_reorder":   aug_attribute_reorder,
}

METHOD_PRIORITY = [
    "description_snippet",
    "price_context",
    "highlights_fusion",
    "category_prefix",
    "keyword_drop",
    "title_clean",
    "attribute_reorder",
]


# BUILD AUGMENTED DATASET

def build_augmented_dataset(
    csv_path: str,
    n_samples: int = 4000,
    category_filter: str = None,
    aug_methods: list = None,
    seed: int = 42,
    deduplicate: bool = True,
) -> pd.DataFrame:
    random.seed(seed)
    np.random.seed(seed)

    df = load_data(csv_path)
    if category_filter:
        df = df[df["category_1"].str.strip() == category_filter].reset_index(drop=True)
        log.info("Filtered to category '%s': %d rows", category_filter, len(df))

    df_sample = df.sample(n=min(n_samples, len(df)), random_state=seed).reset_index(drop=True)
    log.info("Sampled %d rows for augmentation", len(df_sample))

    methods = aug_methods or list(AUGMENTERS.keys())
    records = []
    for idx, row in df_sample.iterrows():
        for method in methods:
            row_pos = df_sample.index.get_loc(idx)
            method_seed = seed * 10_000 + row_pos
            if method in ("keyword_drop", "attribute_reorder"):
                text = AUGMENTERS[method](row, seed=method_seed)
            else:
                text = AUGMENTERS[method](row)
            if text is None or len(text.strip()) < 5:
                text = aug_title_clean(row)
            records.append({
                "original_index": row["h_index"],
                "image_url":      row["image_links"],
                "method":         method,
                "augmented_text": text.strip(),
                "category_2":     row["category_2"],
                "category_3":     row["category_3"],
                "selling_price":  row["selling_price"],
            })

    result = pd.DataFrame(records)

    if deduplicate:
        before = len(result)
        priority_map = {m: i for i, m in enumerate(METHOD_PRIORITY)}
        result["_priority"] = result["method"].map(lambda m: priority_map.get(m, len(METHOD_PRIORITY)))
        result = result.sort_values(["original_index", "_priority"])
        result = result.drop_duplicates(subset=["original_index", "augmented_text"], keep="first")
        result = result.drop(columns=["_priority"]).reset_index(drop=True)
        log.info("Dedup: %d → %d rows (%d removed)", before, len(result), before - len(result))

    return result


# INSPECT / BENCHMARK UTILITIES

def inspect_row(index: int, csv_path: str) -> None:
    df = load_data(csv_path)
    if index >= len(df):
        print(f"Index {index} out of range ({len(df)} rows).")
        return
    row = df.iloc[index]
    sep = "─" * 70
    print(sep)
    print(f"ROW {index}  |  {row['category_2'].strip()} > {row['category_3'].strip()}")
    print(sep)
    print(f"  RAW TITLE   : {row['title']}")
    print(f"  PRICE       : ₹{row['selling_price']}  (MRP ₹{row['mrp']})")
    print(f"  HIGHLIGHTS  : {str(row['highlights'])[:100]}")
    print(f"  DESCRIPTION : {str(row['description'])[:120]}")
    print(sep)
    print("AUGMENTATIONS:\n")
    for name, fn in AUGMENTERS.items():
        result = fn(row)
        status = "✓" if result else "✗ (fallback)"
        print(f"  [{name:22s}] {status}")
        print(f"    → {result if result else aug_title_clean(row)}\n")


def benchmark(csv_path: str, n_samples: int = 500, seed: int = 42) -> pd.DataFrame:
    random.seed(seed)
    df = load_data(csv_path)
    df_s = df.sample(n=min(n_samples, len(df)), random_state=seed)
    stats = defaultdict(list)
    for _, row in df_s.iterrows():
        title_words = set(_clean_title(row["title"]).lower().split())
        for name, fn in AUGMENTERS.items():
            result = fn(row)
            stats[name + "_raw"].append(result)
            if result:
                words = result.lower().split()
                stats[name + "_len"].append(len(result))
                stats[name + "_div"].append(len(set(words)) / max(len(words), 1))
                stats[name + "_ovl"].append(len(set(words) & title_words) / max(len(title_words), 1))

    rows = []
    for name in AUGMENTERS:
        raw = stats[name + "_raw"]
        non_none = [r for r in raw if r is not None]
        coverage = len(non_none) / len(raw) * 100
        rows.append({
            "method":          name,
            "coverage_%":      round(coverage, 1),
            "avg_length":      round(np.mean(stats[name + "_len"]) if non_none else 0, 1),
            "vocab_diversity": round(np.mean(stats[name + "_div"]) if non_none else 0, 3),
            "title_overlap":   round(np.mean(stats[name + "_ovl"]) if non_none else 0, 3),
        })

    result_df = pd.DataFrame(rows)
    r = result_df.copy()
    for col in ["coverage_%", "avg_length", "vocab_diversity"]:
        mn, mx = r[col].min(), r[col].max()
        r[col + "_n"] = (r[col] - mn) / (mx - mn + 1e-9)
    r["novelty_n"] = 1 - (r["title_overlap"] - r["title_overlap"].min()) / \
                     (r["title_overlap"].max() - r["title_overlap"].min() + 1e-9)
    result_df["composite_score"] = (
        0.35 * r["coverage_%_n"] + 0.20 * r["avg_length_n"] +
        0.25 * r["vocab_diversity_n"] + 0.20 * r["novelty_n"]
    ).round(3)
    return result_df.sort_values("composite_score", ascending=False).reset_index(drop=True)


# CLI  (called by dvc.yaml augment stage and Airflow augment_texts task)

def _parse_args():
    p = argparse.ArgumentParser(description="Augment a raw Flipkart CSV.")
    p.add_argument("--input",    required=True, help="Path to raw incoming CSV.")
    p.add_argument("--output",   required=True, help="Path to write augmented CSV.")
    p.add_argument("--n_samples", type=int, default=4000)
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--category_filter", default=None)
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args()
    df_out = build_augmented_dataset(
        csv_path        = args.input,
        n_samples       = args.n_samples,
        seed            = args.seed,
        category_filter = args.category_filter,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(args.output, index=False)
    log.info("Saved %d rows → %s", len(df_out), args.output)