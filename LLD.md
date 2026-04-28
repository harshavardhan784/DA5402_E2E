# Low-Level Design (LLD)
## CLIP Product Retrieval — Continual Learning MLOps Pipeline

---

## 1. API Endpoint Definitions (I/O Specification)

This section defines every HTTP endpoint exposed by the **Search API** (`src/search_api.py` + `routes/`), running on port 8000. All request and response bodies are JSON unless noted.

---

### 1.1 Search Endpoints

#### `POST /search/text`

**Purpose:** Encode a text query and return the top-K most similar products from both FAISS indexes.

**Request:**
```json
{
  "query": "string (required, non-empty)",
  "top_k": "integer (default: 10)"
}
```

**Response (200):**
```json
{
  "image_results": [
    {
      "original_index": 1234,
      "image_url": "https://...",
      "title": "Samsung Galaxy S23 ...",
      "category_2": "Electronics",
      "category_3": "Smartphones",
      "score": 0.872
    }
  ],
  "text_results": [ /* same shape, ranked by text-embedding similarity */ ]
}
```

**Errors:**
- `400` — empty query string
- `422` — missing required field
- `500` — CLIP encoding failure or FAISS error

---

#### `POST /search/image`

**Purpose:** Encode a base64 image and return top-K visually similar products.

**Request:**
```json
{
  "image_b64": "string (base64-encoded JPEG/PNG, required)",
  "top_k": "integer (default: 10)"
}
```

**Response:** Same shape as `/search/text`.

**Errors:**
- `400` — empty `image_b64`
- `422` — missing field
- `500` — PIL decode failure or FAISS error

---

#### `POST /search/multimodal`

**Purpose:** Average text and image embeddings (whichever are present) and search.

**Request:**
```json
{
  "query": "string (optional)",
  "image_b64": "string (optional base64)",
  "top_k": "integer (default: 10)"
}
```

**Constraint:** At least one of `query` or `image_b64` must be non-empty.

**Response:** Same shape as `/search/text`.

**Errors:**
- `400` — both `query` and `image_b64` are empty
- `500` — encoding or FAISS failure

---

### 1.2 Management Endpoints

#### `GET /health`

**Purpose:** Liveness probe. Returns FAISS index state and MLflow reachability.

**Response (200 always):**
```json
{
  "status": "ok",
  "img_vectors": 12500,
  "txt_vectors": 12500,
  "meta_entries": 12500,
  "dim": 512,
  "mlflow_healthy": true,
  "mlflow_url": "http://localhost:5001"
}
```

---

#### `GET /ready`

**Purpose:** Readiness probe. Returns `503` until FAISS index is loaded.

**Response (200):**
```json
{ "status": "ready", "vectors": 12500 }
```

**Response (503):**
```json
{ "status": "not_ready", "reason": "empty index" }
```

---

#### `POST /reload`

**Purpose:** Hot-swap both FAISS indexes from disk without restarting the server. Called by the Airflow DAG `reload_faiss_index` task after each pipeline run.

**Request:** Empty body `{}`.

**Response (200):**
```json
{ "status": "ok", "vector_count": 12500 }
```

**Errors:**
- `500` — index files not found or integrity check failure (ntotal ≠ len(meta))

---

### 1.3 Pipeline Endpoints

#### `POST /pipeline/trigger`

**Purpose:** Trigger the Airflow DAG `clip_product_retrieval_pipeline` via its REST API.

**Request:**
```json
{ "conf": {} }
```

**Response (200):**
```json
{ "dag_run_id": "manual__2024-01-01T00:00:00+00:00", "state": "queued", ... }
```

**Errors:**
- `502` — Airflow API unreachable or returned non-2xx

---

#### `GET /pipeline/status`

**Purpose:** Return the most recent DAG run record.

**Response (200):**
```json
{
  "dag_run_id": "scheduled__2024-01-01",
  "state": "success",
  "start_date": "2024-01-01T10:00:00",
  "end_date": "2024-01-01T10:12:34"
}
```

---

### 1.4 Terminal / Operational Endpoints

These endpoints expose non-sensitive system state to the UI terminal without requiring container shell access.

#### `GET /terminal/faiss-info`

**Purpose:** Return current FAISS index metadata.

**Response (200):**
```json
{
  "ntotal": 12500,
  "dimension": 512,
  "index_path": "/opt/project/data/faiss/index_img.bin",
  "meta_count": 12500,
  "sample": [ { "original_index": 1, "image_url": "...", "title": "..." } ]
}
```

---

#### `GET /terminal/env`

**Purpose:** Expose a whitelisted set of non-secret environment variables for debugging.

**Response (200):** Object with keys: `DATA_DIR`, `SRC_DIR`, `ROOT_DIR`, `MLFLOW_TRACKING_URI`, `MLFLOW_SERVE_URL`, `SEARCH_API_URL`, `PYTHONPATH`, `HOSTNAME`.

---

#### `GET /terminal/logs?tail=N`

**Purpose:** Return the last N lines of the uvicorn log file. Max `tail=500`.

**Response (200):**
```json
{ "lines": ["2024-01-01 10:00:00 INFO ..."], "total_lines": 1240 }
```

---

#### `POST /terminal/dvc-status`

**Purpose:** Run `dvc status` in `ROOT_DIR` and return stdout/stderr.

**Request:** Empty body.

**Response (200):**
```json
{ "stdout": "Data and pipelines are up to date.", "stderr": "", "returncode": 0 }
```

---

#### `POST /terminal/dvc-push`

**Purpose:** Run `dvc push` to sync DVC-tracked data to remote.

**Response (200):** Same shape as `/terminal/dvc-status`.

---

#### `GET /metrics`

**Purpose:** Prometheus metrics scrape endpoint. Exposed automatically by `prometheus_fastapi_instrumentator`.

**Response:** Plain text Prometheus exposition format (content-type: `text/plain`).

---

## 2. Module Breakdown

### 2.1 `src/augment_texts.py`

**Purpose:** Converts a raw Flipkart product CSV into a multi-text training set for CLIP contrastive learning.

**Core data contract:**

| Input column | Required | Notes |
|---|---|---|
| `h_index` | Yes | Unique product identifier → becomes `original_index` |
| `title` | Yes | Raw product title |
| `image_links` | Yes | URL of product image |
| `category_2`, `category_3` | No | Used by `aug_category_prefix` |
| `selling_price`, `mrp` | No | Used by `aug_price_context` |
| `highlights` | No | Used by `aug_highlights_fusion` |
| `description` | No | Used by `aug_description_snippet` |

**Output columns:** `original_index`, `image_url`, `method`, `augmented_text`, `category_2`, `category_3`, `selling_price`

**The 7 augmentation strategies:**
```
title_clean         — noise removal (NBSP, brackets, pipes) + whitespace normalisation
category_prefix     — "Electronics > Smartphones: Samsung Galaxy S23"
price_context       — "mid-range Samsung Galaxy under ₹45000"
highlights_fusion   — title keywords + parsed highlight attributes
description_snippet — first 25 words of description (filtered for >70% title overlap)
keyword_drop        — title with 30% of non-stopwords randomly dropped
attribute_reorder   — colour/size attributes shuffled within title
```

**Key function:** `build_augmented_dataset(csv_path, n_samples=4000, seed=42) -> pd.DataFrame`

---

### 2.2 `src/clip_experiments.py`

**Purpose:** All CLIP training logic, evaluation, checkpoint management, and MLflow artifact logging.

**`Config` dataclass — key fields:**
```python
model_name:             str   = "ViT-B-32"
pretrained:             str   = "openai"
mode:                   str   = "linear_probe"   # zero_shot | linear_probe | finetune
epochs:                 int   = 10
batch_size:             int   = 32
lr:                     float = 1e-3
embed_dim:              int   = 512
probe_hidden:           Optional[int] = None     # None → linear (no hidden layer)
probe_dropout:          float = 0.10
unfreeze_last_n_blocks: int   = 2               # finetune mode only
replay_ratio:           float = 0.20
```

**`LinearProbeHead`:**
```
image_head: Linear(512→512, bias=False)
text_head:  same architecture
init:       eye_ (identity initialisation)
forward:    (img_f, txt_f) → (F.normalize(image_head(img_f)), F.normalize(text_head(txt_f)))
```

**`InfoNCELoss`:**
- Symmetric cross-entropy over image↔text similarity matrix
- Uses model's `logit_scale` parameter, clamped at max=100

**Training loop (`run_linear_probe`):**
1. CLIP backbone set to `eval()`, frozen
2. AdamW optimiser on probe parameters only
3. Cosine LR schedule with linear warmup
4. Gradient clipping at `grad_clip=1.0`
5. Checkpoints at epochs in `{2, 4, 8, ..., epochs}` + best Recall@1

**Evaluation metrics:**
- `Recall@K` for K ∈ {1, 5, 10}
- `Consistency@K` for K ∈ {1, 5, 10}

**MLflow artifact layout per run:**
```
artifacts/
  model/                       ← mlflow.pytorch.log_model
  checkpoints/
    probe_ep2.pt, probe_ep4.pt, ...
    probe_best.pt
  embeddings/                  ← val image + text .npy files
  drift/
    drift_reference.json
    baseline_text_centroid.npy
    baseline_image_centroid.npy
  confusion/
    retrieval_failures.csv
  provenance/
  eval_metrics_<label>.json
  requirements.txt
```

---

### 2.3 `src/drift_detection.py`

**Purpose:** Compute a scalar drift score comparing new week's data against a stored baseline run.

**Drift score formula:**
```
drift_score = 0.20 × embedding_signal
            + 0.40 × recall_signal
            + 0.40 × consistency_signal

embedding_signal  = mean(centroid_shift_text, centroid_shift_image,
                         pairwise_delta_text, pairwise_delta_image)

centroid_shift    = clip(1 − cosine(new_centroid, baseline_centroid), 0, 1)
pairwise_delta    = clip(|new_mean − baseline_mean| / 0.20, 0, 1)

recall_signal     = mean(relative_drop(R@1), relative_drop(R@5))
consistency_signal = mean(relative_drop(C@1), relative_drop(C@5))

drift_detected    = drift_score >= 0.30
```

**Thresholds:**
```python
DRIFT_SCORE_THRESHOLD  = 0.30
DRIFT_METRIC_THRESHOLD = 0.15
PAIRWISE_NORM_RANGE    = 0.20
```

---

### 2.4 `src/replay_buffer.py`

**Purpose:** Build a training CSV mixing current-week data with strategically sampled old data.

**Index offset scheme:**
```python
random_df["original_index"] += 1_000_000 × week_number
hard_df["original_index"]   += 2_000_000 × week_number
```

**Output:** `pd.concat([new_df, random_df, hard_df])` deduplicated on `(original_index, augmented_text)`.

---

### 2.5 `src/faiss_loader.py`

**Purpose:** Thread-safe singleton managing the FAISS index and metadata.

**State:**
```python
_lock:      threading.Lock
_img_index: faiss.Index      # IndexFlatIP on image embeddings
_txt_index: faiss.Index      # IndexFlatIP on text embeddings
_meta:      list[dict]
```

**Hot-swap (`reload_index`):**
```python
def reload_index() -> int:
    new_img, new_txt, new_meta = _load()   # load outside lock
    with _lock:                            # atomic swap
        _img_index = new_img
        _txt_index = new_txt
        _meta      = new_meta
    return new_img.ntotal
```

**Invariant check on load:** `img_index.ntotal == len(meta)` — raises `RuntimeError` if violated.

---

### 2.6 `src/search_api.py` + `routes/`

**FastAPI app structure:**
```
app (lifespan: CLIP + FAISS loaded eagerly at startup)
├── GET  /health                    → routes/health.py
├── GET  /ready                     → routes/health.py
├── POST /pipeline/trigger          → routes/pipeline.py
├── GET  /pipeline/status           → routes/pipeline.py
├── POST /reload                    → routes/search.py
├── POST /search/text               → routes/search.py
├── POST /search/image              → routes/search.py
├── POST /search/multimodal         → routes/search.py
├── GET  /terminal/faiss-info       → routes/terminal.py
├── GET  /terminal/env              → routes/terminal.py
├── GET  /terminal/logs             → routes/terminal.py
├── POST /terminal/dvc-status       → routes/terminal.py
├── POST /terminal/dvc-push         → routes/terminal.py
└── GET  /metrics                   → prometheus_fastapi_instrumentator
```

**Search flow (text query example):**
```
1. Receive { "query": "blue running shoes", "top_k": 10 }
2. Tokenise query with CLIP tokenizer
3. Encode with CLIP text encoder (torch.inference_mode)
4. L2-normalise embedding → float32 (1, 512)
5. img_index.search(embedding, top_k) → distances + indices
6. txt_index.search(embedding, top_k) → distances + indices
7. Lookup meta[indices] → product metadata
8. Return { "image_results": [...], "text_results": [...] }
```

---

### 2.7 `src/register_clip_model.py`

**Artifact bundle:**
```python
artifacts = {
    "clip_checkpoint":        "checkpoints/probe_best.pt",
    "faiss_index":            "data/faiss/index_img.bin",
    "faiss_meta":             "data/faiss/meta.json",
    "model_config":           "model_config.json",
    "corpus_text_embeddings": "corpus_text_embeddings.npy",
}
```

**Model Registry flow:**
1. `mlflow.pyfunc.log_model(registered_model_name="clip_product_retrieval")`
2. Get latest version number via `client.search_model_versions(...)`
3. `client.transition_model_version_stage(..., stage="Production")`

---

### 2.8 Airflow DAG (`dags/clip_pipeline_dag.py`)

**DAG configuration:**
```python
schedule         = timedelta(minutes=5)
max_active_runs  = 1
catchup          = False
```

**XCom key map:**

| Key | Pushed by | Consumed by |
|---|---|---|
| `incoming_csv` | `detect_new_data` | `augment_texts`, `drift_check`, `log_provenance` |
| `week_label` | `detect_new_data` | all downstream tasks |
| `new_csv_path` | `augment_texts` | `build_replay_buffer`, `drift_check` |
| `corpus_csv_path` | `augment_texts` | `update_faiss_index` |
| `replay_csv_path` | `build_replay_buffer` | `initial_train`, `retrain_model` |
| `baseline_run_id` | `build_replay_buffer` | `branch_on_baseline`, `drift_check` |
| `drift_detected` | `drift_check` | `branch_on_drift` |
| `drift_score` | `drift_check` | `branch_on_drift`, `skip_training` |
| `new_run_id` | `initial_train` / `retrain_model` / `skip_training` | `update_faiss_index`, `log_provenance`, `register_model` |

**Branch logic:**
```
branch_on_baseline:
  baseline_run_id is None  →  "initial_train"
  baseline_run_id exists   →  "drift_check"

branch_on_drift:
  drift_detected = True    →  "retrain_model"
  drift_detected = False   →  "skip_training"
```

**Trigger rule:** `update_faiss_index`, `log_provenance`, `register_model`, `start_serving`, `reload_faiss_index` all use `NONE_FAILED_MIN_ONE_SUCCESS` — run regardless of which branch was taken.

---

### 2.9 `templates/index.html`

**Purpose:** Standalone single-page application. No build step, no framework, no server-side dependency. Open directly in any browser.

**API coupling:** All backend calls go through `const API = 'http://localhost:8000'` (single configurable constant). The UI is independent of backend implementation — it only depends on the REST API contract.

**Pages and their API calls:**

| Page | Endpoints called |
|------|-----------------|
| Dashboard | `GET /health`, `POST /pipeline/trigger`, `POST /reload` |
| Product Search | `POST /search/text`, `POST /search/image`, `POST /search/multimodal` |
| Pipeline Viz | `GET /pipeline/status`, `POST /pipeline/trigger` |
| Terminal | Any GET/POST endpoint (free-form and shortcut buttons) |
| Live Metrics | `GET /health`, `POST /reload` |
| Drift Watch | Static formula display (MLflow integration optional) |
| User Manual | Static content — no API calls |

**Key UI components:**

- `navigate(page)` — SPA router, switches visible page without page reload
- `runSearch()` — dispatches to correct `/search/*` endpoint based on active tab
- `renderResults(data)` — renders dual `image_results` / `text_results` card grids
- `termPrint(type, text)` — appends styled lines to the terminal output buffer
- `runTermCmd()` — parses free-form `GET /path` or `POST /path {body}` and dispatches
- `renderPipeline()` — builds interactive DAG node row with click-to-detail
- `refreshDashboard()` — polls `/health` every 30 seconds and updates all stat boxes

---

### 2.10 Docker Compose Services

| Service | Image | Key env vars | Volumes |
|---|---|---|---|
| `postgres` | postgres:15 | `POSTGRES_MULTIPLE_DATABASES=airflow,mlflow` | `postgres_data` |
| `mlflow` | custom (python:3.11-slim) | `MLFLOW_SERVER_ALLOWED_HOSTS=*` | `mlflow_artifacts` |
| `airflow-init` | airflow-clip | — | project root, logs, plugins |
| `airflow-webserver` | airflow-clip | `MLFLOW_TRACKING_URI=http://mlflow:5000` | project root, logs |
| `airflow-scheduler` | airflow-clip | same as webserver | project root, logs |
| `mlflow-serve` | custom serve | `MLFLOW_TRACKING_URI=http://mlflow:5000` | project root, mlflow_artifacts |
| `search-api` | custom serve | `MLFLOW_SERVE_URL=http://mlflow-serve:5001` | project root |

---

## 3. File I/O Map

```
data/
├── incoming/
│   └── weekN.csv                ← USER SUPPLIED (trigger for pipeline)
├── processed/
│   ├── weekN_augmented.csv      ← written by augment_texts task
│   └── weekN_replay.csv         ← written by build_replay_buffer task
├── full_corpus.csv              ← growing, appended by augment_texts each week
├── image_cache/
│   └── <md5>.jpg                ← written by load_image() during training
├── corpus_embeddings/
│   ├── corpus_image_embeddings.npy
│   ├── corpus_text_embeddings.npy
│   ├── corpus_orig_idx.npy
│   └── corpus_embeddings.parquet
├── faiss/
│   ├── index_img.bin            ← written by update_faiss_index
│   ├── index_txt.bin            ← written by update_faiss_index
│   └── meta.json                ← written by update_faiss_index, read by faiss_loader
└── meta/
    ├── last_run_id.txt
    └── logs_for_revert.yaml
templates/
└── index.html                   ← standalone frontend — open in browser
```

---

## 4. Error Handling Strategy

| Failure | Behaviour |
|---|---|
| Image download fails | `load_image()` returns None; `collate_skip_none` drops the sample |
| DVC pull exits non-zero | Logged as WARNING, pipeline continues |
| MLflow connection error | Task raises exception → Airflow marks task FAILED, retries once (30s delay) |
| FAISS index/meta count mismatch | `RuntimeError` raised in `_load()`, search API startup fails |
| `start_serving` container not found | Logs warning, does not raise |
| `reload_faiss_index` health check timeout | Raises `RuntimeError` after 18 × 10s = 3 minutes |
| Search API: empty query | Returns HTTP 400 with descriptive detail |
| Search API: CLIP encode fails | Returns HTTP 500 with exception message (logged at ERROR level) |

---

## 5. Logging Strategy

- All modules use `logging.getLogger(__name__)` — no `print()` statements.
- FastAPI app logs at startup, shutdown, and on every exception.
- Search API: INFO logs on every request with latency; ERROR logs on exceptions.
- Airflow tasks: each task logs to Airflow task log (viewable in Airflow UI).
- Uvicorn access logs → file at `/tmp/uvicorn.log` (retrievable via `GET /terminal/logs`).

---

## 6. Coding Standards

- Python style: PEP 8, enforced via type hints throughout.
- Pydantic models for all FastAPI request/response schemas.
- Dataclasses (`@dataclass`) for training configuration (`Config`, `DriftBaseline`).
- All I/O paths use `pathlib.Path`, not raw strings.
- All subprocess calls capture stdout/stderr; return codes are checked.
- Unit tests in `tests/` using `pytest` with `pytest-mock` for FAISS/CLIP mocking.

---

## 7. Reproducibility Protocol

**Per-run state committed to git:**
```
params.yaml          — week, prev_week, run_id, week_number, mlflow_tracking_uri
dvc.lock             — hash of all DVC stage inputs/outputs
data/meta/logs_for_revert.yaml — per-week {mlflow_run_id, git_commit, dvc_lock_md5}
```

**Tag format:** `weekN/run-XXXXXXXX` (e.g. `week2/run-abc12345`)

**Restoration:**
```bash
./reproduce.sh week2/run-abc12345
# Internally: git checkout tag, dvc pull --force, prints params.yaml + MLflow run_id
```

**DVC repro:**
```bash
dvc repro              # full pipeline (augment + replay + train)
dvc repro augment replay  # skip training
```