# Screen Cast Link

https://drive.google.com/file/d/1Q6r0TE4fds5ujRquPjMAcowLaUOB5ZvQ/view?usp=sharing

# CLIP Product Retrieval Pipeline

An end-to-end MLOps system for continual learning on product image-text retrieval using OpenAI's CLIP model (ViT-B/32). The pipeline ingests weekly product CSVs, augments text descriptions, detects embedding drift, retrains a linear probe head when drift is detected, and serves results via a FAISS-backed search API — all orchestrated by Apache Airflow with full MLflow experiment tracking and DVC data versioning.

---

## Architecture Overview

```
Weekly CSV → Airflow DAG → Augment → Replay Buffer → [Drift Check] → Train/Skip
                                                              ↓
                                              FAISS Index → Search API (FastAPI)
                                                              ↓
                                              MLflow Model Registry → mlflow-serve
```

The system has three operational modes for the CLIP model:
- **Zero-shot**: Plain CLIP, no adaptation (baseline evaluation)
- **Linear probe**: Frozen CLIP backbone + trainable projection head (default)
- **Finetune**: Last N transformer blocks unfrozen + InfoNCE loss

---

## Repository Structure

```
.
├── dags/
│   └── clip_pipeline_dag.py       # Airflow DAG — 11-task pipeline definition
├── src/
│   ├── augment_texts.py           # 7 rule-based text augmentation strategies
│   ├── clip_experiments.py        # CLIP training modes + MLflow logging
│   ├── clip_mlflow_wrapper.py     # MLflow pyfunc wrapper for serving
│   ├── drift_detection.py         # Weighted drift score computation
│   ├── faiss_loader.py            # Thread-safe FAISS singleton
│   ├── replay_buffer.py           # Continual learning replay buffer
│   ├── register_clip_model.py     # MLflow Model Registry registration
│   └── search_api.py              # FastAPI app entry point
├── routes/
│   ├── health.py                  # GET /health
│   ├── pipeline.py                # POST /pipeline/trigger
│   ├── search.py                  # POST /search/{text,image,multimodal}
│   └── terminal.py                # GET /terminal/...
├── docker/
│   ├── airflow/
│   │   ├── Dockerfile.airflow
│   │   └── requirements.airflow.txt
│   ├── mlflow/
│   │   ├── Dockerfile.mlflow
│   │   └── requirements.mlflow.txt
│   └── serve/
│       ├── Dockerfile.serve
│       └── requirements.serve.txt
├── scripts/
│   └── create-multiple-dbs.sh     # Postgres multi-DB init script
├── docker-compose.yaml
├── dvc.yaml                       # 3-stage DVC pipeline (augment→replay→train)
├── params.yaml                    # DVC params (week, run_id, prev_week, ...)
└── reproduce.sh                   # Git-tag based state restoration script
```

---

## Prerequisites

- Docker & Docker Compose v2
- 8 GB RAM minimum (16 GB recommended for training)
- ~10 GB disk for images, embeddings, and MLflow artifacts

---

## Quick Start

### 1. Clone and configure

```bash
git clone <repo-url>
cd DA5402_E2E
```

### 2. Place your first weekly CSV

```bash
mkdir -p data/incoming
cp /path/to/your/products.csv data/incoming/week1.csv
```

The CSV must have columns: `h_index`, `title`, `image_links`, `category_2`, `category_3`, `selling_price`, `mrp`, `highlights`, `description`.

### 3. Start the stack

```bash
docker compose up -d postgres mlflow airflow-init
docker compose up -d airflow-webserver airflow-scheduler
```

### 4. Access the UIs

| Service | URL | Credentials |
|---|---|---|
| Airflow | http://localhost:8080 | admin / admin |
| MLflow | http://localhost:5000 | — |

### 5. Trigger the pipeline

In the Airflow UI, enable and trigger the `clip_product_retrieval_pipeline` DAG manually, or wait for it to auto-run (scheduled every 5 minutes, picks up any new CSV in `data/incoming/`).

### 6. Start the serving stack (after first pipeline run)

```bash
docker compose --profile serving up -d
```

The search API will be available at http://localhost:8000.

---

## Pipeline DAG Tasks

| # | Task | Description |
|---|---|---|
| 1 | `detect_new_data` | Scans `data/incoming/` for unprocessed CSVs |
| 2 | `dvc_pull_latest` | Initialises Git/DVC, configures local remote, pulls tracked data |
| 3 | `augment_texts` | Applies 7 augmentation strategies, updates `full_corpus.csv` |
| 4 | `build_replay_buffer` | Day-0: writes augmented data directly. Week N: mixes hard negatives |
| 5 | `branch_on_baseline` | Branches to `initial_train` (Day-0) or `drift_check` (Week N) |
| 6a | `initial_train` | Trains linear probe, logs to MLflow, saves drift reference |
| 6b | `drift_check` | Computes weighted drift score against baseline run |
| 6c | `branch_on_drift` | Routes to `retrain_model` or `skip_training` |
| 7 | `update_faiss_index` | Embeds full corpus, builds FAISS IndexFlatIP, logs to MLflow |
| 8 | `dvc_add_image_cache` | Tracks `data/image_cache/` via DVC, pushes to local remote |
| 9 | `log_provenance` | Tags MLflow run with git commit + DVC lock hash, updates `params.yaml` |
| 10 | `register_model` | Registers model in MLflow Model Registry, promotes to Production |
| 11 | `start_serving` | Restarts `mlflow_serve` and `search_api` Docker containers |
| 12 | `reload_faiss_index` | Calls `POST /reload` to hot-swap FAISS index in running search API |

---

## Search API Endpoints

```
POST /search/text          { "query": "blue running shoes", "top_k": 10 }
POST /search/image         { "image_b64": "<base64 string>", "top_k": 10 }
POST /search/multimodal    { "query": "...", "image_b64": "...", "top_k": 10 }
GET  /health
POST /reload               # hot-swap FAISS index without restart
POST /pipeline/trigger     # trigger Airflow DAG via REST
```

---

## Drift Detection

The drift score is a weighted combination of three signals:

| Signal | Weight | Measures |
|---|---|---|
| Embedding shift | 20% | Centroid shift + pairwise cosine delta |
| Recall drop | 40% | Relative Recall@1 and Recall@5 degradation |
| Consistency drop | 40% | Relative Consistency@1 and Consistency@5 degradation |

`drift_detected = drift_score >= 0.30`

---

## Reproducibility

To restore the exact state of any past week:

```bash
# List available tags
git tag

# Restore week2's exact state
./reproduce.sh week2/run-abc12345

# This checks out the git tag, runs dvc pull, and prints the MLflow run_id
```

DVC stages can also be re-executed for any week by editing `params.yaml` and running:

```bash
dvc repro              # full pipeline (augment + replay + train)
dvc repro augment replay  # skip training
```

---

## Configuration

Key parameters in `params.yaml`:

```yaml
mlflow_tracking_uri: http://mlflow:5000
mlflow_experiment: clip_product_retrieval
run_id: null          # filled automatically after each training run
week: ""              # current week label, e.g. "week2"
week_number: 0        # numeric week index
prev_week: ""         # previous week label for replay mixing
```

Training hyperparameters are set inside the DAG task functions (`initial_train`, `retrain_model`) and can be modified in `clip_pipeline_dag.py`:

```python
cfg = Config(
    epochs     = 2,
    batch_size = 32,
    lr         = 1e-3,
    mode       = "linear_probe",
)
```

---

## Known Issues & Fixes

| Issue | Fix |
|---|---|
| MLflow 403 "Invalid Host header" | Add `MLFLOW_SERVER_ALLOWED_HOSTS: "*"` to mlflow service in docker-compose.yaml |
| DVC git "dubious ownership" | `RUN git config --global --add safe.directory /opt/project` in Dockerfile.airflow |
| `dvc add data/image_cache` blocked | Remove `data/image_cache/` from `.dvcignore` |
| `dvc pull data/incoming/week1.csv` fails | Use `dvc pull --force` (incoming CSVs are user-supplied, not DVC outputs) |

---

## Tech Stack

| Component | Technology |
|---|---|
| Model | OpenCLIP ViT-B/32 (OpenAI weights) |
| Orchestration | Apache Airflow 3.2 |
| Experiment tracking | MLflow 3.10 |
| Data versioning | DVC |
| Vector search | FAISS (IndexFlatIP) |
| Search API | FastAPI + uvicorn |
| Databases | PostgreSQL 15 (Airflow + MLflow metadata) |
| Containerisation | Docker Compose |