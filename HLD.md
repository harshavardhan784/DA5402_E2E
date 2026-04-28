# High-Level Design (HLD)
## CLIP Product Retrieval — Continual Learning MLOps Pipeline

**Course:** DA5402 — End-to-End Machine Learning Systems
**Stack:** OpenCLIP · Apache Airflow · MLflow · DVC · FAISS · FastAPI · Docker

---

## 1. Purpose and Scope

This system enables a production e-commerce platform to maintain high-quality image-text product retrieval as the product catalogue changes week over week. It ingests raw product CSVs, adapts a CLIP vision-language model to the current catalogue distribution, and serves retrieval results via a REST API — without requiring a full model retrain unless the data distribution has meaningfully shifted.

**In scope:** Data ingestion, text augmentation, drift detection, continual learning, vector indexing, model serving, end-to-end reproducibility, a standalone frontend UI, and full monitoring stack.

**Out of scope:** User authentication, A/B testing, GPU cluster scheduling, and product catalogue management.

---

## 2. System Goals

| Goal | Mechanism |
|------|-----------|
| Adapt to catalogue drift without full retraining every week | Drift-gated retraining — only retrain when drift score ≥ 0.30 |
| Prevent catastrophic forgetting | Replay buffer mixing old hard negatives with new data |
| Full reproducibility of any past week's state | Git tags + DVC data versioning + MLflow run tracking |
| Zero-downtime model updates | FAISS hot-swap via `POST /reload`, container restart |
| Decoupled training and serving | MLflow Model Registry as the handoff point |
| Non-technical user access | Standalone web UI (`ui/index.html`) with search, pipeline viz, terminal, and user manual |
| Strict loose coupling between UI and backend | Frontend communicates **only** via REST API calls to `localhost:8000` |

---

## 3. Architecture Diagram

The system is divided into four planes: **Training**, **Serving**, **UI/Monitoring**, and **Data/State**.

```
┌──────────────────────────── TRAINING PLANE ─────────────────────────────────┐
│                                                                              │
│  data/incoming/weekN.csv                                                     │
│         │                                                                    │
│         ▼                                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                  Apache Airflow DAG                                  │    │
│  │           clip_product_retrieval_pipeline (5-min schedule)           │    │
│  │                                                                      │    │
│  │  detect → dvc_pull → augment → replay_buffer → branch_baseline      │    │
│  │                                                      │               │    │
│  │                                              ┌───────┴───────┐      │    │
│  │                                              │               │      │    │
│  │                                         [Day-0]          [Week N]   │    │
│  │                                        initial_train    drift_check │    │
│  │                                              │               │      │    │
│  │                                              │         ┌─────┴──┐  │    │
│  │                                              │      retrain   skip  │    │
│  │                                              └────────┬────────┘   │    │
│  │                                                       │             │    │
│  │          update_faiss → dvc_cache → log_provenance → register      │    │
│  │                                                       │             │    │
│  │                                             start_serving → reload  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                     │                                        │
│                                     ▼                                        │
│                    ┌────────────────────────────┐                           │
│                    │   MLflow Tracking Server    │                           │
│                    │   (PostgreSQL backend)      │                           │
│                    │  Experiments & runs         │                           │
│                    │  Artifacts (checkpoints,    │                           │
│                    │    embeddings, FAISS index) │                           │
│                    │  Model Registry             │                           │
│                    └────────────────────────────┘                           │
└──────────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────── SERVING PLANE ─────────────────────────────────┐
│                                                                              │
│   ┌──────────────────┐        ┌───────────────────────────────────────┐    │
│   │  mlflow-serve    │        │       search-api (FastAPI)             │    │
│   │  port 5001       │◀───────│       port 8000                        │    │
│   │                  │        │                                        │    │
│   │  CLIPRetriever   │        │  POST /search/text                     │────┼──▶ Client
│   │  pyfunc model    │        │  POST /search/image                    │    │
│   │                  │        │  POST /search/multimodal               │    │
│   └──────────────────┘        │  GET  /health   GET  /ready            │    │
│                                │  POST /reload   (FAISS hot-swap)       │    │
│                                │  GET  /terminal/faiss-info             │    │
│                                │  GET  /terminal/logs                   │    │
│                                │  POST /terminal/dvc-status             │    │
│                                │  POST /terminal/dvc-push               │    │
│                                │  GET  /metrics  (Prometheus)           │    │
│                                └───────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────── UI / MONITORING PLANE ───────────────────────────┐
│                                                                              │
│   ui/index.html  (standalone, no build step — open directly in browser)     │
│   ├── Dashboard    — live health, service status, API latency, actions      │
│   ├── Product Search — text / image / multimodal with result cards          │
│   ├── Pipeline Viz — DAG task graph, XCom map, branch logic, DAG run state  │
│   ├── Terminal     — live ops terminal + clickable command reference        │
│   ├── Live Metrics — /health poll (30s), links to Grafana / Prometheus      │
│   ├── Drift Watch  — score breakdown bars, formula, threshold annotation    │
│   └── User Manual  — step-by-step guide for non-technical users            │
│                                                                              │
│   Prometheus :9090  ←  scrapes :8000/metrics every 15s                     │
│   Grafana    :3001  ←  request rate, latency histogram, error rate          │
│   Alertmanager:9093 ←  fires if error_rate > 5% or service down            │
└──────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────── DATA / STATE PLANE ──────────────────────────────┐
│                                                                              │
│  data/incoming/weekN.csv           → weekly trigger for the DAG             │
│  data/processed/weekN_augmented.csv → 7 text variants per product           │
│  data/processed/weekN_replay.csv   → new + hard negatives from old week     │
│  data/full_corpus.csv              → growing cumulative product catalogue   │
│  data/image_cache/<md5>.jpg        → DVC-tracked image cache                │
│  data/faiss/index_img.bin          → FAISS image index (hot-swappable)      │
│  data/faiss/index_txt.bin          → FAISS text index (hot-swappable)       │
│  data/faiss/meta.json              → parallel product metadata list         │
│  params.yaml                       → DVC week state (for dvc repro)         │
│  data/meta/logs_for_revert.yaml    → per-week git commit + run_id log       │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Component Responsibilities

### 4.1 Airflow DAG (`dags/clip_pipeline_dag.py`)

Central orchestrator. Detects new data, runs all 12 pipeline stages in dependency order, manages branching logic (Day-0 vs Week N; drift detected vs not), and coordinates the handoff between training and serving. Runs on a 5-minute schedule with `max_active_runs=1`.

### 4.2 Text Augmentation (`src/augment_texts.py`)

Converts raw product titles into a richer training set using 7 deterministic strategies: title cleaning, category prefix, price context, highlights fusion, description snippet, keyword drop, and attribute reorder. Creates multiple supervision signals per image for contrastive learning.

### 4.3 Replay Buffer (`src/replay_buffer.py`)

Prevents catastrophic forgetting by mixing current-week data with hard negatives and random samples from previous weeks. Index offsets (`1M × week_number` for random, `2M × week_number` for hard negatives) prevent `original_index` collisions.

### 4.4 Drift Detection (`src/drift_detection.py`)

Computes a scalar drift score (0–1) as a weighted combination: embedding centroid shift + pairwise cosine distribution change (20%), Recall@K degradation (40%), and Consistency@K degradation (40%). Retraining triggers at 0.30.

### 4.5 CLIP Training (`src/clip_experiments.py`)

Three training modes around OpenCLIP ViT-B/32. Default is linear probe: frozen CLIP backbone + trainable 512→512 projection head with symmetric InfoNCE loss. Every run logs exponential-epoch checkpoints, drift reference JSON, and embedding centroids to MLflow.

### 4.6 MLflow Tracking Server

Stores all experiment metadata in PostgreSQL and all artifacts in a Docker volume (`mlflow_artifacts`). Acts as the canonical model handoff point between training and serving.

### 4.7 FAISS Index + Search API (`src/search_api.py`, `routes/`)

Thread-safe FAISS singleton (`faiss_loader.py`) with two `IndexFlatIP` instances (image and text). Three search modes: text-only, image-only, multimodal. Hot-swap via `POST /reload`. Prometheus instrumentation via `prometheus_fastapi_instrumentator`.

### 4.8 DVC + Git Versioning

DVC tracks `data/image_cache/` in a local remote. Git tags of the form `weekN/run-XXXXXXXX` snapshot `params.yaml`, `dvc.lock`, and `logs_for_revert.yaml`, enabling `./reproduce.sh <tag>` to restore any past week's exact state.

### 4.9 Monitoring Stack

Prometheus scrapes `:8000/metrics` every 15 seconds. Grafana visualises request rate, latency histograms, and error rates. Alertmanager fires when error rate exceeds 5% or any service goes down. Postgres Exporter provides DB-level metrics.

### 4.10 Frontend UI (`ui/index.html`)

A standalone single-page application with no build step or framework dependency. Communicates with the backend **exclusively** via REST API calls to `http://localhost:8000` — demonstrating strict loose coupling. Features: live dashboard, product search (text/image/multimodal), interactive pipeline visualisation, live ops terminal with command reference, monitoring links, drift watchdog, and full user manual.

---

## 5. Data Flow

```
Raw CSV (Flipkart product data)
    │
    ▼
[augment_texts]  7 strategies × N products → weekN_augmented.csv
    │
    ▼
[build_replay_buffer]
    ├── Day-0:  augmented.csv → replay.csv (direct)
    └── Week N: new + random_old + hard_negatives → replay.csv
    │
    ▼
[initial_train / retrain_model]
    CLIP ViT-B/32 (frozen) + LinearProbeHead (InfoNCE, AdamW, cosine LR)
    → MLflow: checkpoints, drift_reference.json, centroids
    │
    ▼
[update_faiss_index]
    Full corpus → CLIP embeddings → FAISS IndexFlatIP × 2
    → index_img.bin + index_txt.bin + meta.json
    │
    ▼
[register_model → start_serving → reload_faiss]
    CLIPRetrieverModel → MLflow Registry (Production)
    → POST /reload → new FAISS index hot-swapped in-memory
    │
    ▼
[User via UI]
    ui/index.html → REST → search-api :8000 → FAISS → results
```

---

## 6. Deployment Topology

```
Host Machine (Docker Compose)
├── postgres           port 5432   Airflow + MLflow metadata DB
├── mlflow             port 5000   Tracking server + artifact server
├── airflow-init       one-shot    DB migration + admin user creation
├── airflow-webserver  port 8080   DAG UI + REST API
├── airflow-scheduler  —           DAG scheduler (5-min tick)
├── mlflow-serve       port 5001   MLflow pyfunc model server
├── search-api         port 8000   FastAPI search + ops API
├── prometheus         port 9090   Metrics scraper
├── grafana            port 3001   Metrics dashboards
├── alertmanager       port 9093   Alert routing
└── postgres-exporter  port 9187   DB metrics for Prometheus
```

---

## 7. Key Design Decisions

**Standalone HTML UI with strict REST-only coupling.** No shared code between frontend and backend. The API base URL is a single configurable constant in `ui/index.html`. This satisfies the assignment's requirement for a loose-coupled, independently deployable UI and backend.

**Linear probe over full finetune by default.** ~1M parameters vs 86M, less overfitting risk on small weekly batches (~4,000 samples), faster per-epoch training, no catastrophic forgetting of CLIP's pretrained features.

**Weighted drift score over single metric.** Data quality issues (→ high recall signal, low embedding signal) and genuine distribution shift (→ high embedding signal) are distinguishable. The combined score is robust to single-signal noise.

**Two separate FAISS indexes (image + text).** Allows the multimodal fusion to query each modality independently and combine scores. Supports the `image_results` / `text_results` dual-list response that the UI renders.

**DVC + Git tags alongside MLflow.** MLflow tracks model lineage; DVC tracks data lineage; Git tags bind both at a single commit hash for full reproducibility.

**Separate mlflow-serve and search-api containers.** Model inference logic (MLflow pyfunc) and API routing (FastAPI) can be updated, scaled, or replaced independently.

**Terminal endpoints in the API (`/terminal/*`).** Expose system state to the UI terminal without SSH access to containers, filtered to non-sensitive information only.

---

## 8. ML and Business Metrics

| Type | Metric | Target |
|------|--------|--------|
| ML | Recall@1 | ≥ 0.80 (linear probe, Day-0) |
| ML | Recall@5 | ≥ 0.95 |
| ML | Consistency@1 | ≥ 0.75 |
| ML | Drift Score | < 0.30 (no retrain needed) |
| Business | Search API latency | < 200 ms (p95) |
| Business | Error rate | < 5% (Alertmanager threshold) |
| Business | Index availability | 100% (hot-swap, no restart) |