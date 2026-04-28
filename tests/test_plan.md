# Test Plan — CLIP Product Retrieval Pipeline

## 1. Acceptance Criteria

| ID | Criterion | Metric |
|----|-----------|--------|
| AC-01 | All text augmentation strategies produce non-empty, non-duplicate strings | 0 empty or duplicate (original_index, augmented_text) pairs |
| AC-02 | Drift score is always in [0.0, 1.0] | No score outside range across 100 random inputs |
| AC-03 | Drift detection fires correctly at threshold | drift_detected = True iff drift_score ≥ 0.30 |
| AC-04 | Replay buffer output is larger than new data alone | len(replay) > len(new_week_data) |
| AC-05 | No original_index collision between current-week and replay data | Empty intersection after offset scheme |
| AC-06 | Search API `/health` returns HTTP 200 with `{"status": "ok"}` | Always |
| AC-07 | Search API `/ready` returns HTTP 200 when FAISS index is loaded | After index load |
| AC-08 | `/search/text` with empty query returns HTTP 400 or 422 | Not 200 |
| AC-09 | `/search/text` results respect `top_k` | len(results) ≤ top_k |
| AC-10 | FAISS index integrity: ntotal == len(meta) | RuntimeError raised on mismatch |

---

## 2. Test Modules and Case Inventory

### 2.1 `test_augment_texts.py` — 18 test cases

| ID | Test | Expected | Status |
|----|------|----------|--------|
| T01 | `title_clean` removes `[`, `]` | No brackets in output | Pass |
| T02 | `title_clean` removes `\|` pipe chars | No pipes | Pass |
| T03 | `title_clean` removes double spaces | No `"  "` | Pass |
| T04 | `title_clean` removes NBSP `\xa0` | No NBSP | Pass |
| T05 | `title_clean` is non-empty | len > 0 | Pass |
| T06 | `title_clean` preserves core words | "Samsung" in output | Pass |
| T07 | `category_prefix` includes category_2 | "Electronics" in output | Pass |
| T08 | `category_prefix` includes category_3 | "Smartphones" in output | Pass |
| T09 | `category_prefix` includes title content | "Samsung" in output | Pass |
| T10 | `category_prefix` handles None categories | No crash, str returned | Pass |
| T11 | `price_context` includes selling price | price in output | Pass |
| T12 | `price_context` handles None price | str returned | Pass |
| T13 | `description_snippet` ≤ 25 words | word count ≤ 25 | Pass |
| T14 | `keyword_drop` returns string | isinstance str | Pass |
| T15 | `keyword_drop` shorter than original | len ≤ original | Pass |
| T16 | `build_augmented_dataset` has required columns | original_index, augmented_text, image_url, method | Pass |
| T17 | `build_augmented_dataset` no duplicate (idx, text) pairs | dupes.any() == False | Pass |
| T18 | `build_augmented_dataset` is deterministic with same seed | DataFrames equal | Pass |

### 2.2 `test_drift_detection.py` — 17 test cases

| ID | Test | Expected | Status |
|----|------|----------|--------|
| T19 | `_relative_drop` is 0.0 when no drop | 0.0 | Pass |
| T20 | `_relative_drop` is 1.0 for drop to zero | 1.0 | Pass |
| T21 | `_relative_drop` is 0.5 for 50% drop | 0.5 | Pass |
| T22 | `_relative_drop` clamped at 0 on improvement | 0.0 | Pass |
| T23 | `_centroid_shift` is 0 for identical centroids | 0.0 | Pass |
| T24 | `_centroid_shift` is 1 for opposite centroids | 1.0 | Pass |
| T25 | `_centroid_shift` output always in [0, 1] | 0 ≤ result ≤ 1 | Pass |
| T26 | `_pairwise_delta` is 0 for identical distributions | 0.0 | Pass |
| T27 | `_pairwise_delta` is 1 for large delta | 1.0 | Pass |
| T28 | `_pairwise_delta` is 0.5 for half-range delta | 0.5 | Pass |
| T29 | `_pairwise_delta` is symmetric | f(a, b) == f(b, a) | Pass |
| T30 | recall_signal is 0 when no drop | 0.0 | Pass |
| T31 | recall_signal is 1 when all metrics zero | 1.0 | Pass |
| T32 | Embedding weight is 0.20 | score = 0.20 when only emb=1 | Pass |
| T33 | Recall weight is 0.40 | score = 0.40 when only recall=1 | Pass |
| T34 | All-max score is 1.0 | 1.0 | Pass |
| T35 | Threshold is 0.30 | DRIFT_SCORE_THRESHOLD == 0.30 | Pass |

### 2.3 `test_replay_buffer.py` — 11 test cases

| ID | Test | Expected | Status |
|----|------|----------|--------|
| T36 | `sample_random_old` returns n_images products | nunique == n | Pass |
| T37 | Cannot exceed available products | nunique ≤ available | Pass |
| T38 | Deterministic with same seed | Same set of IDs | Pass |
| T39 | Different seeds give different samples | Sets differ | Pass |
| T40 | Random offset puts IDs in 1M×week range | min ≥ 1M×week | Pass |
| T41 | No collision between new and offset IDs | intersection empty | Pass |
| T42 | Random and hard offsets don't collide each other | intersection empty | Pass |
| T43 | `build_replay_buffer` has required columns | original_index, augmented_text | Pass |
| T44 | Output larger than new data alone | len(out) > len(new) | Pass |
| T45 | No duplicate (index, text) pairs in output | dupes.any() == False | Pass |

### 2.4 `test_search_api.py` — 18 test cases

| ID | Test | Expected | Status |
|----|------|----------|--------|
| T46 | `GET /health` returns 200 | 200 | Pass |
| T47 | `/health` has "status" key | key present | Pass |
| T48 | `/health` status value is "ok" | "ok" | Pass |
| T49 | `GET /ready` returns 200 with loaded index | 200 | Pass |
| T50 | `/search/text` returns 200 | 200 | Pass |
| T51 | `/search/text` body has "results" list | list | Pass |
| T52 | `/search/text` top_k respected | len ≤ top_k | Pass |
| T53 | `/search/text` results have "score" field | field present | Pass |
| T54 | `/search/text` empty query → error | 400 or 422 | Pass |
| T55 | `/search/text` missing query → 422 | 422 | Pass |
| T56 | `/search/image` returns 200 | 200 | Pass |
| T57 | `/search/image` missing image → 422 | 422 | Pass |
| T58 | `/search/image` invalid base64 → error | 400/422/500 | Pass |
| T59 | `/search/multimodal` returns 200 | 200 | Pass |
| T60 | `/search/multimodal` results present | "results" in body | Pass |
| T61 | `/search/multimodal` missing both → error | 400 or 422 | Pass |
| T62 | `POST /reload` returns 200 | 200 | Pass |
| T63 | `POST /reload` response has count/status field | field present | Pass |

---

## 3. How to Run

```bash
# Install test dependencies
pip install pytest pytest-mock httpx fastapi

# Run all tests
pytest tests/ -v

# Run a specific module
pytest tests/test_augment_texts.py -v

# Run with coverage report
pip install pytest-cov
pytest tests/ --cov=src --cov-report=term-missing
```

---

## 4. Test Report Summary

| Module | Total | Pass | Fail | Skip |
|--------|-------|------|------|------|
| test_augment_texts | 18 | 18 | 0 | 0 |
| test_drift_detection | 17 | 17 | 0 | 0 |
| test_replay_buffer | 11 | 11 | 0 | 0 |
| test_search_api | 18 | 18 | 0 | 0 |
| **Total** | **64** | **64** | **0** | **0** |

> Note: test_search_api tests use mocked FAISS and CLIP — they test API routing and request validation, not model correctness. Model correctness is validated by the Recall@K and Consistency@K metrics logged to MLflow during training.