"""
tests/test_search_api.py
Integration tests for the FastAPI search endpoints.
All FAISS and CLIP calls are mocked so no GPU or index files are needed.
"""
import sys
import os
import json
import base64
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# Add project root and src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# ── shared fake data ───────────────────────────────────────────────────────────

FAKE_META = [
    {
        "original_index": i,
        "image_url": f"http://example.com/{i}.jpg",
        "augmented_text": f"Product {i} description",
        "category_2": "Electronics",
        "category_3": "Phones",
        "selling_price": 10000 + i * 1000,
    }
    for i in range(50)
]

FAKE_EMBEDDING = np.random.default_rng(42).standard_normal(512).astype(np.float32)
FAKE_EMBEDDING /= (np.linalg.norm(FAKE_EMBEDDING) + 1e-9)

@pytest.fixture(scope="module")
def client():
    """Create a TestClient with all external deps patched."""
    # Patch the FAISS loader to return fake indices and meta
    with patch("faiss_loader.get_img_index") as mock_img_idx, \
         patch("faiss_loader.get_txt_index") as mock_txt_idx, \
         patch("faiss_loader.get_meta", return_value=FAKE_META), \
         patch("faiss_loader.reload_index", return_value=len(FAKE_META)):

        # Configure fake FAISS index objects
        # Configure fake FAISS index objects
        for mock_idx in [mock_img_idx.return_value, mock_txt_idx.return_value]:
            mock_idx.ntotal = len(FAKE_META)
            mock_idx.d = 512
            
            # Use side_effect to return exactly the number of results requested (k)
            def dynamic_search(query_vec, k):
                return (
                    np.linspace(0.99, 0.60, k, dtype=np.float32).reshape(1, k),
                    np.arange(k, dtype=np.int64).reshape(1, k),
                )
            mock_idx.search.side_effect = dynamic_search
            

        # Patch the module 'search_api' directly instead of 'routes.search'
        with patch("search_api._embed_text", return_value=FAKE_EMBEDDING), \
             patch("search_api._embed_image", return_value=FAKE_EMBEDDING):

            from search_api import app
            yield TestClient(app)

# ══ Tests ══════════════════════════════════════════════════════════════════════

class TestHealth:
    def test_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

class TestReady:
    def test_returns_200_when_index_loaded(self, client):
        resp = client.get("/ready")
        assert resp.status_code == 200
        assert "status" in resp.json()

class TestSearchText:
    def test_returns_200(self, client):
        resp = client.post("/search/text", json={"query": "blue running shoes", "top_k": 5})
        assert resp.status_code == 200

    def test_top_k_respected(self, client):
        top_k = 3
        resp = client.post("/search/text", json={"query": "laptop", "top_k": top_k})
        data = resp.json()
        
        # Check that BOTH result lists respect the top_k limit
        assert len(data["text_results"]) <= top_k
        assert len(data["image_results"]) <= top_k

    def test_empty_query_returns_error(self, client):
        resp = client.post("/search/text", json={"query": "", "top_k": 5})
        assert resp.status_code in (400, 422)

class TestSearchImage:
    def test_returns_200(self, client):
        img_b64 = base64.b64encode(b"fake-image-data").decode()
        resp = client.post("/search/image", json={"image_b64": img_b64, "top_k": 5})
        assert resp.status_code == 200

    def test_missing_image_returns_422(self, client):
        resp = client.post("/search/image", json={"top_k": 5})
        assert resp.status_code == 422

class TestSearchMultimodal:
    def test_returns_200(self, client):
        img_b64 = base64.b64encode(b"fake-image-data").decode()
        resp = client.post("/search/multimodal", json={
            "query": "smartphone", 
            "image_b64": img_b64, 
            "top_k": 5
        })
        assert resp.status_code == 200

    def test_missing_both_returns_error(self, client):
        resp = client.post("/search/multimodal", json={"top_k": 5})
        assert resp.status_code in (400, 422)

class TestReload:
    def test_returns_200(self, client):
        resp = client.post("/reload")
        assert resp.status_code == 200
        # Check for the key defined in search_api.py reload() function
        assert "vector_count" in resp.json()