"""
tests/test_augment_texts.py
Comprehensive unit tests for src/augment_texts.py augmentation strategies and dataset builder.
"""
import sys
import os
import pandas as pd
import pytest
from pathlib import Path

# Add src to path so we can import augment_texts
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from augment_texts import (
    aug_title_clean,
    aug_category_prefix,
    aug_price_context,
    aug_highlights_fusion,
    aug_description_snippet,
    aug_keyword_drop,
    aug_attribute_reorder,
    build_augmented_dataset
)

# ── Helpers & Fixtures ────────────────────────────────────────────────────────

def _make_row(**kwargs):
    defaults = {
        "h_index": 1,
        "title": "Samsung Galaxy S23 [128GB] | 50MP Camera",
        "image_links": "http://example.com/img.jpg",
        "category_2": "Electronics",
        "category_3": "Smartphones",
        "selling_price": 45000,
        "mrp": 50000,
        "highlights": "5G | 50MP Camera | 256GB Storage",
        "description": "Flagship smartphone with advanced AI-powered camera system.",
    }
    defaults.update(kwargs)
    return pd.Series(defaults)


def _make_df(n=10, **row_kwargs):
    rows = []
    for i in range(n):
        r = _make_row(h_index=i + 1, title=f"Product {i} [32GB]", **row_kwargs)
        rows.append(r)
    return pd.DataFrame(rows)

@pytest.fixture
def sample_row():
    return _make_row()

# ══ Strategy Tests ════════════════════════════════════════════════════════════

class TestAugmentationStrategies:
    
    def test_title_clean(self):
        row = _make_row(title="Samsung [128GB] | Phone \xa0 5G")
        result = aug_title_clean(row)
        assert "[" not in result and "]" not in result
        assert "|" not in result
        assert "\xa0" not in result
        assert "  " not in result # No double spaces
        assert "Samsung" in result

    def test_category_prefix(self, sample_row):
        result = aug_category_prefix(sample_row)
        assert "Electronics" in result
        assert "Smartphones" in result
        assert "Samsung" in result

    def test_price_context(self):
        row = _make_row(selling_price=45000)
        result = aug_price_context(row)
        assert "45000" in result or "45,000" in result

    def test_description_snippet_limits_length(self):
        long_desc = " ".join([f"word{i}" for i in range(60)])
        row = _make_row(description=long_desc, title="Unique Object")
        result = aug_description_snippet(row)
        if result:
            assert len(result.split()) <= 25

    def test_keyword_drop(self):
        row = _make_row(title="Samsung Galaxy S23 Smartphone Flagship Camera")
        result = aug_keyword_drop(row)
        assert isinstance(result, str)
        assert len(result) <= len(row["title"])
        assert len(result.strip()) > 0

    def test_attribute_reorder(self):
        row = _make_row(title="Blue Samsung Galaxy Phone")
        result = aug_attribute_reorder(row)
        original_words = set(row["title"].lower().split())
        result_words = set(result.lower().split())
        # Reorder should preserve the majority of keywords
        assert len(result_words.intersection(original_words)) >= len(original_words) * 0.7

# ══ Dataset Builder Tests ════════════════════════════════════════════════════

class TestBuildAugmentedDataset:
    
    def test_output_structure(self, tmp_path):
        df = _make_df(n=5)
        csv_path = tmp_path / "input.csv"
        df.to_csv(csv_path, index=False)
        
        result = build_augmented_dataset(str(csv_path), n_samples=5)
        
        for col in ["original_index", "augmented_text", "image_url", "method"]:
            assert col in result.columns
        assert len(result) > 5  # Multiplication check
        assert pd.api.types.is_integer_dtype(result["original_index"])

    def test_no_duplicates(self, tmp_path):
        df = _make_df(n=1)
        csv_path = tmp_path / "single.csv"
        df.to_csv(csv_path, index=False)
        
        result = build_augmented_dataset(str(csv_path), n_samples=5)
        dupes = result.duplicated(subset=["original_index", "augmented_text"])
        assert not dupes.any()

    def test_deterministic_seed(self, tmp_path):
        df = _make_df(n=10)
        csv_path = tmp_path / "seed_test.csv"
        df.to_csv(csv_path, index=False)
        
        r1 = build_augmented_dataset(str(csv_path), n_samples=10, seed=42)
        r2 = build_augmented_dataset(str(csv_path), n_samples=10, seed=42)
        
        pd.testing.assert_frame_equal(
            r1.sort_values(["original_index", "method"]).reset_index(drop=True),
            r2.sort_values(["original_index", "method"]).reset_index(drop=True)
        )

    def test_n_samples_limit(self, tmp_path):
        df = _make_df(n=50)
        csv_path = tmp_path / "limit_test.csv"
        df.to_csv(csv_path, index=False)
        
        result = build_augmented_dataset(str(csv_path), n_samples=10)
        assert result["original_index"].nunique() <= 10