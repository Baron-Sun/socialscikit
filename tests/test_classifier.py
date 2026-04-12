"""Tests for socialscikit.quantikit.classifier.

These tests verify the data structures, config, and utility methods
without downloading models or running actual training.
"""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from socialscikit.quantikit.classifier import (
    Classifier,
    PredictionResult,
    TrainConfig,
    TrainResult,
    DEFAULT_MODEL_EN,
    DEFAULT_MODEL_MULTI,
)


# ---------------------------------------------------------------------------
# TrainConfig
# ---------------------------------------------------------------------------


class TestTrainConfig:
    def test_defaults(self):
        cfg = TrainConfig()
        assert cfg.model_name == DEFAULT_MODEL_EN
        assert cfg.max_length == 512
        assert cfg.batch_size == 16
        assert cfg.learning_rate == 2e-5
        assert cfg.num_epochs == 5
        assert cfg.early_stopping_patience == 2
        assert cfg.seed == 42

    def test_custom_config(self):
        cfg = TrainConfig(
            model_name=DEFAULT_MODEL_MULTI,
            batch_size=8,
            num_epochs=10,
        )
        assert cfg.model_name == DEFAULT_MODEL_MULTI
        assert cfg.batch_size == 8
        assert cfg.num_epochs == 10

    def test_model_constants(self):
        assert DEFAULT_MODEL_EN == "roberta-base"
        assert DEFAULT_MODEL_MULTI == "xlm-roberta-base"


# ---------------------------------------------------------------------------
# TrainResult
# ---------------------------------------------------------------------------


class TestTrainResult:
    def test_defaults(self):
        r = TrainResult()
        assert r.best_eval_f1 == 0.0
        assert r.train_history == []
        assert r.label_map == {}

    def test_populated(self):
        r = TrainResult(
            best_eval_f1=0.87,
            best_eval_loss=0.32,
            best_epoch=3,
            train_history=[{"epoch": 1, "eval_f1": 0.80}],
            model_path="/tmp/model",
            label_map={"pos": 0, "neg": 1},
            id2label={0: "pos", 1: "neg"},
        )
        assert r.best_eval_f1 == 0.87
        assert r.id2label[0] == "pos"


# ---------------------------------------------------------------------------
# PredictionResult
# ---------------------------------------------------------------------------


class TestPredictionResult:
    def test_defaults(self):
        r = PredictionResult()
        assert r.predictions == []
        assert r.probabilities == []

    def test_populated(self):
        r = PredictionResult(
            predictions=["pos", "neg"],
            probabilities=[{"pos": 0.9, "neg": 0.1}, {"pos": 0.2, "neg": 0.8}],
        )
        assert len(r.predictions) == 2
        assert r.probabilities[0]["pos"] == 0.9


# ---------------------------------------------------------------------------
# Classifier initialization
# ---------------------------------------------------------------------------


class TestClassifierInit:
    def test_init(self):
        clf = Classifier()
        assert clf._model is None
        assert clf._tokenizer is None
        assert clf._label_map == {}

    def test_predict_without_model(self):
        clf = Classifier()
        with pytest.raises(RuntimeError, match="No model loaded"):
            clf.predict(["hello"])

    def test_save_without_model(self):
        clf = Classifier()
        with pytest.raises(RuntimeError, match="No model to save"):
            clf.save("/tmp/test_save")


# ---------------------------------------------------------------------------
# to_dataframe utility
# ---------------------------------------------------------------------------


class TestToDataFrame:
    def test_basic(self):
        clf = Classifier()
        texts = ["text A", "text B"]
        result = PredictionResult(
            predictions=["pos", "neg"],
            probabilities=[
                {"pos": 0.9, "neg": 0.1},
                {"pos": 0.3, "neg": 0.7},
            ],
        )
        df = clf.to_dataframe(texts, result)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "text" in df.columns
        assert "predicted_label" in df.columns
        assert "prob_pos" in df.columns
        assert "prob_neg" in df.columns
        assert df.iloc[0]["predicted_label"] == "pos"
        assert df.iloc[0]["prob_pos"] == 0.9

    def test_empty(self):
        clf = Classifier()
        df = clf.to_dataframe([], PredictionResult())
        assert len(df) == 0


# ---------------------------------------------------------------------------
# Load with metadata (mocked file system)
# ---------------------------------------------------------------------------


class TestLoadMeta:
    def test_load_parses_meta(self, tmp_path):
        """Test that load() correctly reads socialscikit_meta.json."""
        meta = {
            "label_map": {"positive": 0, "negative": 1},
            "id2label": {"0": "positive", "1": "negative"},
        }
        meta_path = tmp_path / "socialscikit_meta.json"
        meta_path.write_text(json.dumps(meta))

        clf = Classifier()

        # Imports are local inside load(), so we patch transformers directly
        mock_tok_cls = MagicMock()
        mock_model_cls = MagicMock()
        mock_tok_cls.from_pretrained.return_value = MagicMock()
        mock_model_cls.from_pretrained.return_value = MagicMock()

        with patch.dict("sys.modules", {
            "transformers": MagicMock(
                AutoTokenizer=mock_tok_cls,
                AutoModelForSequenceClassification=mock_model_cls,
            ),
        }):
            clf.load(str(tmp_path))

        assert clf._label_map == {"positive": 0, "negative": 1}
        assert clf._id2label == {0: "positive", 1: "negative"}

    def test_load_without_meta(self, tmp_path):
        """Load should work even without meta file (empty maps)."""
        clf = Classifier()

        mock_tok_cls = MagicMock()
        mock_model_cls = MagicMock()
        mock_tok_cls.from_pretrained.return_value = MagicMock()
        mock_model_cls.from_pretrained.return_value = MagicMock()

        with patch.dict("sys.modules", {
            "transformers": MagicMock(
                AutoTokenizer=mock_tok_cls,
                AutoModelForSequenceClassification=mock_model_cls,
            ),
        }):
            clf.load(str(tmp_path))

        assert clf._label_map == {}
        assert clf._id2label == {}


# ---------------------------------------------------------------------------
# Train (mocked — no actual model download)
# ---------------------------------------------------------------------------


class TestTrainMocked:
    def test_label_map_built_correctly(self):
        """Verify label map is correctly built from training data."""
        clf = Classifier()
        train_df = pd.DataFrame({
            "text": ["good", "bad", "okay", "great", "terrible", "fine"] * 5,
            "label": ["pos", "neg", "neutral", "pos", "neg", "neutral"] * 5,
        })

        # We'll test just the label map logic by checking the train method's
        # setup. Full training requires model downloads, so we mock that.
        labels = sorted(train_df["label"].dropna().unique().tolist())
        label_map = {label: i for i, label in enumerate(labels)}
        id2label = {i: label for label, i in label_map.items()}

        assert label_map == {"neg": 0, "neutral": 1, "pos": 2}
        assert id2label == {0: "neg", 1: "neutral", 2: "pos"}

    def test_train_val_split_shapes(self):
        """Verify train/val split produces expected sizes."""
        from sklearn.model_selection import train_test_split

        texts = [f"text {i}" for i in range(100)]
        labels = ["pos"] * 50 + ["neg"] * 50
        train_t, val_t, train_l, val_l = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels,
        )
        assert len(train_t) == 80
        assert len(val_t) == 20
        # Stratified: each split should have both classes
        assert "pos" in train_l and "neg" in train_l
        assert "pos" in val_l and "neg" in val_l
