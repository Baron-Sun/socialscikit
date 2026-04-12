"""Fine-tuning pipeline — train and predict with HuggingFace transformers.

Supports RoBERTa (English) and XLM-RoBERTa (multilingual) for sequence
classification. Handles train/val splitting, tokenisation, training with
early stopping, prediction, and model saving/loading.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Default models
DEFAULT_MODEL_EN = "roberta-base"
DEFAULT_MODEL_MULTI = "xlm-roberta-base"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    """Training hyper-parameters."""

    model_name: str = DEFAULT_MODEL_EN
    max_length: int = 512
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    early_stopping_patience: int = 2
    eval_steps: int | None = None  # None = evaluate each epoch
    output_dir: str = "./socialscikit_model"
    seed: int = 42


@dataclass
class TrainResult:
    """Result of a fine-tuning run."""

    best_eval_f1: float = 0.0
    best_eval_loss: float = 0.0
    best_epoch: int = 0
    train_history: list[dict] = field(default_factory=list)
    model_path: str = ""
    label_map: dict[str, int] = field(default_factory=dict)
    id2label: dict[int, str] = field(default_factory=dict)


@dataclass
class PredictionResult:
    """Batch prediction output."""

    predictions: list[str] = field(default_factory=list)
    probabilities: list[dict[str, float]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


class Classifier:
    """HuggingFace transformer fine-tuning pipeline.

    Usage::

        clf = Classifier()
        result = clf.train(
            train_df=train_df,
            text_col="text",
            label_col="label",
            config=TrainConfig(num_epochs=3),
        )
        preds = clf.predict(texts=["new text to classify"])
    """

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._label_map: dict[str, int] = {}
        self._id2label: dict[int, str] = {}
        self._config: TrainConfig | None = None

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------

    def train(
        self,
        train_df: pd.DataFrame,
        text_col: str = "text",
        label_col: str = "label",
        val_df: pd.DataFrame | None = None,
        val_ratio: float = 0.2,
        config: TrainConfig | None = None,
    ) -> TrainResult:
        """Fine-tune a transformer model on labeled data.

        Parameters
        ----------
        train_df : pd.DataFrame
            Training data with text and label columns.
        text_col, label_col : str
            Column names.
        val_df : pd.DataFrame or None
            Explicit validation set. If None, splits from train_df.
        val_ratio : float
            Fraction of train_df used for validation (if val_df is None).
        config : TrainConfig or None
            Training configuration. Defaults to TrainConfig().

        Returns
        -------
        TrainResult
        """
        from datasets import Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            EarlyStoppingCallback,
            Trainer,
            TrainingArguments,
        )
        from sklearn.model_selection import train_test_split
        import numpy as np

        config = config or TrainConfig()
        self._config = config

        # --- Build label map ---
        labels = sorted(train_df[label_col].dropna().unique().tolist())
        self._label_map = {label: i for i, label in enumerate(labels)}
        self._id2label = {i: label for label, i in self._label_map.items()}
        num_labels = len(labels)

        # --- Train/val split ---
        if val_df is None:
            train_texts, val_texts, train_labels, val_labels = train_test_split(
                train_df[text_col].tolist(),
                train_df[label_col].tolist(),
                test_size=val_ratio,
                random_state=config.seed,
                stratify=train_df[label_col].tolist(),
            )
        else:
            train_texts = train_df[text_col].tolist()
            train_labels = train_df[label_col].tolist()
            val_texts = val_df[text_col].tolist()
            val_labels = val_df[label_col].tolist()

        train_label_ids = [self._label_map[l] for l in train_labels]
        val_label_ids = [self._label_map[l] for l in val_labels]

        # --- Tokenise ---
        self._tokenizer = AutoTokenizer.from_pretrained(config.model_name)

        train_enc = self._tokenizer(
            train_texts, truncation=True, padding=True, max_length=config.max_length,
        )
        val_enc = self._tokenizer(
            val_texts, truncation=True, padding=True, max_length=config.max_length,
        )

        train_dataset = Dataset.from_dict({
            "input_ids": train_enc["input_ids"],
            "attention_mask": train_enc["attention_mask"],
            "labels": train_label_ids,
        })
        val_dataset = Dataset.from_dict({
            "input_ids": val_enc["input_ids"],
            "attention_mask": val_enc["attention_mask"],
            "labels": val_label_ids,
        })

        # --- Model ---
        self._model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name,
            num_labels=num_labels,
            id2label=self._id2label,
            label2id=self._label_map,
        )

        # --- Training args ---
        training_args = TrainingArguments(
            output_dir=config.output_dir,
            num_train_epochs=config.num_epochs,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            warmup_ratio=config.warmup_ratio,
            weight_decay=config.weight_decay,
            eval_strategy="epoch" if config.eval_steps is None else "steps",
            eval_steps=config.eval_steps,
            save_strategy="epoch" if config.eval_steps is None else "steps",
            save_steps=config.eval_steps,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            save_total_limit=2,
            seed=config.seed,
            logging_steps=50,
            report_to="none",
        )

        # --- Metrics ---
        def compute_metrics(eval_pred):
            logits, label_ids = eval_pred
            preds = np.argmax(logits, axis=-1)
            acc = (preds == label_ids).mean()
            # Macro F1
            f1_scores = []
            for cls_id in range(num_labels):
                tp = ((preds == cls_id) & (label_ids == cls_id)).sum()
                fp = ((preds == cls_id) & (label_ids != cls_id)).sum()
                fn = ((preds != cls_id) & (label_ids == cls_id)).sum()
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                f1_scores.append(f1)
            macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
            return {"accuracy": float(acc), "f1": float(macro_f1)}

        # --- Trainer ---
        callbacks = [EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)]
        trainer = Trainer(
            model=self._model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
        )

        # --- Train ---
        train_output = trainer.train()

        # --- Collect history ---
        history = []
        for entry in trainer.state.log_history:
            if "eval_f1" in entry:
                history.append({
                    "epoch": entry.get("epoch", 0),
                    "eval_loss": entry.get("eval_loss", 0),
                    "eval_f1": entry.get("eval_f1", 0),
                    "eval_accuracy": entry.get("eval_accuracy", 0),
                })

        best_f1 = max((h["eval_f1"] for h in history), default=0.0)
        best_entry = next((h for h in history if h["eval_f1"] == best_f1), {})

        # --- Save ---
        model_path = str(Path(config.output_dir) / "best_model")
        trainer.save_model(model_path)
        self._tokenizer.save_pretrained(model_path)

        return TrainResult(
            best_eval_f1=round(best_f1, 4),
            best_eval_loss=round(best_entry.get("eval_loss", 0), 4),
            best_epoch=int(best_entry.get("epoch", 0)),
            train_history=history,
            model_path=model_path,
            label_map=self._label_map,
            id2label=self._id2label,
        )

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(
        self,
        texts: list[str],
        batch_size: int = 32,
    ) -> PredictionResult:
        """Run predictions on new texts using the trained model.

        Parameters
        ----------
        texts : list[str]
        batch_size : int

        Returns
        -------
        PredictionResult
        """
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("No model loaded. Call train() or load() first.")

        import torch
        import numpy as np

        self._model.eval()
        device = next(self._model.parameters()).device

        all_preds: list[str] = []
        all_probs: list[dict[str, float]] = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i: i + batch_size]
            enc = self._tokenizer(
                batch_texts, truncation=True, padding=True,
                max_length=self._config.max_length if self._config else 512,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                outputs = self._model(**enc)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                pred_ids = np.argmax(probs, axis=-1)

            for j in range(len(batch_texts)):
                label = self._id2label.get(int(pred_ids[j]), "UNKNOWN")
                all_preds.append(label)
                prob_dict = {
                    self._id2label[k]: round(float(probs[j][k]), 4)
                    for k in range(probs.shape[1])
                    if k in self._id2label
                }
                all_probs.append(prob_dict)

        return PredictionResult(predictions=all_preds, probabilities=all_probs)

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save model, tokenizer, and label map to disk."""
        import json

        if self._model is None or self._tokenizer is None:
            raise RuntimeError("No model to save.")

        path = str(path)
        self._model.save_pretrained(path)
        self._tokenizer.save_pretrained(path)

        meta = {"label_map": self._label_map, "id2label": self._id2label}
        with open(os.path.join(path, "socialscikit_meta.json"), "w") as f:
            json.dump(meta, f)

    def load(self, path: str, model_name: str | None = None) -> None:
        """Load a saved model from disk.

        Parameters
        ----------
        path : str
            Directory containing the saved model.
        model_name : str or None
            Original model architecture name (e.g. "roberta-base").
            If None, auto-detected from config.
        """
        import json
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(path)

        meta_path = os.path.join(path, "socialscikit_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            self._label_map = meta.get("label_map", {})
            # JSON keys are strings, convert int keys
            raw_id2label = meta.get("id2label", {})
            self._id2label = {int(k): v for k, v in raw_id2label.items()}
        else:
            self._label_map = {}
            self._id2label = {}

        self._model = AutoModelForSequenceClassification.from_pretrained(path)
        self._config = TrainConfig(model_name=model_name or "unknown")

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def to_dataframe(self, texts: list[str], result: PredictionResult) -> pd.DataFrame:
        """Combine texts and predictions into a DataFrame."""
        rows = []
        for text, label, probs in zip(texts, result.predictions, result.probabilities):
            row = {"text": text, "predicted_label": label}
            row.update({f"prob_{k}": v for k, v in probs.items()})
            rows.append(row)
        return pd.DataFrame(rows)
