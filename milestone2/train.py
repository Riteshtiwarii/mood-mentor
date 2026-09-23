"""Model Training & Fine-Tuning Pipeline for Mood Mentor (Milestone 2).

Fine-tunes BERT and DistilBERT multi-label emotion classification heads
using PyTorch, Hugging Face Transformers, and BCEWithLogitsLoss.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from config import (
    BERT_BASE_NAME,
    BERT_MODEL_DIR,
    DATA_DIR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_SEQ_LEN,
    DEFAULT_NUM_EPOCHS,
    DISTILBERT_BASE_NAME,
    DISTILBERT_MODEL_DIR,
    EMOTION_LABELS,
    NUM_EMOTIONS,
    configure_logging,
)
from emotion_classifier import get_default_device
from exceptions import TrainingError

logger = logging.getLogger("mood_mentor.training")


class EmotionDataset(Dataset):
    """PyTorch Dataset for multi-label emotion classification."""

    def __init__(
        self,
        csv_path: Path,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = DEFAULT_MAX_SEQ_LEN,
    ) -> None:
        if not csv_path.exists():
            raise TrainingError(f"Dataset file not found: {csv_path}")

        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

        missing_cols = [col for col in ["text"] + list(EMOTION_LABELS) if col not in self.df.columns]
        if missing_cols:
            raise TrainingError(f"Missing required columns in dataset: {missing_cols}")

        self.texts = self.df["text"].astype(str).tolist()
        self.labels = self.df[list(EMOTION_LABELS)].values.astype("float32")

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float32)
        return item


def train_single_model(
    model_type: str,
    base_model_name: str,
    output_dir: Path,
    train_csv: Path = DATA_DIR / "emotion_train.csv",
    val_csv: Path = DATA_DIR / "emotion_val.csv",
    epochs: int = DEFAULT_NUM_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    device: Optional[torch.device] = None,
) -> None:
    """Fine-tune a single Transformer model and save artifacts."""
    device = device or get_default_device()
    logger.info("Starting training for %s on %s...", model_type.upper(), device)

    try:
        tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            base_model_name,
            use_fast=True,
        )

        train_dataset = EmotionDataset(train_csv, tokenizer)
        val_dataset = EmotionDataset(val_csv, tokenizer)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        config = AutoConfig.from_pretrained(
            base_model_name,
            num_labels=NUM_EMOTIONS,
            problem_type="multi_label_classification",
            id2label={i: label for i, label in enumerate(EMOTION_LABELS)},
            label2id={label: i for i, label in enumerate(EMOTION_LABELS)},
        )

        model: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            config=config,
            ignore_mismatched_sizes=True,
        )
        model.to(device)

        criterion = nn.BCEWithLogitsLoss()

        # Differentiate backbone parameters from classification head for fast convergence
        head_params = [p for n, p in model.named_parameters() if any(k in n for k in ("classifier", "pre_classifier", "score", "dense"))]
        backbone_params = [p for n, p in model.named_parameters() if not any(k in n for k in ("classifier", "pre_classifier", "score", "dense"))]

        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": learning_rate},
                {"params": head_params, "lr": 1.5e-3},
            ],
            weight_decay=0.01,
        )

        for epoch in range(1, epochs + 1):
            model.train()
            total_train_loss = 0.0

            for batch in train_loader:
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)

            # Validation step
            model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    val_loss = criterion(outputs.logits, labels)
                    total_val_loss += val_loss.item()

            avg_val_loss = total_val_loss / len(val_loader)
            logger.info(
                "[%s Epoch %d/%d] Train Loss: %.4f | Val Loss: %.4f",
                model_type.upper(),
                epoch,
                epochs,
                avg_train_loss,
                avg_val_loss,
            )

        # Save fine-tuned checkpoint
        output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info("Successfully saved fine-tuned %s to '%s'", model_type.upper(), output_dir)

    except Exception as exc:
        raise TrainingError(f"Training failed for model '{model_type}': {exc}") from exc


def main() -> None:
    """CLI execution for training pipeline."""
    parser = argparse.ArgumentParser(description="Fine-tune BERT & DistilBERT for Multi-Label Emotions")
    parser.add_argument(
        "--model",
        choices=["bert", "distilbert", "both"],
        default="both",
        help="Model architecture to train (default: both).",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_NUM_EPOCHS, help="Number of epochs.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size.")
    parser.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE, help="Learning rate.")
    args = parser.parse_args()

    configure_logging()

    if args.model in ("distilbert", "both"):
        train_single_model(
            model_type="distilbert",
            base_model_name=DISTILBERT_BASE_NAME,
            output_dir=DISTILBERT_MODEL_DIR,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
        )

    if args.model in ("bert", "both"):
        train_single_model(
            model_type="bert",
            base_model_name=BERT_BASE_NAME,
            output_dir=BERT_MODEL_DIR,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
        )


if __name__ == "__main__":
    main()
