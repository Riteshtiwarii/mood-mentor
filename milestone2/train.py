"""
Milestone 2: DistilBERT Training Script for 6-Emotion Classification.
Uses PyTorch + Hugging Face Transformers with BCEWithLogitsLoss (Multi-Label Sigmoid).
"""

import os
import json
import logging
import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from dataset_loader import get_dataloaders, TARGET_EMOTIONS, LABEL2ID, ID2LABEL

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def train_model(config_path: str = "milestone2/config.json", output_dir: str = "milestone2/models/distilbert", epochs: int = 2, batch_size: int = 16, lr: float = 3e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load configuration
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        model_name = cfg.get("model_name", "distilbert-base-uncased")
        max_len = cfg.get("max_length", 128)
    else:
        model_name = "distilbert-base-uncased"
        max_len = 128

    logger.info(f"Loading dataset with base model '{model_name}'...")
    train_loader, val_loader, test_loader, tokenizer = get_dataloaders(
        model_name=model_name,
        batch_size=batch_size,
        max_len=max_len
    )

    logger.info(f"Initializing DistilBERT sequence classification model (6 labels, multi-label)...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(TARGET_EMOTIONS),
        problem_type="multi_label_classification",
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)
    criterion = nn.BCEWithLogitsLoss()

    logger.info(f"Starting training for {epochs} epochs ({total_steps} total steps)...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for step, batch in enumerate(train_loader, 1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = criterion(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            if step % 100 == 0 or step == len(train_loader):
                logger.info(f"Epoch {epoch}/{epochs} | Step {step}/{len(train_loader)} | Loss: {loss.item():.4f}")

        avg_train_loss = total_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        logger.info(f"Epoch {epoch} Complete | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # Save model artifacts
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Saving model and tokenizer to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save config metadata
    config_dict = {
        "architectures": ["DistilBertForSequenceClassification"],
        "model_type": "distilbert",
        "num_labels": 6,
        "problem_type": "multi_label_classification",
        "id2label": ID2LABEL,
        "label2id": LABEL2ID
    }
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)

    logger.info(f"Training completed successfully. Model saved to {output_dir}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DistilBERT Emotion Classification Model")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--output_dir", type=str, default="milestone2/models/distilbert", help="Output directory")
    args = parser.parse_args()

    train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_dir=args.output_dir
    )
