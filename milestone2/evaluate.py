"""
Milestone 2: Evaluation Script for 6-Emotion Classification Model.
Computes real empirical metrics (Micro/Macro F1, Precision, Recall, Per-Emotion Metrics)
on the held-out test split of GoEmotions.
"""

import os
import json
import logging
import argparse
import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from dataset_loader import get_dataloaders, TARGET_EMOTIONS, ID2LABEL, LABEL2ID

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def has_weights(path: str) -> bool:
    if not path or not os.path.exists(path):
        return False
    weight_files = ["model.safetensors", "pytorch_model.bin", "model.safetensors.index.json", "pytorch_model.bin.index.json"]
    return os.path.exists(os.path.join(path, "config.json")) and any(os.path.exists(os.path.join(path, wf)) for wf in weight_files)

def evaluate_model(model_dir: str = "milestone2/models/distilbert", threshold: float = 0.3, batch_size: int = 16) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Evaluating model on device: {device}")

    if not has_weights(model_dir):
        logger.warning(f"Fine-tuned model weights not found at '{model_dir}'. Evaluating base 'distilbert-base-uncased' model...")
        model_name = "distilbert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(TARGET_EMOTIONS),
            problem_type="multi_label_classification",
            id2label=ID2LABEL,
            label2id=LABEL2ID
        )
    else:
        logger.info(f"Loading trained model from '{model_dir}'...")
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    model.to(device)
    model.eval()

    _, _, test_loader, _ = get_dataloaders(
        model_name=model_dir if has_weights(model_dir) else "distilbert-base-uncased",
        batch_size=batch_size
    )

    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].cpu().numpy()

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            probs = torch.sigmoid(logits).cpu().numpy()

            binary_preds = (probs >= threshold).astype(float)

            all_probs.append(probs)
            all_preds.append(binary_preds)
            all_targets.append(labels)

    all_probs = np.vstack(all_probs)
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    # Compute metrics
    f1_micro = float(f1_score(all_targets, all_preds, average="micro", zero_division=0))
    f1_macro = float(f1_score(all_targets, all_preds, average="macro", zero_division=0))
    precision_micro = float(precision_score(all_targets, all_preds, average="micro", zero_division=0))
    precision_macro = float(precision_score(all_targets, all_preds, average="macro", zero_division=0))
    recall_micro = float(recall_score(all_targets, all_preds, average="micro", zero_division=0))
    recall_macro = float(recall_score(all_targets, all_preds, average="macro", zero_division=0))
    subset_acc = float(accuracy_score(all_targets, all_preds))

    # Per-emotion metrics
    per_emotion = {}
    for idx, emotion in enumerate(TARGET_EMOTIONS):
        p = float(precision_score(all_targets[:, idx], all_preds[:, idx], zero_division=0))
        r = float(recall_score(all_targets[:, idx], all_preds[:, idx], zero_division=0))
        f = float(f1_score(all_targets[:, idx], all_preds[:, idx], zero_division=0))
        per_emotion[emotion] = {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4)}

    report_data = {
        "evaluation_metrics": {
            "f1_micro": round(f1_micro, 4),
            "f1_macro": round(f1_macro, 4),
            "precision_micro": round(precision_micro, 4),
            "precision_macro": round(precision_macro, 4),
            "recall_micro": round(recall_micro, 4),
            "recall_macro": round(recall_macro, 4),
            "exact_match_ratio": round(subset_acc, 4)
        },
        "per_emotion_metrics": per_emotion,
        "classification_threshold": threshold,
        "total_test_samples": len(all_targets)
    }

    logger.info("=== EVALUATION REPORT ===")
    logger.info(f"Micro F1: {f1_micro:.4f} | Macro F1: {f1_macro:.4f}")
    logger.info(f"Micro Precision: {precision_micro:.4f} | Micro Recall: {recall_micro:.4f}")
    logger.info(f"Subset Accuracy: {subset_acc:.4f}")
    logger.info("Per-Emotion Metrics:")
    for emotion, metrics in per_emotion.items():
        logger.info(f"  - {emotion:10s} -> Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}")

    # Save evaluation report
    os.makedirs("milestone2", exist_ok=True)
    report_path = "milestone2/evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    logger.info(f"Evaluation report saved to {report_path}.")

    return report_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Emotion Classification Model")
    parser.add_argument("--model_dir", type=str, default="milestone2/models/distilbert", help="Path to model directory")
    parser.add_argument("--threshold", type=float, default=0.3, help="Decision threshold for multi-label binary prediction")
    args = parser.parse_args()

    evaluate_model(model_dir=args.model_dir, threshold=args.threshold)
