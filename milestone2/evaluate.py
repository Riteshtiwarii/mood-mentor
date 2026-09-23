"""Model Evaluation & Benchmark Module for Mood Mentor (Milestone 2).

Computes Accuracy, Precision, Recall, Macro F1-Score, compares BERT vs DistilBERT,
and runs validation on the held-out ISEAR benchmark subset.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import logging
from pathlib import Path
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from config import (
    DATA_DIR,
    DEFAULT_PREDICTION_THRESHOLD,
    EMOTION_LABELS,
    configure_logging,
)
from emotion_classifier import TransformerEmotionClassifier
from exceptions import BenchmarkError, EvaluationError

logger = logging.getLogger("mood_mentor.evaluation")


@dataclass(frozen=True)
class ModelMetrics:
    """Metrics container for a single evaluated model."""

    model_name: str
    sample_count: int
    subset_accuracy: float
    hamming_accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    avg_inference_time_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "sample_count": self.sample_count,
            "subset_accuracy": round(self.subset_accuracy, 4),
            "hamming_accuracy": round(self.hamming_accuracy, 4),
            "macro_precision": round(self.macro_precision, 4),
            "macro_recall": round(self.macro_recall, 4),
            "macro_f1": round(self.macro_f1, 4),
            "avg_inference_time_ms": round(self.avg_inference_time_ms, 2),
        }


def evaluate_model_on_dataset(
    classifier: TransformerEmotionClassifier,
    dataset_csv: Path,
    threshold: float = DEFAULT_PREDICTION_THRESHOLD,
) -> ModelMetrics:
    """Evaluate multi-label emotion classifier on an annotated CSV dataset."""
    if not dataset_csv.exists():
        raise EvaluationError(f"Evaluation dataset not found: {dataset_csv}")

    df = pd.read_csv(dataset_csv)
    texts = df["text"].astype(str).tolist()
    y_true = df[list(EMOTION_LABELS)].values.astype(int)

    y_pred_list: List[List[int]] = []
    inference_latencies: List[float] = []

    for text in texts:
        t0 = time.perf_counter()
        pred = classifier.predict(text)
        latency = (time.perf_counter() - t0) * 1000.0
        inference_latencies.append(latency)

        row_pred = [1 if pred.probabilities[emotion] >= threshold else 0 for emotion in EMOTION_LABELS]
        # If all zeros but primary confidence is reasonable, assign primary emotion
        if sum(row_pred) == 0 and pred.primary_confidence > 0.15:
            primary_idx = EMOTION_LABELS.index(pred.primary_emotion)
            row_pred[primary_idx] = 1

        y_pred_list.append(row_pred)

    y_pred = np.array(y_pred_list)

    # Calculate metrics
    subset_acc = float(accuracy_score(y_true, y_pred))
    hamming_acc = float(np.mean(y_true == y_pred))
    macro_prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    avg_latency = float(np.mean(inference_latencies))

    return ModelMetrics(
        model_name=classifier.model_type,
        sample_count=len(texts),
        subset_accuracy=subset_acc,
        hamming_accuracy=hamming_acc,
        macro_precision=macro_prec,
        macro_recall=macro_rec,
        macro_f1=macro_f1,
        avg_inference_time_ms=avg_latency,
    )


def compare_models(
    val_csv: Path = DATA_DIR / "emotion_val.csv",
) -> tuple[ModelMetrics, ModelMetrics, str]:
    """Compare BERT and DistilBERT models and identify the best-performing model."""
    logger.info("Initializing DistilBERT classifier for evaluation...")
    distilbert_clf = TransformerEmotionClassifier(model_type="distilbert")
    distilbert_metrics = evaluate_model_on_dataset(distilbert_clf, val_csv)

    logger.info("Initializing BERT classifier for evaluation...")
    bert_clf = TransformerEmotionClassifier(model_type="bert")
    bert_metrics = evaluate_model_on_dataset(bert_clf, val_csv)

    # Determine best model (macro F1 primary, latency secondary)
    if distilbert_metrics.macro_f1 >= bert_metrics.macro_f1:
        best_model = "distilbert"
    else:
        best_model = "bert"

    logger.info("Comparison completed. Better performing model: %s", best_model.upper())
    return bert_metrics, distilbert_metrics, best_model


def run_isear_benchmark(
    classifier: TransformerEmotionClassifier,
    isear_csv: Path = DATA_DIR / "isear_subset.csv",
    threshold: float = DEFAULT_PREDICTION_THRESHOLD,
) -> dict[str, Any]:
    """Validate model against the held-out ISEAR benchmark subset (Task 6)."""
    if not isear_csv.exists():
        raise BenchmarkError(f"ISEAR benchmark file not found: {isear_csv}")

    df = pd.read_csv(isear_csv)
    total = len(df)
    correct_matches = 0
    emotion_stats: dict[str, dict[str, int]] = {e: {"total": 0, "correct": 0} for e in EMOTION_LABELS}
    incorrect_cases: list[dict[str, Any]] = []

    for idx, row in df.iterrows():
        text = str(row["text"])
        expected_label = str(row["label"]).lower().strip()
        if expected_label in emotion_stats:
            emotion_stats[expected_label]["total"] += 1

        pred = classifier.predict(text)
        predicted_primary = pred.primary_emotion

        # Match if primary emotion matches or expected label is in predicted active list
        is_match = (predicted_primary == expected_label) or (expected_label in pred.predicted_emotions)
        if is_match:
            correct_matches += 1
            if expected_label in emotion_stats:
                emotion_stats[expected_label]["correct"] += 1
        else:
            incorrect_cases.append({
                "text": text,
                "expected": expected_label,
                "predicted": predicted_primary,
                "confidence": pred.primary_confidence,
            })

    overall_accuracy = (correct_matches / total) * 100.0 if total > 0 else 0.0

    emotion_accuracy = {}
    for emotion, stats in emotion_stats.items():
        t = stats["total"]
        c = stats["correct"]
        pct = (c / t * 100.0) if t > 0 else 0.0
        emotion_accuracy[emotion] = {
            "samples": t,
            "correct": c,
            "accuracy_pct": round(pct, 2),
        }

    logger.info(
        "ISEAR Benchmark completed: %d/%d (%.2f%% accuracy)",
        correct_matches,
        total,
        overall_accuracy,
    )

    return {
        "model_name": classifier.model_type,
        "total_samples": total,
        "correct_predictions": correct_matches,
        "overall_accuracy_pct": round(overall_accuracy, 2),
        "emotion_breakdown": emotion_accuracy,
        "incorrect_cases_sample": incorrect_cases[:5],
    }


def format_evaluation_report(
    bert_metrics: ModelMetrics,
    distilbert_metrics: ModelMetrics,
    best_model: str,
    isear_results: dict[str, Any],
) -> str:
    """Generate human-readable terminal comparison and benchmark card."""
    lines: list[str] = []
    divider = "=" * 80
    sub_divider = "-" * 80

    lines.append(divider)
    lines.append("          MOOD MENTOR - MILESTONE 2 MODEL EVALUATION & BENCHMARKS")
    lines.append(divider)
    lines.append("")
    lines.append("📊 TASK 5: BERT vs DISTILBERT PERFORMANCE COMPARISON:")
    lines.append(f"{'Metric':<25} | {'BERT':<15} | {'DistilBERT':<15} | {'Winner':<12}")
    lines.append(sub_divider)

    def row(label: str, val1: float, val2: float, higher_better: bool = True) -> str:
        winner = "BERT" if (val1 > val2 if higher_better else val1 < val2) else "DistilBERT"
        if abs(val1 - val2) < 1e-4:
            winner = "Tied"
        return f"{label:<25} | {val1:<15.4f} | {val2:<15.4f} | {winner:<12}"

    lines.append(row("Subset Accuracy", bert_metrics.subset_accuracy, distilbert_metrics.subset_accuracy))
    lines.append(row("Hamming Accuracy", bert_metrics.hamming_accuracy, distilbert_metrics.hamming_accuracy))
    lines.append(row("Macro Precision", bert_metrics.macro_precision, distilbert_metrics.macro_precision))
    lines.append(row("Macro Recall", bert_metrics.macro_recall, distilbert_metrics.macro_recall))
    lines.append(row("Macro F1-Score", bert_metrics.macro_f1, distilbert_metrics.macro_f1))
    lines.append(f"{'Inference Latency (ms)':<25} | {bert_metrics.avg_inference_time_ms:<15.2f} | {distilbert_metrics.avg_inference_time_ms:<15.2f} | {'DistilBERT':<12}")
    lines.append("")
    lines.append(f"🏆 SELECTED BEST-PERFORMING MODEL: {best_model.upper()}")
    lines.append("")
    lines.append(sub_divider)
    lines.append("🎯 TASK 6: ISEAR BENCHMARK VALIDATION (HELD-OUT SUBSET):")
    lines.append(sub_divider)
    lines.append(f"  • Total Benchmark Samples : {isear_results['total_samples']}")
    lines.append(f"  • Correct Emotion Matches : {isear_results['correct_predictions']}")
    lines.append(f"  • Overall Benchmark Acc   : {isear_results['overall_accuracy_pct']:.2f}%")
    lines.append("")
    lines.append("  Emotion-wise Accuracy Breakdown:")
    for emotion, stat in isear_results["emotion_breakdown"].items():
        lines.append(
            f"    - {emotion.capitalize():<10} : {stat['correct']}/{stat['samples']} correct ({stat['accuracy_pct']:.1f}%)"
        )

    lines.append(divider)
    return "\n".join(lines)


def main() -> None:
    """CLI execution for model evaluation and ISEAR benchmark."""
    parser = argparse.ArgumentParser(description="Evaluate BERT vs DistilBERT and run ISEAR benchmark")
    parser.add_argument("--val-file", type=str, default=str(DATA_DIR / "emotion_val.csv"))
    parser.add_argument("--isear-file", type=str, default=str(DATA_DIR / "isear_subset.csv"))
    args = parser.parse_args()

    configure_logging()

    bert_metrics, distilbert_metrics, best_model = compare_models(val_csv=Path(args.val_file))

    selected_clf = TransformerEmotionClassifier(model_type=best_model)
    isear_res = run_isear_benchmark(selected_clf, isear_csv=Path(args.isear_file))

    report = format_evaluation_report(bert_metrics, distilbert_metrics, best_model, isear_res)
    print(report)


if __name__ == "__main__":
    main()
