"""Mood Mentor - Milestone 2 CLI & Pipeline Orchestrator.

Integrates Ingestion -> Preprocessing -> VADER Baseline Sentiment ->
Transformer Multi-Label Emotion Classification (BERT/DistilBERT) -> Unified Reporting.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

from config import (
    DATA_DIR,
    DEFAULT_CSV_TEXT_COLUMN,
    DEFAULT_PREDICTION_THRESHOLD,
    configure_logging,
)
from emotion_classifier import TransformerEmotionClassifier
from evaluate import compare_models, format_evaluation_report, run_isear_benchmark
from exceptions import MoodMentorError
from ingestion import validate_and_ingest
from preprocessing import preprocess_corpus
from report import format_terminal_report, generate_unified_report, to_dataframe
from sentiment import analyze_batch

logger = logging.getLogger("mood_mentor")


def run_pipeline(
    source: str | list[str],
    source_type: str = "auto",
    csv_column: str = DEFAULT_CSV_TEXT_COLUMN,
    model_type: str = "distilbert",
    threshold: float = DEFAULT_PREDICTION_THRESHOLD,
    output_csv_path: str | None = None,
) -> int:
    """Execute the end-to-end Milestone 2 processing pipeline."""
    try:
        logger.info("Starting Mood Mentor Milestone 2 pipeline...")

        # Stage 1: Ingestion
        raw_texts = validate_and_ingest(
            source=source,
            source_type=source_type,
            text_column=csv_column,
        )
        logger.info("Stage 1 (Ingestion) completed: %d text(s) loaded.", len(raw_texts))

        # Stage 2: Preprocessing
        preprocessed_results = preprocess_corpus(raw_texts)
        logger.info("Stage 2 (Preprocessing) completed.")

        # Stage 3: Baseline Sentiment (VADER)
        sentiment_scores = analyze_batch(raw_texts)
        logger.info("Stage 3 (VADER Sentiment Analysis) completed.")

        # Stage 4: Deep Emotion Classification (Transformer)
        logger.info("Initializing Transformer [%s] for emotion classification...", model_type.upper())
        classifier = TransformerEmotionClassifier(model_type=model_type, threshold=threshold)
        emotion_predictions = classifier.predict_batch(raw_texts)
        logger.info("Stage 4 (Transformer Emotion Analysis) completed.")

        # Stage 5: Unified Reporting
        summary = generate_unified_report(
            raw_texts=raw_texts,
            preprocessed=preprocessed_results,
            sentiments=sentiment_scores,
            emotions=emotion_predictions,
        )
        logger.info("Stage 5 (Report Generation) completed.")

        # Terminal Output
        print(format_terminal_report(summary))

        # Optional CSV Export
        if output_csv_path:
            out_file = Path(output_csv_path)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            df = to_dataframe(summary)
            df.to_csv(out_file, index=False)
            logger.info("Exported unified analysis to CSV at '%s'.", out_file.resolve())

        return 0

    except MoodMentorError as app_err:
        logger.error("Pipeline failed with application error: %s", app_err)
        return 1
    except Exception as exc:
        logger.critical("Unexpected failure occurred: %s", exc, exc_info=True)
        return 2


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        prog="mood_mentor_m2",
        description="Mood Mentor Milestone 2 - Deep Emotion Classification & Benchmark Validation",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-t", "--text", type=str, help="Direct text string to analyze.")
    group.add_argument("-f", "--file", type=str, help="Path to .txt or .csv dataset file.")

    parser.add_argument(
        "-m", "--model",
        choices=["distilbert", "bert"],
        default="distilbert",
        help="Transformer model architecture to use (default: distilbert).",
    )
    parser.add_argument(
        "--csv-col",
        type=str,
        default=DEFAULT_CSV_TEXT_COLUMN,
        help=f"Target column name for CSV input (default: '{DEFAULT_CSV_TEXT_COLUMN}').",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_PREDICTION_THRESHOLD,
        help=f"Confidence threshold for active emotions (default: {DEFAULT_PREDICTION_THRESHOLD}).",
    )
    parser.add_argument("-o", "--output", type=str, help="Path to save report output as CSV.")
    parser.add_argument("--evaluate", action="store_true", help="Run BERT vs DistilBERT benchmark evaluation.")
    parser.add_argument("--isear", action="store_true", help="Run ISEAR benchmark validation on held-out subset.")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level verbosity.",
    )
    return parser.parse_args()


def main() -> None:
    """Main CLI entry point."""
    args = parse_args()
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    configure_logging(level=log_level)

    base_dir = Path(__file__).parent

    # Benchmarks & Evaluation Modes
    if args.evaluate or args.isear:
        logger.info("Executing Benchmark & Evaluation suites...")
        bert_metrics, distilbert_metrics, best_model = compare_models()
        selected_clf = TransformerEmotionClassifier(model_type=best_model)
        isear_results = run_isear_benchmark(selected_clf)
        report = format_evaluation_report(bert_metrics, distilbert_metrics, best_model, isear_results)
        print(report)
        sys.exit(0)

    # Pipeline Processing Modes
    if args.text:
        exit_code = run_pipeline(
            source=args.text,
            source_type="direct",
            model_type=args.model,
            threshold=args.threshold,
            output_csv_path=args.output,
        )
    elif args.file:
        exit_code = run_pipeline(
            source=args.file,
            source_type="auto",
            csv_column=args.csv_col,
            model_type=args.model,
            threshold=args.threshold,
            output_csv_path=args.output,
        )
    else:
        # Default run over sample datasets
        sample_txt = base_dir / "data" / "sample.txt"
        logger.info("No input argument supplied. Running default demonstration over '%s'...", sample_txt.name)
        exit_code = run_pipeline(
            source=str(sample_txt),
            source_type="txt",
            model_type=args.model,
            threshold=args.threshold,
        )

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
