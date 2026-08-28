"""Mood Mentor - Milestone 1 CLI & Pipeline Orchestrator.

Wires Ingestion -> Preprocessing -> VADER Sentiment Analysis -> Report Generation.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from config import DEFAULT_CSV_TEXT_COLUMN, configure_logging
from exceptions import MoodMentorError
from ingestion import validate_and_ingest
from preprocessing import preprocess_corpus
from report import format_terminal_report, generate_sentiment_report, to_dataframe
from sentiment import analyze_batch

logger = logging.getLogger("mood_mentor")


def run_pipeline(
    source: str | list[str],
    source_type: str = "auto",
    csv_column: str = DEFAULT_CSV_TEXT_COLUMN,
    output_csv_path: str | None = None,
) -> int:
    """Execute the end-to-end Milestone 1 processing pipeline.

    Args:
        source: Raw string, file path, or string list.
        source_type: 'auto', 'direct', 'txt', 'csv'.
        csv_column: Column name for CSV ingestion.
        output_csv_path: Optional file path to export analysis results.

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    try:
        logger.info("Starting Mood Mentor Milestone 1 pipeline...")

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

        # Stage 3: Sentiment Analysis
        sentiment_scores = analyze_batch(raw_texts)
        logger.info("Stage 3 (Sentiment Analysis) completed.")

        # Stage 4: Reporting
        report_summary = generate_sentiment_report(
            raw_texts=raw_texts,
            preprocessed_results=preprocessed_results,
            sentiment_scores=sentiment_scores,
        )
        logger.info("Stage 4 (Report Generation) completed.")

        # Display terminal report
        report_text = format_terminal_report(report_summary)
        print(report_text)

        # Optional CSV export
        if output_csv_path:
            df = to_dataframe(report_summary)
            out_file = Path(output_csv_path)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_file, index=False)
            logger.info("Exported results to CSV at '%s'.", out_file.resolve())

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
        prog="mood_mentor",
        description="Mood Mentor Milestone 1 - Text Ingestion & Baseline Sentiment Analysis",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-t", "--text",
        type=str,
        help="Direct text string to analyze.",
    )
    group.add_argument(
        "-f", "--file",
        type=str,
        help="Path to .txt or .csv dataset file.",
    )

    parser.add_argument(
        "--csv-col",
        type=str,
        default=DEFAULT_CSV_TEXT_COLUMN,
        help=f"Target column name in CSV input (default: '{DEFAULT_CSV_TEXT_COLUMN}').",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Optional path to save report output as CSV.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Application logging verbosity level (default: INFO).",
    )

    return parser.parse_args()


def main() -> None:
    """CLI execution entrypoint."""
    args = parse_args()
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    configure_logging(level=log_level)

    base_dir = Path(__file__).parent

    if args.text:
        exit_code = run_pipeline(
            source=args.text,
            source_type="direct",
            output_csv_path=args.output,
        )
    elif args.file:
        exit_code = run_pipeline(
            source=args.file,
            source_type="auto",
            csv_column=args.csv_col,
            output_csv_path=args.output,
        )
    else:
        # Default behavior: run demonstration across sample datasets
        sample_txt = base_dir / "data" / "sample.txt"
        sample_csv = base_dir / "data" / "sample.csv"

        logger.info("No input argument supplied. Running default sample demonstrations...")
        if sample_txt.exists():
            logger.info(">>> Processing sample text file: %s", sample_txt.name)
            run_pipeline(source=str(sample_txt), source_type="txt")

        if sample_csv.exists():
            logger.info(">>> Processing sample CSV file: %s", sample_csv.name)
            run_pipeline(source=str(sample_csv), source_type="csv")

        exit_code = 0

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
