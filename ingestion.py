"""Text Ingestion Module for Mood Mentor (Milestone 1).

Supports direct text strings, .txt files, and .csv datasets with strict input
validation, error handling, and structured logging.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Sequence, Union

from config import DEFAULT_CSV_TEXT_COLUMN, SUPPORTED_FILE_EXTENSIONS
from exceptions import IngestionError

logger = logging.getLogger(__name__)


def ingest_direct_text(text: str) -> list[str]:
    """Ingest and validate a direct text string.

    Args:
        text: Raw input string or multiline text.

    Returns:
        List of non-empty text lines/segments.

    Raises:
        IngestionError: If the input is not a string, or contains only whitespace.
    """
    if not isinstance(text, str):
        raise IngestionError(
            f"Expected string input, received type '{type(text).__name__}'."
        )

    stripped = text.strip()
    if not stripped:
        raise IngestionError("Input text cannot be empty or solely whitespace.")

    # Split multiline strings into individual meaningful lines
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    logger.info("Ingested %d line(s) from direct text input.", len(lines))
    return lines


def ingest_txt_file(filepath: Union[str, Path]) -> list[str]:
    """Ingest and validate lines from a plain text (.txt) file.

    Args:
        filepath: Path to the .txt file.

    Returns:
        List of non-empty text lines.

    Raises:
        IngestionError: If file does not exist, is not a .txt file, or has no valid text.
    """
    path = Path(filepath)

    if not path.exists():
        raise IngestionError(f"File not found: '{path.resolve()}'")

    if not path.is_file():
        raise IngestionError(f"Path is not a regular file: '{path.resolve()}'")

    if path.suffix.lower() != ".txt":
        raise IngestionError(
            f"Invalid file extension '{path.suffix}'. Expected '.txt'."
        )

    try:
        content = path.read_text(encoding="utf-8")
    except Exception as exc:
        raise IngestionError(
            f"Failed to read file '{path.resolve()}': {exc}"
        ) from exc

    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if not lines:
        raise IngestionError(
            f"Text file '{path.name}' contains no valid text content."
        )

    logger.info("Ingested %d valid line(s) from '%s'.", len(lines), path.name)
    return lines


def ingest_csv_file(
    filepath: Union[str, Path],
    text_column: str = DEFAULT_CSV_TEXT_COLUMN,
) -> list[str]:
    """Ingest and validate text entries from a CSV (.csv) file.

    Args:
        filepath: Path to the CSV file.
        text_column: Name of the column containing feedback/text data.

    Returns:
        List of non-empty text strings extracted from the specified column.

    Raises:
        IngestionError: If file is missing, not a CSV, missing the target column,
            or contains no valid text.
    """
    path = Path(filepath)

    if not path.exists():
        raise IngestionError(f"File not found: '{path.resolve()}'")

    if not path.is_file():
        raise IngestionError(f"Path is not a regular file: '{path.resolve()}'")

    if path.suffix.lower() != ".csv":
        raise IngestionError(
            f"Invalid file extension '{path.suffix}'. Expected '.csv'."
        )

    texts: list[str] = []
    try:
        with path.open(mode="r", encoding="utf-8-sig", newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            if not reader.fieldnames:
                raise IngestionError(f"CSV file '{path.name}' is empty or headerless.")

            # Trim whitespace from fieldnames for resilience
            fieldnames = [col.strip() for col in reader.fieldnames if col]
            if text_column not in fieldnames:
                raise IngestionError(
                    f"Column '{text_column}' not found in CSV. Available columns: {fieldnames}"
                )

            for row_idx, row in enumerate(reader, start=2):
                val = row.get(text_column)
                if val is not None and isinstance(val, str):
                    cleaned = val.strip()
                    if cleaned:
                        texts.append(cleaned)
                    else:
                        logger.debug(
                            "Skipping empty text entry at row %d in '%s'.",
                            row_idx,
                            path.name,
                        )
    except IngestionError:
        raise
    except Exception as exc:
        raise IngestionError(
            f"Error processing CSV file '{path.resolve()}': {exc}"
        ) from exc

    if not texts:
        raise IngestionError(
            f"No valid non-empty text rows found under column '{text_column}' in '{path.name}'."
        )

    logger.info(
        "Ingested %d valid text sample(s) from column '%s' in '%s'.",
        len(texts),
        text_column,
        path.name,
    )
    return texts


def validate_and_ingest(
    source: Union[str, Path, Sequence[str]],
    source_type: str = "auto",
    text_column: str = DEFAULT_CSV_TEXT_COLUMN,
) -> list[str]:
    """Unified ingestion gateway with automatic type detection and validation.

    Args:
        source: Text string, file path (.txt or .csv), or sequence of strings.
        source_type: One of 'auto', 'direct', 'txt', 'csv', 'list'.
        text_column: Column name when ingesting from CSV.

    Returns:
        List of validated, non-empty text strings.

    Raises:
        IngestionError: If the source format is invalid or extraction yields no data.
    """
    if source is None:
        raise IngestionError("Source cannot be None.")

    source_type_clean = source_type.strip().lower()

    if isinstance(source, (list, tuple)):
        # List of strings input
        valid_items: list[str] = []
        for idx, item in enumerate(source):
            if not isinstance(item, str):
                logger.warning("Ignoring non-string item at index %d: %r", idx, item)
                continue
            cleaned = item.strip()
            if cleaned:
                valid_items.append(cleaned)
        if not valid_items:
            raise IngestionError("No valid non-empty string entries in input list.")
        logger.info("Ingested %d item(s) from input list.", len(valid_items))
        return valid_items

    if isinstance(source, (str, Path)):
        path_candidate = Path(source)

        # Check if source points to an existing file or explicitly specified as file
        if source_type_clean == "txt" or (
            source_type_clean == "auto"
            and path_candidate.is_file()
            and path_candidate.suffix.lower() == ".txt"
        ):
            return ingest_txt_file(path_candidate)

        if source_type_clean == "csv" or (
            source_type_clean == "auto"
            and path_candidate.is_file()
            and path_candidate.suffix.lower() == ".csv"
        ):
            return ingest_csv_file(path_candidate, text_column=text_column)

        if source_type_clean == "direct":
            return ingest_direct_text(str(source))

        if source_type_clean == "auto":
            # If path candidate exists with unsupported extension, raise error
            if path_candidate.exists() and path_candidate.is_file():
                if path_candidate.suffix.lower() not in SUPPORTED_FILE_EXTENSIONS:
                    raise IngestionError(
                        f"Unsupported file format '{path_candidate.suffix}'. "
                        f"Supported formats: {sorted(SUPPORTED_FILE_EXTENSIONS)}"
                    )

            # Otherwise treat as raw direct string
            return ingest_direct_text(str(source))

    raise IngestionError(
        f"Unsupported source type '{type(source).__name__}' or mode '{source_type}'."
    )
