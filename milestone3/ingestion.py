"""Text & Dataset Ingestion Module for Mood Mentor (Milestone 3).

Supports direct text strings, .txt files, and .csv datasets with user_id extraction,
strict input validation, domain error handling, and structured logging.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Sequence, Union

from config import DEFAULT_CSV_TEXT_COLUMN, DEFAULT_CSV_USER_COLUMN, SUPPORTED_FILE_EXTENSIONS
from exceptions import IngestionError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestedFeedback:
    """Standardized record holding ingested text and user identification."""

    text: str
    user_id: str | None = None
    row_idx: int = 1


def ingest_direct_text(text: str, user_id: str | None = None) -> list[IngestedFeedback]:
    """Ingest and validate direct text input."""
    if not isinstance(text, str):
        raise IngestionError(
            f"Expected string input, received type '{type(text).__name__}'."
        )

    stripped = text.strip()
    if not stripped:
        raise IngestionError("Input text cannot be empty or solely whitespace.")

    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    records = [
        IngestedFeedback(text=line, user_id=user_id, row_idx=idx + 1)
        for idx, line in enumerate(lines)
    ]
    logger.info("Ingested %d feedback item(s) from direct text input.", len(records))
    return records


def ingest_txt_file(
    filepath: Union[str, Path], user_id: str | None = None
) -> list[IngestedFeedback]:
    """Ingest non-empty lines from a plain text (.txt) file."""
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

    records = [
        IngestedFeedback(text=line, user_id=user_id, row_idx=idx + 1)
        for idx, line in enumerate(lines)
    ]
    logger.info("Ingested %d valid line(s) from '%s'.", len(records), path.name)
    return records


def ingest_csv_file(
    filepath: Union[str, Path],
    text_column: str = DEFAULT_CSV_TEXT_COLUMN,
    user_column: str = DEFAULT_CSV_USER_COLUMN,
) -> list[IngestedFeedback]:
    """Ingest text and optional user_id records from a CSV (.csv) file."""
    path = Path(filepath)

    if not path.exists():
        raise IngestionError(f"File not found: '{path.resolve()}'")

    if not path.is_file():
        raise IngestionError(f"Path is not a regular file: '{path.resolve()}'")

    if path.suffix.lower() != ".csv":
        raise IngestionError(
            f"Invalid file extension '{path.suffix}'. Expected '.csv'."
        )

    records: list[IngestedFeedback] = []
    try:
        with path.open(mode="r", encoding="utf-8-sig", newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            if not reader.fieldnames:
                raise IngestionError(f"CSV file '{path.name}' is empty or headerless.")

            fieldnames = [col.strip() for col in reader.fieldnames if col]
            if text_column not in fieldnames:
                raise IngestionError(
                    f"Column '{text_column}' not found in CSV. Available columns: {fieldnames}"
                )

            has_user_col = user_column in fieldnames

            for row_idx, row in enumerate(reader, start=2):
                val = row.get(text_column)
                if val is not None and isinstance(val, str):
                    cleaned = val.strip()
                    if cleaned:
                        uid = (
                            row.get(user_column, "").strip()
                            if has_user_col and row.get(user_column)
                            else None
                        )
                        records.append(
                            IngestedFeedback(
                                text=cleaned, user_id=uid or None, row_idx=row_idx
                            )
                        )
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

    if not records:
        raise IngestionError(
            f"No valid non-empty text rows found under column '{text_column}' in '{path.name}'."
        )

    logger.info(
        "Ingested %d valid feedback record(s) from column '%s' in '%s'.",
        len(records),
        text_column,
        path.name,
    )
    return records


def validate_and_ingest(
    source: Union[str, Path, Sequence[str]],
    source_type: str = "auto",
    text_column: str = DEFAULT_CSV_TEXT_COLUMN,
    user_id: str | None = None,
) -> list[IngestedFeedback]:
    """Unified ingestion gateway with automatic type detection and validation."""
    if source is None:
        raise IngestionError("Source cannot be None.")

    source_type_clean = source_type.strip().lower()

    if isinstance(source, (list, tuple)):
        valid_items: list[IngestedFeedback] = []
        for idx, item in enumerate(source):
            if not isinstance(item, str):
                logger.warning("Ignoring non-string item at index %d: %r", idx, item)
                continue
            cleaned = item.strip()
            if cleaned:
                valid_items.append(
                    IngestedFeedback(text=cleaned, user_id=user_id, row_idx=idx + 1)
                )
        if not valid_items:
            raise IngestionError("No valid non-empty string entries in input list.")
        logger.info("Ingested %d item(s) from input list.", len(valid_items))
        return valid_items

    if isinstance(source, (str, Path)):
        path_candidate = Path(source)

        if source_type_clean == "txt" or (
            source_type_clean == "auto"
            and path_candidate.is_file()
            and path_candidate.suffix.lower() == ".txt"
        ):
            return ingest_txt_file(path_candidate, user_id=user_id)

        if source_type_clean == "csv" or (
            source_type_clean == "auto"
            and path_candidate.is_file()
            and path_candidate.suffix.lower() == ".csv"
        ):
            return ingest_csv_file(path_candidate, text_column=text_column)

        if source_type_clean == "direct":
            return ingest_direct_text(str(source), user_id=user_id)

        if source_type_clean == "auto":
            if path_candidate.exists() and path_candidate.is_file():
                if path_candidate.suffix.lower() not in SUPPORTED_FILE_EXTENSIONS:
                    raise IngestionError(
                        f"Unsupported file format '{path_candidate.suffix}'. "
                        f"Supported formats: {sorted(SUPPORTED_FILE_EXTENSIONS)}"
                    )
            return ingest_direct_text(str(source), user_id=user_id)

    raise IngestionError(
        f"Unsupported source type '{type(source).__name__}' or mode '{source_type}'."
    )
