"""Configuration constants and settings for Mood Mentor - Milestone 2."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final, FrozenSet

# Base Paths
PROJECT_ROOT: Final[Path] = Path(__file__).parent.resolve()
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
MODELS_DIR: Final[Path] = PROJECT_ROOT / "models"
BERT_MODEL_DIR: Final[Path] = MODELS_DIR / "bert"
DISTILBERT_MODEL_DIR: Final[Path] = MODELS_DIR / "distilbert"

# Core Target Emotions for Multi-Label Classification
EMOTION_LABELS: Final[tuple[str, ...]] = (
    "joy",
    "sadness",
    "anger",
    "fear",
    "surprise",
    "disgust",
)

EMOTION_TO_IDX: Final[dict[str, int]] = {label: idx for idx, label in enumerate(EMOTION_LABELS)}
IDX_TO_EMOTION: Final[dict[int, str]] = {idx: label for idx, label in enumerate(EMOTION_LABELS)}
NUM_EMOTIONS: Final[int] = len(EMOTION_LABELS)

# Multi-label probability threshold for predicting an emotion
DEFAULT_PREDICTION_THRESHOLD: Final[float] = 0.35

# Pre-trained base models from HuggingFace
BERT_BASE_NAME: Final[str] = "google/bert_uncased_L-2_H-128_A-2"  # Fast, lightweight BERT for reliable local execution
DISTILBERT_BASE_NAME: Final[str] = "distilbert-base-uncased"      # Standard DistilBERT architecture

# Training Hyperparameters
DEFAULT_MAX_SEQ_LEN: Final[int] = 128
DEFAULT_BATCH_SIZE: Final[int] = 8
DEFAULT_LEARNING_RATE: Final[float] = 5e-5
DEFAULT_HEAD_LEARNING_RATE: Final[float] = 1.5e-3
DEFAULT_NUM_EPOCHS: Final[int] = 8

# VADER Baseline Sentiment Thresholds (from Milestone 1)
POSITIVE_THRESHOLD: Final[float] = 0.05
NEGATIVE_THRESHOLD: Final[float] = -0.05

# Ingestion Defaults
DEFAULT_CSV_TEXT_COLUMN: Final[str] = "text"
SUPPORTED_FILE_EXTENSIONS: Final[FrozenSet[str]] = frozenset({".txt", ".csv"})

# Words to preserve during stopword filtering to prevent sentiment inversion
NEGATION_WORDS: Final[FrozenSet[str]] = frozenset({
    "not", "no", "never", "neither", "nor", "none", "barely", "hardly", "scarcely",
    "cannot", "cant", "can't", "wont", "won't", "dont", "don't", "doesnt", "doesn't",
    "didnt", "didn't", "isnt", "isn't", "arent", "aren't", "wasnt", "wasn't", "werent",
    "weren't", "havent", "haven't", "hasnt", "hasn't", "hadnt", "hadn't", "couldnt",
    "couldn't", "shouldnt", "shouldn't", "wouldnt", "wouldn't", "without", "against",
    "despite",
})

CONTRASTIVE_CONJUNCTIONS: Final[FrozenSet[str]] = frozenset({
    "but", "however", "yet", "although", "though", "nonetheless", "nevertheless", "still",
})

# Required NLTK packages for baseline & token preprocessing
REQUIRED_NLTK_PACKAGES: Final[tuple[str, ...]] = (
    "vader_lexicon",
    "stopwords",
    "wordnet",
    "punkt",
    "punkt_tab",
)


def configure_logging(level: int = logging.INFO) -> None:
    """Configure structured logging format for Mood Mentor Milestone 2."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
