"""Configuration constants and settings for Mood Mentor - Milestone 3."""

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

# Core Target Emotions
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

# Multi-label probability threshold
DEFAULT_PREDICTION_THRESHOLD: Final[float] = 0.35

# Pretrained base models
BERT_BASE_NAME: Final[str] = "google/bert_uncased_L-2_H-128_A-2"
DISTILBERT_BASE_NAME: Final[str] = "distilbert-base-uncased"
DEFAULT_MAX_SEQ_LEN: Final[int] = 128
DEFAULT_BATCH_SIZE: Final[int] = 8
DEFAULT_LEARNING_RATE: Final[float] = 3e-5

# Intensity Analysis Settings (Task 1)
INTENSITY_TIERS: Final[dict[str, tuple[float, float]]] = {
    "Low": (0.0, 0.35),
    "Moderate": (0.35, 0.65),
    "High": (0.65, 0.85),
    "Severe": (0.85, 1.0),
}

# Linguistic Intensifier Keywords (Boost Intensity)
LINGUISTIC_INTENSIFIERS: Final[FrozenSet[str]] = frozenset({
    "extremely", "utterly", "completely", "totally", "unbearably", "severely",
    "absolutely", "deeply", "immensely", "incredibly", "enormously", "terribly",
    "horribly", "excessively", "intensely", "desperately", "overwhelmingly",
    "overwhelmed", "uncontrollably", "violently", "paralyzed", "hopeless",
})

# Linguistic Mitigators (Lower Intensity)
LINGUISTIC_MITIGATORS: Final[FrozenSet[str]] = frozenset({
    "somewhat", "slightly", "a bit", "a little", "mildly", "moderately",
    "kind of", "sort of", "partially", "barely", "hardly", "a touch",
})

# Acute Crisis / Severe Burnout Triage Keywords (Task 1 Severity)
CRISIS_KEYWORDS: Final[FrozenSet[str]] = frozenset({
    "burnout", "panic", "breakdown", "suffocating", "suicidal", "cannot go on",
    "hopeless", "paralyzed", "despair", "breaking down", "toxic", "abusive",
    "fainting", "chest pain", "quitting immediately", "cannot cope",
})

# Recommendation Engine Weights (Task 4 Ranking Model)
WEIGHT_SEMANTIC: Final[float] = 0.30
WEIGHT_EMOTION: Final[float] = 0.25
WEIGHT_INTENSITY: Final[float] = 0.15
WEIGHT_PREFERENCE: Final[float] = 0.20
WEIGHT_COLLABORATIVE: Final[float] = 0.10

# Ranking & Filtering Thresholds
RECENT_RECOMMENDATION_PENALTY: Final[float] = 0.30
MIN_RELEVANCE_THRESHOLD: Final[float] = 0.25
DEFAULT_TOP_K: Final[int] = 3

# Longitudinal Emotional Trend Tracking Settings (Task 6)
TREND_WINDOW_SIZE: Final[int] = 5
TREND_ESCALATION_BOOST: Final[float] = 0.15

# Recommendation Feedback Learning Settings (Task 7)
FEEDBACK_FILE_PATH: Final[Path] = DATA_DIR / "feedback_history.json"
FEEDBACK_ACCEPT_BOOST: Final[float] = 0.15
FEEDBACK_REJECT_PENALTY: Final[float] = 0.20

# Evaluation Benchmarks Settings (Task 9)
EVALUATION_DATASET_PATH: Final[Path] = DATA_DIR / "recommendation_eval_set.json"
EVALUATION_K_VALUES: Final[tuple[int, ...]] = (1, 3, 5)

# VADER Baseline Sentiment Thresholds
POSITIVE_THRESHOLD: Final[float] = 0.05
NEGATIVE_THRESHOLD: Final[float] = -0.05

# Ingestion Defaults
DEFAULT_CSV_TEXT_COLUMN: Final[str] = "text"
DEFAULT_CSV_USER_COLUMN: Final[str] = "user_id"
SUPPORTED_FILE_EXTENSIONS: Final[FrozenSet[str]] = frozenset({".txt", ".csv"})

# Words to preserve during stopword filtering
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

REQUIRED_NLTK_PACKAGES: Final[tuple[str, ...]] = (
    "vader_lexicon",
    "stopwords",
    "wordnet",
    "punkt",
    "punkt_tab",
)


def configure_logging(level: int = logging.INFO) -> None:
    """Configure structured logging format for Mood Mentor Milestone 3."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
