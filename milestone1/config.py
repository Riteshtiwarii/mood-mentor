"""Configuration constants and settings for Mood Mentor - Milestone 1."""

import logging
from pathlib import Path
from typing import Final, FrozenSet
import nltk

# Ensure local venv/nltk_data directory is included in search path
_PROJECT_ROOT = Path(__file__).parent.resolve()
_LOCAL_NLTK_DATA = _PROJECT_ROOT / ".venv" / "nltk_data"
if _LOCAL_NLTK_DATA.exists() and str(_LOCAL_NLTK_DATA) not in nltk.data.path:
    nltk.data.path.insert(0, str(_LOCAL_NLTK_DATA))

# Sentiment Compound Classification Thresholds (VADER standard)
POSITIVE_THRESHOLD: Final[float] = 0.05
NEGATIVE_THRESHOLD: Final[float] = -0.05

# Ingestion Defaults
DEFAULT_CSV_TEXT_COLUMN: Final[str] = "text"
SUPPORTED_FILE_EXTENSIONS: Final[FrozenSet[str]] = frozenset({".txt", ".csv"})

# Words to preserve during stopword filtering to prevent sentiment inversion/distortion
NEGATION_WORDS: Final[FrozenSet[str]] = frozenset({
    "not",
    "no",
    "never",
    "neither",
    "nor",
    "none",
    "barely",
    "hardly",
    "scarcely",
    "cannot",
    "cant",
    "can't",
    "wont",
    "won't",
    "dont",
    "don't",
    "doesnt",
    "doesn't",
    "didnt",
    "didn't",
    "isnt",
    "isn't",
    "arent",
    "aren't",
    "wasnt",
    "wasn't",
    "werent",
    "weren't",
    "havent",
    "haven't",
    "hasnt",
    "hasn't",
    "hadnt",
    "hadn't",
    "couldnt",
    "couldn't",
    "shouldnt",
    "shouldn't",
    "wouldnt",
    "wouldn't",
    "without",
    "against",
    "despite",
})

# Required NLTK corpora / packages
REQUIRED_NLTK_PACKAGES: Final[tuple[str, ...]] = (
    "vader_lexicon",
    "stopwords",
    "wordnet",
    "punkt",
    "punkt_tab",
)


def configure_logging(level: int = logging.INFO) -> None:
    """Configure application-wide structured logging format."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
