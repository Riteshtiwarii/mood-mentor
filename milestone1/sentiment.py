"""VADER Sentiment Analysis Module for Mood Mentor (Milestone 1).

Provides baseline sentiment scoring (compound, pos, neu, neg) and polarity
classification with dynamic scoring, type validation, and domain exception handling.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Sequence

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from config import NEGATIVE_THRESHOLD, POSITIVE_THRESHOLD, REQUIRED_NLTK_PACKAGES
from exceptions import SentimentAnalysisError

logger = logging.getLogger(__name__)


def _ensure_vader_downloaded() -> None:
    """Ensure VADER lexicon is downloaded and available."""
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except (LookupError, FileNotFoundError):
        logger.info("Downloading NLTK vader_lexicon...")
        try:
            nltk.download("vader_lexicon", quiet=True)
        except Exception as exc:
            logger.warning("Could not auto-download vader_lexicon: %s", exc)


_ensure_vader_downloaded()


@dataclass(frozen=True)
class SentimentScore:
    """Immutable data container for sentiment intensity scores and label."""

    compound: float
    pos: float
    neu: float
    neg: float
    label: str

    def to_dict(self) -> dict[str, float | str]:
        """Convert score object to dictionary representation."""
        return {
            "compound": self.compound,
            "pos": self.pos,
            "neu": self.neu,
            "neg": self.neg,
            "label": self.label,
        }


class VaderSentimentAnalyzer:
    """Production-grade wrapper around NLTK's VADER SentimentIntensityAnalyzer."""

    def __init__(
        self,
        positive_threshold: float = POSITIVE_THRESHOLD,
        negative_threshold: float = NEGATIVE_THRESHOLD,
    ) -> None:
        """Initialize the analyzer with configurable decision thresholds.

        Args:
            positive_threshold: Compound score at/above which text is Positive.
            negative_threshold: Compound score at/below which text is Negative.
        """
        if positive_threshold <= negative_threshold:
            raise ValueError("positive_threshold must be strictly greater than negative_threshold.")

        self.positive_threshold = positive_threshold
        self.negative_threshold = negative_threshold

        try:
            self._analyzer = SentimentIntensityAnalyzer()
        except Exception as exc:
            raise SentimentAnalysisError(
                f"Failed to initialize VADER SentimentIntensityAnalyzer: {exc}"
            ) from exc

    def classify_compound(self, compound_score: float) -> str:
        """Classify a compound score into 'Positive', 'Negative', or 'Neutral'.

        Args:
            compound_score: Real-valued score in range [-1.0, 1.0].

        Returns:
            Sentiment category string: 'Positive', 'Negative', or 'Neutral'.
        """
        if compound_score >= self.positive_threshold:
            return "Positive"
        if compound_score <= self.negative_threshold:
            return "Negative"
        return "Neutral"

    def analyze(self, text: str) -> SentimentScore:
        """Analyze text and compute dynamic sentiment metrics.

        Args:
            text: Input text to analyze.

        Returns:
            SentimentScore containing compound, pos, neu, neg, and label.

        Raises:
            SentimentAnalysisError: If text is not a string or analysis fails.
        """
        if not isinstance(text, str):
            raise SentimentAnalysisError(
                f"Expected string input for sentiment analysis, got '{type(text).__name__}'."
            )

        stripped = text.strip()
        if not stripped:
            logger.debug("Received empty/whitespace text. Returning default neutral score.")
            return SentimentScore(
                compound=0.0,
                pos=0.0,
                neu=1.0,
                neg=0.0,
                label="Neutral",
            )

        try:
            scores = self._analyzer.polarity_scores(stripped)
            compound = float(scores["compound"])
            pos = float(scores["pos"])
            neu = float(scores["neu"])
            neg = float(scores["neg"])
            label = self.classify_compound(compound)

            return SentimentScore(
                compound=compound,
                pos=pos,
                neu=neu,
                neg=neg,
                label=label,
            )
        except Exception as exc:
            raise SentimentAnalysisError(
                f"Sentiment scoring failed for text '{text[:40]}...': {exc}"
            ) from exc

    def analyze_batch(self, texts: Sequence[str]) -> list[SentimentScore]:
        """Analyze a sequence of texts.

        Args:
            texts: Sequence of input strings.

        Returns:
            List of SentimentScore objects.

        Raises:
            SentimentAnalysisError: If sequence is invalid.
        """
        if not isinstance(texts, (list, tuple)):
            raise SentimentAnalysisError(
                f"Expected list/tuple of strings, received '{type(texts).__name__}'."
            )

        results: list[SentimentScore] = []
        for idx, text in enumerate(texts):
            if not isinstance(text, str):
                raise SentimentAnalysisError(
                    f"Item at index {idx} is not a string (type: {type(text).__name__})."
                )
            results.append(self.analyze(text))

        logger.info("Computed sentiment scores for %d item(s).", len(results))
        return results


# Module-level default analyzer singleton
_DEFAULT_ANALYZER = VaderSentimentAnalyzer()


def analyze_sentiment(text: str) -> SentimentScore:
    """Analyze a single string with default VADER analyzer."""
    return _DEFAULT_ANALYZER.analyze(text)


def analyze_batch(texts: Sequence[str]) -> list[SentimentScore]:
    """Analyze multiple strings with default VADER analyzer."""
    return _DEFAULT_ANALYZER.analyze_batch(texts)
