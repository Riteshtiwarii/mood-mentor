"""VADER Sentiment Analysis Module for Mood Mentor (Milestone 3 Baseline).

Computes baseline sentiment scores (compound, pos, neu, neg) and polarity classification.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Sequence

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from config import NEGATIVE_THRESHOLD, POSITIVE_THRESHOLD
from exceptions import SentimentAnalysisError

logger = logging.getLogger(__name__)


def _ensure_vader_downloaded() -> None:
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except (LookupError, FileNotFoundError):
        try:
            nltk.download("vader_lexicon", quiet=True)
        except Exception as exc:
            logger.debug("Could not auto-download vader_lexicon: %s", exc)


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
        return {
            "compound": self.compound,
            "pos": self.pos,
            "neu": self.neu,
            "neg": self.neg,
            "label": self.label,
        }


class VaderSentimentAnalyzer:
    """Wrapper around NLTK's VADER SentimentIntensityAnalyzer."""

    def __init__(
        self,
        positive_threshold: float = POSITIVE_THRESHOLD,
        negative_threshold: float = NEGATIVE_THRESHOLD,
    ) -> None:
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
        if compound_score >= self.positive_threshold:
            return "Positive"
        if compound_score <= self.negative_threshold:
            return "Negative"
        return "Neutral"

    def analyze(self, text: str) -> SentimentScore:
        if not isinstance(text, str):
            raise SentimentAnalysisError(
                f"Expected string input for sentiment analysis, got '{type(text).__name__}'."
            )

        stripped = text.strip()
        if not stripped:
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

        logger.info("Computed VADER sentiment scores for %d item(s).", len(results))
        return results


_DEFAULT_ANALYZER = VaderSentimentAnalyzer()


def analyze_sentiment(text: str) -> SentimentScore:
    return _DEFAULT_ANALYZER.analyze(text)


def analyze_batch(texts: Sequence[str]) -> list[SentimentScore]:
    return _DEFAULT_ANALYZER.analyze_batch(texts)
