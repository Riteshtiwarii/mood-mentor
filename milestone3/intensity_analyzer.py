"""Emotion Intensity & Emotional State Analysis Module (Milestone 3 - Task 1).

Analyzes dominant emotion, multiple emotions, dynamic continuous intensity,
severity tiers, positive/negative polarity, and psychological mixed states.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import re
from typing import Any, Dict, List

from config import (
    CRISIS_KEYWORDS,
    EMOTION_LABELS,
    INTENSITY_TIERS,
    LINGUISTIC_INTENSIFIERS,
    LINGUISTIC_MITIGATORS,
)
from emotion_classifier import EmotionPrediction
from exceptions import IntensityAnalysisError
from sentiment import SentimentScore

logger = logging.getLogger(__name__)

EXCLAMATION_PATTERN = re.compile(r"!+")
ALL_CAPS_PATTERN = re.compile(r"\b[A-Z]{3,}\b")


@dataclass(frozen=True)
class EmotionalState:
    """Immutable comprehensive container holding deep emotional state analysis (Task 1)."""

    text: str
    dominant_emotion: str
    multiple_emotions: list[str]
    emotion_confidence: float
    intensity_score: float
    intensity_tier: str
    polarity: str
    compound_score: float
    is_mixed_state: bool
    mixed_state_type: str | None
    severity: str
    crisis_flag: bool
    probabilities: dict[str, float]
    intensity_breakdown: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "dominant_emotion": self.dominant_emotion.capitalize(),
            "multiple_emotions": [e.capitalize() for e in self.multiple_emotions],
            "emotion_confidence": round(self.emotion_confidence, 4),
            "intensity_score": round(self.intensity_score, 4),
            "intensity_tier": self.intensity_tier,
            "polarity": self.polarity,
            "compound_score": round(self.compound_score, 4),
            "is_mixed_state": self.is_mixed_state,
            "mixed_state_type": self.mixed_state_type,
            "severity": self.severity,
            "crisis_flag": self.crisis_flag,
            "probabilities": {k: round(v, 4) for k, v in self.probabilities.items()},
            "intensity_breakdown": {k: round(v, 4) for k, v in self.intensity_breakdown.items()},
        }


class EmotionIntensityAnalyzer:
    """Dynamic emotional intensity and psychological state analyzer."""

    def __init__(self) -> None:
        self.intensifiers = LINGUISTIC_INTENSIFIERS
        self.mitigators = LINGUISTIC_MITIGATORS
        self.crisis_keywords = CRISIS_KEYWORDS

    def _calculate_intensity(
        self,
        text: str,
        primary_confidence: float,
        vader_compound: float,
        active_emotions: list[str],
    ) -> tuple[float, dict[str, float]]:
        """Calculate continuous intensity score (0.0 to 1.0) dynamically."""
        lower_text = text.lower()
        words = set(re.findall(r"\b[a-z'-]+\b", lower_text))

        # Factor 1: Model Confidence Weight (40% baseline)
        confidence_component = primary_confidence * 0.45

        # Factor 2: Sentiment Extremity (30%)
        # Strong positive (+0.8) or strong negative (-0.8) indicates high emotional energy
        sentiment_component = abs(vader_compound) * 0.30

        # Factor 3: Linguistic Modifiers (+/- up to 20%)
        intensifier_matches = len(words.intersection(self.intensifiers))
        mitigator_matches = len(words.intersection(self.mitigators))
        modifier_component = (intensifier_matches * 0.08) - (mitigator_matches * 0.08)

        # Factor 4: Syntactic Emphasis (Punctuation and Capitalization)
        exclamation_count = len(EXCLAMATION_PATTERN.findall(text))
        all_caps_words = len(ALL_CAPS_PATTERN.findall(text))
        emphasis_component = min((exclamation_count * 0.04) + (all_caps_words * 0.05), 0.15)

        # Factor 5: Multi-emotion Synergy
        # Having multiple co-occurring emotions increases psychological complexity and load
        multi_emotion_component = min(len(active_emotions) * 0.03, 0.09)

        raw_intensity = (
            confidence_component
            + sentiment_component
            + modifier_component
            + emphasis_component
            + multi_emotion_component
        )

        # Clamp strictly between 0.05 and 1.00
        final_intensity = max(0.05, min(1.0, raw_intensity))

        breakdown = {
            "confidence_component": confidence_component,
            "sentiment_component": sentiment_component,
            "modifier_component": modifier_component,
            "emphasis_component": emphasis_component,
            "multi_emotion_component": multi_emotion_component,
            "raw_total": raw_intensity,
        }

        return final_intensity, breakdown

    def _determine_intensity_tier(self, intensity_score: float) -> str:
        """Map continuous score into discrete intensity tier."""
        for tier, (low, high) in INTENSITY_TIERS.items():
            if low <= intensity_score <= high:
                return tier
        return "Severe" if intensity_score > 0.85 else "Low"

    def _detect_mixed_state(
        self,
        active_emotions: list[str],
        vader_label: str,
        dominant_emotion: str,
    ) -> tuple[bool, str | None]:
        """Detect and classify nuanced mixed emotional states."""
        active_set = set(e.lower() for e in active_emotions)
        is_mixed = False
        mixed_type = None

        has_joy = "joy" in active_set
        has_fear = "fear" in active_set
        has_anger = "anger" in active_set
        has_sadness = "sadness" in active_set
        has_disgust = "disgust" in active_set

        if has_joy and has_fear:
            is_mixed = True
            mixed_type = "Anticipatory Anxiety (Joy + Fear)"
        elif has_anger and has_sadness:
            is_mixed = True
            mixed_type = "Frustrated Grief / Disheartened Outrage"
        elif has_anger and has_disgust:
            is_mixed = True
            mixed_type = "Moral Outrage & Contempt"
        elif has_fear and has_sadness:
            is_mixed = True
            mixed_type = "Anxious Despair / Burnout Vulnerability"
        elif has_joy and (has_sadness or has_anger or has_disgust):
            is_mixed = True
            mixed_type = "Bittersweet Ambivalence"
        elif len(active_set) >= 2:
            is_mixed = True
            mixed_type = f"Co-occurring {' + '.join(e.capitalize() for e in sorted(active_set))}"
        elif vader_label == "Positive" and dominant_emotion in ("sadness", "anger", "fear", "disgust"):
            is_mixed = True
            mixed_type = "Suppressed Strain (Positive Polarity with Negative Emotion)"

        return is_mixed, mixed_type

    def _evaluate_severity(
        self,
        text: str,
        intensity_score: float,
        polarity: str,
        dominant_emotion: str,
    ) -> tuple[str, bool]:
        """Classify emotional state into actionable severity triage levels."""
        lower_text = text.lower()
        has_crisis_keyword = any(kw in lower_text for kw in self.crisis_keywords)

        # High/Critical triage rules
        if has_crisis_keyword and intensity_score >= 0.60:
            return "Critical", True

        if polarity == "Positive":
            return "Mild", False

        if polarity == "Neutral":
            return "Mild" if intensity_score < 0.5 else "Moderate", False

        # Negative Polarity Severity mapping
        if intensity_score >= 0.85:
            severity = "Critical" if (has_crisis_keyword or dominant_emotion in ("fear", "sadness")) else "High"
            crisis = severity == "Critical"
            return severity, crisis
        elif intensity_score >= 0.65:
            return "High", has_crisis_keyword
        elif intensity_score >= 0.35:
            return "Moderate", False
        else:
            return "Mild", False

    def analyze(
        self,
        text: str,
        emotion_prediction: EmotionPrediction,
        sentiment_score: SentimentScore,
    ) -> EmotionalState:
        """Execute full emotional state and intensity analysis (Task 1)."""
        if not isinstance(text, str):
            raise IntensityAnalysisError(f"Expected string text, got {type(text).__name__}")

        try:
            intensity_score, breakdown = self._calculate_intensity(
                text=text,
                primary_confidence=emotion_prediction.primary_confidence,
                vader_compound=sentiment_score.compound,
                active_emotions=emotion_prediction.predicted_emotions,
            )

            intensity_tier = self._determine_intensity_tier(intensity_score)

            is_mixed, mixed_type = self._detect_mixed_state(
                active_emotions=emotion_prediction.predicted_emotions,
                vader_label=sentiment_score.label,
                dominant_emotion=emotion_prediction.primary_emotion,
            )

            severity, crisis_flag = self._evaluate_severity(
                text=text,
                intensity_score=intensity_score,
                polarity=sentiment_score.label,
                dominant_emotion=emotion_prediction.primary_emotion,
            )

            return EmotionalState(
                text=text,
                dominant_emotion=emotion_prediction.primary_emotion,
                multiple_emotions=emotion_prediction.predicted_emotions,
                emotion_confidence=emotion_prediction.primary_confidence,
                intensity_score=intensity_score,
                intensity_tier=intensity_tier,
                polarity=sentiment_score.label,
                compound_score=sentiment_score.compound,
                is_mixed_state=is_mixed,
                mixed_state_type=mixed_type,
                severity=severity,
                crisis_flag=crisis_flag,
                probabilities=emotion_prediction.probabilities,
                intensity_breakdown=breakdown,
            )

        except Exception as exc:
            raise IntensityAnalysisError(
                f"Failed to compute emotional state for '{text[:40]}...': {exc}"
            ) from exc
