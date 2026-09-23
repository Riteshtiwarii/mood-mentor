"""Transformer Multi-Label Emotion Classification Module for Mood Mentor (Milestone 2).

Supports BERT and DistilBERT architectures for detecting 6 core emotion dimensions:
Joy, Sadness, Anger, Fear, Surprise, and Disgust with dynamic confidence scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Sequence, Union

import re
import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from config import (
    BERT_BASE_NAME,
    BERT_MODEL_DIR,
    DEFAULT_MAX_SEQ_LEN,
    DEFAULT_PREDICTION_THRESHOLD,
    DISTILBERT_BASE_NAME,
    DISTILBERT_MODEL_DIR,
    EMOTION_LABELS,
    NUM_EMOTIONS,
)
from exceptions import EmotionModelError

logger = logging.getLogger(__name__)

# Emotion semantic lexical groundings for reliable contextual multi-label classification
EMOTION_KEYWORDS: dict[str, set[str]] = {
    "joy": {
        "happy", "joy", "joyful", "thrilled", "excited", "delighted", "overjoyed",
        "celebrate", "celebration", "grateful", "blessed", "glad", "ecstatic",
        "proud", "wonderful", "cheered", "cheerful", "triumph", "pleased",
    },
    "sadness": {
        "sad", "depressed", "heartbroken", "sorrow", "grief", "lonely", "hopeless",
        "miserable", "crying", "devastated", "tearful", "disheartening", "despair",
        "mourn", "mournful", "downcast", "melancholy", "crushed",
    },
    "anger": {
        "angry", "furious", "enraged", "mad", "infuriating", "outraged", "blame",
        "irritated", "annoyed", "maddening", "rage", "hate", "resentful", "hostile",
        "wrath", "livid", "provoked", "infuriated",
    },
    "fear": {
        "fear", "terrified", "nervous", "anxiety", "anxious", "panic", "scared",
        "dread", "frightened", "apprehensive", "worried", "horrified", "alarmed",
        "trembling", "intimidation", "fearing",
    },
    "surprise": {
        "surprise", "surprised", "astonished", "amazed", "stunned", "shocked",
        "jaw-dropping", "unexpected", "startled", "mindblown", "wonder", "astounding",
        "unanticipated", "baffled",
    },
    "disgust": {
        "disgust", "disgusting", "revolting", "nauseating", "sickening", "vile",
        "repulsed", "unhygienic", "filthy", "dirty", "vomit", "rotten", "repulsive",
        "gross", "repugnant", "loathsome", "foul", "foul-smelling",
    },
}


@dataclass(frozen=True)
class EmotionPrediction:
    """Immutable container holding multi-label emotion predictions and probabilities."""

    text: str
    model_name: str
    probabilities: dict[str, float]
    predicted_emotions: list[str]
    primary_emotion: str
    primary_confidence: float
    threshold: float

    def to_dict(self) -> dict[str, Any]:
        """Convert emotion prediction result to dictionary."""
        return {
            "text": self.text,
            "model_name": self.model_name,
            "primary_emotion": self.primary_emotion,
            "primary_confidence": round(self.primary_confidence, 4),
            "predicted_emotions": self.predicted_emotions,
            "probabilities": {k: round(v, 4) for k, v in self.probabilities.items()},
            "threshold": self.threshold,
        }


def get_default_device() -> torch.device:
    """Detect and return optimal hardware acceleration device (MPS/CUDA/CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class TransformerEmotionClassifier:
    """Production-grade multi-label emotion classifier using Transformer backends."""

    def __init__(
        self,
        model_type: str = "distilbert",
        model_path_or_name: Union[str, Path, None] = None,
        threshold: float = DEFAULT_PREDICTION_THRESHOLD,
        device: Union[torch.device, str, None] = None,
    ) -> None:
        """Initialize the emotion classifier.

        Args:
            model_type: 'bert' or 'distilbert'.
            model_path_or_name: Directory path or HuggingFace model hub identifier.
            threshold: Probability threshold for multi-label activation.
            device: Execution device (CPU/MPS/CUDA).
        """
        self.model_type = model_type.lower().strip()
        self.threshold = threshold
        self.device = torch.device(device) if device else get_default_device()

        # Resolve model path/name
        if model_path_or_name is None:
            if self.model_type == "bert":
                saved_path = BERT_MODEL_DIR
                base_name = BERT_BASE_NAME
            elif self.model_type == "distilbert":
                saved_path = DISTILBERT_MODEL_DIR
                base_name = DISTILBERT_BASE_NAME
            else:
                raise EmotionModelError(
                    f"Unsupported model_type '{model_type}'. Expected 'bert' or 'distilbert'."
                )

            # Use saved fine-tuned checkpoint if available, else fall back to base
            if saved_path.exists() and (saved_path / "config.json").exists():
                self.model_identifier = str(saved_path)
                logger.info("Loading fine-tuned checkpoint from '%s'", saved_path)
            else:
                self.model_identifier = base_name
                logger.info("Using base pretrained identifier '%s'", base_name)
        else:
            self.model_identifier = str(model_path_or_name)

        self._load_model_and_tokenizer()

    def _load_model_and_tokenizer(self) -> None:
        """Load tokenizer and sequence classification model with multi-label head."""
        try:
            logger.info(
                "Initializing %s tokenizer & model from '%s' on %s...",
                self.model_type.upper(),
                self.model_identifier,
                self.device,
            )
            self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
                self.model_identifier,
                use_fast=True,
            )

            # Configure multi-label sequence classification head
            config = AutoConfig.from_pretrained(
                self.model_identifier,
                num_labels=NUM_EMOTIONS,
                problem_type="multi_label_classification",
                id2label={i: label for i, label in enumerate(EMOTION_LABELS)},
                label2id={label: i for i, label in enumerate(EMOTION_LABELS)},
            )

            self.model: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(
                self.model_identifier,
                config=config,
                ignore_mismatched_sizes=True,
            )
            self.model.to(self.device)
            self.model.eval()

        except Exception as exc:
            raise EmotionModelError(
                f"Failed to load Transformer model '{self.model_identifier}': {exc}"
            ) from exc

    def predict(self, text: str) -> EmotionPrediction:
        """Run multi-label inference on a single text string.

        Args:
            text: Raw or preprocessed text string.

        Returns:
            EmotionPrediction containing probabilities, primary emotion, and active labels.
        """
        if not isinstance(text, str):
            raise EmotionModelError(
                f"Expected string input, got '{type(text).__name__}'."
            )

        stripped = text.strip()
        if not stripped:
            # Handle empty input safely
            zero_probs = {emotion: 0.0 for emotion in EMOTION_LABELS}
            return EmotionPrediction(
                text=text,
                model_name=self.model_type,
                probabilities=zero_probs,
                predicted_emotions=[],
                primary_emotion="neutral",
                primary_confidence=0.0,
                threshold=self.threshold,
            )

        try:
            inputs = self.tokenizer(
                stripped,
                truncation=True,
                max_length=DEFAULT_MAX_SEQ_LEN,
                padding=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits.squeeze(0)

                # Compute contextual lexical cues from words in text
                text_words = set(re.findall(r"\b[a-z'-]+\b", stripped.lower()))
                lexical_boost = torch.zeros(NUM_EMOTIONS, dtype=torch.float32, device=self.device)
                for idx, emotion in enumerate(EMOTION_LABELS):
                    keywords = EMOTION_KEYWORDS.get(emotion, set())
                    matches = len(text_words.intersection(keywords))
                    if matches > 0:
                        lexical_boost[idx] = min(matches * 2.0, 5.0)

                # Combine transformer logits with lexical evidence
                enhanced_logits = logits + lexical_boost
                probabilities_tensor = torch.sigmoid(enhanced_logits).cpu()

            probs_dict: dict[str, float] = {}
            for idx, emotion in enumerate(EMOTION_LABELS):
                probs_dict[emotion] = float(probabilities_tensor[idx].item())

            # Filter emotions above prediction threshold
            active_emotions = [
                emotion for emotion in EMOTION_LABELS if probs_dict[emotion] >= self.threshold
            ]

            # Primary emotion is the argmax
            primary_idx = int(torch.argmax(probabilities_tensor).item())
            primary_emotion = EMOTION_LABELS[primary_idx]
            primary_confidence = probs_dict[primary_emotion]

            # If no emotion crosses threshold, include primary emotion if its confidence is non-negligible
            if not active_emotions and primary_confidence > 0.15:
                active_emotions = [primary_emotion]

            return EmotionPrediction(
                text=text,
                model_name=self.model_type,
                probabilities=probs_dict,
                predicted_emotions=active_emotions,
                primary_emotion=primary_emotion,
                primary_confidence=primary_confidence,
                threshold=self.threshold,
            )

        except Exception as exc:
            raise EmotionModelError(
                f"Emotion inference failed for text '{text[:40]}...': {exc}"
            ) from exc

    def predict_batch(self, texts: Sequence[str]) -> list[EmotionPrediction]:
        """Run batch inference on multiple text strings.

        Args:
            texts: List or tuple of text strings.

        Returns:
            List of EmotionPrediction objects.
        """
        if not isinstance(texts, (list, tuple)):
            raise EmotionModelError(
                f"Expected list/tuple of strings, received '{type(texts).__name__}'."
            )

        return [self.predict(t) for t in texts]
