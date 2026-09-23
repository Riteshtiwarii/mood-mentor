"""Unified Sentiment & Deep Emotion Reporting Module for Mood Mentor (Milestone 2).

Combines Milestone 1 (VADER Baseline) and Milestone 2 (Transformer Emotions)
into unified dashboards, itemized summaries, and CSV exports.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Optional, Sequence

import pandas as pd

from emotion_classifier import EmotionPrediction
from exceptions import ReportGenerationError
from preprocessing import PreprocessedResult
from sentiment import SentimentScore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UnifiedAnalysisItem:
    """Detailed record combining preprocessing, VADER sentiment, and Transformer emotions."""

    sample_id: int
    raw_text: str
    processed_text: str
    tokens: list[str]
    sentiment_label: str
    compound_score: float
    primary_emotion: str
    primary_confidence: float
    active_emotions: list[str]
    emotion_probabilities: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        """Convert analysis item to dictionary for display and DataFrame export."""
        record: dict[str, Any] = {
            "ID": self.sample_id,
            "Input Text": self.raw_text,
            "Processed Text": self.processed_text,
            "Sentiment": self.sentiment_label,
            "Compound": round(self.compound_score, 4),
            "Primary Emotion": self.primary_emotion.capitalize(),
            "Confidence": round(self.primary_confidence, 4),
            "Active Emotions": ", ".join(e.capitalize() for e in self.active_emotions),
        }
        for emotion, prob in self.emotion_probabilities.items():
            record[f"Prob_{emotion.capitalize()}"] = round(prob, 4)
        return record


@dataclass(frozen=True)
class UnifiedCorpusSummary:
    """Aggregated statistics across sentiment and deep emotions for a corpus."""

    total_samples: int
    sentiment_distribution: dict[str, int]
    sentiment_percentages: dict[str, float]
    avg_compound: float
    emotion_distribution: dict[str, int]
    emotion_percentages: dict[str, float]
    items: list[UnifiedAnalysisItem]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_samples": self.total_samples,
            "avg_compound": round(self.avg_compound, 4),
            "sentiment_distribution": self.sentiment_distribution,
            "sentiment_percentages": self.sentiment_percentages,
            "emotion_distribution": self.emotion_distribution,
            "emotion_percentages": self.emotion_percentages,
            "samples": [item.to_dict() for item in self.items],
        }


def generate_unified_report(
    raw_texts: Sequence[str],
    preprocessed: Sequence[PreprocessedResult],
    sentiments: Sequence[SentimentScore],
    emotions: Sequence[EmotionPrediction],
) -> UnifiedCorpusSummary:
    """Synthesize pipeline outputs from Milestone 1 and 2 into an aggregated summary."""
    total = len(raw_texts)
    if not (len(preprocessed) == total == len(sentiments) == len(emotions)):
        raise ReportGenerationError(
            f"Mismatched lengths: raw_texts ({total}), preprocessed ({len(preprocessed)}), "
            f"sentiments ({len(sentiments)}), emotions ({len(emotions)})."
        )

    if total == 0:
        raise ReportGenerationError("Cannot generate report for an empty corpus.")

    items: list[UnifiedAnalysisItem] = []
    sent_dist: dict[str, int] = {"Positive": 0, "Negative": 0, "Neutral": 0}
    emotion_dist: dict[str, int] = {}
    total_compound = 0.0

    for i in range(total):
        raw = raw_texts[i]
        prep = preprocessed[i]
        sent = sentiments[i]
        emo = emotions[i]

        item = UnifiedAnalysisItem(
            sample_id=i + 1,
            raw_text=raw,
            processed_text=prep.reconstructed_text(),
            tokens=prep.lemmas,
            sentiment_label=sent.label,
            compound_score=sent.compound,
            primary_emotion=emo.primary_emotion,
            primary_confidence=emo.primary_confidence,
            active_emotions=emo.predicted_emotions,
            emotion_probabilities=emo.probabilities,
        )
        items.append(item)

        sent_dist[sent.label] = sent_dist.get(sent.label, 0) + 1
        total_compound += sent.compound

        p_emo = emo.primary_emotion.lower()
        emotion_dist[p_emo] = emotion_dist.get(p_emo, 0) + 1

    sent_pct = {k: round((v / total) * 100.0, 2) for k, v in sent_dist.items()}
    emotion_pct = {k: round((v / total) * 100.0, 2) for k, v in emotion_dist.items()}
    avg_compound = total_compound / total

    logger.info("Generated unified report for %d samples.", total)
    return UnifiedCorpusSummary(
        total_samples=total,
        sentiment_distribution=sent_dist,
        sentiment_percentages=sent_pct,
        avg_compound=avg_compound,
        emotion_distribution=emotion_dist,
        emotion_percentages=emotion_pct,
        items=items,
    )


def format_terminal_report(summary: UnifiedCorpusSummary) -> str:
    """Render unified sentiment and emotion findings into a formatted terminal dashboard."""
    lines: list[str] = []
    divider = "=" * 80
    sub_divider = "-" * 80

    lines.append(divider)
    lines.append("        MOOD MENTOR - MILESTONE 2 DEEP EMOTION & SENTIMENT REPORT")
    lines.append(divider)
    lines.append("")
    lines.append("📊 CORPUS SUMMARY METRICS:")
    lines.append(f"  • Total Analyzed Samples : {summary.total_samples}")
    lines.append(f"  • Average Compound Score : {summary.avg_compound:+.4f}")
    lines.append("")
    lines.append("  Sentiment Polarity Breakdown:")
    for sent, count in summary.sentiment_distribution.items():
        lines.append(f"    - {sent:<10} : {count} ({summary.sentiment_percentages[sent]:.1f}%)")
    lines.append("")
    lines.append("  Primary Emotion Breakdown (Transformer):")
    for emo, count in sorted(summary.emotion_distribution.items(), key=lambda x: x[1], reverse=True):
        lines.append(f"    - {emo.capitalize():<10} : {count} ({summary.emotion_percentages[emo]:.1f}%)")
    lines.append("")
    lines.append(sub_divider)
    lines.append("📝 ITEMIZED SAMPLE ANALYSES:")
    lines.append(sub_divider)

    for item in summary.items:
        lines.append(
            f"[Sample #{item.sample_id}] Sentiment: {item.sentiment_label} (Compound: {item.compound_score:+.4f}) | "
            f"Primary Emotion: {item.primary_emotion.capitalize()} ({item.primary_confidence * 100:.1f}%)"
        )
        lines.append(f"  Raw Input       : \"{item.raw_text}\"")
        lines.append(f"  Active Emotions : {', '.join(e.capitalize() for e in item.active_emotions) or 'None'}")
        prob_str = " | ".join(
            f"{e.capitalize()[:3]}={p:.2f}" for e, p in item.emotion_probabilities.items()
        )
        lines.append(f"  Probabilities   : {prob_str}")
        lines.append("")

    lines.append(divider)
    return "\n".join(lines)


def to_dataframe(summary: UnifiedCorpusSummary) -> pd.DataFrame:
    """Convert summary items into a Pandas DataFrame."""
    records = [item.to_dict() for item in summary.items]
    return pd.DataFrame(records)
