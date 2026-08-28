"""Sentiment and Emotion Reporting Module for Mood Mentor (Milestone 1).

Generates structured summaries, per-sample classification reports, terminal
visualization tables, and Pandas DataFrame exports.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Optional, Sequence

import pandas as pd

from exceptions import ReportGenerationError
from preprocessing import PreprocessedResult
from sentiment import SentimentScore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SampleReportItem:
    """Detailed report entry for an individual text sample."""

    sample_id: int
    raw_text: str
    processed_text: str
    tokens: list[str]
    sentiment_label: str
    compound_score: float
    pos_score: float
    neu_score: float
    neg_score: float

    def to_dict(self) -> dict[str, Any]:
        """Convert sample report item to dictionary."""
        return {
            "ID": self.sample_id,
            "Input Text": self.raw_text,
            "Processed Text": self.processed_text,
            "Sentiment": self.sentiment_label,
            "Compound": round(self.compound_score, 4),
            "Pos": round(self.pos_score, 4),
            "Neu": round(self.neu_score, 4),
            "Neg": round(self.neg_score, 4),
        }


@dataclass(frozen=True)
class CorpusReportSummary:
    """Aggregated statistical summary and itemized report for a corpus."""

    total_samples: int
    positive_count: int
    negative_count: int
    neutral_count: int
    positive_pct: float
    negative_pct: float
    neutral_pct: float
    avg_compound: float
    min_compound_sample: Optional[SampleReportItem]
    max_compound_sample: Optional[SampleReportItem]
    items: list[SampleReportItem]

    def to_dict(self) -> dict[str, Any]:
        """Convert corpus summary to structured dictionary."""
        return {
            "summary_metrics": {
                "total_samples": self.total_samples,
                "positive_count": self.positive_count,
                "negative_count": self.negative_count,
                "neutral_count": self.neutral_count,
                "positive_pct": round(self.positive_pct, 2),
                "negative_pct": round(self.negative_pct, 2),
                "neutral_pct": round(self.neutral_pct, 2),
                "avg_compound": round(self.avg_compound, 4),
            },
            "samples": [item.to_dict() for item in self.items],
        }


def generate_sentiment_report(
    raw_texts: Sequence[str],
    preprocessed_results: Sequence[PreprocessedResult],
    sentiment_scores: Sequence[SentimentScore],
) -> CorpusReportSummary:
    """Generate an initial emotion/sentiment classification report.

    Args:
        raw_texts: List of original input texts.
        preprocessed_results: Corresponding preprocessed outputs.
        sentiment_scores: Corresponding sentiment scores.

    Returns:
        CorpusReportSummary containing per-sample records and aggregate statistics.

    Raises:
        ReportGenerationError: If sequence lengths mismatch or data is invalid.
    """
    total = len(raw_texts)
    if len(preprocessed_results) != total or len(sentiment_scores) != total:
        raise ReportGenerationError(
            f"Mismatched input lengths: raw_texts ({total}), "
            f"preprocessed ({len(preprocessed_results)}), "
            f"sentiments ({len(sentiment_scores)})."
        )

    if total == 0:
        raise ReportGenerationError("Cannot generate report for an empty corpus.")

    items: list[SampleReportItem] = []
    pos_count = 0
    neg_count = 0
    neu_count = 0
    total_compound = 0.0

    min_sample: Optional[SampleReportItem] = None
    max_sample: Optional[SampleReportItem] = None

    for i in range(total):
        raw = raw_texts[i]
        prep = preprocessed_results[i]
        sent = sentiment_scores[i]

        item = SampleReportItem(
            sample_id=i + 1,
            raw_text=raw,
            processed_text=prep.reconstructed_text(),
            tokens=prep.lemmas,
            sentiment_label=sent.label,
            compound_score=sent.compound,
            pos_score=sent.pos,
            neu_score=sent.neu,
            neg_score=sent.neg,
        )
        items.append(item)

        if sent.label == "Positive":
            pos_count += 1
        elif sent.label == "Negative":
            neg_count += 1
        else:
            neu_count += 1

        total_compound += sent.compound

        if min_sample is None or sent.compound < min_sample.compound_score:
            min_sample = item
        if max_sample is None or sent.compound > max_sample.compound_score:
            max_sample = item

    pos_pct = (pos_count / total) * 100.0
    neg_pct = (neg_count / total) * 100.0
    neu_pct = (neu_count / total) * 100.0
    avg_compound = total_compound / total

    logger.info(
        "Generated report for %d samples: %d Pos (%.1f%%), %d Neg (%.1f%%), %d Neu (%.1f%%)",
        total,
        pos_count,
        pos_pct,
        neg_count,
        neg_pct,
        neu_count,
        neu_pct,
    )

    return CorpusReportSummary(
        total_samples=total,
        positive_count=pos_count,
        negative_count=neg_count,
        neutral_count=neu_count,
        positive_pct=pos_pct,
        negative_pct=neg_pct,
        neutral_pct=neu_pct,
        avg_compound=avg_compound,
        min_compound_sample=min_sample,
        max_compound_sample=max_sample,
        items=items,
    )


def format_terminal_report(summary: CorpusReportSummary) -> str:
    """Format the corpus report into a clean, visually appealing terminal display."""
    lines: list[str] = []
    divider = "=" * 80
    sub_divider = "-" * 80

    lines.append(divider)
    lines.append("                  MOOD MENTOR - MILESTONE 1 REPORT")
    lines.append(divider)
    lines.append("")
    lines.append("📊 CORPUS SUMMARY METRICS:")
    lines.append(f"  • Total Analyzed Samples : {summary.total_samples}")
    lines.append(
        f"  • Positive Samples       : {summary.positive_count} ({summary.positive_pct:.1f}%)"
    )
    lines.append(
        f"  • Negative Samples       : {summary.negative_count} ({summary.negative_pct:.1f}%)"
    )
    lines.append(
        f"  • Neutral Samples        : {summary.neutral_count} ({summary.neutral_pct:.1f}%)"
    )
    lines.append(f"  • Average Compound Score : {summary.avg_compound:+.4f}")
    lines.append("")

    if summary.max_compound_sample:
        lines.append(
            f"  ▲ Highest Sentiment: [{summary.max_compound_sample.compound_score:+.4f}] "
            f"\"{summary.max_compound_sample.raw_text[:60]}...\""
        )
    if summary.min_compound_sample:
        lines.append(
            f"  ▼ Lowest Sentiment : [{summary.min_compound_sample.compound_score:+.4f}] "
            f"\"{summary.min_compound_sample.raw_text[:60]}...\""
        )
    lines.append("")
    lines.append(sub_divider)
    lines.append("📝 ITEMIZED SAMPLE CLASSIFICATIONS:")
    lines.append(sub_divider)

    for item in summary.items:
        lines.append(f"[Sample #{item.sample_id}] Sentiment: {item.sentiment_label}")
        lines.append(f"  Raw Input   : \"{item.raw_text}\"")
        lines.append(f"  Processed   : \"{item.processed_text}\"")
        lines.append(
            f"  Scores      : Compound={item.compound_score:+.4f} | "
            f"Pos={item.pos_score:.3f} | Neu={item.neu_score:.3f} | Neg={item.neg_score:.3f}"
        )
        lines.append("")

    lines.append(divider)
    return "\n".join(lines)


def to_dataframe(summary: CorpusReportSummary) -> pd.DataFrame:
    """Convert itemized report items into a Pandas DataFrame."""
    records = [item.to_dict() for item in summary.items]
    return pd.DataFrame(records)
