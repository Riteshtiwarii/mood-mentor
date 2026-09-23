"""Unified Reporting Module for Mood Mentor (Milestone 3).

Synthesizes VADER Sentiment (M1), Transformer Multi-Label Emotions (M2),
Dynamic Intensity & Severity Triage (Task 1), and Personalized Ranked Recommendations (Tasks 2, 3, 4, 5).
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, List, Sequence

import pandas as pd

from exceptions import ReportGenerationError
from intensity_analyzer import EmotionalState
from recommendation_engine import RecommendationResult
from user_profile import UserProfile

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Milestone3AnalysisItem:
    """Consolidated record combining sentiment, emotion, intensity, and recommendations."""

    sample_id: int
    user_id: str
    raw_text: str
    sentiment_label: str
    compound_score: float
    dominant_emotion: str
    emotion_confidence: float
    active_emotions: list[str]
    intensity_score: float
    intensity_tier: str
    severity: str
    is_mixed_state: bool
    mixed_state_type: str | None
    top_recommendation_title: str
    top_recommendation_score: float
    top_recommendation_modality: str
    top_recommendation_duration: int
    all_recommendation_titles: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ID": self.sample_id,
            "User_ID": self.user_id,
            "Text": self.raw_text,
            "Sentiment": self.sentiment_label,
            "Compound": round(self.compound_score, 4),
            "Dominant_Emotion": self.dominant_emotion.capitalize(),
            "Confidence": round(self.emotion_confidence, 4),
            "Active_Emotions": ", ".join(e.capitalize() for e in self.active_emotions),
            "Intensity_Score": round(self.intensity_score, 4),
            "Intensity_Tier": self.intensity_tier,
            "Severity": self.severity,
            "Mixed_State": self.mixed_state_type or "No",
            "Top_Recommendation": self.top_recommendation_title,
            "Rec_Score": round(self.top_recommendation_score, 4),
            "Modality": self.top_recommendation_modality,
            "Duration_Min": self.top_recommendation_duration,
            "Ranked_Order": " -> ".join(self.all_recommendation_titles),
        }


@dataclass(frozen=True)
class Milestone3CorpusSummary:
    """Aggregated corpus metrics for Milestone 3 analysis."""

    total_samples: int
    avg_intensity: float
    severity_distribution: dict[str, int]
    dominant_emotion_distribution: dict[str, int]
    top_recommended_activities: dict[str, int]
    items: list[Milestone3AnalysisItem]


def generate_milestone3_report(
    raw_texts: Sequence[str],
    user_ids: Sequence[str | None],
    emotional_states: Sequence[EmotionalState],
    recommendation_results: Sequence[RecommendationResult],
) -> Milestone3CorpusSummary:
    """Assemble items into aggregated report container."""
    total = len(raw_texts)
    if total == 0:
        raise ReportGenerationError("Cannot generate report for an empty corpus.")

    items: list[Milestone3AnalysisItem] = []
    severity_dist: dict[str, int] = {"Mild": 0, "Moderate": 0, "High": 0, "Critical": 0}
    emotion_dist: dict[str, int] = {}
    activity_dist: dict[str, int] = {}
    total_intensity = 0.0

    for idx in range(total):
        text = raw_texts[idx]
        uid = user_ids[idx] or "anonymous_user"
        state = emotional_states[idx]
        rec = recommendation_results[idx]

        top_rec = rec.top_recommendation
        top_title = top_rec.title if top_rec else "None"
        top_score = top_rec.composite_score if top_rec else 0.0
        top_mod = top_rec.modality if top_rec else "N/A"
        top_dur = top_rec.duration_minutes if top_rec else 0
        all_titles = [r.title for r in rec.recommendations]

        item = Milestone3AnalysisItem(
            sample_id=idx + 1,
            user_id=uid,
            raw_text=text,
            sentiment_label=state.polarity,
            compound_score=state.compound_score,
            dominant_emotion=state.dominant_emotion,
            emotion_confidence=state.emotion_confidence,
            active_emotions=state.multiple_emotions,
            intensity_score=state.intensity_score,
            intensity_tier=state.intensity_tier,
            severity=state.severity,
            is_mixed_state=state.is_mixed_state,
            mixed_state_type=state.mixed_state_type,
            top_recommendation_title=top_title,
            top_recommendation_score=top_score,
            top_recommendation_modality=top_mod,
            top_recommendation_duration=top_dur,
            all_recommendation_titles=all_titles,
        )
        items.append(item)

        total_intensity += state.intensity_score
        severity_dist[state.severity] = severity_dist.get(state.severity, 0) + 1
        emotion_dist[state.dominant_emotion] = emotion_dist.get(state.dominant_emotion, 0) + 1
        if top_title != "None":
            activity_dist[top_title] = activity_dist.get(top_title, 0) + 1

    return Milestone3CorpusSummary(
        total_samples=total,
        avg_intensity=total_intensity / total,
        severity_distribution=severity_dist,
        dominant_emotion_distribution=emotion_dist,
        top_recommended_activities=activity_dist,
        items=items,
    )


def format_terminal_report(
    summary: Milestone3CorpusSummary,
    recommendation_results: Sequence[RecommendationResult] | None = None,
) -> str:
    """Format human-readable terminal dashboard for Milestone 3."""
    recommendation_results = recommendation_results or []
    lines: list[str] = []
    divider = "=" * 85
    sub_divider = "-" * 85

    lines.append(divider)
    lines.append("     MOOD MENTOR - MILESTONE 3: AI EMOTIONAL STATE & WELLNESS RECOMMENDATIONS")
    lines.append(divider)
    lines.append("")
    lines.append("📊 EXECUTIVE SUMMARY:")
    lines.append(f"  • Total Analyzed Samples  : {summary.total_samples}")
    lines.append(f"  • Average Intensity Score : {summary.avg_intensity:.4f}")
    lines.append("")
    lines.append("  Severity Triage Breakdown:")
    for sev, count in summary.severity_distribution.items():
        pct = (count / summary.total_samples) * 100.0
        lines.append(f"    - {sev:<10} : {count} ({pct:.1f}%)")
    lines.append("")
    lines.append(sub_divider)
    lines.append("🎯 ITEMIZED EMOTIONAL STATE & PERSONALIZED RECOMMENDATIONS:")
    lines.append(sub_divider)

    for idx, item in enumerate(summary.items):
        rec_res = recommendation_results[idx] if idx < len(recommendation_results) else None
        lines.append(
            f"[Sample #{item.sample_id} | User: {item.user_id}] "
            f"Dominant Emotion: {item.dominant_emotion.capitalize()} ({item.emotion_confidence*100:.1f}%) | "
            f"Intensity: {item.intensity_score:.2f} ({item.intensity_tier}) | Severity: {item.severity}"
        )
        lines.append(f"  Raw Input       : \"{item.raw_text}\"")
        if item.is_mixed_state:
            lines.append(f"  Psychological   : Mixed State Detected -> {item.mixed_state_type}")
        lines.append(f"  Active Emotions : {', '.join(e.capitalize() for e in item.active_emotions)}")

        # Task 6: Longitudinal Trend Display
        if rec_res and rec_res.trend_report and rec_res.trend_report.total_snapshots > 1:
            lines.append(f"  📈 Trend Tracker: {rec_res.trend_report.trend_label} (Slope: {rec_res.trend_report.intensity_trajectory_slope:+.2f})")
            if rec_res.trend_report.repeated_patterns:
                lines.append(f"     Patterns     : {', '.join(rec_res.trend_report.repeated_patterns)}")

        lines.append(
            f"  ⭐ Top Rec       : {item.top_recommendation_title} "
            f"[Score: {item.top_recommendation_score:.3f} | {item.top_recommendation_modality} | {item.top_recommendation_duration} mins]"
        )

        # Task 8: Explainability Layer Display
        if rec_res and rec_res.top_recommendation and rec_res.top_recommendation.explanation:
            exp = rec_res.top_recommendation.explanation
            lines.append(f"  💡 Why Selected : {exp.primary_driver}")
            for r in exp.bullet_reasons[:3]:
                lines.append(f"     • {r}")

        if len(item.all_recommendation_titles) > 1:
            lines.append(f"  Ranked Order    : {' -> '.join(item.all_recommendation_titles)}")
        lines.append("")

    lines.append(divider)
    return "\n".join(lines)


def to_dataframe(summary: Milestone3CorpusSummary) -> pd.DataFrame:
    """Convert summary items into a Pandas DataFrame."""
    records = [item.to_dict() for item in summary.items]
    return pd.DataFrame(records)
