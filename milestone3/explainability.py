"""Recommendation Explainability Module (Milestone 3 - Task 8).

Generates human-interpretable, dynamically constructed rationales explaining
why specific wellness interventions were selected and ranked for the employee.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, List

from exceptions import ExplainabilityError
from intensity_analyzer import EmotionalState
from user_profile import EmotionalTrendReport, UserProfile
from wellness_catalog import WellnessIntervention

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RecommendationExplanation:
    """Immutable data model holding human-readable justifications for a recommendation."""

    item_id: str
    title: str
    primary_driver: str
    bullet_reasons: List[str]
    summary_sentence: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "title": self.title,
            "primary_driver": self.primary_driver,
            "reasons": self.bullet_reasons,
            "summary": self.summary_sentence,
        }


class RecommendationExplainer:
    """Explainable AI (XAI) engine generating dynamic justifications."""

    def explain(
        self,
        item: WellnessIntervention,
        score: float,
        score_breakdown: dict[str, float],
        emotional_state: EmotionalState,
        user_profile: UserProfile,
        trend_report: EmotionalTrendReport | None = None,
    ) -> RecommendationExplanation:
        """Construct multi-factor explanation dynamically based on features and state."""
        reasons: List[str] = []
        dom_emo = emotional_state.dominant_emotion.capitalize()
        intensity = emotional_state.intensity_score
        tier = emotional_state.intensity_tier

        # 1. Primary Driver Classification
        if item.is_crisis_resource or emotional_state.severity == "Critical":
            primary_driver = "Acute Crisis & Emergency De-escalation Protocol"
            reasons.append("Critical severity indicators detected requiring priority clinical/safety intervention.")
        elif dom_emo.lower() in item.target_emotions:
            primary_driver = f"Targeted Relief for {dom_emo}"
            reasons.append(f"Specifically designed to counteract detected {dom_emo} (Confidence: {emotional_state.emotion_confidence*100:.1f}%).")
        elif score_breakdown.get("semantic", 0.0) >= 0.40:
            primary_driver = "Semantic Context & Symptom Match"
            reasons.append("Directly matches the specific physical and psychological symptoms described in your text.")
        else:
            primary_driver = "General Stress Neutralization & Restoration"
            reasons.append("Promotes parasympathetic nervous system restoration.")

        # 2. Emotional Intensity Reason
        if intensity >= 0.70:
            reasons.append(f"Calibrated for {tier} emotional intensity ({intensity:.2f}) to provide immediate grounding.")
        elif intensity <= 0.35:
            reasons.append(f"Gentle, low-effort exercise suited for {tier} emotional intensity ({intensity:.2f}).")
        else:
            reasons.append(f"Effective for {tier} intensity ({intensity:.2f}) workplace pressures.")

        # 3. User Preferences Alignment
        if item.modality in user_profile.preferred_modalities:
            reasons.append(f"Aligns with your preferred '{item.modality.capitalize()}' wellness format.")
        if item.duration_minutes <= user_profile.max_preferred_duration:
            reasons.append(f"Fits within your {user_profile.max_preferred_duration}-minute schedule window (Duration: {item.duration_minutes}m).")

        # 4. Semantic Alignment Reason
        sem_score = score_breakdown.get("semantic", 0.0)
        if sem_score >= 0.35:
            reasons.append(f"Strong semantic relevance ({sem_score*100:.1f}%) to contextual themes in your feedback.")

        # 5. Historical Interaction Reason
        past_ratings = [
            i.rating for i in user_profile.interaction_history
            if i.item_id == item.id
        ]
        if past_ratings:
            avg_r = sum(past_ratings) / len(past_ratings)
            reasons.append(f"You previously completed this activity with a {avg_r:.1f}/5.0 satisfaction rating.")
        elif any(i.rating >= 4.0 for i in user_profile.interaction_history):
            reasons.append("Proven effective among peer employees with similar emotional profiles.")

        # 6. Longitudinal Trend Context
        if trend_report and "Escalating" in trend_report.trend_label:
            reasons.append("Prioritized due to an escalating stress trajectory observed over recent check-ins.")
        elif trend_report and "Chronic" in trend_report.trend_label:
            reasons.append("Recommended to interrupt sustained negative emotional fatigue.")

        # 7. Synthesize Summary Sentence
        summary = (
            f"Selected because {dom_emo} at {tier} intensity was detected, "
            f"matching your preferred {item.modality} format and addressing your immediate emotional needs."
        )

        return RecommendationExplanation(
            item_id=item.id,
            title=item.title,
            primary_driver=primary_driver,
            bullet_reasons=reasons,
            summary_sentence=summary,
        )
