"""Hybrid Recommendation Strategies Module (Milestone 3 - Task 3).

Implements 5 specialized recommendation strategies:
1. Rule-Based Clinical Triage (Crisis / acute panic safety rails)
2. Content-Based Filtering (Intensity band & category alignment)
3. Emotion Vector Similarity (6-D Cosine Distance)
4. Collaborative Filtering (Historical peer interaction patterns)
5. User Preference Matching (Modality & time constraint alignment)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import logging
from typing import Dict, List, Set

import numpy as np

from config import EMOTION_LABELS
from intensity_analyzer import EmotionalState
from user_profile import UserProfile
from wellness_catalog import WellnessIntervention

logger = logging.getLogger(__name__)


class RecommendationStrategy(ABC):
    """Abstract base class for recommendation scoring strategies."""

    @abstractmethod
    def score_items(
        self,
        items: List[WellnessIntervention],
        emotional_state: EmotionalState,
        user_profile: UserProfile,
    ) -> Dict[str, float]:
        """Compute strategy-specific score (0.0 to 1.0) for each wellness item."""
        pass


class RuleBasedTriageStrategy(RecommendationStrategy):
    """Clinical safety rails: triggers priority interventions for severe distress/panic."""

    def score_items(
        self,
        items: List[WellnessIntervention],
        emotional_state: EmotionalState,
        user_profile: UserProfile,
    ) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        is_crisis = emotional_state.crisis_flag or emotional_state.severity == "Critical"
        is_high_intensity = emotional_state.intensity_score >= 0.80

        for item in items:
            if is_crisis:
                # Prioritize crisis EAP and emergency grounding
                if item.is_crisis_resource:
                    scores[item.id] = 1.00
                elif item.id in ("act_box_breathing", "act_grounding_54321", "act_physiological_sigh"):
                    scores[item.id] = 0.95
                else:
                    scores[item.id] = 0.10
            elif is_high_intensity and item.id in ("act_box_breathing", "act_grounding_54321"):
                scores[item.id] = 0.85
            else:
                # Neutral baseline
                scores[item.id] = 0.50 if not item.is_crisis_resource else 0.05

        return scores


class ContentBasedStrategy(RecommendationStrategy):
    """Matches activity metadata, target emotions, and intensity bands."""

    def score_items(
        self,
        items: List[WellnessIntervention],
        emotional_state: EmotionalState,
        user_profile: UserProfile,
    ) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        dom_emo = emotional_state.dominant_emotion.lower()
        active_emos = set(e.lower() for e in emotional_state.multiple_emotions)
        intensity = emotional_state.intensity_score

        for item in items:
            score = 0.0

            # 1. Target Emotion Match (up to 0.60)
            if dom_emo in item.target_emotions:
                score += 0.40
            matching_active = active_emos.intersection(item.target_emotions)
            score += min(len(matching_active) * 0.10, 0.20)

            # 2. Intensity Band Match (up to 0.40)
            if item.min_intensity <= intensity <= item.max_intensity:
                score += 0.40
            else:
                # Proximity penalty
                dist = min(abs(intensity - item.min_intensity), abs(intensity - item.max_intensity))
                score += max(0.0, 0.30 - (dist * 0.5))

            scores[item.id] = min(1.0, score)

        return scores


class EmotionSimilarityStrategy(RecommendationStrategy):
    """Vector similarity between user's 6-D emotion probabilities and intervention target profile."""

    def score_items(
        self,
        items: List[WellnessIntervention],
        emotional_state: EmotionalState,
        user_profile: UserProfile,
    ) -> Dict[str, float]:
        scores: Dict[str, float] = {}

        # Construct user 6-D emotion vector
        user_vec = np.array([emotional_state.probabilities.get(e, 0.0) for e in EMOTION_LABELS])
        user_norm = np.linalg.norm(user_vec)
        if user_norm == 0:
            return {item.id: 0.5 for item in items}

        for item in items:
            # Construct item target 6-D emotion profile
            item_vec = np.zeros(len(EMOTION_LABELS))
            for idx, emo in enumerate(EMOTION_LABELS):
                if emo in item.target_emotions:
                    item_vec[idx] = 1.0

            item_norm = np.linalg.norm(item_vec)
            if item_norm == 0:
                scores[item.id] = 0.3
            else:
                cosine_sim = float(np.dot(user_vec, item_vec) / (user_norm * item_norm))
                scores[item.id] = max(0.0, min(1.0, cosine_sim))

        return scores


class CollaborativeFilteringStrategy(RecommendationStrategy):
    """Utility scoring based on peer interaction ratings and completion histories."""

    def score_items(
        self,
        items: List[WellnessIntervention],
        emotional_state: EmotionalState,
        user_profile: UserProfile,
    ) -> Dict[str, float]:
        scores: Dict[str, float] = {}

        # Aggregate ratings from user's own past interactions or peer signals
        user_ratings: Dict[str, List[float]] = {}
        for interaction in user_profile.interaction_history:
            user_ratings.setdefault(interaction.item_id, []).append(interaction.rating)

        for item in items:
            if item.id in user_ratings:
                avg_rating = sum(user_ratings[item.id]) / len(user_ratings[item.id])
                # Normalize 1-5 rating into 0.0 - 1.0
                scores[item.id] = max(0.0, min(1.0, avg_rating / 5.0))
            else:
                # Neutral cold-start baseline
                scores[item.id] = 0.50

        return scores


class UserPreferenceStrategy(RecommendationStrategy):
    """Alignment with user's explicit preferred modalities and duration limits."""

    def score_items(
        self,
        items: List[WellnessIntervention],
        emotional_state: EmotionalState,
        user_profile: UserProfile,
    ) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        preferred_modalities = user_profile.preferred_modalities
        max_duration = user_profile.max_preferred_duration

        for item in items:
            score = 0.0

            # Modality match (0.60)
            if item.modality in preferred_modalities:
                score += 0.60
            else:
                score += 0.20

            # Duration suitability (0.40)
            if item.duration_minutes <= max_duration:
                score += 0.40
            else:
                # Moderate decay for longer activities
                overtime = item.duration_minutes - max_duration
                score += max(0.0, 0.35 - (overtime * 0.05))

            scores[item.id] = min(1.0, score)

        return scores
