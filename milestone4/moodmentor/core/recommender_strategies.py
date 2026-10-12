"""
Mood Mentor — Hybrid Recommender Strategies
Encapsulates 5 distinct recommendation strategies combined into an ensemble ranker.
"""

from typing import List, Dict, Any
from moodmentor.core.wellness_catalog import WellnessActivity
from moodmentor.core.intensity_analyzer import EmotionalState
from moodmentor.core.user_profile import UserProfile

class RecommenderStrategies:
    @staticmethod
    def rule_based_triage(activity: WellnessActivity, state: EmotionalState) -> float:
        # Clinical safety override
        if state.triage_level == "Critical":
            return 1.0 if activity.id == "act_crisis_helpline" else 0.05
        if activity.id == "act_crisis_helpline":
            return 0.0  # Do not recommend emergency hotline for normal moods
        
        # Duration constraint
        if state.tier in ["High", "Severe"] and activity.duration_mins > 10:
            return 0.4
        return 0.85

    @staticmethod
    def content_based(activity: WellnessActivity, state: EmotionalState) -> float:
        intensity_match = 1.0 if state.tier in activity.target_intensities else 0.25
        emotion_match = 1.0 if state.primary_emotion in [e.lower() for e in activity.target_emotions] else 0.15
        return (intensity_match * 0.5) + (emotion_match * 0.5)

    @staticmethod
    def emotion_vector_similarity(activity: WellnessActivity, emotion_probs: Dict[str, float]) -> float:
        score = sum(emotion_probs.get(e.lower(), 0.0) for e in activity.target_emotions)
        return min(1.0, score * 1.2)

    @staticmethod
    def user_preference(activity: WellnessActivity, profile: UserProfile) -> float:
        affinity = profile.get_modality_affinity(activity.modality)
        time_penalty = 1.0 if activity.duration_mins <= profile.time_limit_mins else 0.6
        return min(1.0, affinity * 0.7 * time_penalty)

    @staticmethod
    def collaborative_peer_pattern(activity: WellnessActivity, state: EmotionalState) -> float:
        # Popular high-efficacy interventions by state
        priors = {
            "fear": {"act_box_breathing": 0.95, "act_54321_grounding": 0.92},
            "anger": {"act_progressive_relaxation": 0.90, "act_mindful_walking": 0.88},
            "sadness": {"act_thought_defusion": 0.88, "act_compassion_meditation": 0.85},
            "joy": {"act_joy_savoring": 0.95}
        }
        return priors.get(state.primary_emotion, {}).get(activity.id, 0.5)
