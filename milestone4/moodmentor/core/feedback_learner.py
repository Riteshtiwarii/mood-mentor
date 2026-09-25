"""
Mood Mentor — Feedback Learner
Implements online learning telemetry: updates modality affinities based on real-time ratings.
"""

from typing import Dict, Any
from moodmentor.core.user_profile import UserProfile

class FeedbackLearner:
    def __init__(self):
        pass

    def record_feedback(self, profile: UserProfile, activity_id: str,
                        modality: str, action: str, rating: float = 3.0) -> Dict[str, Any]:
        """
        action: 'viewed', 'accepted', 'rejected'
        rating: 1.0 to 5.0 stars
        """
        # Adapt affinity based on action and rating
        if action == "accepted":
            effective_rating = max(3.5, rating)
        elif action == "rejected":
            effective_rating = min(2.0, rating)
        else:
            effective_rating = rating

        profile.record_feedback(modality, effective_rating)
        updated_affinity = profile.get_modality_affinity(modality)

        return {
            "user_id": profile.user_id,
            "activity_id": activity_id,
            "modality": modality,
            "action": action,
            "rating": rating,
            "new_affinity": round(updated_affinity, 3)
        }
