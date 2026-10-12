"""
Mood Mentor — User Profile & History
Maintains user engagement preferences, modality affinity, and historical engagement telemetry.
"""

from typing import List, Dict, Any, Optional

class UserProfile:
    def __init__(self, user_id: str, preferred_modalities: Optional[List[str]] = None,
                 time_limit_mins: int = 15):
        self.user_id = user_id
        self.preferred_modalities = preferred_modalities or ["Breathing", "Mindfulness", "Somatic", "Movement", "Journaling"]
        self.time_limit_mins = time_limit_mins
        self.interaction_history: List[Dict[str, Any]] = []
        self.modality_ratings: Dict[str, List[float]] = {}
        self.recently_recommended_ids: List[str] = []

    def log_interaction(self, primary_emotion: str, intensity_score: float, recommended_ids: List[str]):
        self.interaction_history.append({
            "primary_emotion": primary_emotion,
            "intensity_score": intensity_score,
            "recommended_ids": recommended_ids
        })
        self.recently_recommended_ids.extend(recommended_ids)
        self.recently_recommended_ids = self.recently_recommended_ids[-10:]

    def record_feedback(self, modality: str, rating: float):
        if modality not in self.modality_ratings:
            self.modality_ratings[modality] = []
        self.modality_ratings[modality].append(rating)

    def get_modality_affinity(self, modality: str) -> float:
        ratings = self.modality_ratings.get(modality, [])
        if not ratings:
            return 1.0 if modality in self.preferred_modalities else 0.75
        avg_rating = sum(ratings) / len(ratings)
        return max(0.2, min(1.5, avg_rating / 3.0))
