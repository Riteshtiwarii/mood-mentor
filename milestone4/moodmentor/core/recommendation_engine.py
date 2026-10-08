"""
Mood Mentor — Recommendation Engine
Hybrid ensemble recommender ranking activities across 5 algorithmic strategies.
"""

from typing import List, Dict, Any, Optional
from moodmentor.core.wellness_catalog import WellnessCatalog, WellnessActivity
from moodmentor.core.intensity_analyzer import EmotionalState
from moodmentor.core.user_profile import UserProfile
from moodmentor.core.recommender_strategies import RecommenderStrategies
from moodmentor.core.semantic_matcher import SemanticMatcher

class RankedRecommendation:
    def __init__(self, activity: WellnessActivity, score: float,
                 strategy_scores: Dict[str, float], rationale: str):
        self.activity = activity
        self.score = round(score, 4)
        self.strategy_scores = strategy_scores
        self.rationale = rationale

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.activity.id,
            "title": self.activity.title,
            "modality": self.activity.modality,
            "duration_mins": self.activity.duration_mins,
            "score": self.score,
            "description": self.activity.description,
            "clinical_rationale": self.activity.clinical_rationale,
            "rationale": self.rationale,
            "strategy_breakdown": self.strategy_scores
        }

class RecommendationEngine:
    RELEVANCE_PRUNING_THRESHOLD = 0.38

    def __init__(self, catalog: Optional[WellnessCatalog] = None):
        self.catalog = catalog or WellnessCatalog()
        self.semantic_matcher = SemanticMatcher()

    def get_recommendations(self, state: EmotionalState,
                            emotion_probs: Dict[str, float],
                            user_profile: Optional[UserProfile] = None,
                            query_text: str = "",
                            top_k: int = 3) -> List[RankedRecommendation]:
        profile = user_profile or UserProfile(user_id="guest_user")
        activities = self.catalog.get_all_activities()

        # Semantic scores
        semantic_matches = dict(self.semantic_matcher.match(query_text, activities, top_k=len(activities)))

        ranked: List[RankedRecommendation] = []

        # If Critical Crisis: Immediate safety override
        if state.triage_level == "Critical":
            crisis_act = self.catalog.get_by_id("act_crisis_helpline")
            if crisis_act:
                return [RankedRecommendation(
                    activity=crisis_act,
                    score=1.0,
                    strategy_scores={"rule_based": 1.0, "semantic": 1.0},
                    rationale="Immediate Clinical Safety Protocol: Professional crisis intervention surfaced."
                )]

        for act in activities:
            if act.id == "act_crisis_helpline":
                continue

            s_rule = RecommenderStrategies.rule_based_triage(act, state)
            s_content = RecommenderStrategies.content_based(act, state)
            s_vector = RecommenderStrategies.emotion_vector_similarity(act, emotion_probs)
            s_collab = RecommenderStrategies.collaborative_peer_pattern(act, state)
            s_user = RecommenderStrategies.user_preference(act, profile)
            s_semantic = semantic_matches.get(act, 0.0)

            # Weighted combination formula
            composite = (
                s_rule * 0.20 +
                s_content * 0.25 +
                s_vector * 0.20 +
                s_semantic * 0.20 +
                s_user * 0.10 +
                s_collab * 0.05
            )

            # Deduplication penalty for recently recommended items
            if act.id in profile.recently_recommended_ids:
                composite *= 0.75

            if composite >= self.RELEVANCE_PRUNING_THRESHOLD:
                rationale = (
                    f"Selected because {state.primary_emotion.upper()} ({state.tier} intensity) was identified. "
                    f"Matches your affinity for {act.modality} to target {act.target_emotions[0]}."
                )
                ranked.append(RankedRecommendation(
                    activity=act,
                    score=composite,
                    strategy_scores={
                        "rule_based": round(s_rule, 3),
                        "content_based": round(s_content, 3),
                        "emotion_vector": round(s_vector, 3),
                        "semantic": round(s_semantic, 3),
                        "user_affinity": round(s_user, 3)
                    },
                    rationale=rationale
                ))

        ranked.sort(key=lambda x: x.score, reverse=True)

        # Fallback if pruning was too aggressive
        if not ranked and activities:
            fallback_act = activities[0]
            ranked.append(RankedRecommendation(
                activity=fallback_act,
                score=0.5,
                strategy_scores={"fallback": 0.5},
                rationale="Fallback general wellness protocol."
            ))

        return ranked[:top_k]
