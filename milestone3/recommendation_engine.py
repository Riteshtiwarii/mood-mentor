"""Personalized Hybrid Recommendation & Dynamic Ranking Engine (Milestone 3 - Tasks 2, 3, 4, 6, 7, 8).

Fuses Semantic Content Matching, Emotion Profile Similarity, User Preferences,
Collaborative Filtering, Longitudinal Trends, Active Feedback Learning, and XAI Explanations.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Dict, List, Optional

from config import (
    DEFAULT_TOP_K,
    MIN_RELEVANCE_THRESHOLD,
    RECENT_RECOMMENDATION_PENALTY,
    WEIGHT_COLLABORATIVE,
    WEIGHT_EMOTION,
    WEIGHT_INTENSITY,
    WEIGHT_PREFERENCE,
    WEIGHT_SEMANTIC,
)
from exceptions import RecommendationError
from explainability import RecommendationExplainer, RecommendationExplanation
from feedback_learner import FeedbackLearner
from intensity_analyzer import EmotionalState
from recommender_strategies import (
    CollaborativeFilteringStrategy,
    ContentBasedStrategy,
    EmotionSimilarityStrategy,
    RuleBasedTriageStrategy,
    UserPreferenceStrategy,
)
from semantic_matcher import SemanticWellnessMatcher
from user_profile import EmotionalTrendReport, UserProfile, get_user_profile
from wellness_catalog import WellnessCatalogManager, WellnessIntervention, get_wellness_catalog

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RankedRecommendation:
    """Detailed model for an individually ranked wellness recommendation with XAI."""

    rank: int
    item_id: str
    title: str
    category: str
    modality: str
    duration_minutes: int
    composite_score: float
    score_breakdown: dict[str, float]
    instructions: str
    description: str
    is_crisis_resource: bool
    explanation: Optional[RecommendationExplanation] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "item_id": self.item_id,
            "title": self.title,
            "category": self.category.capitalize(),
            "modality": self.modality.capitalize(),
            "duration_minutes": self.duration_minutes,
            "composite_score": round(self.composite_score, 4),
            "score_breakdown": {k: round(v, 4) for k, v in self.score_breakdown.items()},
            "instructions": self.instructions,
            "description": self.description,
            "is_crisis_resource": self.is_crisis_resource,
            "explanation": self.explanation.to_dict() if self.explanation else None,
        }


@dataclass(frozen=True)
class RecommendationResult:
    """Comprehensive container holding ranked recommendations and audit metadata."""

    user_id: str
    top_recommendation: Optional[RankedRecommendation]
    recommendations: List[RankedRecommendation]
    total_candidates_evaluated: int
    filtered_low_relevance_count: int
    deduplicated_count: int
    ranking_order: List[str]
    trend_report: Optional[EmotionalTrendReport] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "top_recommendation": self.top_recommendation.to_dict() if self.top_recommendation else None,
            "recommendations": [r.to_dict() for r in self.recommendations],
            "total_candidates_evaluated": self.total_candidates_evaluated,
            "filtered_low_relevance_count": self.filtered_low_relevance_count,
            "deduplicated_count": self.deduplicated_count,
            "ranking_order": self.ranking_order,
            "trend_report": self.trend_report.to_dict() if self.trend_report else None,
        }


class HybridRecommendationEngine:
    """Production-grade hybrid recommendation and dynamic ranking system."""

    def __init__(
        self,
        catalog_manager: WellnessCatalogManager | None = None,
        semantic_matcher: SemanticWellnessMatcher | None = None,
        feedback_learner: FeedbackLearner | None = None,
        explainer: RecommendationExplainer | None = None,
        relevance_threshold: float = MIN_RELEVANCE_THRESHOLD,
    ) -> None:
        self.catalog = catalog_manager or get_wellness_catalog()
        self.semantic_matcher = semantic_matcher or SemanticWellnessMatcher(self.catalog)
        self.feedback_learner = feedback_learner or FeedbackLearner(catalog_manager=self.catalog)
        self.explainer = explainer or RecommendationExplainer()
        self.relevance_threshold = relevance_threshold

        # Initialize Strategy Components (Task 3)
        self.rule_strategy = RuleBasedTriageStrategy()
        self.content_strategy = ContentBasedStrategy()
        self.emotion_sim_strategy = EmotionSimilarityStrategy()
        self.collab_strategy = CollaborativeFilteringStrategy()
        self.pref_strategy = UserPreferenceStrategy()

    def generate_recommendations(
        self,
        emotional_state: EmotionalState,
        user_profile: UserProfile | None = None,
        top_k: int = DEFAULT_TOP_K,
        exclude_recent: bool = True,
        explain: bool = True,
    ) -> RecommendationResult:
        """Execute hybrid scoring, trend-influence, feedback adaptation, ranking, and explainability."""
        profile = user_profile or get_user_profile()
        items = self.catalog.get_all()

        if not items:
            raise RecommendationError("Wellness catalog contains no items to recommend.")

        # Step 1: Compute scores across all 5 strategies
        semantic_scores = self.semantic_matcher.compute_similarity(emotional_state.text)
        rule_scores = self.rule_strategy.score_items(items, emotional_state, profile)
        content_scores = self.content_strategy.score_items(items, emotional_state, profile)
        emotion_sim_scores = self.emotion_sim_strategy.score_items(items, emotional_state, profile)
        collab_scores = self.collab_strategy.score_items(items, emotional_state, profile)
        pref_scores = self.pref_strategy.score_items(items, emotional_state, profile)

        # Retrieve Longitudinal Trend Analysis (Task 6)
        trend_report = profile.get_trend_report()
        trend_bias = trend_report.recommendation_bias

        candidates: List[dict[str, Any]] = []
        dedup_count = 0
        filtered_count = 0

        is_crisis = emotional_state.crisis_flag or emotional_state.severity == "Critical"

        for item in items:
            s_sem = semantic_scores.get(item.id, 0.0)
            s_rule = rule_scores.get(item.id, 0.5)
            s_cont = content_scores.get(item.id, 0.0)
            s_emo = emotion_sim_scores.get(item.id, 0.0)
            s_col = collab_scores.get(item.id, 0.5)
            s_pref = pref_scores.get(item.id, 0.5)

            # Online Learning: Apply feedback weight modifier (Task 7)
            modality_multiplier = profile.modality_weights.get(item.modality, 1.0)
            adjusted_pref = min(1.0, s_pref * modality_multiplier)

            # Calculate base weighted composite score
            base_score = (
                (WEIGHT_SEMANTIC * s_sem)
                + (WEIGHT_EMOTION * s_emo)
                + (WEIGHT_INTENSITY * s_cont)
                + (WEIGHT_PREFERENCE * adjusted_pref)
                + (WEIGHT_COLLABORATIVE * s_col)
            )

            # Longitudinal Trend Influence (Task 6): Apply trend bias
            t_boost = trend_bias.get(item.modality, 0.0) + trend_bias.get(item.category, 0.0)
            base_score += t_boost

            # Online Feedback modifier for this specific item (Task 7)
            feedback_mod = self.feedback_learner.compute_feedback_modifier(profile.user_id, item.id)
            base_score += feedback_mod

            # Rule-based triage override for crisis states
            if is_crisis:
                base_score = (base_score * 0.4) + (s_rule * 0.6)

            # Deduplication penalty for recently recommended items (Task 4)
            penalty = 0.0
            if exclude_recent and profile.has_recently_received(item.id):
                penalty = RECENT_RECOMMENDATION_PENALTY
                dedup_count += 1

            final_score = max(0.0, min(1.0, base_score - penalty))

            # Filter out low-relevance recommendations (Task 4)
            if final_score < self.relevance_threshold and not (is_crisis and item.is_crisis_resource):
                filtered_count += 1
                continue

            breakdown = {
                "semantic": s_sem,
                "emotion_similarity": s_emo,
                "content_intensity": s_cont,
                "user_preference": adjusted_pref,
                "collaborative": s_col,
                "trend_boost": t_boost,
                "feedback_modifier": feedback_mod,
                "rule_triage": s_rule,
                "penalty": penalty,
            }

            candidates.append({
                "item": item,
                "score": final_score,
                "breakdown": breakdown,
            })

        # Step 2: Dynamic Ranking (Task 4: Strictly sorted descending by composite score)
        candidates.sort(key=lambda c: c["score"], reverse=True)

        ranked_list: List[RankedRecommendation] = []
        for rank_idx, cand in enumerate(candidates[:top_k], start=1):
            item = cand["item"]
            score = cand["score"]
            breakdown = cand["breakdown"]

            # Task 8: Generate dynamic XAI explanation
            explanation = None
            if explain:
                explanation = self.explainer.explain(
                    item=item,
                    score=score,
                    score_breakdown=breakdown,
                    emotional_state=emotional_state,
                    user_profile=profile,
                    trend_report=trend_report,
                )

            ranked_rec = RankedRecommendation(
                rank=rank_idx,
                item_id=item.id,
                title=item.title,
                category=item.category,
                modality=item.modality,
                duration_minutes=item.duration_minutes,
                composite_score=score,
                score_breakdown=breakdown,
                instructions=item.instructions,
                description=item.description,
                is_crisis_resource=item.is_crisis_resource,
                explanation=explanation,
            )
            ranked_list.append(ranked_rec)

            # Record recommendation into user profile history
            profile.record_recommendation(item.id)

        # Record current mood into user longitudinal history (Task 6)
        profile.record_mood(
            dominant_emotion=emotional_state.dominant_emotion,
            intensity_score=emotional_state.intensity_score,
            polarity=emotional_state.polarity,
            severity=emotional_state.severity,
        )

        top_rec = ranked_list[0] if ranked_list else None
        ranking_order = [r.title for r in ranked_list]

        logger.info(
            "Generated %d ranked recommendations for user '%s' (Top: %s | Score: %.3f)",
            len(ranked_list),
            profile.user_id,
            top_rec.title if top_rec else "None",
            top_rec.composite_score if top_rec else 0.0,
        )

        return RecommendationResult(
            user_id=profile.user_id,
            top_recommendation=top_rec,
            recommendations=ranked_list,
            total_candidates_evaluated=len(items),
            filtered_low_relevance_count=filtered_count,
            deduplicated_count=dedup_count,
            ranking_order=ranking_order,
            trend_report=trend_report,
        )
