"""
MoodMentor — Milestone 3 Package
AI Emotional State, 5-Strategy Hybrid Recommendations, and Explainable AI (XAI).
"""

import os
import sys

_pkg_dir = os.path.dirname(os.path.abspath(__file__))
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

from emotion_classifier import TransformerEmotionClassifier, EmotionPrediction
from intensity_analyzer import EmotionIntensityAnalyzer, EmotionalState
from recommendation_engine import HybridRecommendationEngine, RecommendationResult
from recommender_strategies import (
    RuleBasedTriageStrategy,
    ContentBasedStrategy,
    EmotionSimilarityStrategy,
    CollaborativeFilteringStrategy,
    UserPreferenceStrategy
)
from semantic_matcher import SemanticWellnessMatcher
from explainability import RecommendationExplainer, RecommendationExplanation
from user_profile import UserProfile, UserProfileRegistry, EmotionalTrendReport
from feedback_learner import FeedbackLearner
from wellness_catalog import WellnessCatalogManager, WellnessIntervention, get_wellness_catalog
from sentiment import analyze_sentiment, SentimentScore

__all__ = [
    "TransformerEmotionClassifier",
    "EmotionPrediction",
    "EmotionIntensityAnalyzer",
    "EmotionalState",
    "HybridRecommendationEngine",
    "RecommendationResult",
    "RuleBasedTriageStrategy",
    "ContentBasedStrategy",
    "EmotionSimilarityStrategy",
    "CollaborativeFilteringStrategy",
    "UserPreferenceStrategy",
    "SemanticWellnessMatcher",
    "RecommendationExplainer",
    "RecommendationExplanation",
    "UserProfile",
    "UserProfileRegistry",
    "EmotionalTrendReport",
    "FeedbackLearner",
    "WellnessCatalogManager",
    "WellnessIntervention",
    "get_wellness_catalog",
    "analyze_sentiment",
    "SentimentScore"
]
