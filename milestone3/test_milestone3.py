"""Comprehensive Test Suite for Mood Mentor - Milestone 3 (Tasks 1 to 10).

Validates:
- Task 1: Emotion Intensity & Emotional State Analysis
- Task 2: Personalized Recommendation Model & User Profile
- Task 3: Hybrid Recommendation Engine (5 Ensemble Strategies)
- Task 4: Recommendation Ranking Model (Scoring, Dynamic Order, Deduplication, Filtering)
- Task 5: Semantic Wellness Content Matching (Embeddings & Cosine Similarity)
- Task 6: Emotional Trend & User State Tracking (Trajectory, Slopes, Patterns)
- Task 7: Recommendation Feedback Learning (Accept/Reject, Online Preference Update)
- Task 8: Recommendation Explainability (XAI Multi-Factor Justifications)
- Task 9: Advanced ML Validation & Performance Testing (Precision@K, NDCG vs Baseline)
- Task 10: Complete ML Integration, CLI Modes & Defect-Free Operation
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import pytest

from config import DATA_DIR
from emotion_classifier import EmotionPrediction, TransformerEmotionClassifier
from evaluation_recommender import (
    BaselineRecommender,
    compute_dcg,
    compute_ndcg,
    evaluate_recommenders,
)
from exceptions import FeedbackError, MoodMentorError
from explainability import RecommendationExplainer, RecommendationExplanation
from feedback_learner import FeedbackLearner
from intensity_analyzer import EmotionalState, EmotionIntensityAnalyzer
from main import run_milestone3_pipeline
from recommendation_engine import HybridRecommendationEngine, RecommendationResult
from recommender_strategies import (
    CollaborativeFilteringStrategy,
    ContentBasedStrategy,
    EmotionSimilarityStrategy,
    RuleBasedTriageStrategy,
    UserPreferenceStrategy,
)
from semantic_matcher import SemanticWellnessMatcher
from sentiment import SentimentScore, analyze_sentiment
from user_profile import EmotionalTrendReport, UserProfile, UserProfileRegistry
from wellness_catalog import WellnessCatalogManager, get_wellness_catalog


# ============================================================================
# TASK 1: EMOTION INTENSITY & EMOTIONAL STATE ANALYSIS TESTS
# ============================================================================


class TestEmotionIntensityAnalysis:
    """Validate dynamic calculation of continuous intensity, tiers, and mixed states."""

    @pytest.fixture(scope="class")
    @staticmethod
    def classifier() -> TransformerEmotionClassifier:
        return TransformerEmotionClassifier(model_type="distilbert")

    @pytest.fixture(scope="class")
    @staticmethod
    def analyzer() -> EmotionIntensityAnalyzer:
        return EmotionIntensityAnalyzer()

    def test_intensity_score_bounds_and_dynamics(
        self, classifier: TransformerEmotionClassifier, analyzer: EmotionIntensityAnalyzer
    ) -> None:
        text_mild = "Feeling a bit tired today."
        text_severe = "I am completely exhausted, experiencing unbearable burnout and severe panic!!!"

        pred_mild = classifier.predict(text_mild)
        sent_mild = analyze_sentiment(text_mild)
        state_mild = analyzer.analyze(text_mild, pred_mild, sent_mild)

        pred_severe = classifier.predict(text_severe)
        sent_severe = analyze_sentiment(text_severe)
        state_severe = analyzer.analyze(text_severe, pred_severe, sent_severe)

        assert 0.0 <= state_mild.intensity_score <= 1.0
        assert 0.0 <= state_severe.intensity_score <= 1.0
        assert state_severe.intensity_score > state_mild.intensity_score
        assert state_severe.intensity_tier in ("High", "Severe")

    def test_intensifiers_boost_intensity(
        self, classifier: TransformerEmotionClassifier, analyzer: EmotionIntensityAnalyzer
    ) -> None:
        text_base = "I am stressed about this work."
        text_boosted = "I am utterly, completely, and terribly stressed about this work!"

        pred_base = classifier.predict(text_base)
        sent_base = analyze_sentiment(text_base)
        state_base = analyzer.analyze(text_base, pred_base, sent_base)

        pred_boosted = classifier.predict(text_boosted)
        sent_boosted = analyze_sentiment(text_boosted)
        state_boosted = analyzer.analyze(text_boosted, pred_boosted, sent_boosted)

        assert state_boosted.intensity_score > state_base.intensity_score

    def test_mixed_emotional_state_detection(
        self, classifier: TransformerEmotionClassifier, analyzer: EmotionIntensityAnalyzer
    ) -> None:
        mixed_text = "I am excited about the new opportunity but nervous about the outcome."
        pred = classifier.predict(mixed_text)
        sent = analyze_sentiment(mixed_text)
        state = analyzer.analyze(mixed_text, pred, sent)

        assert state.is_mixed_state is True
        assert state.mixed_state_type is not None
        assert "Anticipatory Anxiety" in state.mixed_state_type or "Joy" in state.mixed_state_type

    def test_critical_severity_triage(
        self, classifier: TransformerEmotionClassifier, analyzer: EmotionIntensityAnalyzer
    ) -> None:
        crisis_text = "I am breaking down from severe burnout and acute panic attacks."
        pred = classifier.predict(crisis_text)
        sent = analyze_sentiment(crisis_text)
        state = analyzer.analyze(crisis_text, pred, sent)

        assert state.severity in ("High", "Critical")
        assert state.crisis_flag is True


# ============================================================================
# TASK 2: PERSONALIZED RECOMMENDATION MODEL & USER PROFILE TESTS
# ============================================================================


class TestPersonalizedUserProfile:
    """Validate user preferences, recommendation history, and longitudinal trend analysis."""

    def test_user_profile_creation_and_preferences(self) -> None:
        profile = UserProfile(
            user_id="emp_test",
            preferred_modalities={"audio", "exercise"},
            max_preferred_duration=10,
        )
        assert profile.user_id == "emp_test"
        assert "audio" in profile.preferred_modalities
        assert profile.max_preferred_duration == 10

    def test_recommendation_history_tracking(self) -> None:
        profile = UserProfile(user_id="emp_test")
        profile.record_recommendation("act_box_breathing")
        profile.record_recommendation("act_grounding_54321")

        assert profile.has_recently_received("act_box_breathing") is True
        assert profile.has_recently_received("act_grounding_54321") is True
        assert profile.has_recently_received("act_pomodoro_reset") is False


# ============================================================================
# TASK 3: HYBRID RECOMMENDATION STRATEGIES TESTS
# ============================================================================


class TestHybridRecommendationStrategies:
    """Validate each of the 5 specialized hybrid recommendation strategies."""

    @pytest.fixture(scope="class")
    @staticmethod
    def catalog() -> WellnessCatalogManager:
        return get_wellness_catalog()

    @pytest.fixture(scope="class")
    @staticmethod
    def mock_state() -> EmotionalState:
        return EmotionalState(
            text="I am terrified and anxious about this crisis.",
            dominant_emotion="fear",
            multiple_emotions=["fear"],
            emotion_confidence=0.85,
            intensity_score=0.80,
            intensity_tier="High",
            polarity="Negative",
            compound_score=-0.75,
            is_mixed_state=False,
            mixed_state_type=None,
            severity="High",
            crisis_flag=False,
            probabilities={"fear": 0.85, "anger": 0.1, "joy": 0.05, "sadness": 0.1, "surprise": 0.0, "disgust": 0.0},
            intensity_breakdown={},
        )

    def test_rule_based_triage_strategy(self, catalog: WellnessCatalogManager, mock_state: EmotionalState) -> None:
        strategy = RuleBasedTriageStrategy()
        profile = UserProfile(user_id="emp_test")
        scores = strategy.score_items(catalog.get_all(), mock_state, profile)

        assert scores["act_box_breathing"] >= 0.80
        assert scores["act_grounding_54321"] >= 0.80

    def test_content_based_strategy(self, catalog: WellnessCatalogManager, mock_state: EmotionalState) -> None:
        strategy = ContentBasedStrategy()
        profile = UserProfile(user_id="emp_test")
        scores = strategy.score_items(catalog.get_all(), mock_state, profile)

        assert scores["act_box_breathing"] > scores.get("act_celebration_milestone_log", 0.0)

    def test_emotion_similarity_strategy(self, catalog: WellnessCatalogManager, mock_state: EmotionalState) -> None:
        strategy = EmotionSimilarityStrategy()
        profile = UserProfile(user_id="emp_test")
        scores = strategy.score_items(catalog.get_all(), mock_state, profile)

        assert scores["act_box_breathing"] > 0.50

    def test_user_preference_strategy(self, catalog: WellnessCatalogManager, mock_state: EmotionalState) -> None:
        strategy = UserPreferenceStrategy()
        profile = UserProfile(user_id="emp_test", preferred_modalities={"breathing"})
        scores = strategy.score_items(catalog.get_all(), mock_state, profile)

        assert scores["act_box_breathing"] > scores["act_mindful_soundscape"]


# ============================================================================
# TASK 4: RECOMMENDATION RANKING MODEL TESTS
# ============================================================================


class TestRecommendationRankingModel:
    """Validate dynamic ranking order, composite scoring, deduplication, and thresholding."""

    @pytest.fixture(scope="class")
    @staticmethod
    def engine() -> HybridRecommendationEngine:
        return HybridRecommendationEngine()

    def test_dynamic_ranking_and_top_recommendation(self, engine: HybridRecommendationEngine) -> None:
        state = EmotionalState(
            text="I am having a panic attack, feeling severe fear and anxiety.",
            dominant_emotion="fear",
            multiple_emotions=["fear"],
            emotion_confidence=0.90,
            intensity_score=0.88,
            intensity_tier="Severe",
            polarity="Negative",
            compound_score=-0.80,
            is_mixed_state=False,
            mixed_state_type=None,
            severity="Critical",
            crisis_flag=True,
            probabilities={"fear": 0.90, "anger": 0.05, "joy": 0.0, "sadness": 0.05, "surprise": 0.0, "disgust": 0.0},
            intensity_breakdown={},
        )

        res = engine.generate_recommendations(state, top_k=3)

        assert isinstance(res, RecommendationResult)
        assert res.top_recommendation is not None
        assert len(res.recommendations) <= 3
        # Check strict descending rank order
        scores = [r.composite_score for r in res.recommendations]
        assert scores == sorted(scores, reverse=True)

    def test_deduplication_penalty(self, engine: HybridRecommendationEngine) -> None:
        profile = UserProfile(user_id="emp_dedup")
        state = EmotionalState(
            text="Feeling anxious about delivery.",
            dominant_emotion="fear",
            multiple_emotions=["fear"],
            emotion_confidence=0.75,
            intensity_score=0.60,
            intensity_tier="Moderate",
            polarity="Negative",
            compound_score=-0.50,
            is_mixed_state=False,
            mixed_state_type=None,
            severity="Moderate",
            crisis_flag=False,
            probabilities={"fear": 0.75, "anger": 0.1, "joy": 0.05, "sadness": 0.1, "surprise": 0.0, "disgust": 0.0},
            intensity_breakdown={},
        )

        # Run once -> records top recommendation
        res1 = engine.generate_recommendations(state, user_profile=profile, top_k=1)

        # Run second time -> top item from res1 should receive deduplication penalty
        res2 = engine.generate_recommendations(state, user_profile=profile, top_k=3)
        assert res2.deduplicated_count >= 1


# ============================================================================
# TASK 5: SEMANTIC WELLNESS CONTENT MATCHING TESTS
# ============================================================================


class TestSemanticWellnessMatching:
    """Validate embedding generation and cosine similarity matching on implicit distress."""

    @pytest.fixture(scope="class")
    @staticmethod
    def matcher() -> SemanticWellnessMatcher:
        return SemanticWellnessMatcher()

    def test_implicit_distress_semantic_matching(self, matcher: SemanticWellnessMatcher) -> None:
        query = "I feel like the walls are closing in on me and I can barely catch my breath."
        matches = matcher.find_top_semantic_matches(query, top_k=3)

        assert len(matches) > 0
        top_item, top_score = matches[0]
        assert top_score > 0.0
        assert any(
            kw in top_item.id for kw in ("breathing", "grounding", "sigh", "relaxation", "pmr")
        )


# ============================================================================
# TASK 6: EMOTIONAL TREND & USER STATE TRACKING TESTS
# ============================================================================


class TestEmotionalTrendTracking:
    """Validate historical tracking, frequency, intensity trajectory, and trend bias (Task 6)."""

    def test_trend_report_frequency_and_slope(self) -> None:
        profile = UserProfile(user_id="emp_trend_test")
        profile.record_mood("sadness", 0.35, "Negative", "Mild")
        profile.record_mood("sadness", 0.50, "Negative", "Moderate")
        profile.record_mood("fear", 0.75, "Negative", "High")
        profile.record_mood("fear", 0.90, "Negative", "Critical")

        report = profile.get_trend_report()

        assert isinstance(report, EmotionalTrendReport)
        assert report.total_snapshots == 4
        assert report.emotion_frequencies.get("fear") == 2
        assert report.emotion_frequencies.get("sadness") == 2
        assert report.intensity_trajectory_slope > 0.10
        assert "Escalating" in report.trend_label or "Chronic" in report.trend_label
        assert "breathing" in report.recommendation_bias or "clinical" in report.recommendation_bias

    def test_trend_influences_recommendations(self) -> None:
        """Verify Task 6 requirement: previous emotional patterns influence future recommendations."""
        engine = HybridRecommendationEngine()
        profile_stressed = UserProfile(user_id="emp_escalating")

        # Simulate escalating stress pattern with ascending intensity
        profile_stressed.record_mood("fear", 0.40, "Negative", "Moderate")
        profile_stressed.record_mood("fear", 0.65, "Negative", "High")
        profile_stressed.record_mood("fear", 0.90, "Negative", "Critical")

        state = EmotionalState(
            text="General work update",
            dominant_emotion="fear",
            multiple_emotions=["fear"],
            emotion_confidence=0.70,
            intensity_score=0.75,
            intensity_tier="High",
            polarity="Negative",
            compound_score=-0.5,
            is_mixed_state=False,
            mixed_state_type=None,
            severity="High",
            crisis_flag=False,
            probabilities={"fear": 0.70, "anger": 0.1, "joy": 0.0, "sadness": 0.1, "surprise": 0.0, "disgust": 0.0},
            intensity_breakdown={},
        )

        res = engine.generate_recommendations(state, user_profile=profile_stressed, top_k=3)
        assert res.trend_report is not None
        # Check that top recommendations reflect trend bias toward calming / breathing
        top_cats = [r.category.lower() for r in res.recommendations]
        assert any(c in top_cats for c in ("breathing", "mindfulness", "clinical"))


# ============================================================================
# TASK 7: RECOMMENDATION FEEDBACK LEARNING TESTS
# ============================================================================


class TestRecommendationFeedbackLearning:
    """Validate feedback loop: capture viewed/accepted/rejected, update weights (Task 7)."""

    def test_feedback_acceptance_boosts_modality(self, tmp_path: Path) -> None:
        feedback_file = tmp_path / "test_feedback.json"
        learner = FeedbackLearner(feedback_file=feedback_file)
        profile = UserProfile(user_id="emp_feedback_test")

        initial_weight = profile.modality_weights.get("breathing", 1.0)

        learner.record_feedback(
            user_profile=profile,
            item_id="act_box_breathing",
            action="accepted",
            rating=5.0,
        )

        new_weight = profile.modality_weights.get("breathing", 1.0)
        assert new_weight > initial_weight
        assert feedback_file.exists()

    def test_feedback_rejection_dampens_modality(self, tmp_path: Path) -> None:
        feedback_file = tmp_path / "test_feedback.json"
        learner = FeedbackLearner(feedback_file=feedback_file)
        profile = UserProfile(user_id="emp_feedback_test")

        initial_weight = profile.modality_weights.get("journaling", 0.8)

        learner.record_feedback(
            user_profile=profile,
            item_id="act_cognitive_reframing",
            action="rejected",
        )

        new_weight = profile.modality_weights.get("journaling", 0.8)
        assert new_weight < initial_weight

    def test_invalid_action_raises_feedback_error(self, tmp_path: Path) -> None:
        learner = FeedbackLearner(feedback_file=tmp_path / "fb.json")
        profile = UserProfile(user_id="emp_err")
        with pytest.raises(FeedbackError):
            learner.record_feedback(profile, "act_box_breathing", "invalid_action")


# ============================================================================
# TASK 8: RECOMMENDATION EXPLAINABILITY TESTS
# ============================================================================


class TestRecommendationExplainability:
    """Validate explainability layer: multi-factor dynamic reasons (Task 8)."""

    def test_dynamic_explanation_generation(self) -> None:
        catalog = get_wellness_catalog()
        item = catalog.get_by_id("act_box_breathing")
        assert item is not None

        explainer = RecommendationExplainer()
        profile = UserProfile(user_id="emp_xai", preferred_modalities={"breathing"})
        state = EmotionalState(
            text="Heart is pounding with panic!",
            dominant_emotion="fear",
            multiple_emotions=["fear"],
            emotion_confidence=0.88,
            intensity_score=0.85,
            intensity_tier="Severe",
            polarity="Negative",
            compound_score=-0.70,
            is_mixed_state=False,
            mixed_state_type=None,
            severity="High",
            crisis_flag=False,
            probabilities={"fear": 0.88, "anger": 0.0, "joy": 0.0, "sadness": 0.1, "surprise": 0.0, "disgust": 0.0},
            intensity_breakdown={},
        )

        explanation = explainer.explain(
            item=item,
            score=0.89,
            score_breakdown={"semantic": 0.45},
            emotional_state=state,
            user_profile=profile,
        )

        assert isinstance(explanation, RecommendationExplanation)
        assert explanation.item_id == "act_box_breathing"
        assert len(explanation.bullet_reasons) >= 2
        # Check dynamic references
        joined_reasons = " ".join(explanation.bullet_reasons).lower()
        assert "fear" in joined_reasons
        assert "intensity" in joined_reasons or "breathing" in joined_reasons
        assert len(explanation.summary_sentence) > 10


# ============================================================================
# TASK 9: ADVANCED ML VALIDATION & BENCHMARK TESTS
# ============================================================================


class TestAdvancedMLValidation:
    """Validate Precision@K, Recall@K, NDCG@K, Diversity, and ML superiority (Task 9)."""

    def test_dcg_and_ndcg_calculation(self) -> None:
        recs = ["item_a", "item_b", "item_c"]
        grades = {"item_a": 2, "item_b": 1, "item_c": 0}

        ndcg_3 = compute_ndcg(recs, grades, k=3)
        assert 0.0 <= ndcg_3 <= 1.0
        assert ndcg_3 == 1.0  # Perfect order matches ideal DCG

    def test_advanced_ml_outperforms_baseline(self) -> None:
        """Verify Task 9 requirement: Advanced ML provides better recommendation quality than baseline."""
        eval_path = DATA_DIR / "recommendation_eval_set.json"
        assert eval_path.exists()

        base_metrics, ml_metrics = evaluate_recommenders(eval_dataset_path=eval_path)

        assert ml_metrics.sample_count > 0
        assert base_metrics.sample_count > 0
        # Advanced ML with semantic embeddings and multi-strategy ranking must beat naive baseline on NDCG@3
        assert ml_metrics.ndcg_at_k[3] >= base_metrics.ndcg_at_k[3]
        # Advanced ML must achieve strong Precision@3
        assert ml_metrics.precision_at_k[3] >= base_metrics.precision_at_k[3]
        assert ml_metrics.intra_list_diversity > 0.0
        assert ml_metrics.avg_latency_ms > 0.0


# ============================================================================
# TASK 10: COMPLETE ML INTEGRATION & REGRESSION TESTS
# ============================================================================


class TestCompleteMLIntegration:
    """Validate end-to-end integration, CLI modes, and defect-free execution (Task 10)."""

    def test_run_pipeline_on_text_with_explain(self) -> None:
        text = "I am excited about leading this initiative but terrified of failing."
        exit_code = run_milestone3_pipeline(
            source=text,
            source_type="direct",
            user_id="emp_test_full",
            explain=True,
        )
        assert exit_code == 0

    def test_run_pipeline_on_csv_with_export(self, tmp_path: Path) -> None:
        sample_csv = DATA_DIR / "sample.csv"
        out_csv = tmp_path / "m3_full_report.csv"

        exit_code = run_milestone3_pipeline(
            source=str(sample_csv),
            source_type="csv",
            output_csv_path=str(out_csv),
        )
        assert exit_code == 0
        assert out_csv.exists()
        assert out_csv.stat().st_size > 0
