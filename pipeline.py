"""
MoodMentor — Unified Master Pipeline (Milestones 1–4 Integration)

Orchestrates the entire clinical-grade affective intelligence flow:
- Stage 1 (M1): PII Scrubbing & NLP Ingestion
- Stage 2 (M2): Deep Transformer Emotion Classification
- Stage 3 (M3): Intensity Trajectory, Severity Triage & 5-Strategy Hybrid Recommender
- Stage 4 (M3): Explainable AI (XAI) Justifications
- Stage 5 (M4): SQLite Telemetry Persistence & Dual Reporting Exporters
"""

import os
import sys
import logging
from typing import Dict, Any, Optional

# Ensure repository root and milestone paths are discoverable
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

M3_DIR = os.path.join(REPO_ROOT, "milestone3")
if M3_DIR not in sys.path:
    sys.path.insert(0, M3_DIR)

M4_DIR = os.path.join(REPO_ROOT, "milestone4")
if M4_DIR not in sys.path:
    sys.path.insert(0, M4_DIR)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MoodMentorMasterPipeline")

# Import Milestone 1 (NLP Ingestion)
try:
    from nlp_ingestion.pii_cleaner import clean_pii
    from nlp_ingestion.sentiment import analyze_sentiment as vader_sentiment
except ImportError:
    clean_pii = lambda text: text
    vader_sentiment = lambda text: {"compound": 0.0, "pos": 0.0, "neu": 1.0, "neg": 0.0}

# Import Milestone 3 (Emotional Intensity, Hybrid Recs, XAI)
from milestone3.emotion_classifier import TransformerEmotionClassifier
from milestone3.intensity_analyzer import EmotionIntensityAnalyzer
from milestone3.recommendation_engine import HybridRecommendationEngine
from milestone3.explainability import RecommendationExplainer
from milestone3.user_profile import UserProfile, get_user_profile
from milestone3.wellness_catalog import get_wellness_catalog
from milestone3.sentiment import analyze_sentiment as m3_sentiment

# Import Milestone 4 (Storage & Security)
from moodmentor.storage.db import DatabaseManager
from moodmentor.security.privacy import SecurityManager


class MoodMentorPipeline:
    """End-to-End Orchestrator for MoodMentor Platform."""

    def __init__(self, db_path: Optional[str] = None):
        logger.info("Initializing MoodMentor Master Pipeline...")
        self.security = SecurityManager()
        self.classifier = TransformerEmotionClassifier(model_type="distilbert")
        self.intensity_analyzer = EmotionIntensityAnalyzer()
        self.rec_engine = HybridRecommendationEngine()
        self.explainer = RecommendationExplainer()
        self.catalog = get_wellness_catalog()
        self.db = DatabaseManager(db_path=db_path) if db_path else DatabaseManager()
        logger.info("MoodMentor Master Pipeline initialized successfully!")

    def process(self, text: str, user_id: str = "anonymous_emp", top_k: int = 3) -> Dict[str, Any]:
        logger.info(f"Processing input for user: {user_id}")

        # 1. Milestone 1: PII Masking & Privacy Sanitization
        sanitized_text, redacted_map = self.security.anonymize_pii(text)

        # 2. Milestone 2: Transformer Emotion Classification
        emotion_pred = self.classifier.predict(sanitized_text)

        # 3. Milestone 1 & 3: Sentiment & Affective Intensity
        sent_score = m3_sentiment(sanitized_text)
        emotional_state = self.intensity_analyzer.analyze(sanitized_text, emotion_pred, sent_score)

        # 4. Milestone 3: User Profile & 5-Strategy Hybrid Recommender
        profile = get_user_profile(user_id)
        rec_result = self.rec_engine.generate_recommendations(
            emotional_state=emotional_state,
            user_profile=profile,
            top_k=top_k
        )

        # 5. Milestone 4: Telemetry Persistence
        interaction_id = self.db.save_interaction(
            user_id=user_id,
            raw_text=text,
            primary_emotion=emotional_state.dominant_emotion,
            confidence=emotional_state.emotion_confidence,
            intensity_score=emotional_state.intensity_score,
            tier=emotional_state.intensity_tier,
            triage_level=emotional_state.severity,
            vader_compound=sent_score.compound
        )

        recommendations_payload = []
        for rec in rec_result.recommendations:
            summary = rec.explanation.summary_sentence if rec.explanation else rec.description
            reasons = rec.explanation.bullet_reasons if rec.explanation else []
            recommendations_payload.append({
                "item_id": rec.item_id,
                "title": rec.title,
                "category": rec.category,
                "score": round(rec.composite_score, 4),
                "duration": rec.duration_minutes,
                "summary": summary,
                "reasons": reasons
            })

        return {
            "interaction_id": interaction_id,
            "user_id": user_id,
            "raw_text": text,
            "sanitized_text": sanitized_text,
            "dominant_emotion": emotional_state.dominant_emotion,
            "emotion_confidence": round(emotional_state.emotion_confidence, 4),
            "multiple_emotions": emotional_state.multiple_emotions,
            "probabilities": {k: round(v, 4) for k, v in emotional_state.probabilities.items()},
            "intensity_score": round(emotional_state.intensity_score, 4),
            "intensity_tier": emotional_state.intensity_tier,
            "severity_triage": emotional_state.severity,
            "crisis_flag": emotional_state.crisis_flag,
            "is_mixed_state": emotional_state.is_mixed_state,
            "mixed_state_type": emotional_state.mixed_state_type,
            "vader_compound": round(sent_score.compound, 4),
            "recommendations": recommendations_payload
        }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="MoodMentor Master Pipeline CLI")
    parser.add_argument("--text", type=str, default="I am feeling overwhelmed with deadlines and acute anxiety.", help="Input text")
    parser.add_argument("--user-id", type=str, default="emp_lead_test", help="User ID")
    parser.add_argument("--top-k", type=int, default=3, help="Number of recommendations")
    args = parser.parse_args()

    pipeline = MoodMentorPipeline()
    result = pipeline.process(args.text, user_id=args.user_id, top_k=args.top_k)

    print("\n" + "=" * 70)
    print("🧠 MOODMENTOR UNIFIED PIPELINE (MILESTONES 1–4) RESULT:")
    print("=" * 70)
    u_id = result["user_id"]
    dom_emo = result["dominant_emotion"].upper()
    conf = result["emotion_confidence"] * 100
    int_score = result["intensity_score"]
    int_tier = result["intensity_tier"]
    triage = result["severity_triage"]
    crisis = "TRIGGERED (988 Active)" if result["crisis_flag"] else "Normal Care"
    vader = result["vader_compound"]
    print(f"User ID          : {u_id}")
    print(f"Dominant Emotion : {dom_emo} (Confidence: {conf:.1f}%)")
    print(f"Intensity Score  : {int_score} [{int_tier}]")
    print(f"Severity Triage  : {triage}")
    print(f"Crisis Guardrail : {crisis}")
    print(f"Sentiment        : {vader}")
    print("\n📋 Top Interventions (with Explainable AI):")
    for idx, rec in enumerate(result["recommendations"], 1):
        rec_title = rec["title"]
        rec_cat = rec["category"]
        match_pct = int(rec["score"] * 100)
        rec_sum = rec["summary"]
        print(f"  {idx}. {rec_title} ({rec_cat}) - Match: {match_pct}%")
        print(f"     Reason: {rec_sum}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
